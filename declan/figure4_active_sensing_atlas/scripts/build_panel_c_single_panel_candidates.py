"""Build single-panel promotion candidates for Figure 4C.

Each PNG is a candidate for the one promoted Panel C. Existing observer
subpanels are reused where they already express a single claim; one focused
matched-static variant is redrawn from the cached values with only the
empirical-prior joint observer.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
PANEL_C = ATLAS / "figures" / "panel_C"
OUT_DIR = PANEL_C / "promotion_candidates"

COLORS = {
    "known": "#242a2f",
    "zero": "#8e9aa6",
    "empirical": "#2f8f6a",
    "ou": "#3366aa",
    "matched_static_response": "#2f8f6a",
    "hard_negative_structure": "#8064a2",
    "muted": "#65717a",
}


@dataclass(frozen=True)
class Candidate:
    slug: str
    title: str
    source_png: str | None
    recommendation: str
    boundary: str


CANDIDATES = (
    Candidate(
        slug="4C_candidate_1_matched_static_rescue_current",
        title="Matched-static rescue",
        source_png="C3_matched_static_rescue.png",
        recommendation="Faithful to the current contract: zero-eye collapses, latent-eye joint recovers much of the known-eye gap, and both empirical and OU priors are visible.",
        boundary="OU here is a trajectory prior, not the Panel B motion-family control, but it may distract as the main visual.",
    ),
    Candidate(
        slug="4C_candidate_2_empirical_prior_rescue_clean",
        title="Empirical-prior rescue",
        source_png=None,
        recommendation="Selected provisional 4C: empirical-prior marginalization rescues image identity under matched-static distractors.",
        boundary="Omits the OU-prior robustness check; keep that in caption, supplement, or companion prose.",
    ),
    Candidate(
        slug="4C_candidate_3_accuracy_ordering_context",
        title="Accuracy ordering context",
        source_png="C2_accuracy_ordering.png",
        recommendation="Best if Panel C must show that the ordering holds across hard-negative and matched-static candidate sets.",
        boundary="Wider and denser; the matched-static rescue can be visually less immediate.",
    ),
    Candidate(
        slug="4C_candidate_4_scale_gap_guardrail",
        title="Scale rescue guardrail",
        source_png="C5_scale_gap_guardrail.png",
        recommendation="Most caveat-forward option: shows rescue grows where zero-eye failure is larger.",
        boundary="Too indirect for the first C panel unless the rescue claim is already stated elsewhere.",
    ),
    Candidate(
        slug="4C_candidate_5_joint_feature_posterior_recovery",
        title="Joint feature-posterior recovery",
        source_png=None,
        recommendation="Selected revised 4C: latent-eye joint posterior recovers local feature encoding above zero-eye without known eye position.",
        boundary="Feature-posterior endpoint, not image-identity accuracy; parallel-versus-orthogonal ordering is scale-dependent.",
    ),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _copy_source_candidate(candidate: Candidate) -> tuple[Path, pd.DataFrame | None]:
    assert candidate.source_png is not None
    source = PANEL_C / candidate.source_png
    if not source.exists():
        raise FileNotFoundError(source)
    out = OUT_DIR / f"{candidate.slug}.png"
    shutil.copy2(source, out)
    pdf = source.with_suffix(".pdf")
    if pdf.exists():
        shutil.copy2(pdf, out.with_suffix(".pdf"))

    values_path = {
        "C3_matched_static_rescue.png": "panel_C_matched_static_rescue_values.csv",
        "C2_accuracy_ordering.png": "panel_C_accuracy_ordering_values.csv",
        "C5_scale_gap_guardrail.png": "panel_C_scale_gap_guardrail_values.csv",
    }.get(candidate.source_png)
    values = None
    if values_path:
        values = pd.read_csv(PANEL_C / values_path).assign(candidate=candidate.slug)
    return out, values


def _build_empirical_prior_rescue() -> tuple[Path, pd.DataFrame]:
    values = pd.read_csv(PANEL_C / "panel_C_matched_static_rescue_values.csv")
    block = values[values["observer"].isin(["zero eye", "joint empirical", "known eye"])].copy()
    order = ["zero eye", "joint empirical", "known eye"]
    block["observer"] = pd.Categorical(block["observer"], categories=order, ordered=True)
    block = block.sort_values("observer")

    zero = float(block[block["observer"] == "zero eye"]["accuracy"].iloc[0])
    joint = float(block[block["observer"] == "joint empirical"]["accuracy"].iloc[0])
    known = float(block[block["observer"] == "known eye"]["accuracy"].iloc[0])
    recovery = (joint - zero) / (known - zero)

    fig, ax = plt.subplots(figsize=(3.7, 2.9), constrained_layout=True)
    x = np.arange(len(block))
    colors = [COLORS["zero"], COLORS["empirical"], COLORS["known"]]
    labels = ["zero eye", "latent-eye joint\nempirical prior", "known eye"]
    ax.bar(x, block["accuracy"].to_numpy(dtype=float), color=colors, width=0.64)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("image-identification accuracy")
    ax.set_title("Matched-static rescue at 1.0x")
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    _clean_axis(ax)

    for idx, row in enumerate(block.itertuples()):
        ax.text(idx, float(row.accuracy) + 0.025, f"{row.accuracy:.3f}", ha="center", va="bottom", fontsize=7.6)
    ax.text(
        0.03,
        0.96,
        f"{recovery:.0%} of known-zero gap recovered",
        ha="left",
        va="top",
        fontsize=7.8,
        color="#303840",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#d8dde3", "linewidth": 0.7},
    )

    out = OUT_DIR / "4C_candidate_2_empirical_prior_rescue_clean.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return out, block.assign(
        candidate="4C_candidate_2_empirical_prior_rescue_clean",
        recovery_fraction=recovery,
    )


def _feature_posterior_candidate() -> tuple[Path, pd.DataFrame]:
    image = OUT_DIR / "4C_candidate_5_joint_feature_posterior_recovery.png"
    values = OUT_DIR / "4C_candidate_5_joint_feature_posterior_recovery_values.csv"
    if not image.exists() or not values.exists():
        raise FileNotFoundError(
            "Run scripts/build_joint_feature_posterior_panel.py before rebuilding the 4C candidate sheet."
        )
    return image, pd.read_csv(values).assign(candidate="4C_candidate_5_joint_feature_posterior_recovery")


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

    rows = (len(CANDIDATES) + 1) // 2
    width = 2600
    margin, gap = 62, 36
    thumb_w, thumb_h = 1210, 650
    height = 215 + rows * (thumb_h + 330) + 80
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 45), "Figure 4C Single-Panel Promotion Candidates", font=font(46, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Choose the one panel that should carry latent trajectory inference.",
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

    out = OUT_DIR / "4C_single_panel_candidate_sheet.png"
    sheet.save(out, optimize=True)
    return out


def _write_readme(paths: list[Path]) -> None:
    readme = OUT_DIR / "README.md"
    lines = [
        "# Figure 4C Single-Panel Promotion Candidates",
        "",
        "Status: candidate 5 selected provisionally for the promoted joint feature-posterior panel.",
        "",
        "![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_single_panel_candidate_sheet.png)",
        "",
        "## Recommendation",
        "",
        "Selected provisional 4C: candidate 5, `4C_candidate_5_joint_feature_posterior_recovery.png`. It reflects the newer joint-model endpoint: zero-eye feature recovery falls as motion scale grows, while latent-eye joint inference remains stable without being given the measured eye trace. Candidate 2 remains useful historical image-identity context, and candidates 1, 3, and 4 remain guardrails/supporting views.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `4C_single_panel_candidate_sheet.png`",
            "- `4C_single_panel_candidate_values.csv`",
            "",
            "## Claim Boundary",
            "",
            "These panels use an exact finite trajectory-table/posterior observer. The promoted 4C endpoint is absolute feature recovery under latent eye position, not image-identity accuracy or a gain normalized to a moving zero-eye baseline. Zero-eye scores the moved observation under a zero-eye-motion assumption; latent-eye joint hides the measured eye trace and marginalizes over candidate trajectories. It does not show that the animal computes this posterior or that the posterior identifies the true eye trajectory.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    value_rows: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        if candidate.slug == "4C_candidate_2_empirical_prior_rescue_clean":
            path, values = _build_empirical_prior_rescue()
        elif candidate.slug == "4C_candidate_5_joint_feature_posterior_recovery":
            path, values = _feature_posterior_candidate()
        else:
            path, values = _copy_source_candidate(candidate)
        paths.append(path)
        if values is not None:
            value_rows.append(values)

    pd.concat(value_rows, ignore_index=True, sort=False).to_csv(OUT_DIR / "4C_single_panel_candidate_values.csv", index=False)
    sheet = _make_contact_sheet(paths)
    _write_readme(paths)
    for path in paths + [sheet, OUT_DIR / "README.md", OUT_DIR / "4C_single_panel_candidate_values.csv"]:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()

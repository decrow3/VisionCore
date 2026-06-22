"""Build single-panel promotion candidates for Figure 4E.

Each PNG is a candidate for the one promoted Panel E behavior-geometry bridge.
Existing behavior subpanels are reused directly so the review surface compares
the current claim surfaces without creating a composite.
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
PANEL_E = ATLAS / "figures" / "panel_E"
OUT_DIR = PANEL_E / "promotion_candidates"
WINDOWS_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)

COLORS = {
    "edge": "#244f7a",
    "count": "#dfe3e8",
    "all": "#8e9aa6",
    "reliable": "#2f8f6a",
    "high": "#3366aa",
    "grid": "#d8dde3",
}


@dataclass(frozen=True)
class Candidate:
    slug: str
    title: str
    source_png: str | None
    values_csv: str | None
    recommendation: str
    boundary: str


CANDIDATES = (
    Candidate(
        slug="4E_candidate_1_alignment_strength",
        title="Alignment strength",
        source_png="E2_behavior_alignment_strength.png",
        values_csv="panel_E_alignment_strength_values.csv",
        recommendation="Cleanest statistical headline: free-viewing FEM axes align modestly but reliably with local edges, strongest for high-confidence windows.",
        boundary="Abstract effect-size view; endpoint/null diagnostics should remain in caption or supplement.",
    ),
    Candidate(
        slug="4E_candidate_2_parallel_zone_enrichment",
        title="Parallel-zone enrichment",
        source_png="E3_parallel_zone_enrichment.png",
        values_csv="panel_E_endpoint_enrichment_values.csv",
        recommendation="Most intuitive behavior read: endpoint directions are enriched near edge-parallel and depleted near orthogonal.",
        boundary="Uses binned endpoint zones; keep transformed-null diagnostic visible elsewhere.",
    ),
    Candidate(
        slug="4E_candidate_3a_image_coherence_focus",
        title="Image-coherence focus",
        source_png=None,
        values_csv=None,
        recommendation="Carries the 3A message directly: FEM-edge alignment rises when the local image orientation is coherent.",
        boundary="Confidence-gated read; pair with all-window/reliable values or endpoint-null guardrail in caption.",
    ),
    Candidate(
        slug="4E_candidate_3b_fem_anisotropy_focus",
        title="FEM-anisotropy focus",
        source_png=None,
        values_csv=None,
        recommendation="Companion reliability view: alignment is stronger when the measured FEM cloud is anisotropic.",
        boundary="More about measurement reliability than image geometry; use if 3A needs a paired robustness check.",
    ),
    Candidate(
        slug="4E_candidate_3c_polar_alignment_rose",
        title="Polar alignment rose",
        source_png=None,
        values_csv=None,
        recommendation="Polar option: high-confidence drift axes concentrate around the edge-parallel axis relative to a uniform axial baseline.",
        boundary="More intuitive for directionality, but less quantitative than 3A or the endpoint/null diagnostic.",
    ),
    Candidate(
        slug="4E_candidate_5_confidence_dependence_full",
        title="Full confidence diagnostic",
        source_png="E7_confidence_signed_delta_diagnostic.png",
        values_csv="panel_E_alignment_strength_values.csv",
        recommendation="Full source diagnostic for confidence dependence, including 3A, 3B, confidence grid, and signed-delta distribution.",
        boundary="Dense diagnostic; better as support unless the main claim needs reliability dependence.",
    ),
    Candidate(
        slug="4E_candidate_6_endpoint_null_diagnostic",
        title="Endpoint/null diagnostic",
        source_png="E8_endpoint_null_diagnostic.png",
        values_csv="panel_E_endpoint_enrichment_values.csv",
        recommendation="Most rigorous endpoint-zone guardrail: shows enrichment relative to the uniform-angle transformed null.",
        boundary="Dense provenance panel; less immediate as the single behavior headline.",
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


def _confidence_curve(metric_col: str, *, title: str, xlabel: str, out_stem: str) -> tuple[Path, pd.DataFrame]:
    windows = pd.read_csv(WINDOWS_CSV)
    work = windows[windows["image_feature_ok"].astype(bool)].copy()
    work[metric_col] = pd.to_numeric(work[metric_col])
    work["drift_edge_cos2"] = pd.to_numeric(work["drift_edge_cos2"])
    work = work[np.isfinite(work[metric_col]) & np.isfinite(work["drift_edge_cos2"])].copy()

    bins = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, float | int | str]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        if hi == bins[-1]:
            block = work[(work[metric_col] >= lo) & (work[metric_col] <= hi)].copy()
        else:
            block = work[(work[metric_col] >= lo) & (work[metric_col] < hi)].copy()
        if block.empty:
            continue
        values = block["drift_edge_cos2"].to_numpy(dtype=float)
        mean = float(values.mean())
        sem95 = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "metric": metric_col,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "bin_center": float((lo + hi) / 2.0),
                "mean_edge_alignment_index": mean,
                "ci95_low": mean - sem95,
                "ci95_high": mean + sem95,
                "n_windows": int(len(block)),
                "n_sessions": int(block["session"].nunique()),
            }
        )

    values = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    count_ax = ax.twinx()
    ax.set_zorder(count_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    count_ax.bar(
        values["bin_center"],
        values["n_windows"],
        width=0.075,
        color=COLORS["count"],
        edgecolor="none",
        zorder=0,
        label="window count",
    )
    count_ax.set_ylabel("window count")
    count_ax.tick_params(axis="y", labelsize=7.0, colors="#6f7a83")
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_color("#c8ced4")

    y = values["mean_edge_alignment_index"].to_numpy(dtype=float)
    lo = values["ci95_low"].to_numpy(dtype=float)
    hi = values["ci95_high"].to_numpy(dtype=float)
    ax.errorbar(
        values["bin_center"],
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color=COLORS["edge"],
        marker="o",
        lw=1.9,
        capsize=0,
        zorder=3,
    )
    ax.axhline(0.0, color="#242a2f", lw=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.04, 0.36)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("edge-following alignment")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.set_title(title)
    _clean_axis(ax)

    out = OUT_DIR / f"{out_stem}.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out, values


def _signed_axial_delta_deg(delta: pd.Series) -> pd.Series:
    return (pd.to_numeric(delta) + 90.0) % 180.0 - 90.0


def _polar_alignment_rose() -> tuple[Path, pd.DataFrame]:
    windows = pd.read_csv(WINDOWS_CSV)
    work = windows[windows["image_feature_ok"].astype(bool)].copy()
    work["signed_delta_deg"] = _signed_axial_delta_deg(work["drift_edge_delta_deg"])
    work["abs_delta_deg"] = work["signed_delta_deg"].abs()
    work = work[np.isfinite(work["signed_delta_deg"])].copy()

    subsets = [
        ("All windows", work, COLORS["all"]),
        (
            "High confidence",
            work[(work["image_orientation_coherence"].astype(float) >= 0.50) & (work["anisotropy"].astype(float) >= 0.50)],
            COLORS["high"],
        ),
    ]
    bins = np.linspace(0.0, 180.0, 19)
    centers = (bins[:-1] + bins[1:]) / 2.0
    rows: list[dict[str, float | int | str]] = []

    fig, ax = plt.subplots(figsize=(3.35, 3.35), subplot_kw={"projection": "polar"}, constrained_layout=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_full = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(theta_full, np.ones_like(theta_full), color="#242a2f", lw=0.8, linestyle="--", label="uniform")

    for label, subset, color in subsets:
        axial = (subset["signed_delta_deg"].to_numpy(dtype=float) + 180.0) % 180.0
        counts, _ = np.histogram(axial, bins=bins)
        expected = len(axial) / (len(bins) - 1)
        ratio = counts.astype(float) / expected if expected > 0 else counts.astype(float)
        theta_deg = np.concatenate([centers, centers + 180.0, [centers[0] + 360.0]])
        ratio_full = np.concatenate([ratio, ratio, [ratio[0]]])
        theta = np.deg2rad(theta_deg)
        ax.plot(theta, ratio_full, color=color, lw=2.0, marker="o", markersize=3.2, label=label)
        ax.fill(theta, ratio_full, color=color, alpha=0.08)
        for center, count, value in zip(centers, counts, ratio, strict=True):
            rows.append(
                {
                    "subset": label,
                    "theta_center_deg": float(center),
                    "n_windows": int(len(axial)),
                    "bin_count": int(count),
                    "observed_expected_ratio": float(value),
                }
            )

    ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
    ax.set_rlim(0.0, 2.8)
    ax.set_rticks([1.0, 2.0])
    ax.set_rlabel_position(135)
    ax.grid(color=COLORS["grid"], lw=0.8)
    ax.set_title("Drift axis relative to local edge", pad=18)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.0)

    out = OUT_DIR / "4E_candidate_3c_polar_alignment_rose.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out, pd.DataFrame(rows)


def _copy_candidate(candidate: Candidate) -> tuple[Path, pd.DataFrame]:
    if candidate.slug == "4E_candidate_3a_image_coherence_focus":
        path, values = _confidence_curve(
            "image_orientation_coherence",
            title="Eye drift follows clearer local edges",
            xlabel="local edge coherence",
            out_stem=candidate.slug,
        )
        return path, values.assign(candidate=candidate.slug)
    if candidate.slug == "4E_candidate_3b_fem_anisotropy_focus":
        path, values = _confidence_curve(
            "anisotropy",
            title="Alignment rises with FEM anisotropy",
            xlabel="FEM anisotropy",
            out_stem=candidate.slug,
        )
        return path, values.assign(candidate=candidate.slug)
    if candidate.slug == "4E_candidate_3c_polar_alignment_rose":
        path, values = _polar_alignment_rose()
        return path, values.assign(candidate=candidate.slug)

    assert candidate.source_png is not None
    source = PANEL_E / candidate.source_png
    if not source.exists():
        raise FileNotFoundError(source)
    out = OUT_DIR / f"{candidate.slug}.png"
    shutil.copy2(source, out)
    pdf = source.with_suffix(".pdf")
    if pdf.exists():
        shutil.copy2(pdf, out.with_suffix(".pdf"))

    if candidate.values_csv is None:
        values = pd.DataFrame([{"candidate": candidate.slug}])
    else:
        values = pd.read_csv(PANEL_E / candidate.values_csv).assign(candidate=candidate.slug)
    return out, values


def _make_contact_sheet(paths: list[Path]) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def display_id(candidate: Candidate) -> str:
        prefix = "4E_candidate_"
        token = candidate.slug.removeprefix(prefix).split("_", 1)[0]
        return token.upper()

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
    draw.text((margin, 45), "Figure 4E Single-Panel Promotion Candidates", font=font(46, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Choose the one panel that should carry the behavior-geometry bridge.",
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
        draw.text((x, y + thumb_h + 24), f"{display_id(candidate)}. {candidate.title}", font=font(25, True), fill=(20, 26, 32))
        wrap(draw, (x, y + thumb_h + 64), candidate.recommendation, font(22), thumb_w)
        wrap(draw, (x, y + thumb_h + 136), f"Boundary: {candidate.boundary}", font(20), thumb_w)

    out = OUT_DIR / "4E_single_panel_candidate_sheet.png"
    sheet.save(out, optimize=True)
    return out


def _write_readme(paths: list[Path]) -> None:
    readme = OUT_DIR / "README.md"
    lines = [
        "# Figure 4E Single-Panel Promotion Candidates",
        "",
        "Status: draft candidates for choosing one promoted behavior-geometry bridge panel; 3A/3B focused variants and 3C polar view added after review.",
        "",
        "![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_E/promotion_candidates/4E_single_panel_candidate_sheet.png)",
        "",
        "## Recommendation",
        "",
        "Candidate 3A carries the confidence-dependence message most directly: FEM-edge alignment rises when image orientation coherence is high. Candidate 3B is the paired FEM-anisotropy reliability view. Candidate 3C is a polar/rose option for the same edge-relative directionality. Candidate 1 remains the compact statistical headline, candidate 2 is the most intuitive endpoint-zone read, and candidates 5-6 are dense diagnostics/guardrails.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `4E_single_panel_candidate_sheet.png`",
            "- `4E_single_panel_candidate_values.csv`",
            "",
            "## Claim Boundary",
            "",
            "Panel E supports modest but reliable contour-following geometry in measured FEM/fixation-cloud axes. It is a behavioral bridge, not a causal intervention and not proof that the current V1-twin objective explains behavior beyond raw edge geometry.",
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
        path, values = _copy_candidate(candidate)
        paths.append(path)
        value_rows.append(values)

    pd.concat(value_rows, ignore_index=True, sort=False).to_csv(OUT_DIR / "4E_single_panel_candidate_values.csv", index=False)
    sheet = _make_contact_sheet(paths)
    _write_readme(paths)
    for path in paths + [sheet, OUT_DIR / "README.md", OUT_DIR / "4E_single_panel_candidate_values.csv"]:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()

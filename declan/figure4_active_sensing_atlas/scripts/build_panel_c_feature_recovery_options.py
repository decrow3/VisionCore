"""Build focused Figure 4C feature-recovery option panels.

These options are all drawn from the newer feature-posterior endpoint. They are
meant for choosing how Panel C should communicate latent-eye feature recovery
without making the old gain-over-moving-zero-baseline plot the main visual.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
OUT_DIR = ATLAS / "figures" / "panel_C" / "promotion_candidates" / "feature_recovery_options"
SOURCE_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1"
    / "feature_posterior_summary.csv"
)
FEATURE_COMPACT_MECHANISM_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1"
    / "feature_compact_mechanism_summary.csv"
)

INK = "#1f252b"
MUTED = "#68727d"
GRID = "#dfe4e9"
BLUE = "#244f7a"
GREEN = "#2f8f6a"
PURPLE = "#8064a2"
GRAY = "#7b8288"
LIGHT = "#eef2f5"


@dataclass(frozen=True)
class Option:
    stem: str
    title: str
    read: str
    boundary: str


OPTIONS = (
    Option(
        "4C_option_1_zeroed_vs_compact_subspace",
        "Zero-eye vs compact subspace",
        "Cleanest correction: zero-eye recovery falls with motion scale, compact-subspace recovery stays flat.",
        "Does not foreground the two compact sources; the compact-subspace spread is only a faint band.",
    ),
    Option(
        "4C_option_2_compact_sources_explicit",
        "Compact sources explicit",
        "Shows zero-eye, compact source A, compact source B, and known-eye ceiling on the same recovery axis.",
        "Honest but busier; both compact sources are useful here, so this should not be read as a single-source result.",
    ),
    Option(
        "4C_option_3_observer_scale_heatmap",
        "Observer-by-scale heatmap",
        "Compactly shows zero-eye degrading while compact-subspace priors sit near the known-eye ceiling.",
        "Most diagnostic; less immediately graph-like than a line panel.",
    ),
    Option(
        "4C_option_4_scale_robustness",
        "Scale robustness",
        "Turns the result into slopes: zero-eye recovery drops by about 0.19 cosine, compact-subspace priors stay near flat.",
        "Best for the mechanism point, but needs caption support because it plots change from 0.5x.",
    ),
    Option(
        "4C_option_5_compact_subspace_rescue",
        "Compact subspace rescue",
        "The least busy compact-subspace panel: zero-eye recovery falls, compact-subspace recovery stays near 0.87, known eye is the ceiling.",
        "Omits the second compact-source comparator; use only if surrounding materials carry the component check.",
    ),
    Option(
        "4C_option_6_compact_necessity_audit",
        "Feature-space compact removal",
        "Direct audit in the promoted metric: compact-only stays high, while compact-removed collapses toward zero-eye recovery.",
        "The panel shows necessity for this projection, not proof that the animal computes the posterior.",
    ),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)


def _selected_rows() -> pd.DataFrame:
    rows = pd.read_csv(SOURCE_CSV)
    selected = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["prior_family"].isin(["axis_edge_parallel", "axis_edge_orthogonal"]))
    ].copy()
    if len(selected) != 6:
        raise ValueError(f"Expected 6 primary feature-posterior rows, found {len(selected)}")
    selected["scale_label"] = selected["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"})
    selected["compact_source_label"] = selected["prior_family"].map(
        {
            "axis_edge_parallel": "compact source A",
            "axis_edge_orthogonal": "compact source B",
        }
    )
    return selected.sort_values(["observation_scale", "prior_family"])


def _compact_necessity_summary() -> pd.DataFrame:
    rows = pd.read_csv(FEATURE_COMPACT_MECHANISM_SUMMARY)
    selected = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["likelihood_scale"] == 1.0)
        & (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["k_dim"].astype(int) == 10)
        & (
            rows["response_variant"].isin(["zero_static", "compact_only", "compact_removed", "known_eye", "full_exact", "compact_addback"])
        )
    ].copy()
    selected["display_label"] = selected["response_variant"].map(
        {
            "zero_static": "zero eye",
            "full_exact": "full joint",
            "compact_only": "compact only",
            "compact_removed": "compact removed",
            "compact_addback": "compact addback",
            "known_eye": "known eye",
        }
    )
    summary = (
        selected.groupby(["observation_scale", "response_variant", "display_label"], as_index=False)
        .agg(
            mean_feature_cosine=("mean_feature_cosine", "mean"),
            mean_feature_neg_mse=("mean_feature_neg_mse", "mean"),
            mean_candidate_true_mass=("mean_candidate_true_mass", "mean"),
            median_candidate_N_eff_fraction=("median_candidate_N_eff_fraction", "mean"),
            median_clipped_rate_fraction=("median_clipped_rate_fraction", "mean"),
            n_rows=("n_trial_rows", "sum"),
        )
        .sort_values(["observation_scale", "response_variant"])
    )
    return summary.assign(scale_label=lambda d: d["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"}))


def _scale_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("observation_scale", as_index=False)
        .agg(
            zero_mean_cosine=("zero_mean_cosine", "mean"),
            compact_mean_cosine=("joint_mean_cosine", "mean"),
            compact_min_cosine=("joint_mean_cosine", "min"),
            compact_max_cosine=("joint_mean_cosine", "max"),
            known_mean_cosine=("known_mean_cosine", "mean"),
        )
        .sort_values("observation_scale")
        .assign(scale_label=lambda d: d["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"}))
    )


def _x_for(scales: pd.Series) -> np.ndarray:
    return scales.map({0.5: 0, 1.0: 1, 2.0: 2}).astype(float).to_numpy()


def _save(fig: plt.Figure, stem: str) -> Path:
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_option_1(rows: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(3.7, 2.85), constrained_layout=True)
    x = _x_for(summary["observation_scale"])
    ax.plot(x, summary["zero_mean_cosine"], color=GRAY, marker="o", lw=2.0, label="zero eye")
    ax.plot(x, summary["compact_mean_cosine"], color=GREEN, marker="o", lw=2.2, label="compact subspace")
    ax.fill_between(
        x,
        summary["compact_min_cosine"],
        summary["compact_max_cosine"],
        color=GREEN,
        alpha=0.14,
        linewidth=0,
    )
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.50, 0.92)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_title("Compact subspace stabilizes recovery")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    return _save(fig, OPTIONS[0].stem), summary.assign(option=OPTIONS[0].stem)


def _plot_option_2(rows: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(3.7, 2.85), constrained_layout=True)
    x = _x_for(summary["observation_scale"])
    ax.plot(x, summary["zero_mean_cosine"], color=GRAY, marker="o", lw=1.8, label="zero eye")
    styles = {
        "axis_edge_parallel": ("compact source A", GREEN),
        "axis_edge_orthogonal": ("compact source B", PURPLE),
    }
    for family, (label, color) in styles.items():
        block = rows[rows["prior_family"] == family].sort_values("observation_scale")
        ax.plot(_x_for(block["observation_scale"]), block["joint_mean_cosine"], color=color, marker="o", lw=2.0, label=label)
    ax.plot(x, summary["known_mean_cosine"], color=INK, lw=1.4, linestyle=":", label="known eye")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.50, 0.98)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_title("Compact subspace priors recover features")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    return _save(fig, OPTIONS[1].stem), rows.assign(option=OPTIONS[1].stem)


def _plot_option_3(rows: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    scales = [0.5, 1.0, 2.0]
    matrix = []
    labels = ["zero eye", "compact source A", "compact source B", "known eye"]
    for scale in scales:
        block = rows[rows["observation_scale"] == scale]
        matrix.append(
            [
                float(block["zero_mean_cosine"].iloc[0]),
                float(block[block["prior_family"] == "axis_edge_parallel"]["joint_mean_cosine"].iloc[0]),
                float(block[block["prior_family"] == "axis_edge_orthogonal"]["joint_mean_cosine"].iloc[0]),
                float(block["known_mean_cosine"].iloc[0]),
            ]
        )
    arr = np.asarray(matrix).T
    fig, ax = plt.subplots(figsize=(3.7, 2.85), constrained_layout=True)
    im = ax.imshow(arr, vmin=0.50, vmax=0.96, cmap="viridis", aspect="auto")
    ax.set_xticks(range(3), ["0.5x", "1x", "2x"])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("motion scale")
    ax.set_title("Feature recovery by observer")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7.4, color="white" if arr[i, j] < 0.75 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("feature recovery (cosine)")
    return _save(fig, OPTIONS[2].stem), pd.DataFrame(
        {
            "option": OPTIONS[2].stem,
            "observer": np.repeat(labels, 3),
            "observation_scale": scales * len(labels),
            "feature_recovery": arr.reshape(-1),
        }
    )


def _plot_option_4(rows: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(3.7, 2.85), constrained_layout=True)
    baseline = 0.5
    x = _x_for(summary["observation_scale"])
    zero_delta = summary["zero_mean_cosine"] - float(summary.loc[summary["observation_scale"] == baseline, "zero_mean_cosine"].iloc[0])
    ax.plot(x, zero_delta, color=GRAY, marker="o", lw=2.0, label="zero eye")
    output_rows = [
        {
            "option": OPTIONS[3].stem,
            "observer": "zero eye",
            "observation_scale": scale,
            "delta_from_0p5x": delta,
        }
        for scale, delta in zip(summary["observation_scale"], zero_delta, strict=True)
    ]
    for family, label, color in [
        ("axis_edge_parallel", "compact source A", GREEN),
        ("axis_edge_orthogonal", "compact source B", PURPLE),
    ]:
        block = rows[rows["prior_family"] == family].sort_values("observation_scale")
        base = float(block.loc[block["observation_scale"] == baseline, "joint_mean_cosine"].iloc[0])
        delta = block["joint_mean_cosine"] - base
        ax.plot(_x_for(block["observation_scale"]), delta, color=color, marker="o", lw=2.0, label=label)
        output_rows.extend(
            {
                "option": OPTIONS[3].stem,
                "observer": label,
                "observation_scale": scale,
                "delta_from_0p5x": value,
            }
            for scale, value in zip(block["observation_scale"], delta, strict=True)
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(-0.22, 0.04)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("change from 0.5x (cosine)")
    ax.set_title("Compact subspace resists scale disruption")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    return _save(fig, OPTIONS[3].stem), pd.DataFrame(output_rows)


def _plot_option_5(rows: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(3.7, 2.85), constrained_layout=True)
    x = _x_for(summary["observation_scale"])
    compact_source_a = rows[rows["prior_family"] == "axis_edge_parallel"].sort_values("observation_scale")
    ax.plot(x, summary["zero_mean_cosine"], color=GRAY, marker="o", lw=1.8, label="zero eye")
    ax.plot(
        _x_for(compact_source_a["observation_scale"]),
        compact_source_a["joint_mean_cosine"],
        color=GREEN,
        marker="o",
        lw=2.2,
        label="compact subspace",
    )
    ax.plot(x, summary["known_mean_cosine"], color=INK, lw=1.4, linestyle=":", label="known eye")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.50, 0.98)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_title("Compact subspace rescues latent-eye features")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    value = summary[["observation_scale", "zero_mean_cosine", "known_mean_cosine"]].merge(
        compact_source_a[["observation_scale", "joint_mean_cosine"]].rename(columns={"joint_mean_cosine": "compact_source_a_mean_cosine"}),
        on="observation_scale",
        how="left",
    )
    return _save(fig, OPTIONS[4].stem), value.assign(option=OPTIONS[4].stem)


def _plot_option_6(rows: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    del rows, summary
    values = _compact_necessity_summary()
    fig, ax = plt.subplots(figsize=(3.7, 2.85), constrained_layout=True)
    x_map = {0.5: 0, 1.0: 1, 2.0: 2}
    styles = {
        "zero_static": ("zero eye", GRAY, "-"),
        "compact_only": ("compact subspace", GREEN, "-"),
        "compact_removed": ("compact removed", PURPLE, "-"),
        "known_eye": ("known eye", INK, ":"),
    }
    for variant, (label, color, linestyle) in styles.items():
        block = values[values["response_variant"] == variant].sort_values("observation_scale")
        ax.plot(
            block["observation_scale"].map(x_map).astype(float).to_numpy(),
            block["mean_feature_cosine"].to_numpy(dtype=float),
            color=color,
            marker="o",
            lw=2.1 if variant == "compact_only" else 1.9,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.48, 0.98)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_title("Compact removal collapses feature recovery")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    return _save(fig, OPTIONS[5].stem), values.assign(option=OPTIONS[5].stem)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, max_width: int) -> int:
    x, y = xy
    line = ""
    line_h = int(getattr(font, "size", 18) * 1.18)
    for word in text.split():
        candidate = word if not line else f"{line} {word}"
        width = font.getbbox(candidate)[2] - font.getbbox(candidate)[0]
        if width <= max_width:
            line = candidate
        else:
            if line:
                draw.text((x, y), line, font=font, fill=(45, 49, 54))
                y += line_h
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=(45, 49, 54))
        y += line_h
    return y


def _make_sheet(paths: list[Path]) -> Path:
    width = 2600
    margin, gap = 62, 36
    thumb_w, thumb_h = 1210, 590
    row_step = 890
    rows_n = (len(paths) + 1) // 2
    height = 215 + (rows_n - 1) * row_step + thumb_h + 270
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 44), "Figure 4C Feature-Recovery Options", font=_font(46, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Options 1-5 use the original feature-posterior rows; option 6 uses the feature-space compact-removal audit.",
        font=_font(25),
        fill=(73, 80, 88),
    )
    draw.line((margin, 168, width - margin, 168), fill=(183, 190, 198), width=2)
    for i, (option, path) in enumerate(zip(OPTIONS, paths, strict=True)):
        row = i // 2
        col = i % 2
        x = margin + col * (thumb_w + gap)
        y = 215 + row * row_step
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(191, 199, 207), fill="white")
        image = Image.open(path).convert("RGBA")
        scale = min((thumb_w - 34) / image.width, (thumb_h - 34) / image.height)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + 17 + (thumb_w - 34 - image.width) // 2, y + 17), image)
        draw.text((x, y + thumb_h + 24), f"{i + 1}. {option.title}", font=_font(25, True), fill=(20, 26, 32))
        _wrap(draw, (x, y + thumb_h + 64), option.read, _font(21), thumb_w)
        _wrap(draw, (x, y + thumb_h + 140), f"Boundary: {option.boundary}", _font(19), thumb_w)
    out = OUT_DIR / "4C_feature_recovery_option_sheet.png"
    sheet.save(out, optimize=True)
    return out


def _write_readme(paths: list[Path], sheet: Path) -> None:
    lines = [
        "# Figure 4C Feature-Recovery Options",
        "",
        "Focused option sheet for the feature-posterior Panel C revision.",
        "Options 1-5 use the promoted hard-negative pyramid k=8 feature-posterior rows.",
        "Option 6 uses the matching feature-space compact-only / compact-removed / addback decomposition.",
        "",
        f"![Option sheet]({sheet})",
        "",
        "## Recommended Reads",
        "",
        "- Option 1 is the cleanest corrected main-panel read: zero-eye recovery falls while compact-subspace recovery stays stable.",
        "- Option 2 is the strongest explicit compact-source version, but it also shows that both compact sources are useful here.",
        "- Option 3 is the most diagnostic compact summary.",
        "- Option 4 is the clearest scale-robustness mechanism view.",
        "- Option 5 is the clean compact-subspace sufficiency panel, retained as a simpler fallback.",
        "- Option 6 is the selected feature-space compact-removal panel: compact-only stays high while compact-removed falls toward zero-eye recovery.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `4C_feature_recovery_option_sheet.png`",
            "- `4C_feature_recovery_option_values.csv`",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows()
    summary = _scale_summary(rows)
    builders = [_plot_option_1, _plot_option_2, _plot_option_3, _plot_option_4, _plot_option_5, _plot_option_6]
    paths: list[Path] = []
    values: list[pd.DataFrame] = []
    for builder in builders:
        path, value = builder(rows, summary)
        paths.append(path)
        values.append(value)
    pd.concat(values, ignore_index=True, sort=False).to_csv(OUT_DIR / "4C_feature_recovery_option_values.csv", index=False)
    sheet = _make_sheet(paths)
    _write_readme(paths, sheet)
    for path in paths + [sheet, OUT_DIR / "README.md", OUT_DIR / "4C_feature_recovery_option_values.csv"]:
        print(path)


def main() -> None:
    build()


if __name__ == "__main__":
    main()

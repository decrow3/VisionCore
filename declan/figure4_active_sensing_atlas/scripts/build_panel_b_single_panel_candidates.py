"""Build single-panel promotion candidates for Figure 4B.

Each generated PNG is a candidate for the one promoted Panel B. Existing atlas
subpanels are reused where they already express a clean single claim. The
current promoted candidate should be redrawn from the corrected n384 pyramid
k16 static-mean posthoc, using delta_mean as the primary readout.
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
PANEL_B = ATLAS / "figures" / "panel_B"
OUT_DIR = PANEL_B / "promotion_candidates"
AGGREGATE_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched"
)
RELIDS_DIR = AGGREGATE_DIR / "incremental_static_plus_motion_relids"
POWER_RERUN_DIR = Path(
    os.environ.get(
        "PANEL_B_POWER_RERUN_DIR",
        str(
            REPO_ROOT
            / "outputs"
            / "fixation_statistics_by_stimulus_all_sessions_after_review"
            / "backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1"
            / "incremental_staticmean_plus_motion_info_decode_bootstrap_b50_source_trial_validated_20260630"
        ),
    )
)
POSE_UNAWARE_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_pose_unaware_production_n384_empirical_k8_seed0"
    / "pose_unaware_staticmean_plus_motion_info_source_trial_b50_20260630"
)


COLORS = {
    "empirical": "#244f7a",
    "ou": "#d07a22",
    "brownian": "#707070",
    "rotated": "#8064a2",
    "gabor": "#244f7a",
    "pyramid": "#2f8f6a",
    "pose_unaware": "#b23a48",
    "dark": "#242a2f",
    "muted": "#65717a",
}
MOTION_LABELS = {
    "ou": "OU control",
    "brownian": "random drift",
    "rotated": "rotated drift",
    "pose_unaware": "pose-unaware drift",
}
NO_OU_CONTROL_FAMILIES = ("brownian", "rotated")
NO_OU_ABSOLUTE_FAMILIES = ("empirical", "brownian", "rotated")
POWER_RERUN_FAMILIES = ("empirical", "ou", "brownian", "rotated")
ALLOW_LEGACY_MSE_FALLBACK = os.environ.get("PANEL_B_ALLOW_LEGACY_MSE", "").strip() == "1"
INFO_GAIN_COL = "incremental_gain_info_diag_bits"
INFO_CONTRAST_COL = "incremental_gain_delta_info_diag_bits"
INFO_LO_COL = "info_diag_ci95_low"
INFO_HI_COL = "info_diag_ci95_high"
REQUIRED_INFORMATION_CI_METHOD = "decode_pipeline_group_bootstrap_point_centered"
LEGACY_GAIN_COL = "incremental_gain_neg_mse"
LEGACY_CONTRAST_COL = "incremental_gain_delta_neg_mse"
LEGACY_LO_COL = "ci95_low"
LEGACY_HI_COL = "ci95_high"


@dataclass(frozen=True)
class Candidate:
    slug: str
    title: str
    source_png: str | None
    recommendation: str
    boundary: str


CANDIDATES = (
    Candidate(
        slug="4B_candidate_1_gain_over_static_audited",
        title="Archived gain over static",
        source_png="B3_empirical_gain_vs_static.png",
        recommendation="Archived legacy view retained only for comparison with the new information-axis recompute.",
        boundary="Uses deterministic -MSE units; do not promote as the current 4B information panel.",
    ),
    Candidate(
        slug="4B_candidate_2_empirical_minus_controls",
        title="No-OU specificity check",
        source_png=None,
        recommendation="Use only if the main panel must foreground empirical-minus-generic controls.",
        boundary="Historical no-OU control view; use only as older guardrail context.",
    ),
    Candidate(
        slug="4B_candidate_3_power_rerun_absolute_gain",
        title="Corrected information readout",
        source_png=None,
        recommendation="Promoted 4B target: diagonal Gaussian decoder information gain over stabilized/static in bits.",
        boundary="Pose-unaware proxy is omitted unless it has been recomputed on the same information axis; full-covariance log-det stays supplemental.",
    ),
    Candidate(
        slug="4B_candidate_4_k16_tworeadout_preview",
        title="Corrected control context",
        source_png=None,
        recommendation="Companion view for the corrected delta-mean control contrasts.",
        boundary="OU is not shown here; use the dedicated OU audit if discussing it.",
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


def _scale_value(scale_id: str) -> float:
    return float(str(scale_id).replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(value: float) -> str:
    return f"{value:g}x"


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _errbar(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    y_col: str,
    lo_col: str,
    hi_col: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    block = df.sort_values("scale")
    x = block["scale"].to_numpy(dtype=float)
    y = block[y_col].to_numpy(dtype=float)
    lo = block[lo_col].to_numpy(dtype=float)
    hi = block[hi_col].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        marker=marker,
        markersize=4.4,
        linewidth=1.9,
        capsize=0,
        color=color,
        label=label,
    )


def _metric_columns(df: pd.DataFrame, *, contrast: bool) -> tuple[str, str, str, str, str]:
    info_value = INFO_CONTRAST_COL if contrast else INFO_GAIN_COL
    legacy_value = LEGACY_CONTRAST_COL if contrast else LEGACY_GAIN_COL
    info_cols = (info_value, INFO_LO_COL, INFO_HI_COL)
    if all(col in df.columns for col in info_cols):
        _require_decode_bootstrap_information(df)
        _require_finite_metric(df, info_cols, metric_name="decoder-information")
        ylabel = (
            "information-gain contrast,\n$\\Delta\\Delta\\hat{I}$ (bits)"
            if contrast
            else "feature information gain over stabilized,\n$\\Delta\\hat{I}$ (bits) - decoder lower bound"
        )
        return info_value, INFO_LO_COL, INFO_HI_COL, ylabel, "info_diag_bits"
    legacy_cols = (legacy_value, LEGACY_LO_COL, LEGACY_HI_COL)
    if ALLOW_LEGACY_MSE_FALLBACK and all(col in df.columns for col in legacy_cols):
        _require_finite_metric(df, legacy_cols, metric_name="legacy -MSE")
        ylabel = "legacy gain contrast (-MSE)" if contrast else "legacy feature-decoding gain (-MSE)"
        return legacy_value, LEGACY_LO_COL, LEGACY_HI_COL, ylabel, "legacy_neg_mse"
    missing = ", ".join(info_cols)
    raise ValueError(
        "Panel 4B now requires recomputed decoder-information columns "
        f"({missing}). Rerun the incremental static-plus-motion summary, or set "
        "PANEL_B_ALLOW_LEGACY_MSE=1 for an explicitly legacy QC render."
    )


def _require_finite_metric(df: pd.DataFrame, cols: tuple[str, str, str], *, metric_name: str) -> None:
    values = df.loc[:, list(cols)].to_numpy(dtype=float)
    finite_rows = np.all(np.isfinite(values), axis=1)
    if values.size == 0 or not np.any(finite_rows):
        raise ValueError(f"No finite {metric_name} rows are available for Panel 4B plotting columns {cols}")
    if not np.all(finite_rows):
        bad = int(np.size(finite_rows) - np.sum(finite_rows))
        raise ValueError(f"{bad} Panel 4B rows have non-finite {metric_name} y/CI values in columns {cols}")
    y = df[cols[0]].to_numpy(dtype=float)
    lo = df[cols[1]].to_numpy(dtype=float)
    hi = df[cols[2]].to_numpy(dtype=float)
    outside = finite_rows & ((y < lo - 1e-9) | (y > hi + 1e-9))
    if np.any(outside):
        bad = df.loc[outside, [col for col in ("motion_summary", "family", "lhs_family", "rhs_family", "scale_id", "latent", "k") if col in df.columns]].head(6)
        raise ValueError(
            f"{int(np.sum(outside))} Panel 4B {metric_name} rows have point estimates outside their CI bars "
            f"for columns {cols}. First bad rows:\n{bad.to_string(index=False)}"
        )


def _require_decode_bootstrap_information(df: pd.DataFrame) -> None:
    if "information_ci_method" not in df.columns:
        raise ValueError("Panel 4B information tables must include `information_ci_method` provenance")
    methods = set(df["information_ci_method"].dropna().astype(str))
    if methods != {REQUIRED_INFORMATION_CI_METHOD}:
        raise ValueError(
            "Panel 4B promoted information axis requires point-centered decode-bootstrap CIs; "
            f"found information_ci_method={sorted(methods)}"
        )
    if "n_information_bootstrap_success" in df.columns:
        successes = df["n_information_bootstrap_success"].to_numpy(dtype=float)
        if np.any(~np.isfinite(successes)) or np.any(successes < 2):
            raise ValueError("Panel 4B information tables need at least two successful decode-bootstrap replicates")


def _has_information_gain(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in (INFO_GAIN_COL, INFO_LO_COL, INFO_HI_COL))


def _copy_source_candidate(candidate: Candidate) -> Path:
    assert candidate.source_png is not None
    source = PANEL_B / candidate.source_png
    if not source.exists():
        raise FileNotFoundError(source)
    out = OUT_DIR / f"{candidate.slug}.png"
    shutil.copy2(source, out)
    pdf = source.with_suffix(".pdf")
    if pdf.exists():
        shutil.copy2(pdf, out.with_suffix(".pdf"))
    return out


def _relids_gain_block(
    *,
    latent: str,
    k: int,
    families: tuple[str, ...],
) -> pd.DataFrame:
    gain = pd.read_csv(RELIDS_DIR / "incremental_gain_vs_static.csv")
    block = gain[
        (gain["motion_summary"] == "temporal_pca")
        & (gain["latent"] == latent)
        & (gain["k"] == k)
        & (gain["family"].isin(families))
    ].copy()
    if block.empty:
        raise ValueError(f"Missing temporal_pca gain rows for {latent} k={k}")
    block["scale"] = block["scale_id"].map(_scale_value)
    return block


def _relids_contrast_block(
    *,
    latent: str,
    k: int,
    rhs_families: tuple[str, ...],
) -> pd.DataFrame:
    contrasts = pd.read_csv(RELIDS_DIR / "incremental_gain_contrasts.csv")
    block = contrasts[
        (contrasts["motion_summary"] == "temporal_pca")
        & (contrasts["lhs_family"] == "empirical")
        & (contrasts["latent"] == latent)
        & (contrasts["k"] == k)
        & (contrasts["rhs_family"].isin(rhs_families))
    ].copy()
    if block.empty:
        raise ValueError(f"Missing temporal_pca contrast rows for {latent} k={k}")
    block["scale"] = block["scale_id"].map(_scale_value)
    return block


def _power_gain_block(families: tuple[str, ...] = POWER_RERUN_FAMILIES) -> pd.DataFrame:
    gain = pd.read_csv(POWER_RERUN_DIR / "incremental_gain_vs_static.csv")
    block = gain[
        (gain["motion_summary"] == "delta_mean")
        & (gain["latent"] == "pyramid_local_field")
        & (gain["k"].astype(int) == 16)
        & (gain["family"].isin(families))
    ].copy()
    if block.empty:
        raise ValueError("Missing production n384 pyramid k16 delta_mean gain rows")
    block["scale"] = block["scale_id"].map(_scale_value)
    return block


def _pose_unaware_proxy_block() -> pd.DataFrame:
    proxy = pd.read_csv(POSE_UNAWARE_DIR / "pose_unaware_train_mean_test_samples_proxy.csv")
    block = proxy[
        (proxy["observer"] == "pose_unaware_train_mean_test_hidden_samples")
        & (proxy["motion_summary"] == "delta_mean")
        & (proxy["family"] == "empirical")
        & (proxy["latent"] == "pyramid_local_field")
        & (proxy["k"].astype(int) == 16)
    ].copy()
    if block.empty:
        raise ValueError("Missing pose-unaware n384 pyramid k16 delta_mean proxy rows")
    block["scale"] = block["scale_id"].map(_scale_value)
    block["family"] = "pose_unaware"
    return block


def _power_contrast_block(rhs_families: tuple[str, ...] = ("brownian", "rotated")) -> pd.DataFrame:
    contrasts = pd.read_csv(POWER_RERUN_DIR / "incremental_gain_contrasts.csv")
    block = contrasts[
        (contrasts["motion_summary"] == "delta_mean")
        & (contrasts["lhs_family"] == "empirical")
        & (contrasts["latent"] == "pyramid_local_field")
        & (contrasts["k"].astype(int) == 16)
        & (contrasts["rhs_family"].isin(rhs_families))
    ].copy()
    if block.empty:
        raise ValueError("Missing production n384 pyramid k16 delta_mean contrast rows")
    block["scale"] = block["scale_id"].map(_scale_value)
    return block


def _build_control_contrasts_no_ou() -> tuple[Path, pd.DataFrame]:
    contrast_block = _relids_contrast_block(
        latent="gabor_local_field",
        k=4,
        rhs_families=NO_OU_CONTROL_FAMILIES,
    )
    y_col, lo_col, hi_col, ylabel, metric = _metric_columns(contrast_block, contrast=True)

    fig, ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    for rhs, marker in [("brownian", "s"), ("rotated", "^")]:
        _errbar(
            ax,
            contrast_block[contrast_block["rhs_family"] == rhs],
            y_col=y_col,
            lo_col=lo_col,
            hi_col=hi_col,
            label=f"empirical - {MOTION_LABELS[rhs]}",
            color=COLORS[rhs],
            marker=marker,
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_title("Empirical minus generic controls")
    ax.set_xlabel("motion scale")
    ax.set_ylabel(ylabel)
    scales = sorted(contrast_block["scale"].unique())
    ax.set_xticks(scales, [_scale_label(v) for v in scales])
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    _clean_axis(ax)

    out = OUT_DIR / "4B_candidate_2_empirical_minus_controls.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return out, contrast_block.assign(candidate_metric="empirical_minus_control_no_ou", plotted_metric=metric)


def _build_absolute_guardrail_no_ou() -> tuple[Path, pd.DataFrame]:
    gain_block = _relids_gain_block(
        latent="gabor_local_field",
        k=4,
        families=NO_OU_ABSOLUTE_FAMILIES,
    )
    y_col, lo_col, hi_col, ylabel, metric = _metric_columns(gain_block, contrast=False)

    fig, ax = plt.subplots(figsize=(3.55, 2.75), constrained_layout=True)
    for family, marker in [("empirical", "o"), ("brownian", "s"), ("rotated", "^")]:
        _errbar(
            ax,
            gain_block[gain_block["family"] == family],
            y_col=y_col,
            lo_col=lo_col,
            hi_col=hi_col,
            label="recorded drift" if family == "empirical" else MOTION_LABELS[family],
            color=COLORS[family],
            marker=marker,
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_title("Retinal motion adds feature information")
    ax.set_xlabel("motion scale")
    ax.set_ylabel(ylabel)
    scales = sorted(gain_block["scale"].unique())
    ax.set_xticks(scales, [_scale_label(v) for v in scales])
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    _clean_axis(ax)

    out = OUT_DIR / "4B_candidate_3_absolute_gain_guardrail.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return out, gain_block.assign(candidate_metric="absolute_gain_no_ou", plotted_metric=metric)


def _build_power_rerun_absolute_gain() -> tuple[Path, pd.DataFrame]:
    gain_block = _power_gain_block(("empirical", "brownian", "rotated"))
    y_col, lo_col, hi_col, ylabel, metric = _metric_columns(gain_block, contrast=False)
    pose_block = _pose_unaware_proxy_block()
    check_pose = _has_information_gain(pose_block) or ALLOW_LEGACY_MSE_FALLBACK
    pose_plotted = False

    fig, ax = plt.subplots(figsize=(3.75, 2.95), constrained_layout=True)
    for family, marker in [("empirical", "o"), ("brownian", "s"), ("rotated", "^")]:
        _errbar(
            ax,
            gain_block[gain_block["family"] == family],
            y_col=y_col,
            lo_col=lo_col,
            hi_col=hi_col,
            label="recorded drift" if family == "empirical" else MOTION_LABELS[family],
            color=COLORS[family],
            marker=marker,
        )
    if check_pose:
        pose_y_col, pose_lo_col, pose_hi_col, _, pose_metric = _metric_columns(pose_block, contrast=False)
        if pose_metric == metric:
            pose_plotted = True
            _errbar(
                ax,
                pose_block,
                y_col=pose_y_col,
                lo_col=pose_lo_col,
                hi_col=pose_hi_col,
                label=MOTION_LABELS["pose_unaware"],
                color=COLORS["pose_unaware"],
                marker="v",
            )
            for line in ax.lines[-1:]:
                line.set_linestyle("--")
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_title("Motion enhances feature information")
    ax.set_xlabel("motion scale")
    ax.set_ylabel(ylabel)
    scales = sorted(
        set(gain_block["scale"].unique()).union(set(pose_block["scale"].unique()) if pose_plotted else set())
    )
    ax.set_xticks(scales, [_scale_label(v) for v in scales])
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    _clean_axis(ax)

    out = OUT_DIR / "4B_candidate_3_power_rerun_absolute_gain.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    values = pd.concat(
        [
            gain_block.assign(candidate_metric="power_rerun_absolute_gain_motion_rendered", plotted_metric=metric),
            pose_block.assign(
                candidate_metric="pose_unaware_hidden_samples_proxy",
                plotted_metric=(metric if pose_plotted else "not_plotted_missing_information_recompute"),
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return out, values


def _build_k16_preview() -> tuple[Path, pd.DataFrame]:
    gain_block = _power_gain_block(("empirical",))
    contrast_block = _power_contrast_block()
    gain_y_col, gain_lo_col, gain_hi_col, gain_ylabel, gain_metric = _metric_columns(gain_block, contrast=False)
    contrast_y_col, contrast_lo_col, contrast_hi_col, contrast_ylabel, contrast_metric = _metric_columns(
        contrast_block,
        contrast=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    ax = axes[0]
    _errbar(
        ax,
        gain_block,
        y_col=gain_y_col,
        lo_col=gain_lo_col,
        hi_col=gain_hi_col,
        label="recorded drift",
        color=COLORS["pyramid"],
        marker="s",
    )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_title("Information gain over static mean")
    ax.set_xlabel("motion scale")
    ax.set_ylabel(gain_ylabel)
    ax.set_xticks(sorted(gain_block["scale"].unique()), [_scale_label(v) for v in sorted(gain_block["scale"].unique())])
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    _clean_axis(ax)

    ax = axes[1]
    for rhs, color, marker in [("brownian", COLORS["brownian"], "s"), ("rotated", COLORS["rotated"], "^")]:
        _errbar(
            ax,
            contrast_block[contrast_block["rhs_family"] == rhs],
            y_col=contrast_y_col,
            lo_col=contrast_lo_col,
            hi_col=contrast_hi_col,
            label=f"empirical - {MOTION_LABELS[rhs]}",
            color=color,
            marker=marker,
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_title("Empirical minus generic controls")
    ax.set_xlabel("motion scale")
    ax.set_ylabel(contrast_ylabel)
    ax.set_xticks(sorted(contrast_block["scale"].unique()), [_scale_label(v) for v in sorted(contrast_block["scale"].unique())])
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    _clean_axis(ax)

    fig.suptitle("Corrected power rerun: pyramid k=16 delta mean information", fontsize=10.5)
    out = OUT_DIR / "4B_candidate_4_k16_tworeadout_preview.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    values = pd.concat(
        [
            gain_block.assign(candidate_metric="gain_over_static", plotted_metric=gain_metric),
            contrast_block.assign(candidate_metric="empirical_minus_control", plotted_metric=contrast_metric),
        ],
        ignore_index=True,
        sort=False,
    )
    return out, values


def _make_contact_sheet(paths: list[Path], candidate_values: pd.DataFrame) -> Path:
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

    width, height = 2600, 2100
    margin, gap = 62, 36
    thumb_w, thumb_h = 1210, 650
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 45), "Figure 4B Single-Panel Promotion Candidates", font=font(46, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Choose the one panel that should carry aggregate FEM feature utility.",
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

    out = OUT_DIR / "4B_single_panel_candidate_sheet.png"
    sheet.save(out, optimize=True)
    return out


def _write_readme(paths: list[Path], sheet: Path) -> None:
    readme = OUT_DIR / "README.md"
    lines = [
        "# Figure 4B Single-Panel Promotion Candidates",
        "",
        "Status: draft candidates for choosing one promoted aggregate-FEM panel.",
        "",
        "![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/4B_single_panel_candidate_sheet.png)",
        "",
        "## Recommendation",
        "",
        "Candidate 3 is the promoted strict source-trial grouped information-axis target after the incremental static-plus-motion summaries are recomputed. The plotted quantity is diagonal Gaussian decoder information gain over the stabilized/static baseline in bits, with point-centered decode-bootstrap CIs. The pose-unaware hidden-sample proxy is now plotted on the same information axis.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `4B_single_panel_candidate_sheet.png`",
            "- `4B_single_panel_candidate_values.csv`",
            "",
            "## Claim Boundary",
            "",
            "The promoted axis is a Gaussian variational decoder lower-bound increment, not an absolute mutual-information estimate. Headline panels use the diagonal residual-variance form in bits; full-covariance Ledoit-Wolf log-det values are supplemental robustness. Legacy `-MSE` candidates remain archive/QC only and require `PANEL_B_ALLOW_LEGACY_MSE=1` to render from old tables.",
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
        if candidate.slug in {
            "4B_candidate_1_gain_over_static_audited",
            "4B_candidate_2_empirical_minus_controls",
        } and not ALLOW_LEGACY_MSE_FALLBACK:
            continue
        if candidate.slug == "4B_candidate_2_empirical_minus_controls":
            path, values = _build_control_contrasts_no_ou()
            value_rows.append(values.assign(candidate=candidate.slug))
        elif candidate.slug == "4B_candidate_3_power_rerun_absolute_gain":
            path, values = _build_power_rerun_absolute_gain()
            value_rows.append(values.assign(candidate=candidate.slug))
        elif candidate.source_png is None:
            path, values = _build_k16_preview()
            value_rows.append(values.assign(candidate=candidate.slug))
        else:
            path = _copy_source_candidate(candidate)
        paths.append(path)
    if value_rows:
        pd.concat(value_rows, ignore_index=True, sort=False).to_csv(OUT_DIR / "4B_single_panel_candidate_values.csv", index=False)
    else:
        (OUT_DIR / "4B_single_panel_candidate_values.csv").write_text("", encoding="utf-8")
    sheet = _make_contact_sheet(paths, pd.DataFrame())
    _write_readme(paths, sheet)
    for path in paths + [sheet, OUT_DIR / "README.md", OUT_DIR / "4B_single_panel_candidate_values.csv"]:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()

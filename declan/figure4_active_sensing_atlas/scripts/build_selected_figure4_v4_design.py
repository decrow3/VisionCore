"""Build a design-first selected Figure 4 composite.

Unlike ``build_selected_figure4.py``, this draft does not paste five standalone
promotion-candidate PNGs into a grid. It redraws the quantitative panels in one
shared visual system so each panel is sized and labelled for its place in the
composite.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch
from PIL import Image

try:  # pragma: no cover - script-mode import path fallback
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
FIGURES = ATLAS / "figures"
OUT_DIR = FIGURES / "composites"

A_PNG = FIGURES / "panel_A" / "promotion_candidates" / "4A_candidate_3_real_high_contrast_positive.png"
B_GAIN_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1"
    / "incremental_staticmean_plus_motion_tworeadout_v2"
    / "incremental_gain_vs_static.csv"
)
B_POSE_UNAWARE_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_pose_unaware_production_n384_empirical_k8_seed0"
    / "pose_unaware_staticmean_plus_motion_delta_v1"
    / "pose_unaware_train_mean_test_samples_proxy.csv"
)
C_POSTERIOR_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1"
    / "feature_compact_mechanism_summary.csv"
)
D_STABILITY_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
    / "stability_summary.csv"
)
D_FEATURE_SUMMARY_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2"
    / "feature_posterior_summary.csv"
)
D_FEATURE_CONTRAST_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2"
    / "feature_axis_contrasts.csv"
)
D_FEATURE_HARDNEG_CONTRAST_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1"
    / "feature_axis_contrasts.csv"
)
D_THUMBNAIL_VALUES_CSV = (
    FIGURES
    / "panel_D"
    / "story_options"
    / "4D_row17_row18_visible_rail_fit_orientation_values.csv"
)
E_WINDOWS_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)

INK = "#1f252b"
MUTED = "#68727d"
GRID = "#dfe4e9"
BLUE = "#244f7a"
GREEN = "#2f8f6a"
PURPLE = "#8064a2"
GRAY = "#747a80"
LIGHT_GRAY = "#e5e9ed"
ORANGE = "#d07a22"
RED = "#b23a48"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
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
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=3.0, width=0.8, color=INK)


def _panel_header(ax: plt.Axes, label: str, title: str, subtitle: str) -> None:
    title_x = 0.105
    ax.text(
        0.0,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=BLUE,
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        title_x,
        1.115,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=10.2,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        title_x,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=7.5,
        clip_on=False,
    )


def _load_a_image() -> np.ndarray:
    image = Image.open(A_PNG).convert("RGB")
    # Remove the standalone candidate title while preserving the real-data
    # provenance and the visual movie grammar.
    crop = image.crop((0, 150, image.width, image.height - 130))
    return np.asarray(crop)


def _norm_image(image: np.ndarray, *, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _crop_centered(image: np.ndarray, center_xy: tuple[float, float], size: int) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = center_xy
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(width - size, x0))
    y0 = max(0, min(height - size, y0))
    return image[y0 : y0 + size, x0 : x0 + size]


def _axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(axis_deg)
    along = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    across = np.asarray([np.cos(theta + np.pi / 2.0), np.sin(theta + np.pi / 2.0)], dtype=float)
    return along, across


def _add_axis_arrows(ax: plt.Axes, axis_deg: float, *, scale: float = 0.33) -> None:
    along, across = _axis_vectors(axis_deg)
    center = np.asarray([0.50, 0.50])
    for vector, color in [(along, GREEN), (across, PURPLE)]:
        ax.add_patch(
            FancyArrowPatch(
                tuple(center - vector * scale),
                tuple(center + vector * scale),
                arrowstyle="<|-|>",
                mutation_scale=9.0,
                linewidth=1.85,
                color=color,
                transform=ax.transAxes,
            )
        )


def _load_d_thumbnail() -> tuple[np.ndarray, float, dict[str, object]]:
    if D_THUMBNAIL_VALUES_CSV.exists():
        row = pd.read_csv(D_THUMBNAIL_VALUES_CSV).iloc[0]
        axis_deg = float(row["visible_rail_fit_axis_deg"])
        meta = {
            "thumbnail_source": "visible_rail_fit",
            "thumbnail_values_csv": D_THUMBNAIL_VALUES_CSV.relative_to(REPO_ROOT).as_posix(),
            "thumbnail_raw_window_row": int(row["raw_window_row"]),
            "thumbnail_session": str(row["session"]),
            "thumbnail_trial_idx": int(row["trial_idx"]),
            "thumbnail_axis_deg": axis_deg,
            "thumbnail_stored_image_edge_axis_deg": float(row["stored_image_edge_axis_deg"]),
            "thumbnail_image_orientation_coherence": float(row["image_orientation_coherence"]),
        }
    else:
        raw = pd.read_csv(E_WINDOWS_CSV)
        row = raw.iloc[17]
        axis_deg = float(row["image_edge_axis_deg"])
        meta = {
            "thumbnail_source": "stored_average_image_axis",
            "thumbnail_raw_window_row": 17,
            "thumbnail_session": str(row["session"]),
            "thumbnail_trial_idx": int(row["trial_idx"]),
            "thumbnail_axis_deg": axis_deg,
            "thumbnail_image_orientation_coherence": float(row["image_orientation_coherence"]),
        }

    canvas, ppd, screen_shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center_xy_px = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    patch = _crop_centered(canvas, (float(center_xy_px[0]), float(center_xy_px[1])), 190)
    return patch, axis_deg, meta


def _plot_d_thumbnail(ax: plt.Axes) -> dict[str, object]:
    patch, axis_deg, meta = _load_d_thumbnail()
    ax.imshow(_norm_image(patch), cmap="gray", vmin=0, vmax=1)
    _add_axis_arrows(ax, axis_deg)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02,
        -0.06,
        "green along; purple across",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=6.3,
    )
    return meta


def _plot_a(ax: plt.Axes) -> None:
    ax.imshow(_load_a_image())
    ax.set_axis_off()
    _panel_header(
        ax,
        "A",
        "One image becomes a retinal movie",
        "recorded eye drift samples different views of the same scene",
    )


def _plot_b(ax: plt.Axes) -> pd.DataFrame:
    rows = pd.read_csv(B_GAIN_CSV)
    block = rows[
        (rows["motion_summary"] == "delta_mean")
        & (rows["latent"] == "pyramid_local_field")
        & (rows["k"].astype(int) == 16)
        & (rows["family"].isin(["empirical", "brownian", "rotated"]))
    ].copy()
    block["scale"] = block["scale_id"].map(_scale_value)
    pose_rows = pd.read_csv(B_POSE_UNAWARE_CSV)
    pose = pose_rows[
        (pose_rows["observer"] == "pose_unaware_train_mean_test_hidden_samples")
        & (pose_rows["motion_summary"] == "delta_mean")
        & (pose_rows["family"] == "empirical")
        & (pose_rows["latent"] == "pyramid_local_field")
        & (pose_rows["k"].astype(int) == 16)
    ].copy()
    pose["scale"] = pose["scale_id"].map(_scale_value)
    pose["family"] = "pose_unaware"
    styles = {
        "empirical": ("known-eye drift", BLUE, "o", 2.1, 1.0, "-"),
        "pose_unaware": ("pose-unaware", RED, "v", 1.8, 0.95, "--"),
        "brownian": ("random drift", GRAY, "s", 1.5, 0.78),
        "rotated": ("rotated drift", PURPLE, "^", 1.5, 0.78),
    }
    for family in ["empirical", "pose_unaware", "brownian", "rotated"]:
        style = styles[family]
        if len(style) == 6:
            label, color, marker, lw, alpha, linestyle = style
        else:
            label, color, marker, lw, alpha = style
            linestyle = "-"
        data = pose.sort_values("scale") if family == "pose_unaware" else block[block["family"] == family].sort_values("scale")
        x = data["scale"].to_numpy(dtype=float)
        y = data["incremental_gain_neg_mse"].to_numpy(dtype=float)
        lo = data["ci95_low"].to_numpy(dtype=float)
        hi = data["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            color=color,
            marker=marker,
            markersize=4.0,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
            capsize=0,
            label=label,
            zorder=3 if family == "empirical" else 2,
        )
    ax.axhline(0, color=INK, lw=0.8)
    scales = sorted(set(block["scale"].unique()).union(set(pose["scale"].unique())))
    ax.set_xticks(scales, [_scale_label(v) for v in scales])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("delta-mean gain over static mean (-MSE)")
    ax.set_ylim(-7.6, 4.2)
    ax.legend(frameon=False, loc="lower right", borderaxespad=0.0)
    _clean_axis(ax)
    _panel_header(
        ax,
        "B",
        "Motion enhances feature encoding",
        "but only when eye position is known",
    )
    family_order = {"empirical": 0, "pose_unaware": 1, "brownian": 2, "rotated": 3}
    values = pd.concat(
        [
            block.assign(observer="known_eye_static_plus_motion"),
            pose.assign(observer="pose_unaware_train_mean_test_hidden_samples"),
        ],
        ignore_index=True,
        sort=False,
    )
    return (
        values.assign(_family_order=values["family"].map(family_order))
        .sort_values(["scale", "_family_order"])
        .drop(columns="_family_order")
        .reset_index(drop=True)
    )


def _plot_c(ax: plt.Axes) -> pd.DataFrame:
    rows = pd.read_csv(C_POSTERIOR_CSV)
    block = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["k_dim"].astype(int) == 10)
        & (rows["prior_family"].isin(["axis_edge_parallel", "axis_edge_orthogonal"]))
        & (
            rows["response_variant"].isin(
                ["zero_static", "compact_only", "compact_removed", "known_eye", "full_exact", "compact_addback"]
            )
        )
    ].copy()
    summary = (
        block.groupby(["observation_scale", "response_variant"], as_index=False)
        .agg(mean_feature_cosine=("mean_feature_cosine", "mean"))
        .sort_values("observation_scale")
    )
    summary = (
        summary.pivot(index="observation_scale", columns="response_variant", values="mean_feature_cosine")
        .reset_index()
        .rename(
            columns={
                "zero_static": "zero_mean_cosine",
                "compact_only": "compact_subspace_mean_cosine",
                "compact_removed": "compact_removed_mean_cosine",
                "known_eye": "known_mean_cosine",
                "full_exact": "full_joint_mean_cosine",
                "compact_addback": "compact_addback_mean_cosine",
            }
        )
        .sort_values("observation_scale")
    )
    x_map = {0.5: 0, 1.0: 1, 2.0: 2}
    x = summary["observation_scale"].map(x_map).astype(float).to_numpy()
    zero = summary["zero_mean_cosine"].to_numpy(dtype=float)
    compact = summary["compact_subspace_mean_cosine"].to_numpy(dtype=float)
    removed = summary["compact_removed_mean_cosine"].to_numpy(dtype=float)
    known = summary["known_mean_cosine"].to_numpy(dtype=float)
    ax.plot(x, zero, color=GRAY, marker="o", markersize=4.0, lw=1.9, label="zero eye")
    ax.plot(x, compact, color=GREEN, marker="o", markersize=4.2, lw=2.1, label="compact subspace")
    ax.plot(x, removed, color=PURPLE, marker="o", markersize=4.0, lw=1.9, label="compact removed")
    ax.plot(x, known, color=INK, lw=1.3, linestyle=":", label="known eye")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_ylim(0.48, 0.98)
    ax.legend(frameon=False, loc="lower left", borderaxespad=0.0)
    _clean_axis(ax)
    _panel_header(
        ax,
        "C",
        "Compact subspace rescues hidden-eye features",
        "removing the compact subspace collapses recovery",
    )
    return summary.assign(
        scale_label=summary["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"})
    )


def _plot_d(subspec) -> pd.DataFrame:
    summary = pd.read_csv(D_FEATURE_SUMMARY_CSV)
    block = summary[
        (summary["candidate_set_mode"] == "matched_static_response")
        & (summary["observation_scale"].astype(float) == 0.5)
        & (summary["latent"] == "pyramid_local_field")
        & (summary["requested_k"].astype(int) == 8)
        & (summary["prior_family"].isin(["axis_edge_orthogonal", "axis_edge_parallel"]))
    ].copy()
    axis_order = ["axis_edge_orthogonal", "axis_edge_parallel"]
    axis_labels = {"axis_edge_orthogonal": "across\nedge", "axis_edge_parallel": "along\nedge"}
    axis_colors = {"axis_edge_orthogonal": PURPLE, "axis_edge_parallel": GREEN}
    block["axis_order"] = block["prior_family"].map({name: i for i, name in enumerate(axis_order)})
    block = block.sort_values("axis_order")

    contrast = pd.read_csv(D_FEATURE_CONTRAST_CSV)
    contrast_row = contrast[
        (contrast["candidate_set_mode"] == "matched_static_response")
        & (contrast["observation_scale"].astype(float) == 0.5)
        & (contrast["latent"] == "pyramid_local_field")
        & (contrast["requested_k"].astype(int) == 8)
    ].iloc[0]
    hardneg = pd.read_csv(D_FEATURE_HARDNEG_CONTRAST_CSV)
    hardneg_row = hardneg[
        (hardneg["candidate_set_mode"] == "hard_negative_structure")
        & (hardneg["observation_scale"].astype(float) == 0.5)
        & (hardneg["latent"] == "pyramid_local_field")
        & (hardneg["requested_k"].astype(int) == 8)
    ].iloc[0]

    fig = plt.gcf()
    header_ax = fig.add_subplot(subspec)
    header_ax.set_axis_off()
    _panel_header(
        header_ax,
        "D",
        "Along-edge priors recover features",
        "matched-static hidden-eye decoder",
    )

    inner = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=subspec,
        width_ratios=[0.88, 1.22],
        wspace=0.24,
    )
    ax_thumb = fig.add_subplot(inner[0, 0])
    ax = fig.add_subplot(inner[0, 1])
    thumbnail_meta = _plot_d_thumbnail(ax_thumb)

    x = np.arange(block.shape[0], dtype=float)
    y = block["joint_minus_zero_feature_gain"].to_numpy(dtype=float)
    lo = block["joint_minus_zero_feature_gain_ci_low"].to_numpy(dtype=float)
    hi = block["joint_minus_zero_feature_gain_ci_high"].to_numpy(dtype=float)
    colors = [axis_colors[name] for name in block["prior_family"]]
    ax.bar(x, y, color=colors, width=0.58, alpha=0.94, zorder=3)
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=INK, lw=1.0, capsize=0, zorder=4)
    ax.axhline(0, color=INK, lw=0.8)

    diff = float(contrast_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal"])
    diff_lo = float(contrast_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_low"])
    diff_hi = float(contrast_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_high"])
    p_value = float(contrast_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_permutation_p_two_sided"])
    bracket_y = max(float(np.nanmax(hi)), float(np.nanmax(y))) + 1.15
    ax.plot([x[0], x[0], x[1], x[1]], [bracket_y - 0.35, bracket_y, bracket_y, bracket_y - 0.35], color=INK, lw=0.9)
    ax.text(
        float(x.mean()),
        bracket_y + 0.18,
        f"along-across +{diff:.2f}\np={p_value:.3f}",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=6.6,
    )

    guard_diff = float(hardneg_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal"])
    guard_lo = float(hardneg_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_low"])
    guard_hi = float(hardneg_row["mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_high"])
    ax.text(
        0.98,
        0.05,
        f"guardrail: hard-negative\naxis contrast {guard_diff:+.2f}\nCI [{guard_lo:+.2f}, {guard_hi:+.2f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=6.6,
    )

    ax.set_xticks(x, [axis_labels[name] for name in block["prior_family"]])
    ax.set_ylabel("gain [-MSE]", labelpad=1)
    ax.set_ylim(-5.0, max(16.5, bracket_y + 2.6))
    _clean_axis(ax)
    contrast_export = contrast_row.to_frame().T.assign(panel_value_type="matched_static_axis_contrast")
    hardneg_export = hardneg_row.to_frame().T.assign(panel_value_type="hard_negative_guardrail")
    thumbnail_export = pd.DataFrame(
        [{**thumbnail_meta, "panel_value_type": "edge_thumbnail_provenance"}]
    )
    return pd.concat(
        [
            block.drop(columns=["axis_order"]).assign(panel_value_type="matched_static_axis_gain"),
            contrast_export,
            hardneg_export,
            thumbnail_export,
        ],
        ignore_index=True,
        sort=False,
    )


def _behavior_bins() -> pd.DataFrame:
    windows = pd.read_csv(E_WINDOWS_CSV)
    work = windows[windows["image_feature_ok"].astype(bool)].copy()
    work["image_orientation_coherence"] = pd.to_numeric(work["image_orientation_coherence"])
    work["drift_edge_cos2"] = pd.to_numeric(work["drift_edge_cos2"])
    work = work[
        np.isfinite(work["image_orientation_coherence"]) & np.isfinite(work["drift_edge_cos2"])
    ].copy()

    bins = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, float | int]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        if hi == bins[-1]:
            block = work[(work["image_orientation_coherence"] >= lo) & (work["image_orientation_coherence"] <= hi)]
        else:
            block = work[(work["image_orientation_coherence"] >= lo) & (work["image_orientation_coherence"] < hi)]
        if block.empty:
            continue
        values = block["drift_edge_cos2"].to_numpy(dtype=float)
        sem95 = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "bin_center": float((lo + hi) / 2.0),
                "mean_edge_alignment_index": float(values.mean()),
                "ci95_low": float(values.mean() - sem95),
                "ci95_high": float(values.mean() + sem95),
                "n_windows": int(len(block)),
            }
        )
    return pd.DataFrame(rows)


def _plot_e(ax: plt.Axes) -> pd.DataFrame:
    values = _behavior_bins()
    count_ax = ax.twinx()
    ax.set_zorder(count_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    count_ax.bar(
        values["bin_center"],
        values["n_windows"],
        width=0.075,
        color=LIGHT_GRAY,
        edgecolor="none",
        zorder=0,
    )
    count_ax.set_ylabel("window count", color=MUTED)
    count_ax.tick_params(axis="y", labelsize=7.0, colors=MUTED)
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_color("#c5ccd2")

    y = values["mean_edge_alignment_index"].to_numpy(dtype=float)
    lo = values["ci95_low"].to_numpy(dtype=float)
    hi = values["ci95_high"].to_numpy(dtype=float)
    ax.errorbar(
        values["bin_center"],
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color=BLUE,
        marker="o",
        markersize=4.0,
        lw=2.0,
        capsize=0,
        zorder=3,
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.04, 0.36)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("edge-following alignment")
    _clean_axis(ax)
    _panel_header(
        ax,
        "E",
        "Real drift follows coherent edges",
        "behavioral alignment strengthens when the local edge is clear",
    )
    return values


def _write_manifest(path: Path) -> None:
    rows = [
        ("A", "One image becomes a retinal movie", A_PNG.relative_to(ATLAS).as_posix()),
        (
            "B",
            "Motion enhances feature encoding",
            f"{B_GAIN_CSV.relative_to(REPO_ROOT).as_posix()}; "
            f"{B_POSE_UNAWARE_CSV.relative_to(REPO_ROOT).as_posix()}",
        ),
        ("C", "Features survive hidden eye position", C_POSTERIOR_CSV.relative_to(REPO_ROOT).as_posix()),
        (
            "D",
            "Example edge axes plus along-edge feature recovery",
            f"{D_FEATURE_SUMMARY_CSV.relative_to(REPO_ROOT).as_posix()}; "
            f"{D_THUMBNAIL_VALUES_CSV.relative_to(REPO_ROOT).as_posix()}",
        ),
        ("E", "Real drift follows coherent edges", E_WINDOWS_CSV.relative_to(REPO_ROOT).as_posix()),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["panel", "title", "source"])
        writer.writerows(rows)


def _write_caption(path: Path) -> None:
    caption = """# Figure 4 Selected Composite v4

Status: design-first provisional draft, 2026-06-21.

This version redraws the quantitative panels for the composite instead of
pasting standalone promotion-candidate PNGs. The analysis choices are A3,
B n384 k16 corrected delta-mean feature readout, C5 feature-posterior
recovery, D matched-static along-edge feature recovery, and E3A
image-coherence behavior bridge.

Draft legend:

Figure 4. Small fixational eye movements turn a static natural image into an
informative retinal movie. (A) A recorded eye trace samples different retinal
views of the same image. (B) When the exact eye trajectory is known to the
model, recorded drift produces corrected delta-mean feature-response gain
relative to the static mean baseline; when the same empirical motion is treated
as hidden pose, the pose-unaware proxy falls below static. OU controls are held
out of the main trace set pending the dedicated trace/readout audit. (C) That feature
encoding remains recoverable when the
observer must infer features without being given the eye trace; removing the
compact subspace collapses recovery toward the zero-eye curve. (D) In the
matched-static hidden-eye feature decoder, an example natural-image edge shows
the along/across axes, and along-edge trajectory priors recover more feature
signal than matched across-edge priors; hard-negative controls keep this as a
scoped axis result rather than a universal motion policy. (E)
Measured drift shows the same
contour-following geometry most clearly when the local image supplies a
coherent edge axis. The figure supports convergence between useful
retinal-movie structure and measured behavior, not a completed proof of
behavioral optimality.
"""
    path.write_text(caption, encoding="utf-8")


def build() -> list[Path]:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, 9.55), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.45, 1.0, 1.0],
        left=0.055,
        right=0.975,
        top=0.825,
        bottom=0.07,
        hspace=0.74,
        wspace=0.48,
    )

    fig.text(
        0.055,
        0.955,
        "Figure 4. Small eye movements turn images into informative retinal movies",
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.915,
        "A retinal movie creates feature information, which a hidden-eye observer can use; local edge priors shape which motion helps.",
        ha="left",
        va="top",
        fontsize=8.6,
        color=MUTED,
    )
    fig.add_artist(plt.Line2D([0.055, 0.975], [0.89, 0.89], transform=fig.transFigure, color="#c9d0d6", lw=0.8))

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[2, 2:4])

    _plot_a(ax_a)
    b_values = _plot_b(ax_b)
    c_values = _plot_c(ax_c)
    d_values = _plot_d(gs[2, 0:2])
    e_values = _plot_e(ax_e)

    png = OUT_DIR / "figure4_selected_v4.png"
    pdf = OUT_DIR / "figure4_selected_v4.pdf"
    manifest = OUT_DIR / "figure4_selected_v4_manifest.csv"
    caption = OUT_DIR / "figure4_selected_v4_caption.md"
    b_csv = OUT_DIR / "figure4_selected_v4_panel_b_values.csv"
    c_csv = OUT_DIR / "figure4_selected_v4_panel_c_values.csv"
    d_csv = OUT_DIR / "figure4_selected_v4_panel_d_values.csv"
    e_csv = OUT_DIR / "figure4_selected_v4_panel_e_values.csv"

    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    _write_manifest(manifest)
    _write_caption(caption)
    b_values.to_csv(b_csv, index=False)
    c_values.to_csv(c_csv, index=False)
    d_values.to_csv(d_csv, index=False)
    e_values.to_csv(e_csv, index=False)
    return [png, pdf, manifest, caption, b_csv, c_csv, d_csv, e_csv]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()

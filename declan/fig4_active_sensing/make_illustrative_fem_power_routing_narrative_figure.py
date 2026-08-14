#!/usr/bin/env python3
"""Illustrative (non-confirmatory) three-panel narrative figure for the FEM
power-routing story: does FEM motion redistribute image power into temporal
frequency, does receptive-field-local power predict a unit's response, and
does that account for SSI sharpening.

This is a development/display figure, not a scientific-inference artifact. It
exists to sanity-check the narrative shape of
``declan/fig4_active_sensing/FEM_POWER_ROUTING_ANALYSIS_PLAN.md`` while the
plan's own confirmatory stages (1, 3-6) are still pending. Two constraints
follow directly from the plan's current status:

* Panel 1 is rendered fresh from raw image patches and raw eye traces only
  (never from the 3,000-condition spectral cache), because that cache was
  found to store spectra in image-grouped order against matrix-row-order
  identities (Section 2.3 of the plan) and every checkpoint built on it was
  marked ``superseded_do_not_use_for_scientific_inference`` by the Stage 0
  quarantine on 2026-08-13.
* Panel 2 reuses the recorded-grating population checkpoint
  (``rr100_recorded_grating_power_formula_population_v1``), which used real
  recorded trials and real digital-twin grating responses and was never
  built on the invalidated natural-image cache.
* Panel 3 has no valid natural-image evidence yet (Stage 5 has not run) and
  is drawn as an explicit pending schematic, not a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.run_interim_input_spectral_cache import spectral_statistics
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
)

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
RESPONSES = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
GRATING_POPULATION = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_power_formula_population_v1"
OUT = ROOT / "outputs/fig4_active_sensing/illustrative_fem_power_routing_narrative_v1"
PPD = 37.50476617
FRAME_RATE_HZ = 120.0
N_SCORE = 40

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d6ce"
SURFACE = "#fcfcfb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--response-cache-dir", type=Path, default=RESPONSES)
    parser.add_argument("--grating-population-dir", type=Path, default=GRATING_POPULATION)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def select_image(cohort_dir: Path) -> tuple[int, pd.Series]:
    images = pd.read_csv(cohort_dir / "corrected100_images.csv").set_index("image_index")
    valid = images[images.validation_pass == True]  # noqa: E712
    if valid.empty:
        raise ValueError("No validation-passing images in the corrected cohort")
    chosen = valid.sort_values("corrected_reconstruction_orientation_coherence", ascending=False).iloc[0]
    return int(chosen.name), chosen


def select_trace(response_cache_dir: Path) -> tuple[int, np.ndarray, dict[str, object]]:
    flags = pd.read_csv(response_cache_dir / "quality_control/pre_fixation_history_trace_flags.csv")
    clean = flags.loc[
        flags.history_within_selected_fixation.astype(bool)
        & flags.cache_eligibility.eq("clean_within_fixation_history")
    ].copy()
    with np.load(response_cache_dir / "input_cache/corrected_trace_segments.npz", allow_pickle=False) as archive:
        trace_ids = archive["trace_index"].astype(int)
        score = archive["score_xy_deg"].astype(float)
    lookup = {int(value): index for index, value in enumerate(trace_ids)}
    path_length = []
    for trace_id in clean.trace_index.astype(int):
        segment = score[lookup[int(trace_id)]]
        step = np.diff(segment, axis=0)
        path_length.append(float(np.sum(np.hypot(step[:, 0], step[:, 1]))))
    clean["path_length_deg"] = path_length
    clean_sorted = clean.sort_values("path_length_deg").reset_index(drop=True)
    median_row = clean_sorted.iloc[len(clean_sorted) // 2]
    trace_id = int(median_row.trace_index)
    trace_xy = score[lookup[trace_id]]
    criterion = {
        "selection_rule": "median path length among clean-history (within-fixation) traces",
        "trace_index": trace_id,
        "session": str(median_row.session),
        "path_length_deg": float(median_row.path_length_deg),
        "n_clean_traces": int(len(clean)),
    }
    return trace_id, trace_xy, criterion


def render_and_score(common, patch: np.ndarray, trace_xy: np.ndarray, device: str) -> dict[str, np.ndarray]:
    moving_trace = -np.asarray(trace_xy, dtype=np.float32)
    stabilized_trace = np.zeros_like(moving_trace)
    with torch.no_grad():
        moving_movie = render_retinal_frames_lag_zero(common, patch, moving_trace, ppd=PPD, device=device)
        stabilized_movie = render_retinal_frames_lag_zero(common, patch, stabilized_trace, ppd=PPD, device=device)
    moving_np = moving_movie.detach().cpu().numpy().astype(np.float32, copy=False)
    stabilized_np = stabilized_movie.detach().cpu().numpy().astype(np.float32, copy=False)
    moving_radial, _, moving_scalar = spectral_statistics(moving_np, ppd=PPD)
    stabilized_radial, _, stabilized_scalar = spectral_statistics(stabilized_np, ppd=PPD)
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)
    positive_tf = tf_hz[tf_hz > 0]
    return {
        "moving_movie": moving_np,
        "stabilized_movie": stabilized_np,
        "moving_trace": moving_trace,
        "positive_tf_hz": positive_tf,
        "moving_power_by_tf": moving_radial.sum(axis=1),
        "stabilized_power_by_tf": stabilized_radial.sum(axis=1),
        "moving_total_dynamic_power": float(moving_scalar[0]),
        "stabilized_total_dynamic_power": float(stabilized_scalar[0]),
    }


def load_grating_population(grating_population_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(grating_population_dir / "population_formula_summary.csv")
    manifest = json.loads((grating_population_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded = summary.loc[summary.target == "recorded"].set_index("formula")
    formulas = [
        ("rf_local_oriented_direct_f0", "orientation-aware\nRF-local power\n(primary)", BLUE),
        ("rf_local_radial_direct_f0", "orientation-collapsed\nRF-local power", ORANGE),
        ("rf_local_sf_tf_h2", "squared SF/TF\ntuning", AQUA),
        ("rf_local_total_power", "RF-local\ntotal power", YELLOW),
    ]
    rows = []
    for key, label, color in formulas:
        row = recorded.loc[key]
        rows.append(
            {
                "formula": key,
                "label": label,
                "color": color,
                "mean_cv_r2": float(row.session_balanced_mean_cv_r2),
                "ci_low": float(row.session_cluster_ci_low),
                "ci_high": float(row.session_cluster_ci_high),
                "n_units": int(row.n_units),
                "n_sessions": int(row.n_sessions),
            }
        )
    return pd.DataFrame(rows), manifest


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_figure(
    out_dir: Path,
    dpi: int,
    image_index: int,
    image_row: pd.Series,
    trace_id: int,
    trace_criterion: dict,
    patch: np.ndarray,
    render: dict[str, np.ndarray],
    grating_summary: pd.DataFrame,
    grating_manifest: dict,
) -> None:
    fig = plt.figure(figsize=(11.5, 13.5), facecolor=SURFACE)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.55, wspace=0.32, left=0.09, right=0.96, top=0.90, bottom=0.06
    )

    fig.text(
        0.5, 0.965,
        "Does FEM-driven power redistribution explain digital-twin V1 responses and SSI sharpening?",
        ha="center", va="top", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.945,
        "Illustrative narrative figure — development display only, not a scientific-inference result. See caption for what is and is not evidence.",
        ha="center", va="top", fontsize=9.5, color=ORANGE, fontstyle="italic",
    )

    # --- Panel 1a: source image with FEM trace ---
    ax_img = fig.add_subplot(gs[0, 0])
    half = patch.shape[0] / 2.0
    ax_img.imshow(patch, cmap="gray", extent=(-half / PPD, half / PPD, -half / PPD, half / PPD))
    trace = render["moving_trace"]
    trace_plot = -trace
    ax_img.plot(trace_plot[:, 0], trace_plot[:, 1], color=BLUE, linewidth=1.6, zorder=3)
    ax_img.scatter(trace_plot[0, 0], trace_plot[0, 1], color=AQUA, s=28, zorder=4, label="trace start")
    ax_img.scatter(trace_plot[-1, 0], trace_plot[-1, 1], color=ORANGE, s=28, zorder=4, label="trace end")
    ax_img.set_xlim(-51 / (2 * PPD) - 0.25, 51 / (2 * PPD) + 0.25)
    ax_img.set_ylim(-51 / (2 * PPD) - 0.25, 51 / (2 * PPD) + 0.25)
    ax_img.set_xlabel("deg", fontsize=8, color=MUTED)
    ax_img.set_ylabel("deg", fontsize=8, color=MUTED)
    ax_img.tick_params(colors=MUTED, labelsize=7)
    for spine in ax_img.spines.values():
        spine.set_color(GRID)
    legend = ax_img.legend(
        fontsize=7, frameon=True, loc="upper right", labelcolor=MUTED,
        facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
    )
    legend.set_zorder(5)
    ax_img.set_title(
        f"Panel 1a — source image {image_index} and one FEM trace\n"
        f"(session {trace_criterion['session']}, trace {trace_id}, path length {trace_criterion['path_length_deg']:.2f} deg)",
        fontsize=9.5, color=INK, loc="left",
    )

    # --- Panel 1b: temporal power spectrum, moving vs stabilized ---
    ax_tf = fig.add_subplot(gs[0, 1])
    style_axis(ax_tf)
    tf = render["positive_tf_hz"]
    moving_power = render["moving_power_by_tf"]
    floor = max(float(moving_power.max()) * 1e-6, 1e-12)
    ax_tf.plot(tf, np.maximum(moving_power, floor), color=BLUE, linewidth=2.0, label="moving retina (real FEM trace)")
    ax_tf.axhline(floor, color=ORANGE, linewidth=2.0, linestyle="--", label="stabilized retina (dynamic power ≈ 0)")
    ax_tf.set_yscale("log")
    ax_tf.set_xlabel("temporal frequency (Hz)", fontsize=8.5, color=MUTED)
    ax_tf.set_ylabel("retinal image power (a.u., log scale)", fontsize=8.5, color=MUTED)
    ax_tf.legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=MUTED)
    moving_total = render["moving_total_dynamic_power"]
    stabilized_total = render["stabilized_total_dynamic_power"]
    ax_tf.set_title(
        "Panel 1b — FEM redistributes power to temporal frequency\n"
        f"total dynamic power: moving={moving_total:.3g}, stabilized={stabilized_total:.3g}",
        fontsize=9, color=INK, loc="left",
    )

    # --- Panel 2: recorded-grating population, held-out R^2 by power formula ---
    ax_bar = fig.add_subplot(gs[1, :])
    style_axis(ax_bar)
    x = np.arange(len(grating_summary))
    bars = ax_bar.bar(
        x, grating_summary.mean_cv_r2, width=0.6, color=grating_summary.color,
        yerr=[
            grating_summary.mean_cv_r2 - grating_summary.ci_low,
            grating_summary.ci_high - grating_summary.mean_cv_r2,
        ],
        error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 3}, zorder=3,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(grating_summary.label, fontsize=8.5, color=MUTED)
    ax_bar.set_ylabel("held-out fraction of firing-rate\nvariance explained (session-balanced $R^2$)", fontsize=8.5, color=MUTED)
    ax_bar.axhline(0.0, color=GRID, linewidth=1.0)
    for rect, value in zip(bars, grating_summary.mean_cv_r2):
        ax_bar.text(
            rect.get_x() + rect.get_width() / 2, value + 0.003, f"{value:.3f}",
            ha="center", va="bottom", fontsize=8, color=INK,
        )
    n_units = int(grating_summary.n_units.iloc[0])
    n_sessions = int(grating_summary.n_sessions.iloc[0])
    ax_bar.set_title(
        "Panel 2 — receptive-field-local power predicts recorded firing rate on held-out grating trials\n"
        f"digital-twin grating tuning, recorded responses, n={n_units} units × {n_sessions} sessions, "
        "5-fold held-out trials, error bars: session-balanced 95% bootstrap CI\n"
        "(this is real, validated evidence — the only quantitative result behind this figure that is not illustrative)",
        fontsize=9.5, color=INK, loc="left",
    )

    # --- Panel 3: SSI schematic, explicitly pending ---
    ax_ssi = fig.add_subplot(gs[2, :])
    ax_ssi.set_facecolor(SURFACE)
    ax_ssi.set_xlim(0, 1)
    ax_ssi.set_ylim(0, 1)
    ax_ssi.axis("off")
    ax_ssi.add_patch(
        plt.Rectangle((0.02, 0.05), 0.96, 0.85, fill=False, edgecolor=GRID, linewidth=1.2, linestyle="--")
    )
    ax_ssi.text(
        0.5, 0.82,
        "Panel 3 — does power-derived local activation explain FEM-induced SSI sharpening?",
        ha="center", va="top", fontsize=10.5, color=INK, fontweight="bold",
    )
    ax_ssi.text(
        0.5, 0.62,
        "No figure is shown here because no valid natural-image evidence exists yet.",
        ha="center", va="top", fontsize=10, color=ORANGE, fontstyle="italic",
    )
    ax_ssi.text(
        0.5, 0.40,
        "The natural-image spectral cache behind the earlier SSI-adjacent checkpoints was found to be row-misaligned\n"
        "(2,980 of 3,000 rows) and was quarantined on 2026-08-13. The plan's own decision gate (Section 9) requires the\n"
        "receptive-field calibration (Stage 3), frozen power-to-rate calibration, and held-out SSI evaluation (Stage 5)\n"
        "before any map-derived SSI number can be reported — none of those have run yet.",
        ha="center", va="top", fontsize=8.5, color=MUTED,
    )

    caption = (
        "Panels 1a-1b are rendered fresh from the raw image cohort and raw eye-trace cache for one illustrative "
        "example (image and trace selected by the saved criteria above; not a population claim). Panel 2 reproduces "
        "outputs/fig4_active_sensing/rr100_recorded_grating_power_formula_population_v1, a completed, valid checkpoint "
        "built from recorded grating trials. Panel 3 is a schematic placeholder, not data. None of these panels "
        "substitute for the plan's Stage 1 (corrected natural-image spectral cache), Stage 3 (receptive-field "
        "calibration), Stage 4 (targeted activation maps), or Stage 5 (SSI diagnostic)."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7.2, color=MUTED, wrap=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "illustrative_fem_power_routing_narrative.png", dpi=dpi, facecolor=SURFACE)
    fig.savefig(out_dir / "illustrative_fem_power_routing_narrative.pdf", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_index, image_row = select_image(args.cohort_dir)
    trace_id, trace_xy, trace_criterion = select_trace(args.response_cache_dir)

    with np.load(Path(str(image_row.corrected_patch_npz)), allow_pickle=False) as archive:
        patch = _standardize_uint_like(np.asarray(archive[str(image_row.corrected_patch_key)], dtype=np.float32))

    common = _load_twin_common()
    render = render_and_score(common, patch, trace_xy, args.device)

    grating_summary, grating_manifest = load_grating_population(args.grating_population_dir)

    plot_figure(
        args.out_dir, args.dpi, image_index, image_row, trace_id, trace_criterion, patch, render,
        grating_summary, grating_manifest,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "illustrative_fem_power_routing_narrative",
        "status": "illustrative_development_display_not_scientific_inference",
        "plan": "declan/fig4_active_sensing/FEM_POWER_ROUTING_ANALYSIS_PLAN.md",
        "panel_1_source": {
            "kind": "freshly_rendered_single_example_no_cache_dependency",
            "image_index": image_index,
            "image_selection_rule": "validation_pass image with highest corrected_reconstruction_orientation_coherence",
            "trace_index": trace_id,
            "trace_selection": trace_criterion,
            "cohort_dir": file_identity(args.cohort_dir / "corrected100_images.csv"),
            "trace_flags": file_identity(args.response_cache_dir / "quality_control/pre_fixation_history_trace_flags.csv"),
            "trace_segments": file_identity(args.response_cache_dir / "input_cache/corrected_trace_segments.npz"),
            "moving_total_dynamic_power": render["moving_total_dynamic_power"],
            "stabilized_total_dynamic_power": render["stabilized_total_dynamic_power"],
        },
        "panel_2_source": {
            "kind": "reused_valid_completed_checkpoint",
            "dir": str(args.grating_population_dir),
            "manifest": grating_manifest,
        },
        "panel_3_source": {
            "kind": "schematic_placeholder_no_data",
            "reason": "Stage 5 (SSI prediction and spatial-mechanism diagnostic) has not run",
        },
        "superseded_inputs_excluded": [
            "outputs/fig4_active_sensing/rr100_corrected_three_round_spectral_cache_v1",
            "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_*_clean_history*",
        ],
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {args.out_dir / 'illustrative_fem_power_routing_narrative.png'}")


if __name__ == "__main__":
    main()

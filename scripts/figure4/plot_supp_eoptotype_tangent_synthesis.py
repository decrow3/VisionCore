#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from VisionCore.paths import VISIONCORE_ROOT


BLUE = "#0072B2"
GRAY = "#4D4D4D"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#7B3294"


def configure_style() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "exports": out_dir / "exports",
        "source_tables": out_dir / "source_tables",
        "qc": out_dir / "qc",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=12)


def save_all(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(VISIONCORE_ROOT))
    except ValueError:
        return str(path)


def write_caption_and_methods(
    paths: dict[str, Path],
    *,
    local_j: pd.DataFrame,
    closure: pd.DataFrame,
    step_sens: pd.DataFrame,
    contrasts: pd.DataFrame,
    phase_summary: pd.DataFrame,
    occupancy: pd.DataFrame,
) -> None:
    b_first = local_j.sort_values("bin_mid_px").iloc[0]
    b_last = local_j.sort_values("bin_mid_px").iloc[-1]
    sample_fd = closure[
        (closure["source_label"] == "Samplewise FD")
        & (closure["control_label"] == "global")
    ].iloc[0]
    sample_fd_strict = closure[
        (closure["source_label"] == "Samplewise FD")
        & (closure["control_label"] == "global+PC1")
    ].iloc[0]
    compact = closure[
        (closure["source_label"] == "Compact k=10")
        & (closure["control_label"] == "global")
    ].iloc[0]
    compact_strict = closure[
        (closure["source_label"] == "Compact k=10")
        & (closure["control_label"] == "global+PC1")
    ].iloc[0]
    c60 = contrasts[contrasts["window"].astype(int) == 60].sort_values("logmar")
    interpretable_c60 = c60
    if "effect_status" in c60.columns:
        interpretable_c60 = c60[~c60["effect_status"].astype(str).str.contains("render", case=False, na=False)]
    d_neg = interpretable_c60.loc[interpretable_c60["delta_accuracy"].idxmin()]
    d_pos = interpretable_c60.loc[interpretable_c60["delta_accuracy"].idxmax()]
    e_rows = phase_summary[np.isclose(phase_summary["logmar"].astype(float), -0.35)]
    occ = occupancy[np.isclose(occupancy["logmar"].astype(float), -0.35)]

    caption_lines = [
        "# Supplemental Figure Caption: E-optotype tangent synthesis",
        "",
        "**Figure Sx. Controlled E-optotype geometry links local translation tangents, finite FEM covariance, and identity/pose mimicry.**",
        "",
        "**A. Local derivative, finite covariance.** Schematic of the interpretive distinction used throughout the supplement. A single image-translation Jacobian is a local derivative and should not be treated as a global linear model over a full finite eye-trace cloud. The relevant stronger object is the family of local translation generators and its induced covariance footprint.",
        "",
        (
            "**B. Pointwise prediction is local.** Cross-session local-J validation in four Allen sessions shows that pointwise Taylor prediction is strongest at small pairwise displacements "
            f"(mean R2 = {float(b_first['r2_lin_local_mean']):.2f} at {float(b_first['bin_mid_px']):.2f} model px) and degrades at larger offsets "
            f"(mean R2 = {float(b_last['r2_lin_local_mean']):.2f} at {float(b_last['bin_mid_px']):.2f} model px). "
            "In contrast, the translation-tangent covariance-capture delta remains positive over the tested displacement range, separating pointwise local linearization from finite-cloud covariance geometry."
        ),
        "",
        (
            "**C. Finite-cloud covariance closure.** Matched recorded/twin finite-difference analyses show that samplewise translation-tangent covariance captures a reliable component of recorded FEM covariance over unit-shuffle controls "
            f"after removing both global-rate and target-PC1 modes (effect = {float(sample_fd_strict['effect_unit_mean']):.2f}, bootstrap CI "
            f"[{float(sample_fd_strict['effect_unit_boot_ci_low']):.2f}, {float(sample_fd_strict['effect_unit_boot_ci_high']):.2f}]; "
            f"global-rate-only effect = {float(sample_fd['effect_unit_mean']):.2f}, CI "
            f"[{float(sample_fd['effect_unit_boot_ci_low']):.2f}, {float(sample_fd['effect_unit_boot_ci_high']):.2f}]). "
            f"Projecting the samplewise finite-difference predictions into the compact k=10 tangent subspace preserves the effect "
            f"(global+PC1 effect = {float(compact_strict['effect_unit_mean']):.2f}; global-rate-only effect = {float(compact['effect_unit_mean']):.2f}). "
            "Inset: Allen-session finite-difference step sensitivity for 0.25, 0.5, and 1.0 model px."
        ),
        "",
        (
            "**D. E-optotype scale dependence.** Canonical four-way E-orientation decoding compares real FEM traces with stationary controls using the validated mono-population Model A time-mean-rate decoder. "
            f"Real-minus-stationary accuracy is most negative at LogMAR {float(d_neg['logmar']):.2f} "
            f"(delta = {float(d_neg['delta_accuracy']):.2f}) and becomes most positive at LogMAR {float(d_pos['logmar']):.2f} "
            f"(delta = {float(d_pos['delta_accuracy']):.2f}); LogMAR -0.40 is retained as a render-limit control."
        ),
        "",
        (
            "**E. Translation mimicry is pair-specific.** At LogMAR -0.35, mean phase-resolved mimicry varies across ordered E-orientation pairs "
            f"(range {float(e_rows['mimicry_mean'].min()):.2f}-{float(e_rows['mimicry_mean'].max()):.2f}), indicating that retinal translations can overlap identity differences in a pair-dependent way."
        ),
        "",
        (
            "**F. Mimicry varies over retinal phase.** Phase-resolved mimicry for a representative orientation pair shows strong spatial heterogeneity across subpixel retinal phase. "
            f"Inset compares center-phase mimicry with real-FEM occupancy-weighted mimicry across pairs and scales (n = {len(occ)} ordered pairs at LogMAR -0.35), showing why center-phase summaries are not a sufficient description of the finite eye-trace distribution."
        ),
        "",
        "**Interpretation boundary.** E-optotypes are used here as controlled artificial stimuli. The figure supports the geometry and caveat language for the manuscript supplement; it is not the primary biological evidence for the recorded V1 covariance claim.",
    ]
    (paths["root"] / "FigureS_eoptotype_tangent_synthesis_caption.md").write_text("\n".join(caption_lines) + "\n")

    source_root = VISIONCORE_ROOT / "outputs"
    methods_lines = [
        "# Companion Methods Note: Supplemental E-optotype tangent synthesis",
        "",
        "## Purpose",
        "",
        "This note documents the analysis streams used to build the supplemental E-optotype/tangent synthesis figure. The figure is intended as a controlled-stimulus supplement that connects three previously separate results: local Jacobian validity, finite-cloud translation-tangent covariance closure, and E-orientation mimicry over retinal phase.",
        "",
        "The bounded claim is that pointwise Taylor prediction is local, but the covariance geometry induced by a family of local translation generators can remain stable and useful at the scale of measured FEM traces. The E-optotype analyses are not used as the main biological evidence; they provide an artificial but interpretable stimulus family for stress-testing the geometry.",
        "",
        "## Reproducibility",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python scripts/figure4/plot_supp_eoptotype_tangent_synthesis.py",
        "```",
        "",
        "The script writes figure exports to:",
        "",
        f"- `{rel(paths['exports'] / 'FigureS_eoptotype_tangent_synthesis.png')}`",
        f"- `{rel(paths['exports'] / 'FigureS_eoptotype_tangent_synthesis.pdf')}`",
        f"- `{rel(paths['exports'] / 'FigureS_eoptotype_tangent_synthesis.svg')}`",
        "",
        "It also writes source tables under:",
        "",
        f"- `{rel(paths['source_tables'])}`",
        "",
        "## Input Data Streams",
        "",
        "No new model inference is performed by this plotting script. It consumes previously generated summaries:",
        "",
        f"- Local Jacobian validation: `{rel(source_root / 'jacobian_predictive_framework' / 'cross_session_local_J_v2' / 'cross_session_summary.json')}`",
        f"- Full finite-difference covariance closure: `{rel(source_root / 'matched_twin_covariance_closure_finite_difference' / 'finite_difference_bootstrap_summary.csv')}`",
        f"- Compact k=10 covariance closure: `{rel(source_root / 'matched_twin_covariance_closure_finite_difference_compact_k10' / 'finite_difference_bootstrap_summary.csv')}`",
        f"- Allen finite-difference step 0.25 px: `{rel(source_root / 'matched_twin_covariance_closure_fd_allen_step025' / 'finite_difference_metric_summary.csv')}`",
        f"- Allen finite-difference step 0.50 px: `{rel(source_root / 'matched_twin_covariance_closure_fd_allen_full' / 'finite_difference_metric_summary.csv')}`",
        f"- Allen finite-difference step 1.00 px: `{rel(source_root / 'matched_twin_covariance_closure_fd_allen_step100' / 'finite_difference_metric_summary.csv')}`",
        f"- Canonical E-optotype discrimination: `{rel(source_root / 'figure4_reconciliation' / 'canonical_discrimination' / 'canonical_real_minus_stabilized.csv')}`",
        f"- Phase-landscape mimicry summary: `{rel(source_root / 'figure4_reconciliation' / 'validated_mimicry' / 'phase_landscape_summary.csv')}`",
        f"- Pairwise phase-resolved mimicry: `{rel(source_root / 'figure4_reconciliation' / 'validated_mimicry' / 'pairwise_mimicry_by_phase.csv')}`",
        f"- Occupancy-weighted mimicry: `{rel(source_root / 'figure4_reconciliation' / 'validated_mimicry' / 'occupancy_weighted_mimicry.csv')}`",
        "",
        "## Panel Methods",
        "",
        "### Panel A: Conceptual schematic",
        "",
        "Panel A is a programmatic schematic drawn in matplotlib. It does not encode measured values. Its role is to keep the interpretation boundary explicit: one local Jacobian is not a global linear model of a finite FEM cloud, but a family of local translation generators can still induce a finite covariance footprint.",
        "",
        "### Panel B: Local-J validation versus finite covariance geometry",
        "",
        "The script reads `pairwise_by_session` from the cross-session local-J JSON summary. For each displacement bin, it averages session-level pointwise linear-prediction R2 and covariance-capture delta across sessions, using SEM across sessions for error bars. The left axis plots pointwise R2; the right axis plots the tangent covariance-capture delta over shuffle. This panel is designed to show that pointwise Taylor prediction and covariance geometry are related but not identical regimes.",
        "",
        "Source table:",
        "",
        f"- `{rel(paths['source_tables'] / 'panel_b_local_j_summary.csv')}`",
        "",
        "### Panel C: Finite-difference covariance closure",
        "",
        "The main bars use rows with `target_variant == psd` and `k == 2` from the matched recorded/twin finite-difference closure summaries. The plotted basis sources are `fd_mean_tangent_matrix`, `fd_sample_eye_trace_cov`, and `fd_sample_eye_trace_xfit_compact_k10_cov`. Bars show the mean effect over unit-shuffled controls, with bootstrap confidence intervals from the summary tables. The three projection controls are no projection control, global-rate control, and global-rate plus target-PC1 control.",
        "Caption text leads with the most conservative global-rate plus target-PC1 control; the larger global-rate-only number is labeled separately to avoid conflating the two closure controls.",
        "",
        "The inset uses Allen-session finite-difference metric summaries for step sizes 0.25, 0.5, and 1.0 model px. It plots `capture_mean` for `basis_source == fd_sample_eye_trace_cov`, `projection_control == global_rate`, `target_variant == psd`, and `k == 2`.",
        "",
        "Source tables:",
        "",
        f"- `{rel(paths['source_tables'] / 'panel_c_closure_bars.csv')}`",
        f"- `{rel(paths['source_tables'] / 'panel_c_step_sensitivity.csv')}`",
        "",
        "### Panel D: Canonical E-optotype discrimination",
        "",
        "Panel D uses the validated mono-population Model A canonical discrimination summary. The plot is restricted to the 60-frame window and the four-way orientation task. It compares real FEM traces with stationary controls as a function of LogMAR. The inset plots real-minus-stationary accuracy and its bootstrap interval. LogMAR -0.40 is shaded as a render-limit control rather than interpreted as the main behavioral regime.",
        "",
        "Source table:",
        "",
        f"- `{rel(paths['source_tables'] / 'panel_d_canonical_window60.csv')}`",
        "",
        "### Panel E: Ordered-pair mimicry matrix",
        "",
        "Panel E filters the phase-landscape summary to LogMAR -0.35 and plots mean mimicry for ordered source-target E-orientation pairs. The diagonal is left undefined because the mimicry question concerns whether translations of one identity resemble another identity.",
        "",
        "Source table:",
        "",
        f"- `{rel(paths['source_tables'] / 'panel_e_phase_summary.csv')}`",
        "",
        "### Panel F: Phase-resolved and occupancy-weighted mimicry",
        "",
        "The main heatmap plots phase-resolved mimicry for the representative ordered pair `90_vs_180` at LogMAR -0.35. Phase coordinates are shown in arcminutes. The inset compares center-phase mimicry with real-FEM occupancy-weighted mimicry across ordered pairs and LogMAR values, highlighting that a fixed-center phase can misrepresent the mimicry encountered along the measured finite trace distribution.",
        "",
        "Source tables:",
        "",
        f"- `{rel(paths['source_tables'] / 'panel_f_pairwise_mimicry_by_phase_logmar_m035.csv')}`",
        f"- `{rel(paths['source_tables'] / 'panel_f_occupancy.csv')}`",
        "",
        "## Controls and Interpretation Boundaries",
        "",
        "- Pointwise first-order prediction is interpreted locally. The figure avoids claiming that one baseline Jacobian globally linearizes the full FEM response cloud.",
        "- Finite-cloud covariance closure is interpreted as a partial geometry bridge. The closure effects are compared against unit-shuffled controls and projection controls, but are not claimed to explain all recorded FEM covariance.",
        "- The compact k=10 result is interpreted as preservation of the controlled covariance effect after projection into a compact tangent subspace, not as evidence for a complete biological identity between the twin and recorded V1.",
        "- E-optotype discrimination is treated as a controlled artificial-stimulus result. It is useful for showing scale dependence and mimicry geometry, but it is not the main ecological or biological evidence.",
        "- The render-limit LogMAR -0.40 condition is retained as a control and should not anchor the main interpretation.",
    ]
    (paths["root"] / "FigureS_eoptotype_tangent_synthesis_methods.md").write_text("\n".join(methods_lines) + "\n")


def load_cross_session_local_j(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for session, bins in payload["pairwise_by_session"].items():
        for row in bins:
            r = dict(row)
            r["session"] = session
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("bin_mid_px", as_index=False)
        .agg(
            r2_lin_local_mean=("r2_lin_local_mean", "mean"),
            r2_lin_local_sem=("r2_lin_local_mean", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(len(x)))),
            cosine_local_mean=("cosine_local_mean", "mean"),
            capture_V_J_local_delta_mean=("capture_V_J_local_delta_mean", "mean"),
            capture_V_J_local_delta_sem=("capture_V_J_local_delta_mean", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(len(x)))),
            n_sessions=("session", "nunique"),
        )
        .sort_values("bin_mid_px")
    )
    return agg


def load_closure_bars(full_path: Path, compact_path: Path) -> pd.DataFrame:
    full = pd.read_csv(full_path)
    compact = pd.read_csv(compact_path)
    rows: list[pd.Series] = []
    specs = [
        (full, "fd_mean_tangent_matrix", "Mean tangent"),
        (full, "fd_sample_eye_trace_cov", "Samplewise FD"),
        (compact, "fd_sample_eye_trace_xfit_compact_k10_cov", "Compact k=10"),
    ]
    controls = [("none", "none"), ("global_rate", "global"), ("global_rate+target_pc1", "global+PC1")]
    for df, source, label in specs:
        for ctrl, ctrl_label in controls:
            hit = df[
                (df["target_variant"] == "psd")
                & (df["projection_control"] == ctrl)
                & (df["basis_source"] == source)
                & (df["k"].astype(int) == 2)
            ]
            if hit.empty:
                continue
            row = hit.iloc[0].copy()
            row["source_label"] = label
            row["control_label"] = ctrl_label
            rows.append(row)
    return pd.DataFrame(rows)


def load_step_sensitivity(roots: dict[float, Path]) -> pd.DataFrame:
    rows = []
    for step, path in roots.items():
        df = pd.read_csv(path)
        hit = df[
            (df["target_variant"] == "psd")
            & (df["projection_control"] == "global_rate")
            & (df["basis_source"] == "fd_sample_eye_trace_cov")
            & (df["k"].astype(int) == 2)
        ]
        if not hit.empty:
            row = hit.iloc[0].to_dict()
            row["fd_step_px"] = step
            rows.append(row)
    return pd.DataFrame(rows)


def plot_panel_a(ax: plt.Axes) -> None:
    ax.axis("off")
    t = np.linspace(-2.4, 2.4, 200)
    x = t
    y = 0.28 * t**2 - 0.15 * t
    ax.plot(x, y, color="0.35", lw=1.6)
    p0 = np.array([-0.25, 0.28 * (-0.25) ** 2 - 0.15 * (-0.25)])
    slope = 0.56 * (-0.25) - 0.15
    line_x = np.linspace(-1.25, 0.75, 2)
    line_y = p0[1] + slope * (line_x - p0[0])
    ax.plot(line_x, line_y, color=BLUE, lw=1.5)
    cloud = np.array(
        [
            [-0.65, 0.10],
            [-0.35, 0.02],
            [-0.05, 0.08],
            [0.18, 0.22],
            [0.42, 0.36],
            [0.70, 0.55],
        ]
    )
    ax.scatter(cloud[:, 0], cloud[:, 1], s=18, color=ORANGE, alpha=0.85, zorder=3)
    ax.add_patch(Ellipse((0.04, 0.22), width=1.55, height=0.75, angle=18, fill=False, lw=1.0, ls="--", edgecolor=ORANGE, alpha=0.8))
    ax.scatter([p0[0]], [p0[1]], s=28, color=BLUE, zorder=4)
    ax.annotate("local tangent", xy=(-0.65, line_y[0]), xytext=(-2.1, 0.9), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=7)
    ax.annotate("finite eye-trace cloud", xy=(0.55, 0.48), xytext=(0.35, 1.35), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=7)
    ax.text(
        -2.35,
        -0.74,
        "One Jacobian is local.\nA tangent family can leave a finite-cloud covariance footprint.",
        fontsize=7,
        linespacing=1.15,
    )
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(-0.9, 1.75)
    ax.set_title("Local derivative, finite covariance")


def plot_panel_b(ax: plt.Axes, local_j: pd.DataFrame) -> None:
    x = local_j["bin_mid_px"].to_numpy(float)
    r2 = local_j["r2_lin_local_mean"].to_numpy(float)
    r2_sem = local_j["r2_lin_local_sem"].to_numpy(float)
    cap = local_j["capture_V_J_local_delta_mean"].to_numpy(float)
    cap_sem = local_j["capture_V_J_local_delta_sem"].to_numpy(float)
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.errorbar(x, r2, yerr=r2_sem, marker="o", ms=3.5, lw=1.0, color=BLUE, label="Taylor R2")
    ax.set_xlabel("Pairwise displacement (model px)")
    ax.set_ylabel("Pointwise prediction R2", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(-4.6, 1.05)
    ax.grid(alpha=0.14, lw=0.5)
    ax2 = ax.twinx()
    ax2.errorbar(x, cap, yerr=cap_sem, marker="s", ms=3.2, lw=1.0, color=GREEN, label="tangent capture over shuffle")
    ax2.set_ylabel("Covariance capture delta", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.set_ylim(0.0, max(0.34, float(np.nanmax(cap + cap_sem) * 1.25)))
    ax.set_title("Pointwise prediction is local")
    ax.text(0.03, 0.08, "4 Allen sessions; mean +/- SEM", transform=ax.transAxes, fontsize=6.3)


def plot_panel_c(ax: plt.Axes, closure: pd.DataFrame, step_sens: pd.DataFrame) -> None:
    ax.axis("off")
    main = ax.inset_axes([0.00, 0.03, 0.68, 0.92])
    inset = ax.inset_axes([0.76, 0.16, 0.22, 0.62])
    order = ["Mean tangent", "Samplewise FD", "Compact k=10"]
    colors = {"none": BLUE, "global": GREEN, "global+PC1": PURPLE}
    x0 = np.arange(len(order), dtype=float)
    width = 0.22
    for j, ctrl in enumerate(["none", "global", "global+PC1"]):
        vals = []
        lo = []
        hi = []
        for label in order:
            hit = closure[(closure["source_label"] == label) & (closure["control_label"] == ctrl)]
            if hit.empty:
                vals.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
            else:
                r = hit.iloc[0]
                vals.append(float(r["effect_unit_mean"]))
                lo.append(float(r["effect_unit_mean"] - r["effect_unit_boot_ci_low"]))
                hi.append(float(r["effect_unit_boot_ci_high"] - r["effect_unit_mean"]))
        vals_arr = np.asarray(vals, dtype=float)
        yerr = np.vstack([lo, hi]).astype(float)
        main.bar(x0 + (j - 1) * width, vals_arr, width=width, yerr=yerr, color=colors[ctrl], alpha=0.86, label=ctrl, capsize=2, lw=0)
    main.axhline(0, color="0.5", lw=0.8)
    main.set_xticks(x0)
    main.set_xticklabels(order, rotation=20, ha="right")
    main.set_ylabel("Effect over unit shuffle")
    main.set_title("Finite-cloud covariance closure")
    main.grid(axis="y", alpha=0.14, lw=0.5)
    main.legend(frameon=False, fontsize=5.8, loc="upper left")

    if not step_sens.empty:
        s = step_sens.sort_values("fd_step_px")
        inset.plot(s["fd_step_px"], s["capture_mean"], marker="o", color=GRAY, lw=1.0, ms=3.2)
        inset.set_xlabel("FD step (px)", fontsize=6)
        inset.set_ylabel("", fontsize=6, labelpad=1)
        inset.tick_params(labelsize=6)
        inset.set_title("Allen step\nsensitivity", fontsize=6.5)
        inset.grid(alpha=0.12, lw=0.4)


def plot_panel_d(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    c60 = contrasts[contrasts["window"].astype(int) == 60].sort_values("logmar")
    x = c60["logmar"].to_numpy(float)
    real = c60["real_accuracy"].to_numpy(float)
    stab = c60["stabilized_accuracy"].to_numpy(float)
    delta = c60["delta_accuracy"].to_numpy(float)
    lo = c60["delta_ci_low"].to_numpy(float)
    hi = c60["delta_ci_high"].to_numpy(float)
    ax.plot(x, real, color=BLUE, marker="o", lw=1.0, ms=3.4, label="real FEM")
    ax.plot(x, stab, color=GRAY, marker="o", lw=1.0, ms=3.4, label="stationary")
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("4-way accuracy")
    ax.set_ylim(0.58, 0.94)
    ax.grid(alpha=0.14, lw=0.5)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    ax.axvspan(-0.405, -0.395, color="0.9", zorder=-1)
    ax.text(-0.399, 0.925, "render\ncontrol", fontsize=5.5, ha="center", va="top")
    inset = ax.inset_axes([0.10, 0.10, 0.43, 0.35])
    inset.axhline(0, color="0.35", lw=0.75)
    inset.plot(x, delta, color="black", marker="o", lw=0.85, ms=2.8)
    inset.fill_between(x, lo, hi, color="black", alpha=0.14)
    inset.set_title("real - stationary", fontsize=6.5)
    inset.tick_params(labelsize=5.8)
    inset.set_ylim(min(-0.14, float(np.nanmin(lo)) - 0.01), max(0.08, float(np.nanmax(hi)) + 0.01))
    inset.grid(alpha=0.10, lw=0.4)
    ax.set_title("E-optotype reveals scale dependence")


def mimicry_matrix(summary: pd.DataFrame, logmar: float, value_col: str = "mimicry_mean") -> tuple[np.ndarray, list[int]]:
    orientations = [0, 90, 180, 270]
    mat = np.full((len(orientations), len(orientations)), np.nan)
    rows = summary[np.isclose(summary["logmar"].astype(float), float(logmar))]
    for _, r in rows.iterrows():
        a_s, b_s = str(r["orientation_pair"]).split("_vs_")
        a = int(a_s)
        b = int(b_s)
        if a in orientations and b in orientations:
            mat[orientations.index(a), orientations.index(b)] = float(r[value_col])
    return mat, orientations


def plot_panel_e(ax: plt.Axes, phase_summary: pd.DataFrame) -> None:
    mat, orientations = mimicry_matrix(phase_summary, -0.35)
    im = ax.imshow(mat, vmin=0.0, vmax=0.6, cmap="Blues")
    ax.set_xticks(range(len(orientations)), orientations)
    ax.set_yticks(range(len(orientations)), orientations)
    ax.set_xlabel("target orientation")
    ax.set_ylabel("source orientation")
    ax.set_title("Translation mimicry is pair-specific")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j or not np.isfinite(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.0, color="black" if mat[i, j] < 0.42 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mean mimicry")


def plot_panel_f(ax: plt.Axes, pairwise: pd.DataFrame, occupancy: pd.DataFrame) -> None:
    logmar = -0.35
    rows = pairwise[(np.isclose(pairwise["logmar"].astype(float), logmar)) & (pairwise["orientation_pair"] == "90_vs_180")]
    if rows.empty:
        rows = pairwise[np.isclose(pairwise["logmar"].astype(float), logmar)]
    xs = np.sort(rows["phase_x"].unique().astype(float))
    ys = np.sort(rows["phase_y"].unique().astype(float))
    grid = np.full((len(xs), len(ys)), np.nan)
    if "orientation_pair" in rows.columns and "90_vs_180" not in set(rows["orientation_pair"]):
        grouped = rows.groupby(["phase_x", "phase_y"])["mimicry_fraction"].mean().reset_index()
    else:
        grouped = rows
    for _, r in grouped.iterrows():
        xi = int(np.argmin(np.abs(xs - float(r["phase_x"]))))
        yi = int(np.argmin(np.abs(ys - float(r["phase_y"]))))
        grid[xi, yi] = float(r["mimicry_fraction"])
    im = ax.imshow(grid.T, origin="lower", extent=[xs.min() * 60, xs.max() * 60, ys.min() * 60, ys.max() * 60], cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xlabel("phase x (arcmin)")
    ax.set_ylabel("phase y (arcmin)")
    ax.set_title("Mimicry varies over retinal phase")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mimicry")

    inset = ax.inset_axes([0.62, 0.08, 0.30, 0.30])
    inset.scatter(occupancy["center_mimicry"], occupancy["weighted_mimicry_mean"], c=occupancy["logmar"], cmap="viridis", s=12, alpha=0.85)
    lo = min(float(occupancy["center_mimicry"].min()), float(occupancy["weighted_mimicry_mean"].min()))
    hi = max(float(occupancy["center_mimicry"].max()), float(occupancy["weighted_mimicry_mean"].max()))
    inset.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="0.5")
    inset.set_xlabel("", fontsize=5.4, labelpad=0.5)
    inset.set_ylabel("", fontsize=5.4, labelpad=0.5)
    inset.tick_params(labelsize=5.0, pad=1)
    inset.set_title("occupancy", fontsize=6.0)
    inset.grid(alpha=0.10, lw=0.4)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a synthetic supplemental figure for E-optotype tangent/mimicry results.")
    p.add_argument("--out-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "supp_eoptotype_tangent_synthesis")
    p.add_argument("--dpi", type=int, default=450)
    args = p.parse_args()

    configure_style()
    paths = ensure_dirs(args.out_dir)

    local_j = load_cross_session_local_j(VISIONCORE_ROOT / "outputs" / "jacobian_predictive_framework" / "cross_session_local_J_v2" / "cross_session_summary.json")
    closure = load_closure_bars(
        VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_finite_difference" / "finite_difference_bootstrap_summary.csv",
        VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_finite_difference_compact_k10" / "finite_difference_bootstrap_summary.csv",
    )
    step_sens = load_step_sensitivity(
        {
            0.25: VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_fd_allen_step025" / "finite_difference_metric_summary.csv",
            0.50: VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_fd_allen_full" / "finite_difference_metric_summary.csv",
            1.00: VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_fd_allen_step100" / "finite_difference_metric_summary.csv",
        }
    )
    contrasts = pd.read_csv(VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "canonical_discrimination" / "canonical_real_minus_stabilized.csv")
    phase_summary = pd.read_csv(VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "validated_mimicry" / "phase_landscape_summary.csv")
    pairwise = pd.read_csv(VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "validated_mimicry" / "pairwise_mimicry_by_phase.csv")
    occupancy = pd.read_csv(VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "validated_mimicry" / "occupancy_weighted_mimicry.csv")

    local_j.to_csv(paths["source_tables"] / "panel_b_local_j_summary.csv", index=False)
    closure.to_csv(paths["source_tables"] / "panel_c_closure_bars.csv", index=False)
    step_sens.to_csv(paths["source_tables"] / "panel_c_step_sensitivity.csv", index=False)
    contrasts[contrasts["window"].astype(int) == 60].to_csv(paths["source_tables"] / "panel_d_canonical_window60.csv", index=False)
    phase_summary.to_csv(paths["source_tables"] / "panel_e_phase_summary.csv", index=False)
    pairwise[np.isclose(pairwise["logmar"].astype(float), -0.35)].to_csv(
        paths["source_tables"] / "panel_f_pairwise_mimicry_by_phase_logmar_m035.csv",
        index=False,
    )
    occupancy.to_csv(paths["source_tables"] / "panel_f_occupancy.csv", index=False)

    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=False)
    mosaic = [["A", "B", "C"], ["D", "E", "F"]]
    axd = fig.subplot_mosaic(mosaic, gridspec_kw={"wspace": 0.36, "hspace": 0.45})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.08)

    plot_panel_a(axd["A"])
    plot_panel_b(axd["B"], local_j)
    plot_panel_c(axd["C"], closure, step_sens)
    plot_panel_d(axd["D"], contrasts)
    plot_panel_e(axd["E"], phase_summary)
    plot_panel_f(axd["F"], pairwise, occupancy)
    for letter, ax in axd.items():
        panel_letter(ax, letter)

    fig.suptitle("Supplement: controlled E-optotype geometry links local tangents, finite FEM covariance, and identity/pose mimicry", fontsize=12, y=0.985)
    save_all(fig, paths["exports"] / "FigureS_eoptotype_tangent_synthesis", args.dpi)
    plt.close(fig)

    write_caption_and_methods(
        paths,
        local_j=local_j,
        closure=closure,
        step_sens=step_sens,
        contrasts=contrasts,
        phase_summary=phase_summary,
        occupancy=occupancy,
    )

    qc_lines = [
        "# Supplemental E-optotype tangent synthesis",
        "",
        "## Generated manuscript notes",
        "- `FigureS_eoptotype_tangent_synthesis_caption.md`: panel-by-panel caption draft.",
        "- `FigureS_eoptotype_tangent_synthesis_methods.md`: companion methods/provenance note.",
        "",
        "## Source streams",
        "- Panel B: `outputs/jacobian_predictive_framework/cross_session_local_J_v2/cross_session_summary.json`.",
        "- Panel C: finite-difference matched recorded/twin covariance closure summaries.",
        "- Panel D: canonical validated mono Model A E-optotype discrimination contrasts.",
        "- Panels E/F: validated mono-population mimicry and phase-landscape tables.",
        "",
        "## Interpretation boundary",
        "This supplement treats E-optotypes as a controlled artificial stimulus family. It supports the geometry and caveat language, but it is not the main biological evidence.",
        "",
        "Main intended takeaway: pointwise Taylor prediction is local, yet local translation-generator families have finite-cloud covariance consequences; mimicry maps where pose variation overlaps identity differences.",
    ]
    (paths["qc"] / "README.md").write_text("\n".join(qc_lines) + "\n")
    print(f"Saved supplement figure to {paths['exports']}")


if __name__ == "__main__":
    main()

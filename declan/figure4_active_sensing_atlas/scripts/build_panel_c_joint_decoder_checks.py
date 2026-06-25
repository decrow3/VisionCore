"""Build cache-only diagnostic checks for Figure 4C.

The promoted 4C panel uses the feature-posterior compact-subspace endpoint.
This script makes a deliberately diagnostic figure pack that compares that
endpoint against the older exact trajectory-table image-identity observer and
basic compact-intervention QC.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "declan" / "figure4_active_sensing_atlas" / "figures" / "panel_C" / "diagnostics"
FEATURE_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1"
)
FEATURE_SUMMARY = FEATURE_DIR / "feature_compact_mechanism_summary.csv"
FEATURE_UNCERTAINTY = FEATURE_DIR / "feature_compact_mechanism_uncertainty.csv"
FEATURE_QC = FEATURE_DIR / "feature_compact_mechanism_qc.csv"
OBSERVER_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1"
    / "observer_summary.csv"
)

INK = "#1f252b"
GRID = "#d9dee5"
ZERO = "#66717d"
FULL = "#235789"
COMPACT = "#2f8f6a"
REMOVED = "#8a5ca8"
KNOWN = "#111827"
OU = "#b35c2e"

VARIANT_LABELS = {
    "zero_static": "zero eye",
    "compact_only": "compact only",
    "compact_removed": "compact removed",
    "full_exact": "full joint",
    "compact_addback": "addback",
    "known_eye": "known eye",
}
VARIANT_COLORS = {
    "zero_static": ZERO,
    "compact_only": COMPACT,
    "compact_removed": REMOVED,
    "full_exact": FULL,
    "compact_addback": "#4b5563",
    "known_eye": KNOWN,
}
AXIS_LABELS = {
    "axis_edge_parallel": "axis prior A",
    "axis_edge_orthogonal": "axis prior B",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scale_x(scales: pd.Series) -> np.ndarray:
    return scales.astype(float).map({0.5: 0.0, 1.0: 1.0, 2.0: 2.0}).to_numpy()


def _load_feature_summary() -> pd.DataFrame:
    rows = pd.read_csv(FEATURE_SUMMARY)
    selected = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["k_dim"].astype(int) == 10)
        & (
            rows["response_variant"].isin(
                ["zero_static", "compact_only", "compact_removed", "full_exact", "compact_addback", "known_eye"]
            )
        )
    ].copy()
    if len(selected) != 36:
        raise ValueError(f"Expected 36 feature-summary rows, found {len(selected)}")
    selected["variant_label"] = selected["response_variant"].map(VARIANT_LABELS)
    selected["axis_label"] = selected["prior_family"].map(AXIS_LABELS)
    selected["scale_label"] = selected["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"})
    return selected


def _aggregate_feature_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["observation_scale", "response_variant", "variant_label"], as_index=False)
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


def _load_uncertainty() -> pd.DataFrame:
    rows = pd.read_csv(FEATURE_UNCERTAINTY)
    selected = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["k_dim"].astype(int) == 10)
        & (rows["metric"] == "feature_cosine")
        & (
            rows["contrast"].isin(
                [
                    "compact_only_minus_compact_removed",
                    "full_exact_minus_compact_removed",
                    "compact_removed_minus_zero_static",
                    "compact_only_minus_zero_static",
                ]
            )
        )
    ].copy()
    selected["axis_label"] = selected["prior_family"].map(AXIS_LABELS)
    return selected


def _load_observer() -> pd.DataFrame:
    rows = pd.read_csv(OBSERVER_SUMMARY)
    selected = rows[
        (rows["trajectory_prior_mode"] == "leave_one_out")
        & (rows["likelihood_scale"].astype(float) == 1.0)
        & (rows["observation_scale"].astype(float) == rows["prior_scale"].astype(float))
        & (rows["observation_scale"].astype(float).isin([0.5, 1.0]))
        & (rows["candidate_set_mode"].isin(["hard_negative_structure", "matched_static_response"]))
        & (rows["prior_family"].isin(["empirical", "ou"]))
    ].copy()
    if len(selected) != 8:
        raise ValueError(f"Expected 8 observer rows, found {len(selected)}")
    return selected.sort_values(["candidate_set_mode", "observation_scale", "prior_family"])


def _write_values(
    feature_rows: pd.DataFrame,
    feature_agg: pd.DataFrame,
    uncertainty_rows: pd.DataFrame,
    observer_rows: pd.DataFrame,
) -> None:
    feature_rows.to_csv(OUT_DIR / "panel_C_joint_decoder_feature_rows.csv", index=False)
    feature_agg.to_csv(OUT_DIR / "panel_C_joint_decoder_feature_summary.csv", index=False)
    uncertainty_rows.to_csv(OUT_DIR / "panel_C_joint_decoder_contrasts.csv", index=False)
    observer_rows.to_csv(OUT_DIR / "panel_C_joint_decoder_observer_accuracy.csv", index=False)


def _plot_feature_curves(ax: plt.Axes, agg: pd.DataFrame) -> None:
    order = ["zero_static", "compact_removed", "compact_only", "full_exact", "known_eye"]
    for variant in order:
        block = agg[agg["response_variant"] == variant].sort_values("observation_scale")
        ax.plot(
            _scale_x(block["observation_scale"]),
            block["mean_feature_cosine"],
            marker="o",
            lw=2.0 if variant in {"compact_only", "compact_removed", "zero_static"} else 1.5,
            color=VARIANT_COLORS[variant],
            linestyle=":" if variant == "known_eye" else "-",
            label=VARIANT_LABELS[variant],
        )
    ax.set_title("A. promoted feature endpoint")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.48, 0.98)
    ax.legend(frameon=False, loc="lower left", ncol=1)
    _clean_axis(ax)


def _plot_contrasts(ax: plt.Axes, uncertainty: pd.DataFrame) -> None:
    contrast_order = [
        ("compact_only_minus_compact_removed", "compact only - removed", COMPACT),
        ("full_exact_minus_compact_removed", "full joint - removed", FULL),
        ("compact_removed_minus_zero_static", "removed - zero", REMOVED),
    ]
    offsets = [-0.18, 0.0, 0.18]
    for (contrast, label, color), offset in zip(contrast_order, offsets, strict=True):
        block = (
            uncertainty[uncertainty["contrast"] == contrast]
            .groupby("observation_scale", as_index=False)
            .agg(
                mean=("mean_lhs_minus_rhs", "mean"),
                lo=("mean_lhs_minus_rhs_ci_low", "mean"),
                hi=("mean_lhs_minus_rhs_ci_high", "mean"),
            )
            .sort_values("observation_scale")
        )
        x = _scale_x(block["observation_scale"]) + offset
        y = block["mean"].to_numpy(dtype=float)
        yerr = np.vstack([y - block["lo"].to_numpy(dtype=float), block["hi"].to_numpy(dtype=float) - y])
        ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.6, capsize=2.5, color=color, label=label)
    ax.axhline(0, color="#6b7280", lw=0.9)
    ax.set_title("B. paired feature-cosine contrasts")
    ax.set_ylabel("mean difference")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.legend(frameon=False, loc="upper left")
    _clean_axis(ax)


def _plot_posterior(ax: plt.Axes, agg: pd.DataFrame) -> None:
    order = ["zero_static", "compact_removed", "compact_only", "full_exact", "known_eye"]
    for variant in order:
        block = agg[agg["response_variant"] == variant].sort_values("observation_scale")
        ax.plot(
            _scale_x(block["observation_scale"]),
            block["median_candidate_N_eff_fraction"],
            marker="o",
            lw=1.7,
            color=VARIANT_COLORS[variant],
            linestyle=":" if variant == "known_eye" else "-",
            label=VARIANT_LABELS[variant],
        )
    ax.set_title("C. posterior concentration")
    ax.set_ylabel("median N_eff / K")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.20, 0.85)
    _clean_axis(ax)


def _plot_qc(ax: plt.Axes, rows: pd.DataFrame, agg: pd.DataFrame) -> None:
    pivot = (
        agg.pivot(index="observation_scale", columns="response_variant", values="mean_feature_cosine")
        .reset_index()
        .sort_values("observation_scale")
    )
    addback_delta = (pivot["full_exact"] - pivot["compact_addback"]).abs()
    clipped = agg[agg["response_variant"].isin(["compact_only", "compact_removed"])].copy()
    clipped = (
        clipped.groupby(["observation_scale", "response_variant"], as_index=False)
        .agg(median_clipped_rate_fraction=("median_clipped_rate_fraction", "mean"))
        .sort_values(["response_variant", "observation_scale"])
    )
    ax2 = ax.twinx()
    ax.bar(_scale_x(pivot["observation_scale"]) - 0.12, addback_delta, width=0.22, color="#4b5563", label="abs full-addback")
    for variant, offset in [("compact_only", 0.08), ("compact_removed", 0.22)]:
        block = clipped[clipped["response_variant"] == variant]
        ax2.plot(
            _scale_x(block["observation_scale"]) + offset,
            block["median_clipped_rate_fraction"],
            marker="o",
            lw=1.6,
            color=VARIANT_COLORS[variant],
            label=f"{VARIANT_LABELS[variant]} clipped",
        )
    max_prior_error = rows["prior_addback_reconstruction_max_abs_error"].dropna().astype(float).max()
    max_known_error = rows["known_addback_reconstruction_max_abs_error"].dropna().astype(float).max()
    ax.text(
        0.02,
        0.90,
        f"raw addback max <= {max(max_prior_error, max_known_error):.1e}",
        transform=ax.transAxes,
        color=INK,
        fontsize=7.5,
    )
    ax.set_title("D. addback and clipping QC")
    ax.set_ylabel("abs cosine delta")
    ax2.set_ylabel("clipped rate fraction")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0, 0.01)
    ax2.set_ylim(0, 0.05)
    _clean_axis(ax)
    ax2.spines["top"].set_visible(False)


def _plot_observer(ax: plt.Axes, observer: pd.DataFrame) -> None:
    candidate_mode = "matched_static_response"
    sub = observer[observer["candidate_set_mode"] == candidate_mode]
    scales = [0.5, 1.0]
    x = np.arange(len(scales), dtype=float)
    known = []
    zero = []
    empirical = []
    ou = []
    neff_emp = []
    for scale in scales:
        block = sub[sub["observation_scale"].astype(float) == scale]
        known.append(float(block["known_eye_accuracy"].iloc[0]))
        zero.append(float(block["zero_eye_accuracy"].iloc[0]))
        empirical.append(float(block[block["prior_family"] == "empirical"]["joint_eye_accuracy"].iloc[0]))
        ou.append(float(block[block["prior_family"] == "ou"]["joint_eye_accuracy"].iloc[0]))
        neff_emp.append(float(block[block["prior_family"] == "empirical"]["median_N_eff_fraction"].iloc[0]))
    ax.plot(x, known, marker="o", color=KNOWN, lw=1.7, linestyle=":", label="known eye")
    ax.plot(x, zero, marker="o", color=ZERO, lw=1.9, label="zero eye")
    ax.plot(x, empirical, marker="o", color=COMPACT, lw=2.0, label="joint empirical")
    ax.plot(x, ou, marker="o", color=OU, lw=2.0, label="joint OU")
    ax.set_title("E. older image-identity observer")
    ax.set_ylabel("accuracy")
    ax.set_xticks(x, ["0.5x", "1x"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)


def _plot_axis_detail(ax: plt.Axes, rows: pd.DataFrame) -> None:
    width = 0.13
    variants = ["zero_static", "compact_removed", "compact_only", "full_exact", "known_eye"]
    offsets = np.linspace(-2 * width, 2 * width, len(variants))
    for axis_i, (axis_family, axis_label) in enumerate(AXIS_LABELS.items()):
        base = axis_i * 4.0 + np.array([0.0, 1.0, 2.0])
        for variant, offset in zip(variants, offsets, strict=True):
            block = rows[(rows["prior_family"] == axis_family) & (rows["response_variant"] == variant)].sort_values(
                "observation_scale"
            )
            ax.bar(base + offset, block["mean_feature_cosine"], width=width, color=VARIANT_COLORS[variant], alpha=0.95)
        ax.text(base.mean(), 0.505, axis_label, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_title("F. axis-prior detail")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["0.5x", "1x", "2x", "0.5x", "1x", "2x"])
    ax.set_ylim(0.48, 0.98)
    _clean_axis(ax)


def _plot_check_sheet(
    feature_rows: pd.DataFrame,
    feature_agg: pd.DataFrame,
    uncertainty_rows: pd.DataFrame,
    observer_rows: pd.DataFrame,
    qc_rows: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), constrained_layout=True)
    _plot_feature_curves(axes[0, 0], feature_agg)
    _plot_contrasts(axes[0, 1], uncertainty_rows)
    _plot_posterior(axes[0, 2], feature_agg)
    _plot_qc(axes[1, 0], qc_rows, feature_agg)
    _plot_observer(axes[1, 1], observer_rows)
    _plot_axis_detail(axes[1, 2], feature_rows)
    fig.suptitle("Figure 4C joint-decoder checks", fontsize=12.5, color=INK)
    out = OUT_DIR / "panel_C_joint_decoder_check_sheet.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_axis_curves(feature_rows: pd.DataFrame) -> Path:
    variants = ["zero_static", "compact_removed", "compact_only", "full_exact", "known_eye"]
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.0), sharey=True, constrained_layout=True)
    for ax, (axis_family, axis_label) in zip(axes, AXIS_LABELS.items(), strict=True):
        for variant in variants:
            block = feature_rows[
                (feature_rows["prior_family"] == axis_family) & (feature_rows["response_variant"] == variant)
            ].sort_values("observation_scale")
            ax.plot(
                _scale_x(block["observation_scale"]),
                block["mean_feature_cosine"],
                marker="o",
                lw=2.0 if variant in {"compact_only", "compact_removed"} else 1.5,
                linestyle=":" if variant == "known_eye" else "-",
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
        ax.set_title(axis_label)
        ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
        ax.set_xlabel("motion scale")
        _clean_axis(ax)
    axes[0].set_ylabel("feature recovery (cosine)")
    axes[0].set_ylim(0.48, 0.98)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    out = OUT_DIR / "panel_C_joint_decoder_axis_detail.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _write_readme(paths: list[Path]) -> None:
    text = f"""# Panel C Joint-Decoder Diagnostic Checks

Cache-only diagnostics for the Figure 4C joint observer / joint decoder result.

## Inputs

- `{FEATURE_SUMMARY}`
- `{FEATURE_UNCERTAINTY}`
- `{FEATURE_QC}`
- `{OBSERVER_SUMMARY}`

## Outputs

- `{paths[0].name}`: six-panel check sheet for feature recovery, compact-removal contrasts, posterior concentration, addback/clipping QC, older image-identity observer accuracy, and axis-prior detail.
- `{paths[1].name}`: split axis-prior feature-recovery curves.
- `panel_C_joint_decoder_feature_summary.csv`
- `panel_C_joint_decoder_contrasts.csv`
- `panel_C_joint_decoder_observer_accuracy.csv`
- `panel_C_joint_decoder_feature_rows.csv`

These figures are diagnostics, not replacement promotion candidates.
"""
    (OUT_DIR / "panel_C_joint_decoder_checks_README.md").write_text(text, encoding="utf-8")


def build() -> list[Path]:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_rows = _load_feature_summary()
    feature_agg = _aggregate_feature_summary(feature_rows)
    uncertainty_rows = _load_uncertainty()
    observer_rows = _load_observer()
    qc_rows = pd.read_csv(FEATURE_QC)
    _write_values(feature_rows, feature_agg, uncertainty_rows, observer_rows)
    paths = [
        _plot_check_sheet(feature_rows, feature_agg, uncertainty_rows, observer_rows, qc_rows),
        _plot_axis_curves(feature_rows),
    ]
    _write_readme(paths)
    return paths + [
        OUT_DIR / "panel_C_joint_decoder_check_sheet.pdf",
        OUT_DIR / "panel_C_joint_decoder_axis_detail.pdf",
        OUT_DIR / "panel_C_joint_decoder_feature_summary.csv",
        OUT_DIR / "panel_C_joint_decoder_contrasts.csv",
        OUT_DIR / "panel_C_joint_decoder_observer_accuracy.csv",
        OUT_DIR / "panel_C_joint_decoder_feature_rows.csv",
        OUT_DIR / "panel_C_joint_decoder_checks_README.md",
    ]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()

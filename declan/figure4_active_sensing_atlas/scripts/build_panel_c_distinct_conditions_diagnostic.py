"""Plot all distinct Panel C feature-posterior interpretation conditions."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
OUT_DIR = ATLAS / "figures" / "panel_C" / "diagnostics"
SOURCE_CSV = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1"
    / "feature_compact_mechanism_summary.csv"
)

INK = "#1f252b"
GRID = "#dfe4e9"
BLUE = "#244f7a"
GREEN = "#2f8f6a"
PURPLE = "#8064a2"
GRAY = "#7b8288"
ORANGE = "#d07a22"
TEAL = "#287c89"

DISPLAY = {
    "zero_static": {
        "label": "static reference",
        "color": GRAY,
        "linestyle": "-",
        "marker": "o",
        "zorder": 3,
    },
    "full_exact": {
        "label": "latent-eye full joint",
        "color": BLUE,
        "linestyle": "-",
        "marker": "o",
        "zorder": 5,
    },
    "compact_only": {
        "label": "compact only",
        "color": GREEN,
        "linestyle": "-",
        "marker": "o",
        "zorder": 6,
    },
    "compact_removed": {
        "label": "compact removed",
        "color": PURPLE,
        "linestyle": "-",
        "marker": "o",
        "zorder": 4,
    },
    "compact_addback": {
        "label": "compact addback",
        "color": TEAL,
        "linestyle": "--",
        "marker": "s",
        "zorder": 7,
    },
    "known_eye": {
        "label": "known-trace control",
        "color": INK,
        "linestyle": ":",
        "marker": "o",
        "zorder": 8,
    },
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _selected_rows() -> pd.DataFrame:
    rows = pd.read_csv(SOURCE_CSV)
    selected = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["k_dim"].astype(int) == 10)
        & (rows["likelihood_scale"].astype(float) == 1.0)
        & (rows["response_variant"].isin(DISPLAY))
        & (rows["prior_family"].isin(["axis_edge_parallel", "axis_edge_orthogonal"]))
    ].copy()
    expected = 3 * 2 * len(DISPLAY)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected rows, found {len(selected)}")
    selected["condition_label"] = selected["response_variant"].map(lambda key: DISPLAY[str(key)]["label"])
    selected["scale_label"] = selected["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"})
    return selected


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["observation_scale", "scale_label", "response_variant", "condition_label"], as_index=False)
        .agg(
            mean_feature_cosine=("mean_feature_cosine", "mean"),
            min_feature_cosine=("mean_feature_cosine", "min"),
            max_feature_cosine=("mean_feature_cosine", "max"),
            mean_feature_neg_mse=("mean_feature_neg_mse", "mean"),
            mean_candidate_true_mass=("mean_candidate_true_mass", "mean"),
            median_candidate_N_eff_fraction=("median_candidate_N_eff_fraction", "mean"),
            median_clipped_rate_fraction=("median_clipped_rate_fraction", "mean"),
            n_trial_rows=("n_trial_rows", "sum"),
        )
        .sort_values(["observation_scale", "response_variant"])
    )
    return summary


def build() -> list[Path]:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows()
    summary = _summary(rows)

    fig, ax = plt.subplots(figsize=(5.4, 3.65), constrained_layout=True)
    x_map = {0.5: 0, 1.0: 1, 2.0: 2}
    for variant, style in DISPLAY.items():
        block = summary[summary["response_variant"] == variant].sort_values("observation_scale")
        x = block["observation_scale"].map(x_map).astype(float).to_numpy()
        y = block["mean_feature_cosine"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            marker=str(style["marker"]),
            markersize=4.8,
            lw=2.0 if variant not in {"compact_addback", "known_eye"} else 1.7,
            label=str(style["label"]),
            zorder=int(style["zorder"]),
        )

    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_ylim(0.48, 0.98)
    ax.set_title("Panel C distinct feature-posterior conditions")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=1.0, handlelength=2.1)
    ax.text(
        0.98,
        0.05,
        "addback overlaps full joint",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=ORANGE,
        fontsize=7.4,
    )

    png = OUT_DIR / "panel_C_distinct_condition_feature_recovery.png"
    pdf = png.with_suffix(".pdf")
    values = OUT_DIR / "panel_C_distinct_condition_feature_recovery_values.csv"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(values, index=False)
    return [png, pdf, values]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()

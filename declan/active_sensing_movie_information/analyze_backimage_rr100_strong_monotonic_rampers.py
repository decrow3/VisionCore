"""Analyze RR100 units with strong monotonic SSI increases from 1x to 3x."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AXIS_RUN_DIR = Path(
    "outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1"
)
DEFAULT_TUNING_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
)
DEFAULT_OUT_DIR = DEFAULT_TUNING_DIR / "strong_monotonic_1x_to_3x_rampers"

SSI_COLS = [
    "ssi_at_across_1p0",
    "ssi_at_across_1p5",
    "ssi_at_across_2p0",
    "ssi_at_across_3p0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--delta-quantile",
        type=float,
        default=0.90,
        help="Use strictly monotonic units with delta_3_minus_1 at or above this usable-unit quantile.",
    )
    parser.add_argument(
        "--min-step",
        type=float,
        default=0.0,
        help="Minimum allowed step for 1->1.5, 1.5->2, and 2->3 monotonicity.",
    )
    return parser.parse_args()


def ordered_unique(values: pd.Series) -> list[float]:
    return [float(v) for v in sorted(pd.to_numeric(values, errors="coerce").dropna().unique())]


def hypergeom_p_ge(k: int, successes: int, draws: int, population: int) -> float:
    if k <= 0:
        return 1.0
    total = math.comb(population, draws)
    high = min(successes, draws)
    return float(
        sum(math.comb(successes, i) * math.comb(population - successes, draws - i) / total for i in range(k, high + 1))
    )


def add_curve_metrics(curves: pd.DataFrame, min_step: float) -> pd.DataFrame:
    out = curves.copy()
    for col in SSI_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["delta_3_minus_1"] = out["ssi_at_across_3p0"] - out["ssi_at_across_1p0"]
    out["step_1_to_1p5"] = out["ssi_at_across_1p5"] - out["ssi_at_across_1p0"]
    out["step_1p5_to_2"] = out["ssi_at_across_2p0"] - out["ssi_at_across_1p5"]
    out["step_2_to_3"] = out["ssi_at_across_3p0"] - out["ssi_at_across_2p0"]
    step_cols = ["step_1_to_1p5", "step_1p5_to_2", "step_2_to_3"]
    out["monotonic_1_to_3"] = out[step_cols].ge(float(min_step)).all(axis=1)
    out["min_step_1_to_3"] = out[step_cols].min(axis=1)
    return out


def add_condition_rates(curves: pd.DataFrame, unit_ssi: pd.DataFrame) -> pd.DataFrame:
    needed = unit_ssi[
        (
            np.isclose(pd.to_numeric(unit_ssi["along_scale"], errors="coerce"), 1.0)
            & pd.to_numeric(unit_ssi["across_scale"], errors="coerce").isin([1.0, 3.0])
        )
    ].copy()
    rate = needed.pivot_table(
        index=["unit_index", "unit_label"],
        columns="across_scale",
        values=["unit_mean_rate_mean", "unit_ssi_bits_per_spike_mean"],
        aggfunc="first",
    )
    rate.columns = [
        f"{metric}_across_{str(scale).replace('.', 'p')}"
        for metric, scale in rate.columns.to_flat_index()
    ]
    rate = rate.reset_index()
    out = curves.merge(rate, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    for col in [
        "unit_mean_rate_mean_across_1p0",
        "unit_mean_rate_mean_across_3p0",
        "unit_ssi_bits_per_spike_mean_across_1p0",
        "unit_ssi_bits_per_spike_mean_across_3p0",
    ]:
        if col in out:
            vals = out[col].dropna().sort_values().to_numpy(dtype=float)
            if vals.size:
                out[f"{col}_percentile"] = [float(np.mean(vals <= float(v)) * 100.0) if np.isfinite(v) else np.nan for v in out[col]]
    return out


def enrichment_rows(
    selected: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    selected_label: str,
    universe_label: str,
) -> list[dict[str, object]]:
    tests: list[tuple[str, pd.Series, pd.Series]] = [
        (
            "dynamic_peak_sf_eq_0p8",
            selected["dynamic_peak_spatial_cpd_by_amp"].eq(0.8),
            universe["dynamic_peak_spatial_cpd_by_amp"].eq(0.8),
        ),
        (
            "dynamic_peak_tf_ge_12p8",
            selected["dynamic_peak_temporal_hz_by_amp"].ge(12.8),
            universe["dynamic_peak_temporal_hz_by_amp"].ge(12.8),
        ),
        (
            "dynamic_peak_sf_0p8_and_tf_ge_12p8",
            selected["dynamic_peak_spatial_cpd_by_amp"].eq(0.8)
            & selected["dynamic_peak_temporal_hz_by_amp"].ge(12.8),
            universe["dynamic_peak_spatial_cpd_by_amp"].eq(0.8)
            & universe["dynamic_peak_temporal_hz_by_amp"].ge(12.8),
        ),
        (
            "static_peak_sf_eq_0p0125",
            selected["static_peak_spatial_cpd_by_mean_rate"].eq(0.0125),
            universe["static_peak_spatial_cpd_by_mean_rate"].eq(0.0125),
        ),
        (
            "static_peak_sf_le_0p8",
            selected["static_peak_spatial_cpd_by_mean_rate"].le(0.8),
            universe["static_peak_spatial_cpd_by_mean_rate"].le(0.8),
        ),
        (
            "prior_osi_lt_0p2",
            selected["prior_orientation_selectivity_index"].lt(0.2),
            universe["prior_orientation_selectivity_index"].lt(0.2),
        ),
        (
            "rate_1x_bottom_quartile",
            selected["unit_mean_rate_mean_across_1p0_percentile"].le(25.0),
            universe["unit_mean_rate_mean_across_1p0_percentile"].le(25.0),
        ),
    ]
    out: list[dict[str, object]] = []
    n = int(len(selected))
    population = int(len(universe))
    for name, sel_mask, universe_mask in tests:
        k = int(sel_mask.fillna(False).sum())
        successes = int(universe_mask.fillna(False).sum())
        out.append(
            {
                "comparison": name,
                "selected_label": selected_label,
                "universe_label": universe_label,
                "selected_count": k,
                "selected_n": n,
                "universe_count": successes,
                "universe_n": population,
                "selected_fraction": float(k / n) if n else np.nan,
                "universe_fraction": float(successes / population) if population else np.nan,
                "hypergeom_p_ge": hypergeom_p_ge(k, successes, n, population) if population >= n else np.nan,
            }
        )
    return out


def plot_overview(selected: pd.DataFrame, usable: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
    selected_color = "#b84d5f"
    universe_color = "0.74"

    ax = axes[0, 0]
    curves = usable.sort_values("delta_3_minus_1", ascending=False).reset_index(drop=True)
    ax.plot(np.arange(len(curves)) + 1, curves["delta_3_minus_1"], color="0.55", lw=1.3)
    ax.scatter(selected["delta_rank"], selected["delta_3_minus_1"], color=selected_color, s=55, zorder=4)
    for _, row in selected.iterrows():
        ax.annotate(row["unit_label"], (row["delta_rank"], row["delta_3_minus_1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(float(selected["strong_delta_threshold"].iloc[0]), ls=":", color="0.2")
    ax.set_title("3x - 1x SSI increase")
    ax.set_xlabel("unit rank by delta")
    ax.set_ylabel("bits/spike")

    ax = axes[0, 1]
    for _, row in selected.sort_values("delta_3_minus_1", ascending=False).iterrows():
        xs = [1.0, 1.5, 2.0, 3.0]
        ys = [
            row["ssi_at_across_1p0"],
            row["ssi_at_across_1p5"],
            row["ssi_at_across_2p0"],
            row["ssi_at_across_3p0"],
        ]
        ax.plot(xs, ys, marker="o", lw=1.4, label=str(row["unit_label"]))
    ax.set_title("Selected SSI curves")
    ax.set_xlabel("across scale; along=1")
    ax.set_ylabel("SSI bits/spike")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 2]
    ax.scatter(
        usable["unit_mean_rate_mean_across_1p0"],
        usable["delta_3_minus_1"],
        s=20,
        color="0.65",
        alpha=0.65,
        label="usable units",
    )
    ax.scatter(
        selected["unit_mean_rate_mean_across_1p0"],
        selected["delta_3_minus_1"],
        s=65,
        color=selected_color,
        edgecolor="white",
        label="strong monotonic",
    )
    for _, row in selected.iterrows():
        ax.annotate(row["unit_label"], (row["unit_mean_rate_mean_across_1p0"], row["delta_3_minus_1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Rate at 1x vs ramp")
    ax.set_xlabel("mean rate at across=1")
    ax.set_ylabel("3x - 1x SSI")
    ax.legend(frameon=False, fontsize=8)

    for ax, col, title, xlabel in [
        (axes[1, 0], "dynamic_peak_spatial_cpd_by_amp", "Dynamic peak SF", "cpd"),
        (axes[1, 1], "dynamic_peak_temporal_hz_by_amp", "Dynamic peak TF", "Hz"),
        (axes[1, 2], "static_peak_spatial_cpd_by_mean_rate", "Static peak SF", "cpd"),
    ]:
        vals = ordered_unique(usable[col])
        x = np.arange(len(vals))
        all_counts = usable[col].value_counts().reindex(vals, fill_value=0)
        sel_counts = selected[col].value_counts().reindex(vals, fill_value=0)
        ax.bar(x - 0.18, all_counts.to_numpy(), width=0.36, color=universe_color, label="usable RR100")
        ax.bar(x + 0.18, sel_counts.to_numpy(), width=0.36, color=selected_color, label="strong monotonic")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in vals], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("unit count")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "BackImage RR100 strong monotonic 1x-to-3x SSI rampers\n"
        "strictly nondecreasing over 1, 1.5, 2, 3x and above the configured delta quantile",
        fontsize=14,
    )
    fig.savefig(out_dir / "backimage_rr100_strong_monotonic_rampers_overview.png", dpi=180)
    fig.savefig(out_dir / "backimage_rr100_strong_monotonic_rampers_overview.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.axis_run_dir / "backimage_contour_axis_rr100_unit_zscore_curves.csv")
    unit_ssi = pd.read_csv(args.axis_run_dir / "unit_ssi_table.csv")
    tuning = pd.read_csv(args.tuning_dir / "frequency_tuning_summary.csv")

    curves = add_curve_metrics(curves, min_step=float(args.min_step))
    curves = add_condition_rates(curves, unit_ssi)
    usable = curves.merge(tuning, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    if usable["scalar_readout"].isna().any():
        missing = usable.loc[usable["scalar_readout"].isna(), "unit_label"].tolist()
        raise ValueError(f"Missing frequency tuning rows for usable curve units: {missing}")

    threshold = float(usable["delta_3_minus_1"].quantile(float(args.delta_quantile)))
    selected = usable[(usable["monotonic_1_to_3"]) & (usable["delta_3_minus_1"] >= threshold)].copy()
    selected = selected.sort_values("delta_3_minus_1", ascending=False).reset_index(drop=True)
    ranked = usable.sort_values("delta_3_minus_1", ascending=False).reset_index(drop=True)
    rank_lookup = {int(row.unit_index): int(i + 1) for i, row in ranked.iterrows()}
    selected["delta_rank"] = [rank_lookup[int(v)] for v in selected["unit_index"]]
    selected["strong_delta_threshold"] = threshold
    selected["strong_monotonic_contract"] = (
        f"monotonic_1_to_3 with every step >= {float(args.min_step):g} and "
        f"delta_3_minus_1 >= usable-unit quantile {float(args.delta_quantile):g} ({threshold:.6g})"
    )

    display_cols = [
        "delta_rank",
        "unit_index",
        "unit_label",
        "delta_3_minus_1",
        "step_1_to_1p5",
        "step_1p5_to_2",
        "step_2_to_3",
        "ssi_at_across_1p0",
        "ssi_at_across_1p5",
        "ssi_at_across_2p0",
        "ssi_at_across_3p0",
        "unit_mean_rate_mean_across_1p0",
        "unit_mean_rate_mean_across_1p0_percentile",
        "unit_mean_rate_mean_across_3p0",
        "unit_mean_rate_mean_across_3p0_percentile",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "static_peak_orientation_deg_by_mean_rate",
        "static_peak_spatial_cpd_by_mean_rate",
        "dynamic_peak_orientation_deg_by_amp",
        "dynamic_peak_spatial_cpd_by_amp",
        "dynamic_peak_temporal_hz_by_amp",
        "dynamic_peak_response_amp",
        "strong_delta_threshold",
        "strong_monotonic_contract",
    ]
    selected[display_cols].to_csv(args.out_dir / "strong_monotonic_ramping_units.csv", index=False)

    enrichment = []
    enrichment.extend(
        enrichment_rows(
            selected,
            usable,
            selected_label="strong_monotonic_usable_top_delta",
            universe_label="usable_curve_units",
        )
    )
    all_rr100_with_curve_rates = tuning.merge(
        curves[
            [
                "unit_index",
                "unit_label",
                "unit_mean_rate_mean_across_1p0_percentile",
            ]
        ],
        on=["unit_index", "unit_label"],
        how="left",
        validate="one_to_one",
    )
    enrichment.extend(
        enrichment_rows(
            selected,
            all_rr100_with_curve_rates,
            selected_label="strong_monotonic_usable_top_delta",
            universe_label="all_rr100_frequency_tuning_units",
        )
    )
    pd.DataFrame(enrichment).to_csv(args.out_dir / "strong_monotonic_trend_enrichment.csv", index=False)

    plot_overview(selected, usable, args.out_dir)

    print(f"usable units: {len(usable)}")
    print(f"delta quantile threshold ({args.delta_quantile:g}): {threshold:.6g}")
    print(f"selected strong monotonic units: {len(selected)}")
    print(selected[display_cols].to_string(index=False))
    print(f"Wrote strong monotonic ramper outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

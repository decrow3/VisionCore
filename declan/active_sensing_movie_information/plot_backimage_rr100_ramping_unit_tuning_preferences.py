"""Summarize SF/TF tuning preferences for BackImage RR100 ramping SSI units."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RAMP_SELECTION = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/"
    "ramp_unit_image_maps_top6_img6_v1/ramping_unit_selection.csv"
)
DEFAULT_TUNING_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
)
DEFAULT_OUT_DIR = DEFAULT_TUNING_DIR / "ramping_unit_tuning_preferences"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramp-selection-csv", type=Path, default=DEFAULT_RAMP_SELECTION)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=6)
    return parser.parse_args()


def ordered_unique(values: pd.Series) -> list[float]:
    return [float(v) for v in sorted(pd.to_numeric(values, errors="coerce").dropna().unique())]


def circular_orientation_delta_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return np.abs(((aa - bb + 90.0) % 180.0) - 90.0)


def load_joined(ramp_selection_csv: Path, tuning_dir: Path, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ramp = pd.read_csv(ramp_selection_csv).head(int(top_n)).copy()
    summary = pd.read_csv(tuning_dir / "frequency_tuning_summary.csv")
    grouped = pd.read_csv(tuning_dir / "frequency_tuning_grouped.csv")

    joined = ramp.merge(summary, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    if joined["scalar_readout"].isna().any():
        missing = joined.loc[joined["scalar_readout"].isna(), "unit_label"].tolist()
        raise ValueError(f"Missing frequency tuning rows for ramping units: {missing}")

    joined["prior_to_static_delta_deg"] = circular_orientation_delta_deg(
        joined["prior_preferred_orientation_deg"], joined["static_peak_orientation_deg_by_mean_rate"]
    )
    joined["prior_to_dynamic_delta_deg"] = circular_orientation_delta_deg(
        joined["prior_preferred_orientation_deg"], joined["dynamic_peak_orientation_deg_by_amp"]
    )
    joined["static_to_dynamic_delta_deg"] = circular_orientation_delta_deg(
        joined["static_peak_orientation_deg_by_mean_rate"], joined["dynamic_peak_orientation_deg_by_amp"]
    )
    joined["rank"] = np.arange(1, len(joined) + 1)

    return joined, summary, grouped.merge(joined[["unit_index", "rank"]], on="unit_index", how="inner")


def plot_preference_overview(joined: pd.DataFrame, all_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    ramp_color = "#cf4c4c"
    all_color = "0.72"

    ax = axes[0, 0]
    bins = np.arange(0, 181, 15)
    ax.hist(all_summary["prior_preferred_orientation_deg"], bins=bins, color=all_color, label="all RR100")
    ax.scatter(
        joined["prior_preferred_orientation_deg"],
        np.full(len(joined), ax.get_ylim()[1] * 0.08),
        s=70,
        color=ramp_color,
        edgecolor="white",
        zorder=5,
        label="rampers",
    )
    for _, row in joined.iterrows():
        ax.annotate(row["unit_label"], (row["prior_preferred_orientation_deg"], ax.get_ylim()[1] * 0.12), ha="center", fontsize=8)
    ax.set_title("Prior orientation preference")
    ax.set_xlabel("deg")
    ax.set_ylabel("unit count")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.scatter(
        joined["prior_preferred_orientation_deg"],
        joined["static_peak_orientation_deg_by_mean_rate"],
        s=80 + 140 * joined["positive_ramp_score"] / joined["positive_ramp_score"].max(),
        c=joined["prior_orientation_selectivity_index"],
        cmap="viridis",
        edgecolor="black",
    )
    ax.plot([0, 180], [0, 180], ls=":", color="0.65")
    for _, row in joined.iterrows():
        ax.annotate(row["unit_label"], (row["prior_preferred_orientation_deg"], row["static_peak_orientation_deg_by_mean_rate"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Static peak ori vs prior ori")
    ax.set_xlabel("prior preferred orientation (deg)")
    ax.set_ylabel("static peak orientation (deg)")
    ax.set_xlim(-4, 184)
    ax.set_ylim(-4, 184)

    ax = axes[0, 2]
    ax.scatter(
        joined["prior_preferred_orientation_deg"],
        joined["dynamic_peak_orientation_deg_by_amp"],
        s=80 + 140 * joined["positive_ramp_score"] / joined["positive_ramp_score"].max(),
        c=joined["dynamic_peak_temporal_hz_by_amp"],
        cmap="plasma",
        edgecolor="black",
    )
    ax.plot([0, 180], [0, 180], ls=":", color="0.65")
    for _, row in joined.iterrows():
        ax.annotate(row["unit_label"], (row["prior_preferred_orientation_deg"], row["dynamic_peak_orientation_deg_by_amp"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Dynamic peak ori vs prior ori")
    ax.set_xlabel("prior preferred orientation (deg)")
    ax.set_ylabel("dynamic peak orientation (deg)")
    ax.set_xlim(-4, 184)
    ax.set_ylim(-4, 184)

    for ax, col, title in [
        (axes[1, 0], "static_peak_spatial_cpd_by_mean_rate", "Static peak SF"),
        (axes[1, 1], "dynamic_peak_spatial_cpd_by_amp", "Dynamic peak SF"),
        (axes[1, 2], "dynamic_peak_temporal_hz_by_amp", "Dynamic peak TF"),
    ]:
        vals = ordered_unique(all_summary[col])
        x = np.arange(len(vals))
        all_counts = all_summary[col].value_counts().reindex(vals, fill_value=0)
        ramp_counts = joined[col].value_counts().reindex(vals, fill_value=0)
        ax.bar(x - 0.18, all_counts.to_numpy(), width=0.36, color=all_color, label="all RR100")
        ax.bar(x + 0.18, ramp_counts.to_numpy(), width=0.36, color=ramp_color, label="rampers")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in vals], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("unit count")
        ax.set_xlabel("cpd" if "spatial" in col else "Hz")
        ax.legend(frameon=False)

    fig.suptitle(
        "BackImage RR100 positive SSI rampers: tuning preference overview\n"
        "ramping units from instantaneous-map SSI across sweep; tuning from center-pixel grating probe",
        fontsize=14,
    )
    fig.savefig(out_dir / "backimage_rr100_ramping_unit_tuning_preference_overview.png", dpi=180)
    fig.savefig(out_dir / "backimage_rr100_ramping_unit_tuning_preference_overview.pdf")
    plt.close(fig)


def plot_ramping_unit_heatmaps(joined: pd.DataFrame, grouped: pd.DataFrame, out_dir: Path) -> None:
    spatial_vals = ordered_unique(grouped["spatial_cpd"])
    temporal_vals = ordered_unique(grouped["temporal_hz"])

    n_rows = len(joined)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, max(2.1 * n_rows, 7.5)), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_i, (_, unit) in enumerate(joined.iterrows()):
        unit_grouped = grouped[grouped["unit_index"] == int(unit["unit_index"])].copy()
        best_static_ori = float(unit["static_peak_orientation_deg_by_mean_rate"])
        best_dynamic_ori = float(unit["dynamic_peak_orientation_deg_by_amp"])

        ax = axes[row_i, 0]
        static = (
            unit_grouped[
                (unit_grouped["probe_orientation_deg"] == best_static_ori)
                & (unit_grouped["temporal_hz"] == 0.0)
            ]
            .set_index("spatial_cpd")
            .reindex(spatial_vals)
        )
        ax.plot(spatial_vals, static["mean_rate"], marker="o", color="0.15")
        ax.axvline(float(unit["static_peak_spatial_cpd_by_mean_rate"]), ls=":", color="0.5")
        ax.set_xscale("log")
        ax.set_title(f"{unit['unit_label']} static SF, ori {best_static_ori:g} deg")
        ax.set_ylabel("mean center rate")
        ax.set_xlabel("spatial frequency (cpd)")

        ax = axes[row_i, 1]
        dyn = unit_grouped[unit_grouped["probe_orientation_deg"] == best_dynamic_ori]
        heat = np.full((len(temporal_vals), len(spatial_vals)), np.nan, dtype=float)
        for ti, tf in enumerate(temporal_vals):
            for si, sf in enumerate(spatial_vals):
                match = dyn[(dyn["temporal_hz"] == tf) & (dyn["spatial_cpd"] == sf)]
                if not match.empty:
                    heat[ti, si] = float(match["mean_rate"].iloc[0])
        denom = np.nanmax(heat) - np.nanmin(heat)
        norm_heat = (heat - np.nanmin(heat)) / denom if denom > 0 else heat * 0
        im = ax.imshow(norm_heat, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.scatter(
            [spatial_vals.index(float(unit["dynamic_peak_spatial_cpd_by_amp"]))],
            [temporal_vals.index(float(unit["dynamic_peak_temporal_hz_by_amp"]))],
            marker="x",
            color="cyan",
            s=55,
            linewidths=1.7,
        )
        ax.set_xticks(np.arange(len(spatial_vals)))
        ax.set_xticklabels([f"{v:g}" for v in spatial_vals], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(temporal_vals)))
        ax.set_yticklabels([f"{v:g}" for v in temporal_vals])
        ax.set_title(f"SF x TF mean rate, dyn ori {best_dynamic_ori:g} deg")
        ax.set_xlabel("spatial frequency (cpd)")
        ax.set_ylabel("temporal frequency (Hz)")

        ax = axes[row_i, 2]
        sf = float(unit["dynamic_peak_spatial_cpd_by_amp"])
        tf_curve = (
            dyn[dyn["spatial_cpd"] == sf]
            .set_index("temporal_hz")
            .reindex([v for v in temporal_vals if v > 0])
        )
        ax.plot(tf_curve.index.to_numpy(dtype=float), tf_curve["response_amp_rms"], marker="o", color="#23879a")
        ax.axvline(float(unit["dynamic_peak_temporal_hz_by_amp"]), ls=":", color="0.5")
        ax.set_xscale("log")
        ax.set_title(f"TF amp at dyn SF {sf:g} cpd")
        ax.set_xlabel("temporal frequency (Hz)")
        ax.set_ylabel("response amp")

        axes[row_i, 0].text(
            -0.35,
            0.5,
            (
                f"rank {int(unit['rank'])}\n"
                f"{unit['unit_label']}\n"
                f"ramp {unit['positive_ramp_score']:.3f}\n"
                f"prior {unit['prior_preferred_orientation_deg']:.1f} deg\n"
                f"OSI {unit['prior_orientation_selectivity_index']:.2f}"
            ),
            transform=axes[row_i, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
        )

    cbar = fig.colorbar(im, ax=axes[:, 1], shrink=0.8)
    cbar.set_label("norm mean rate within unit")
    fig.suptitle(
        "BackImage RR100 ramping SSI units: SF/TF tuning probes\n"
        "heatmaps include TF=0 static row; cyan x marks dynamic amplitude peak",
        fontsize=14,
    )
    fig.savefig(out_dir / "backimage_rr100_ramping_unit_sf_tf_heatmaps.png", dpi=180)
    fig.savefig(out_dir / "backimage_rr100_ramping_unit_sf_tf_heatmaps.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joined, all_summary, grouped = load_joined(args.ramp_selection_csv, args.tuning_dir, args.top_n)

    display_cols = [
        "rank",
        "unit_index",
        "unit_label",
        "positive_ramp_score",
        "z_slope_vs_across_scale",
        "absolute_dynamic_range",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "static_peak_orientation_deg_by_mean_rate",
        "static_peak_spatial_cpd_by_mean_rate",
        "static_peak_mean_rate",
        "dynamic_peak_orientation_deg_by_amp",
        "dynamic_peak_spatial_cpd_by_amp",
        "dynamic_peak_temporal_hz_by_amp",
        "dynamic_peak_response_amp",
        "prior_to_static_delta_deg",
        "prior_to_dynamic_delta_deg",
        "static_to_dynamic_delta_deg",
    ]
    joined[display_cols].to_csv(args.out_dir / "ramping_unit_tuning_preferences.csv", index=False)

    plot_preference_overview(joined, all_summary, args.out_dir)
    plot_ramping_unit_heatmaps(joined, grouped, args.out_dir)

    print(f"Wrote ramping-unit tuning preference outputs to {args.out_dir}")
    print(joined[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

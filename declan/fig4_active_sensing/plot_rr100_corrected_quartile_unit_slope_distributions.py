#!/usr/bin/env python3
"""Plot per-unit corrected-cache path-slope distributions by SF quartile."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.make_rr100_corrected_quartile_population_figure import (
    COLORS,
    GROUPS,
    LABELS,
)


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_011_n012_quartile_snapshot_v1"
)
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_rounds000_011_v2/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_no_bottom_row_rounds000_011_v2_unit_assignments.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_quartile_unit_slope_distributions_v1"
STEM = "corrected_quartile_unit_path_slope_distributions_rounds000_011_v1"
SEED = 20260813


def residualize(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    return values - pd.Series(values).groupby(image_ids).transform("mean").to_numpy()


def slope(x: np.ndarray, y: np.ndarray, image_ids: np.ndarray) -> float:
    xx = residualize(x, image_ids)
    yy = residualize(y, image_ids)
    return float(np.dot(xx, yy) / np.dot(xx, xx))


def population_slope(
    moving_info: np.ndarray,
    moving_spikes: np.ndarray,
    baseline_info: np.ndarray,
    baseline_spikes: np.ndarray,
    units: np.ndarray,
    x: np.ndarray,
    image_ids: np.ndarray,
    use: np.ndarray,
) -> float:
    moving = np.asarray(moving_info[:, units], dtype=float).sum(axis=1) / np.maximum(
        np.asarray(moving_spikes[:, units], dtype=float).sum(axis=1), 1e-12
    )
    baseline = baseline_info[:, units].sum(axis=1) / np.maximum(
        baseline_spikes[:, units].sum(axis=1), 1e-12
    )
    delta = moving - baseline[image_ids]
    return slope(x[use], delta[use], image_ids[use])


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv")
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index")
    condition = condition.merge(
        traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin"]],
        on="trace_index",
        validate="many_to_one",
    )
    assignments = pd.read_csv(ASSIGNMENTS)
    assignments = assignments[assignments.sf_quartile.isin(GROUPS)].copy()
    moving_ssi = np.load(ASSEMBLED / "moving_movie_ssi_bits_per_spike.npy", mmap_mode="r")
    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as archive:
        baseline_ssi = np.asarray(archive["movie_ssi_bits_per_spike"], dtype=float)
        baseline_spikes = np.asarray(archive["expected_spikes"], dtype=float)
    baseline_info = baseline_ssi * baseline_spikes
    image_ids = condition.image_index.to_numpy(int)
    paths = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    strong_by_image = images.corrected_reconstruction_orientation_coherence.to_numpy(float) >= 0.20
    masks = {
        "all_images": np.ones(len(condition), dtype=bool),
        "strong_contours": strong_by_image[image_ids],
    }

    rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    for group in GROUPS:
        units = assignments.loc[assignments.sf_quartile.eq(group), "rr100_index"].to_numpy(int)
        for scope, use in masks.items():
            pop = population_slope(
                moving_info, moving_spikes, baseline_info, baseline_spikes,
                units, paths, image_ids, use,
            )
            loo_values = []
            for omitted in units:
                kept = units[units != omitted]
                loo_values.append(
                    population_slope(
                        moving_info, moving_spikes, baseline_info, baseline_spikes,
                        kept, paths, image_ids, use,
                    )
                )
            population_rows.append(
                {
                    "scope": scope,
                    "sf_quartile": group,
                    "n_units": int(len(units)),
                    "spike_weighted_population_slope": pop,
                    "leave_one_out_min": float(np.min(loo_values)),
                    "leave_one_out_max": float(np.max(loo_values)),
                }
            )
            for unit in units:
                delta = np.asarray(moving_ssi[:, unit], dtype=float) - baseline_ssi[image_ids, unit]
                rows.append(
                    {
                        "scope": scope,
                        "sf_quartile": group,
                        "rr100_index": int(unit),
                        "preferred_sf_cpd": float(
                            assignments.loc[assignments.rr100_index.eq(unit), "preferred_sf_cpd"].iloc[0]
                        ),
                        "preferred_tf_hz": float(
                            assignments.loc[assignments.rr100_index.eq(unit), "preferred_tf_hz"].iloc[0]
                        ),
                        "within_image_path_slope": slope(paths[use], delta[use], image_ids[use]),
                        "mean_delta_ssi": float(delta[use].mean()),
                        "mean_expected_spikes": float(np.asarray(moving_spikes[use, unit], dtype=float).mean()),
                    }
                )
    units = pd.DataFrame(rows)
    populations = pd.DataFrame(population_rows)
    return units, populations


def draw(units: pd.DataFrame, populations: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharey=True, constrained_layout=True)
    scope_titles = {
        "all_images": "A  All corrected images",
        "strong_contours": "B  Corrected strong-contour cohort",
    }
    influential = {"strong_contours": {54, 18}, "all_images": {54, 18}}
    for ax, scope in zip(axes, ("all_images", "strong_contours"), strict=True):
        frame = units[units.scope.eq(scope)]
        for position, group in enumerate(GROUPS):
            sub = frame[frame.sf_quartile.eq(group)].copy()
            values = sub.within_image_path_slope.to_numpy(float)
            parts = ax.violinplot(
                values,
                positions=[position],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.45,
            )
            for body in parts["bodies"]:
                body.set_facecolor(COLORS[group])
                body.set_edgecolor(COLORS[group])
                body.set_alpha(0.16)
            jitter = rng.uniform(-0.16, 0.16, len(sub))
            edge = ["black" if int(unit) in influential[scope] else "white" for unit in sub.rr100_index]
            linewidth = [1.2 if int(unit) in influential[scope] else 0.45 for unit in sub.rr100_index]
            ax.scatter(
                position + jitter,
                values,
                s=34,
                color=COLORS[group],
                edgecolors=edge,
                linewidths=linewidth,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot([position - 0.22, position + 0.22], [median, median], color="black", lw=2.0, zorder=4)
            pop = populations[populations.scope.eq(scope) & populations.sf_quartile.eq(group)].iloc[0]
            ax.scatter(
                position,
                pop.spike_weighted_population_slope,
                marker="D",
                s=62,
                facecolor="white",
                edgecolor=COLORS[group],
                linewidth=1.8,
                zorder=5,
            )
            ax.vlines(
                position + 0.30,
                pop.leave_one_out_min,
                pop.leave_one_out_max,
                color=COLORS[group],
                lw=2.0,
                alpha=0.75,
                zorder=2,
            )
            ax.plot(
                [position + 0.25, position + 0.35],
                [pop.leave_one_out_min, pop.leave_one_out_min],
                color=COLORS[group], lw=1.2,
            )
            ax.plot(
                [position + 0.25, position + 0.35],
                [pop.leave_one_out_max, pop.leave_one_out_max],
                color=COLORS[group], lw=1.2,
            )
            for row in sub[sub.rr100_index.isin(influential[scope])].itertuples(index=False):
                ax.annotate(
                    f"u{int(row.rr100_index):03d}",
                    (position, row.within_image_path_slope),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=7,
                )
        ax.axhline(0, color="0.55", lw=0.8)
        ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("preferred-SF quartile after recorded-fit gate")
        ax.set_title(scope_titles[scope], loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("per-unit within-image path slope\n(bits/spike/arcmin)")
    axes[0].text(
        0.02, 0.02,
        "points: units   black bar: unit median\nopen diamond: spike-weighted population   side bar: leave-one-unit-out range",
        transform=axes[0].transAxes, fontsize=7, va="bottom",
    )
    fig.suptitle(
        "Corrected-cache path effects are heterogeneous within SF quartiles\n"
        "12 complete balanced rounds · 61 recorded-validated units",
        fontsize=13, weight="bold",
    )
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(OUT / f"{STEM}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    units, populations = compute()
    units.to_csv(OUT / "per_unit_path_slopes.csv", index=False)
    populations.to_csv(OUT / "population_and_leave_one_out_slopes.csv", index=False)
    draw(units, populations)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "quartile_unit_distribution_checkpoint_complete",
        "scope": "12 complete corrected-cache rounds; interim",
        "point_estimand": "per-unit movie SSI minus image-matched stabilized SSI, regressed on corrected path after within-image residualization",
        "population_estimand": "spike-weighted population SSI slope for the same conditions",
        "outputs": {
            "figure_pdf": str((OUT / f"{STEM}.pdf").resolve()),
            "figure_png": str((OUT / f"{STEM}.png").resolve()),
            "per_unit_values": str((OUT / "per_unit_path_slopes.csv").resolve()),
            "population_values": str((OUT / "population_and_leave_one_out_slopes.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

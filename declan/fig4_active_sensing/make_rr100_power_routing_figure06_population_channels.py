#!/usr/bin/env python3
"""Figure 06: low/high-SF channel summaries and spectral-coverage audit."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1"
DATA = BASE / "data"
TESTS = BASE / "model_tests"
OUT = BASE / "06_population_channels"


def binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, bins + 1))
    centers, means, errors = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
        centers.append(np.nanmedian(x[mask]))
        means.append(np.nanmean(y[mask]))
        errors.append(np.nanstd(y[mask]) / np.sqrt(max(mask.sum(), 1)))
    return np.asarray(centers), np.asarray(means), np.asarray(errors)


def within_unit_condition_values(x: np.ndarray, y: np.ndarray, unit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool within-unit predictor percentiles and standardized outcomes."""
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    for unit_position in np.flatnonzero(unit_mask):
        unit_x = np.asarray(x[:, unit_position], float)
        unit_y = np.asarray(y[:, unit_position], float)
        ranks = np.argsort(np.argsort(unit_x, kind="mergesort"), kind="mergesort") / max(len(unit_x) - 1, 1)
        scale = np.nanstd(unit_y)
        standardized = (unit_y - np.nanmean(unit_y)) / (scale if scale > 1e-12 else 1.0)
        x_values.append(100 * ranks)
        y_values.append(standardized)
    return np.concatenate(x_values), np.concatenate(y_values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "power_routing_joined_arrays.npz", allow_pickle=False) as data:
        d = {key: np.asarray(data[key]) for key in data.files}
    unit_table = pd.read_csv(DATA / "routing_unit_cohort.csv")
    unit_table = unit_table[unit_table.routing_quality_pass].set_index("rr100_index").loc[d["rr100_index"].astype(int)].reset_index()
    scores = pd.read_csv(TESTS / "unit_level_global_routing_hybrid_cv.csv")
    condition_table = pd.read_csv(DATA / "routing_condition_table.csv")
    groups = unit_table.sf_outer_third.fillna("unclassified").astype(str).to_numpy()
    low = groups == "sf_low_half"
    high = groups == "sf_high_half"
    sf, tf = d["sf_centers_cpd"], d["tf_hz"]

    fig = plt.figure(figsize=(16, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    for column, (mask, title) in enumerate([(low, "validated low-SF units"), (high, "validated high-SF units")]):
        ax = fig.add_subplot(gs[0, column])
        if mask.sum():
            surface = np.nanmedian(d["normalized_unit_sensitivity"][mask] ** 2, axis=0)
            im = ax.pcolormesh(sf, tf, surface, shading="nearest", cmap="magma", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="median normalized sensitivity²")
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title=f"{title}\nn={int(mask.sum())}")

    ax = fig.add_subplot(gs[0, 2])
    band = d["routed_band_power"]
    fraction = band / np.maximum(band.sum(axis=2, keepdims=True), np.finfo(float).tiny)
    positions = np.arange(3)
    width = 0.34
    for offset, (mask, label, color) in zip([-width / 2, width / 2], [(low, "low SF", "#0072B2"), (high, "high SF", "#D55E00")]):
        values = fraction[:, mask, :].reshape(-1, 3) if mask.sum() else np.empty((0, 3))
        med = np.nanmedian(values, axis=0) if len(values) else np.full(3, np.nan)
        ax.bar(positions + offset, 100 * med, width=width, color=color, label=label)
    ax.set_xticks(positions, ["≤32", "33–45", "46–56"])
    ax.set(xlabel="TF band (Hz)", ylabel="median routed-power fraction (%)", title="Where predicted channel drive lies")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 3])
    coverage = condition_table.supported_power_fraction_of_all_positive_tf.to_numpy(float)
    ax.hist(100 * coverage, bins=24, color="#6A3D9A", alpha=0.85)
    ax.axvline(100 * np.median(coverage), color="black", lw=2, label=f"median {100*np.median(coverage):.1f}%")
    ax.set(xlabel="positive-TF power inside measured SF×TF support (%)", ylabel="conditions", title="Coverage of the mechanistic test")
    ax.legend(frameon=False)

    for column, outcome in enumerate(["temporal_rms_delta_from_stabilized_hz", "delta_ssi_bits_per_spike"]):
        ax = fig.add_subplot(gs[1, column])
        for mask, label, color in [(low, "low SF", "#0072B2"), (high, "high SF", "#D55E00")]:
            if not mask.sum():
                continue
            x, y = within_unit_condition_values(d["routed_amplitude"], d[outcome], mask)
            centers, means, errors = binned_curve(x, y)
            ax.errorbar(centers, means, yerr=errors, color=color, marker="o", lw=2, label=label)
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.set(xlabel="within-unit routing percentile", ylabel="within-unit activation z-score" if column == 0 else "within-unit ΔSSI z-score",
               title="Condition ordering: activation" if column == 0 else "Condition ordering: information change")
        ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 2])
    primary = scores[scores.outcome.eq("activation_rms_hz")].pivot(index="rr100_index", columns="model", values="crossed_cv_r2")
    primary = primary.join(unit_table.set_index("rr100_index")[["sf_outer_third"]])
    for x, (group, color) in enumerate([("sf_low_half", "#0072B2"), ("sf_high_half", "#D55E00")]):
        values = (primary.loc[primary.sf_outer_third.eq(group), "routing"] - primary.loc[primary.sf_outer_third.eq(group), "global"]).dropna()
        ax.scatter(np.full(len(values), x), values, color=color, s=45, alpha=0.85)
        if len(values):
            ax.plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="black", lw=3)
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xticks([0, 1], ["low SF", "high SF"])
    ax.set(ylabel="routing − global held-out $R^2$", title="Evidence for population-specific routing")

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(0.02, 0.98,
            "Interpretation gate\n\n"
            "Different passbands can receive different TF-band\npower even for the same retinal movie.\n\n"
            "That becomes evidence for selective population routing\nonly if it predicts activation or SSI better than the\nshared global-power predictor.\n\n"
            "The coverage histogram states how much of the retinal\nsignal this measured-passband test can actually address.\n\n"
            "Orientation is intentionally collapsed in this series.", va="top", fontsize=11, linespacing=1.3)

    fig.suptitle("Figure 06 — Do FEMs route power differently to low- and high-SF populations, and does that matter neurally?", fontsize=15, weight="bold")
    fig.savefig(OUT / "figure06_low_high_sf_routing_and_coverage.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "figure06_low_high_sf_routing_and_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "figure06_population_channels_complete",
        "n_low_sf": int(low.sum()),
        "n_high_sf": int(high.sum()),
        "median_supported_power_fraction": float(np.median(coverage)),
        "orientation": "collapsed by design",
        "interpretation_gate": "different predicted band contributions are not called routing unless they improve held-out neural prediction over global power",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

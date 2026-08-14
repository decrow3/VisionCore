#!/usr/bin/env python3
"""Figure 04: fair population comparison of global, routing, and hybrid models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1"
TESTS = BASE / "model_tests"
DATA = BASE / "data"
OUT = BASE / "04_population_prediction"

OUTCOME_LABELS = {
    "activation_rms_hz": "activation magnitude\nRMS[FEM − stabilized]",
    "delta_mean_rate_hz": "mean-rate change",
    "delta_ssi_bits_per_spike": "SSI change",
    "delta_information_bits_spikes": "information-numerator change",
}
MODEL_ORDER = ["global", "routing", "hybrid"]
COLORS = {"global": "#7F7F7F", "routing": "#0072B2", "hybrid": "#D55E00"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(TESTS / "unit_level_global_routing_hybrid_cv.csv")
    units = pd.read_csv(DATA / "routing_unit_cohort.csv")
    units = units[units.routing_quality_pass].copy()
    outcomes = list(OUTCOME_LABELS)
    n_units = scores.rr100_index.nunique()

    fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9])
    for column, outcome in enumerate(outcomes):
        subset = scores[scores.outcome.eq(outcome)]
        ax = fig.add_subplot(gs[0, column])
        for unit in sorted(subset.rr100_index.unique()):
            row = subset[subset.rr100_index.eq(unit)].set_index("model").loc[MODEL_ORDER]
            ax.plot(range(3), row.crossed_cv_correlation, color="0.75", lw=1, alpha=0.8)
            ax.scatter(range(3), row.crossed_cv_correlation, c=[COLORS[m] for m in MODEL_ORDER], s=24, zorder=3)
        medians = subset.groupby("model").crossed_cv_correlation.median().reindex(MODEL_ORDER)
        ax.plot(range(3), medians, color="black", lw=2.7, marker="o", ms=7, zorder=4)
        ax.axhline(0, color="0.3", ls="--", lw=1)
        ax.set_xticks(range(3), ["global\npower", "unit\nrouting", "both"])
        ax.set(ylabel="held-out correlation" if column == 0 else "", title=OUTCOME_LABELS[outcome])

    primary = scores[scores.outcome.eq("activation_rms_hz")].pivot(index="rr100_index", columns="model", values="crossed_cv_r2")
    ax = fig.add_subplot(gs[1, 0])
    for x, model in enumerate(MODEL_ORDER):
        values = primary[model].dropna().to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(x + jitter, values, color=COLORS[model], s=35, alpha=0.85)
        ax.plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="black", lw=3)
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xticks(range(3), ["global", "routing", "both"])
    ax.set(ylabel="held-out $R^2$", title="Absolute prediction of activation magnitude")

    ax = fig.add_subplot(gs[1, 1])
    delta_routing = primary["routing"] - primary["global"]
    delta_hybrid = primary["hybrid"] - primary[["global", "routing"]].max(axis=1)
    ax.scatter(np.zeros(len(delta_routing)), delta_routing, color=COLORS["routing"], s=38, alpha=0.85)
    ax.scatter(np.ones(len(delta_hybrid)), delta_hybrid, color=COLORS["hybrid"], s=38, alpha=0.85)
    ax.plot([-0.2, 0.2], [np.median(delta_routing)] * 2, color="black", lw=3)
    ax.plot([0.8, 1.2], [np.median(delta_hybrid)] * 2, color="black", lw=3)
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xticks([0, 1], ["routing − global", "hybrid − best single"])
    ax.set(ylabel="incremental held-out $R^2$", title="Does unit tuning add information?")

    merged = primary.join(units.set_index("rr100_index")[["extended_rank1_centered_r2", "extended_tf_fit_r2"]], how="left")
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(merged.extended_rank1_centered_r2, merged["routing"] - merged["global"], c=merged.extended_tf_fit_r2, cmap="viridis", s=58, edgecolor="white")
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set(xlabel="joint rank-1 F0 fit $R^2$", ylabel="routing − global held-out $R^2$", title="Fit-quality diagnostic")

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    med_corr = scores.groupby(["outcome", "model"]).crossed_cv_correlation.median().unstack()
    p = med_corr.loc["activation_rms_hz"]
    statement = (
        f"Current completed cohort: n={n_units}\n\n"
        f"Activation median held-out correlation\n"
        f"global: {p['global']:+.2f}\n"
        f"routing: {p['routing']:+.2f}\n"
        f"both: {p['hybrid']:+.2f}\n\n"
        "Each prediction is made for an unseen\nimage group and an unseen trace group.\n\n"
        "A routing mechanism requires routing or\nthe hybrid to improve held-out prediction;\nmap overlap alone is not sufficient evidence."
    )
    ax.text(0.03, 0.98, statement, va="top", fontsize=11, linespacing=1.35)

    fig.suptitle("Figure 04 — Global dynamic power, unit-specific routing, and their combination are tested out of sample", fontsize=15, weight="bold")
    fig.savefig(OUT / "figure04_global_routing_hybrid_population.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "figure04_global_routing_hybrid_population.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "figure04_population_prediction_complete",
        "n_units": int(n_units),
        "cross_validation": "unseen image fifth and unseen trace fifth in every test intersection",
        "primary_outcome": "activation_rms_hz",
        "interpretive_test": "routing is supported only if routing or hybrid improves held-out prediction over global power",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

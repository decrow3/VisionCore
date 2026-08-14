#!/usr/bin/env python3
"""Figure 05: additive versus baseline-scaled (multiplicative) rate models."""
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
OUT = BASE / "05_additive_multiplicative"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TESTS / "unit_level_additive_multiplicative_cv.csv")
    n_units = table.rr100_index.nunique()
    predictor_specs = {
        "global": ["global_additive", "global_multiplicative", "global_additive_plus_multiplicative"],
        "routing": ["routing_additive", "routing_multiplicative", "routing_additive_plus_multiplicative"],
    }
    labels = ["additive", "baseline-scaled", "both"]
    colors = ["#0072B2", "#E69F00", "#D55E00"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for axis, (predictor, models) in zip(axes[:2], predictor_specs.items()):
        subset = table[table.model.isin(models)].pivot(index="rr100_index", columns="model", values="delta_r2_over_baseline")
        for unit in subset.index:
            axis.plot(range(3), subset.loc[unit, models], color="0.75", lw=1)
            axis.scatter(range(3), subset.loc[unit, models], c=colors, s=30, zorder=3)
        medians = subset[models].median(axis=0).to_numpy()
        axis.plot(range(3), medians, color="black", lw=2.8, marker="o", ms=7)
        axis.axhline(0, color="0.3", ls="--", lw=1)
        axis.set_xticks(range(3), labels)
        axis.set(ylabel="Δ held-out $R^2$ over stabilized baseline", title=f"{predictor.capitalize()}-power drive")

    axes[2].axis("off")
    global_rows = table[table.model.str.startswith("global_") & ~table.model.eq("baseline_only")]
    routing_rows = table[table.model.str.startswith("routing_")]
    axes[2].text(
        0.02,
        0.98,
        "What is being distinguished\n\n"
        "additive:\n  moving rate = baseline relation + a·power\n\n"
        "baseline-scaled:\n  moving rate = baseline relation + g·baseline·power\n\n"
        "both:\n  includes both terms\n\n"
        f"Current completed cohort: n={n_units}\n\n"
        "This test applies only to rate. SSI is a\nbits/spike statistic and is not labeled as gain.\n\n"
        "A negative ΔR² means the extra FEM-power\nterm hurts prediction of unseen images/traces.",
        va="top",
        fontsize=11,
        linespacing=1.28,
    )
    fig.suptitle("Figure 05 — Does FEM-related power enter the response additively, multiplicatively, or not detectably?", fontsize=15, weight="bold")
    fig.savefig(OUT / "figure05_additive_multiplicative_rate_test.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "figure05_additive_multiplicative_rate_test.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "figure05_gain_form_complete",
        "n_units": int(n_units),
        "outcome": "moving mean rate",
        "reference_model": "moving rate predicted from stabilized mean rate alone",
        "guardrail": "no additive/multiplicative interpretation is applied to SSI",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

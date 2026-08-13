#!/usr/bin/env python3
"""Tail and leave-one-trace-out sensitivity for corrected SSI checkpoint 21."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import COLORS, HALF_ASSIGNMENTS
from declan.fig4_active_sensing.run_rr100_corrected_ssi_validated_halves_fullbank import (
    GROUPS,
    aggregate_curve,
    bootstrap_population,
    slope,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_ssi_validated_halves_fullbank_checkpoint_21_v1"
OUT_CSV = CHECKPOINT / "checkpoint_21_tail_sensitivity.csv"
OUT_LOO = CHECKPOINT / "checkpoint_21_leave_one_trace_out_slopes.csv"
OUT_PNG = CHECKPOINT / "checkpoint_21_tail_sensitivity.png"


def main() -> None:
    if any(p.exists() for p in (OUT_CSV, OUT_LOO, OUT_PNG)):
        raise FileExistsError("Refusing to overwrite checkpoint-21 sensitivity outputs")
    z = np.load(CHECKPOINT / "corrected_fullbank_unit_sufficient_statistics.npz")
    mn = z["moving_information_numerator"]
    me = z["moving_expected_spikes"]
    bn = z["stabilized_information_numerator"]
    be = z["stabilized_expected_spikes"]
    paths = z["path_length_arcmin"].astype(float)
    assignments = pd.read_csv(HALF_ASSIGNMENTS)
    group_units = {
        g: assignments.loc[assignments.sf_outer_third.eq(g), "rr100_index"].to_numpy(int)
        for g in GROUPS
    }
    masks = {
        "all_32": np.ones(len(paths), dtype=bool),
        "exclude_longest_31": paths < paths.max(),
        "at_or_below_90_arcmin_30": paths <= 90.0,
        "at_or_below_84_arcmin_29": paths <= 84.0,
    }
    rows = []
    for index, (label, mask) in enumerate(masks.items()):
        point = {}
        for group in GROUPS:
            point[group], _ = aggregate_curve(mn[:, mask], me[:, mask], bn, be, np.arange(mn.shape[0]), group_units[group])
        _, trend = bootstrap_population(
            mn[:, mask], me[:, mask], bn, be, paths[mask], group_units,
            n_bootstrap=2000, seed=20260812 + index,
        )
        for _, row in trend.iterrows():
            if row.sf_half in point:
                estimate = slope(paths[mask], point[row.sf_half])
            else:
                estimate = slope(paths[mask], point["sf_low_half"]) - slope(paths[mask], point["sf_high_half"])
            rows.append({
                "sensitivity_set": label, "n_traces": int(mask.sum()),
                "minimum_path_arcmin": float(paths[mask].min()), "maximum_path_arcmin": float(paths[mask].max()),
                "quantity": row.quantity, "sf_half": row.sf_half, "estimate": estimate,
                "bootstrap_median": row.bootstrap_median, "ci_low": row.ci_low, "ci_high": row.ci_high,
                "bootstrap_probability_gt_zero": row.bootstrap_probability_gt_zero,
            })
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT_CSV, index=False)

    loo_rows = []
    for omit in range(len(paths)):
        keep = np.arange(len(paths)) != omit
        estimates = {}
        for group in GROUPS:
            curve, _ = aggregate_curve(mn[:, keep], me[:, keep], bn, be, np.arange(mn.shape[0]), group_units[group])
            estimates[group] = slope(paths[keep], curve)
            loo_rows.append({
                "omitted_trace_index": omit, "omitted_path_arcmin": paths[omit],
                "sf_half": group, "path_slope": estimates[group],
            })
        loo_rows.append({
            "omitted_trace_index": omit, "omitted_path_arcmin": paths[omit],
            "sf_half": "low_minus_high", "path_slope": estimates["sf_low_half"] - estimates["sf_high_half"],
        })
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUT_LOO, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    labels = list(masks)
    for group in GROUPS:
        sub = sensitivity[(sensitivity.quantity.eq("path_slope")) & sensitivity.sf_half.eq(group)].set_index("sensitivity_set").loc[labels]
        x = np.arange(len(labels)) + (-0.08 if group == "sf_low_half" else 0.08)
        axes[0].errorbar(
            x, sub.estimate, yerr=[sub.estimate - sub.ci_low, sub.ci_high - sub.estimate],
            fmt="o-", color=COLORS[group], capsize=3, label=group.replace("sf_", "").replace("_", " "),
        )
    axes[0].axhline(0, color="0.35", ls=":")
    axes[0].set_xticks(np.arange(len(labels)), ["all 32", "omit 117′", "≤90′", "≤84′"], rotation=15)
    axes[0].set(title="Range sensitivity", ylabel="SSI slope (bits/spike/arcmin)")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=0.16)

    for group in ("sf_low_half", "sf_high_half", "low_minus_high"):
        sub = loo[loo.sf_half.eq(group)].sort_values("omitted_path_arcmin")
        color = COLORS.get(group, "#6A3D9A")
        axes[1].plot(sub.omitted_path_arcmin, sub.path_slope, "o", ms=4, color=color, label=group.replace("sf_", "").replace("_", " "))
    axes[1].axhline(0, color="0.35", ls=":")
    axes[1].set(title="Leave-one-trace-out", xlabel="path length of omitted trace (arcmin)", ylabel="refitted path slope")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=0.16)
    fig.suptitle("Checkpoint 21: dependence on the sparse long-path tail", fontweight="bold")
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()

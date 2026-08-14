#!/usr/bin/env python3
"""Auditable unit-role selection for Figure 4 panel B map inspection."""
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

from declan.fig4_active_sensing import audit_rr100_quartile_weighting_and_outliers as audit


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_022_n023_clean_history_snapshot_v1"
)
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_clean_history_rounds000_022_v2/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_clean_history_no_bottom_row_rounds000_022_v2_unit_assignments.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_figure4_b_unit_roles_clean23_v1"


def choose_closest(frame: pd.DataFrame, column: str, target: float) -> pd.Series:
    return frame.iloc[int(np.argmin(np.abs(frame[column].to_numpy(float) - target)))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    audit.ASSEMBLED = ASSEMBLED
    audit.ASSIGNMENTS = ASSIGNMENTS
    data = audit.load_data()
    estimands, units, decomposition = audit.compute_estimands(data)
    strong = units[units.scope.eq("strong_contours")].copy()
    terms = decomposition[decomposition.scope.eq("strong_contours")].copy()
    strong = strong.merge(
        terms[["rr100_index", "sf_quartile", "fixed_share_within_unit_contribution", "spike_composition_contribution", "total_population_term_slope"]],
        on=["rr100_index", "sf_quartile"],
        validate="one_to_one",
    )

    selections: list[dict[str, object]] = []
    for group in audit.GROUPS:
        sub = strong[strong.sf_quartile.eq(group)]
        median = float(sub.unit_path_slope.median())
        row = choose_closest(sub, "unit_path_slope", median)
        selections.append(
            {
                **row.to_dict(),
                "selection_role": f"{audit.LABELS[group].lower()}_typical_unit_slope",
                "criterion_name": "minimum absolute distance to within-quartile median unit path slope",
                "criterion_target": median,
                "algorithmic_selection": True,
            }
        )
    q3 = strong[strong.sf_quartile.eq("sf_q3")]
    row = q3.loc[q3.total_population_term_slope.idxmin()]
    selections.append(
        {
            **row.to_dict(),
            "selection_role": "q3_largest_negative_pooled_population_term",
            "criterion_name": "minimum total pooled-population slope term within Q3",
            "criterion_target": float(row.total_population_term_slope),
            "algorithmic_selection": True,
        }
    )
    selected = pd.DataFrame(selections).drop_duplicates(["selection_role", "rr100_index"])
    strong.to_csv(OUT / "panel_b_per_unit_slope_weight_diagnostics.csv", index=False)
    estimands[estimands.scope.eq("strong_contours")].to_csv(OUT / "panel_b_estimand_sensitivity.csv", index=False)
    selected.to_csv(OUT / "selected_units.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    ax = axes[0]
    for position, group in enumerate(audit.GROUPS):
        sub = strong[strong.sf_quartile.eq(group)]
        jitter = np.linspace(-0.16, 0.16, len(sub))
        ax.scatter(
            position + jitter,
            sub.unit_path_slope * 1e4,
            s=28 + 420 * sub.mean_population_spike_share,
            color=audit.COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
        )
        ax.plot([position - 0.22, position + 0.22], [sub.unit_path_slope.median() * 1e4] * 2, color="black", lw=2)
        chosen = selected[selected.sf_quartile.eq(group)]
        for row in chosen.itertuples(index=False):
            ax.scatter(position, row.unit_path_slope * 1e4, s=92, facecolor="none", edgecolor="black", lw=1.5)
            ax.annotate(f"u{int(row.rr100_index):03d}", (position, row.unit_path_slope * 1e4), xytext=(6, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylabel("unit path slope (×10⁻⁴ bits/spike/arcmin)")
    ax.set_title("A  Units underlying panel B", loc="left", weight="bold")
    ax.text(0.01, 0.01, "point area ∝ mean spike share; black bar = unit median", transform=ax.transAxes, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    order = ["pooled_spike", "fixed_spike_weight", "equal_unit_mean", "conditionwise_unit_median", "median_unit_slope"]
    labels = ["pooled\nspike", "fixed\nweights", "equal-unit\nmean", "conditionwise\nmedian", "median\nunit slope"]
    table = estimands[estimands.scope.eq("strong_contours")]
    for group in audit.GROUPS:
        sub = table[table.sf_quartile.eq(group)].set_index("estimand")
        ax.plot(range(len(order)), sub.loc[order, "path_slope"] * 1e4, marker="o", color=audit.COLORS[group], label=audit.LABELS[group])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(order)), labels)
    ax.set_ylabel("panel-B path slope (×10⁻⁴ bits/spike/arcmin)")
    ax.set_title("B  The conclusion depends on the estimand", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Panel B unit-role selection before targeted response-map rendering\n23 clean-history rounds", fontsize=14, weight="bold")
    fig.savefig(OUT / "panel_b_unit_roles.png", dpi=210, facecolor="white")
    fig.savefig(OUT / "panel_b_unit_roles.pdf", facecolor="white")
    plt.close(fig)

    summary = table.pivot(index="sf_quartile", columns="estimand", values="path_slope").reset_index()
    output_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "panel_b_auditable_unit_role_selection_complete",
        "scope": "23 complete clean-history rounds; strong-contour images only",
        "selection_policy": "typical unit closest to quartile median plus Q3 largest negative pooled-population contributor",
        "next_stage": "targeted stabilized/short/long raw response maps; not a production map cache",
        "outputs": {
            "figure": str((OUT / "panel_b_unit_roles.png").resolve()),
            "selected_units": str((OUT / "selected_units.csv").resolve()),
            "unit_diagnostics": str((OUT / "panel_b_per_unit_slope_weight_diagnostics.csv").resolve()),
            "estimand_sensitivity": str((OUT / "panel_b_estimand_sensitivity.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(selected[["selection_role", "sf_quartile", "rr100_index", "unit_path_slope", "mean_population_spike_share", "total_population_term_slope"]].to_string(index=False))
    print("\n", summary.to_string(index=False))
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write a concise index and current-status interpretation for the figure series."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1"


def main() -> None:
    data_manifest = json.loads((BASE / "data/manifest.json").read_text())
    population = pd.read_csv(BASE / "model_tests/population_global_routing_hybrid_summary.csv")
    gain = pd.read_csv(BASE / "model_tests/population_additive_multiplicative_summary.csv")
    n_units = int(data_manifest["scope"]["units_with_routing_quality_pass"])
    activation = population[population.outcome.eq("activation_rms_hz")].set_index("model")
    coverage = pd.read_csv(BASE / "data/routing_condition_table.csv").supported_power_fraction_of_all_positive_tf.median()
    baseline = float(gain.loc[gain.model.eq("baseline_only"), "median_cv_r2"].iloc[0])
    gain_best = gain[~gain.model.eq("baseline_only")].sort_values("median_delta_r2_over_baseline", ascending=False).iloc[0]

    text = f"""# RR100 FEM power-routing figure series

Generated {datetime.now(timezone.utc).isoformat()} from the corrected 3,000-condition bank (100 images, 1,000 traces, three balanced rounds).

## Reading order

1. [Figure 01 — retinal power redistribution](01_input_redistribution/figure01_retinal_power_redistribution.pdf)  
   Holds the natural image fixed conceptually and shows how corrected FEM traces create nonzero-TF power. No unit response enters.
2. [Figure 02 — the same input through different unit filters](02_unit_filtering/figure02_unit_specific_spectral_routing.pdf)  
   Holds one observed retinal power map fixed and changes only the independently measured native-readout F0 SF×TF passband.
3. [Figure 03 — response examples and dissociations](03_response_examples/figure03_routing_response_examples.pdf)  
   Introduces frozen-model activation and SSI. Auditable selection includes agreement, routing overprediction, and response-without-routing examples.
4. [Figure 04 — population prediction](04_population_prediction/figure04_global_routing_hybrid_population.pdf)  
   Compares global power, unit-specific routing, and their hybrid using predictions for unseen image groups and unseen trace groups.
5. [Figure 05 — additive versus multiplicative rate form](05_additive_multiplicative/figure05_additive_multiplicative_rate_test.pdf)  
   Tests whether power adds to mean rate, scales the stabilized rate, or needs both terms. This language is not applied to SSI.
6. [Figure 06 — low/high-SF channels and coverage](06_population_channels/figure06_low_high_sf_routing_and_coverage.pdf)  
   Shows the population passbands, predicted TF-band routing, within-unit condition ordering, and the fraction of retinal power inside measured support.

## Current quantitative status

- Completed, recorded-validated native tuning cohort in this build: **n={n_units} units**.
- Median supported fraction of positive-TF retinal power: **{100*coverage:.1f}%**.
- Activation-magnitude median held-out correlation:
  - global power: **{activation.loc['global', 'median_cv_correlation']:+.3f}**
  - unit routing: **{activation.loc['routing', 'median_cv_correlation']:+.3f}**
  - hybrid: **{activation.loc['hybrid', 'median_cv_correlation']:+.3f}**
- Stabilized-baseline-only median held-out rate R²: **{baseline:.3f}**.
- Best current extra power term: **{gain_best['model']}**, median ΔR² **{gain_best['median_delta_r2_over_baseline']:+.4f}**.

These values are automatically replaced when the extended native-readout TF probe completes. A visually different routed map is a mechanistic candidate, not evidence of selective neural routing unless it improves held-out response or SSI prediction over global power.

## Contracts and guardrails

- Corrected retinal motion uses the negative `dpi_pix` crop trajectory at 120 Hz.
- The scored spectrum uses the exact 40-frame analyzed segment; history frames are excluded from the input-power estimate.
- Retinal power and all predictors are compared over identical SF×TF support.
- Routing variance is `sum(P * H²)`; routed amplitude is its square root.
- `H` is shape-normalized. Native F0 gain is kept separate because a per-unit free regression slope would absorb it.
- The 60-Hz Nyquist point is a diagnostic control, not a fit point.
- Orientation is deliberately collapsed in this series.
- “Activation magnitude” is temporal RMS of the FEM-minus-stabilized rate timecourses.
- Movie SSI is computed from sufficient statistics; plotted instantaneous SSI is only a time-resolved diagnostic.
- The current response bank is a balanced 3,000-condition interim bank, not the final 100×1,000 cache.
"""
    (BASE / "README.md").write_text(text, encoding="utf-8")
    print(BASE / "README.md")


if __name__ == "__main__":
    main()

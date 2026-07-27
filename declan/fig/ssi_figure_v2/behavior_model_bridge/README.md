# Behavior-Model Bridge Analysis

This folder scopes the next analysis linking the Panel G model dose-response
curves to the real BackImage behavior panels. The goal is to move from a visual
comparison of reference bands to a quantitative test:

```text
Does behavior near coherent contours place eye movements in the model-beneficial
region, or does it mainly avoid the model-damaging tail?
```

## Working Interpretation

The model-side result is that high-SF contour-aligned units are vulnerable to
large contour-normal motion. This holds when the dose axis is unsigned component
path, component RMS excursion, or projected peak-to-peak range. The behavior
result is not that the animal simply accumulates less contour-normal path.
Instead, coherent contours are associated with drift clouds that are more
contour-parallel and narrower in the contour-normal direction.

The bridge should therefore be made with excursion/spread-like quantities first:
component RMS and projected range. Unsigned component path remains useful, but
it is a less direct behavioral alignment metric.

## Main Questions

1. Where does the empirical behavior distribution sit on the model curve?

   For each coherence bin, overlay or integrate the empirical behavior
   distribution against the model dose-response curve. The key distinction is:
   does high-coherence behavior move into a positive/beneficial region, or does
   it mostly remove probability mass from the harmful high-normal-dose tail?

2. Is the effect metric-specific?

   Run the bridge separately for component RMS, projected range, unsigned
   component path, and path/range tortuosity. The strongest mechanistic link to
   H/I should be RMS or range. A result that appears only for unsigned path
   would be harder to connect to the behavior panels.

3. Is the effect population-specific?

   Use the same population splits as the alternative-axis diagnostic:

   - all high-SF units
   - aligned high-SF units
   - oblique high-SF units
   - orthogonal high-SF units
   - all low-SF units

   The expected pattern is strongest contour-normal penalty for aligned
   high-SF units, weaker same-sign effects for all/oblique high-SF units, sign
   flip for orthogonal high-SF units, and a different low-SF pattern where more
   motion is generally beneficial.

4. Does coherence change the predicted SSI?

   Convert each behavior coherence bin into a behavior-weighted model
   prediction:

   ```text
   predicted SSI residual = E_behavior[f_model(component dose)]
   ```

   Compare low vs high coherence for each metric and population.

5. Is the result a tail-avoidance effect?

   Decompose the expectation into dose regions, for example:

   - below the trace-bank q25
   - trace-bank q25-q75
   - above the trace-bank q75
   - extreme high tail

   This tells us whether high-coherence behavior improves the prediction by
   occupying a sweet spot or by avoiding rare damaging large-normal excursions.

6. Do we need a joint model?

   The first pass should use one-dimensional marginal curves because they match
   Panel G. If those predictions are weak or misleading, move to a 2D surface
   using normal and parallel RMS/range jointly:

   ```text
   f_model(normal dose, parallel dose)
   ```

   This would test whether cancellation in marginal curves is hiding an
   interaction.

## Data Sources

Model-side alternative x-axis curves:

```text
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_values.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_last_bin_contrasts.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_populations.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_trace_bank_reference.csv
```

Behavior-side contour-relative summaries:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/contour_motion_component_windows.csv
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_windows.csv
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_alignment_summary.csv
```

The behavior bridge may need one additional behavior metrics table that includes
component RMS, projected range, unsigned component path, and path/range on the
same 0.325 s-equivalent scale as the model trace bank.

## Proposed First Outputs

1. Distribution-on-curve plot

   For each metric, show the model normal/parallel dose-response curves with
   behavior distributions overlaid as shaded densities or rug marks for low and
   high coherence. This answers the visual version of the bridge question.

2. Behavior-weighted prediction plot

   For each population and metric, plot predicted SSI residual by behavior
   coherence bin. The first implementation can treat the model curve as fixed
   and bootstrap behavior windows by session.

3. Tail contribution plot

   Show how much of the predicted SSI residual comes from behavior samples below
   q25, inside q25-q75, above q75, and in the final tail. This separates
   "beneficial sweet spot" from "damage avoidance."

4. Population control sheet

   Repeat the prediction for aligned, oblique, orthogonal, all high-SF, and all
   low-SF populations. The sign-flip control is central: if orthogonal high-SF
   units flip relative to aligned high-SF units, the interpretation is much
   stronger.

## Implemented First Pass

The first-pass bridge is implemented here:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/run_behavior_model_bridge.py
```

Run command:

```text
uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_behavior_model_bridge.py
```

Generated outputs:

```text
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_behavior_snippet_metrics.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_prediction_summary.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_coherence_contrasts.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_tail_contributions.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_distribution_on_curves.pdf

outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_tail_region_occupancy_high_sf_aligned.pdf
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_provenance.json
```

The script computes central 40-sample behavior snippets, matching the 0.325 s
Panel G model trace-bank duration. It projects each behavior snippet onto the
local contour-normal and contour-parallel axes, computes component path, RMS,
projected range, and path/range, then interpolates those empirical doses onto
the Panel G alternative-axis model curves. Summaries are session means with
session-bootstrap intervals.

The distribution-on-curve figures display the model range plus the behavior
99th percentile so rare behavior trace jumps do not collapse the useful axis
range. The full behavior doses are still retained in the CSVs, and the fraction
outside the modeled x-range is reported in
`behavior_model_bridge_prediction_summary.csv`.

## First-Pass Read

For aligned high-SF units, the current one-dimensional bridge does not yet show
a clean coherence-linked rescue on RMS or projected range. High coherence minus
low coherence is approximately:

```text
normal component path: -1.04 pp, CI [-2.23, -0.02]
parallel component path: +0.10 pp, CI [-0.34, +0.55]
normal RMS: +0.06 pp, CI [-0.81, +0.89]
parallel RMS: -0.16 pp, CI [-0.58, +0.28]
normal range: +0.02 pp, CI [-0.73, +0.79]
parallel range: +0.08 pp, CI [-0.33, +0.50]
```

This suggests the current marginal bridge is more consistent with "high
coherence may avoid the most damaging normal excursion tail" than with "high
coherence lands strongly in an SSI-beneficial normal-excursion sweet spot." The
tail contribution table supports that caution: for aligned high-SF normal
RMS/range, high coherence reduces the extreme final-tail occupancy, but it also
leaves substantial mass in the q75-to-tail region, so the behavior-weighted mean
prediction stays near flat.

## Analysis Choices

- Interpolation: start with piecewise-linear interpolation through the binned
  model curve medians. Report the fraction of behavior samples outside the
  modeled x-range rather than silently extrapolating.
- Behavior uncertainty: bootstrap sessions, not windows, for the first pass.
- Model uncertainty: initially use point curves and bootstrap CIs already stored
  in the model values table; later combine behavior and model bootstraps if this
  becomes a figure panel.
- Coherence bins: use the same bins as the behavior plots first. Add finer
  0.1-wide bins only if the first pass is stable.
- Primary metrics: component RMS and projected range.
- Secondary/control metrics: unsigned component path and path/range.

## Decision Criteria

The bridge supports the current story if:

- high-coherence behavior predicts less loss or more gain for aligned high-SF
  units on normal RMS/range axes;
- the effect is weaker in all high-SF units, present but diluted in oblique
  units, and flips or changes sign for orthogonal high-SF units;
- low-SF units show a different pattern consistent with larger motion being
  broadly beneficial;
- the effect is explainable as reduced high-normal-dose occupancy or reduced
  normal excursion/spread, not as reduced accumulated normal path alone.

The bridge weakens the story if:

- behavior distributions mostly occupy a flat region of the model curve;
- predicted SSI barely changes with coherence;
- the result appears only for unsigned path and not for RMS/range;
- aligned and orthogonal high-SF populations do not separate in the expected
  directioutputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_predicted_ssi_by_coherence.pdfon.

## Open Caveats

- Behavior windows and model trace-bank snippets must be placed on the same
  duration scale. Current Panel G trace-bank snippets are 0.325 s.
- Component path, RMS, and range are all contour-relative but not equivalent.
  The figure text should keep these meanings separate.
- If behavior samples fall outside the model dose range, that is itself a result
  and should be shown explicitly.
- A one-dimensional marginal curve may miss interactions between normal and
  parallel components. A 2D reconstruction should be the second-pass test.

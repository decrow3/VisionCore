---
name: map-first-analysis
description: Use an iterative, human-guided, maps-before-summaries workflow for exploratory neuroscience or neural-network analyses involving activation maps, unit responses, spatial or temporal selectivity, stimulus or power redistribution, SSI, difference maps, or a new mechanistic proxy whose meaning must be established visually before population statistics. Also use when debugging a metric by tracing it back to concrete maps or when selecting interesting units for deeper inspection. Do not use for routine reproduction of an already validated analysis unless the user asks to revisit its examples or assumptions.
---

# Map-first analysis

Establish what a proposed mechanism or metric means in concrete examples before building aggregate evidence. Keep the analysis interactive, auditable, and reversible.

## Set the interaction contract

- Default to human checkpoints after the input/mechanism view, the first multi-map comparison, and the interesting-unit drill-down.
- At each checkpoint, show the artifact paths, visible observations, surprises or counterexamples, and the smallest useful next step.
- Pause for user interpretation unless the user explicitly requests autonomous execution. A request to produce one stage does not authorize silently completing later stages.
- Keep proposed outcomes labeled as hypotheses. Let the visible maps contradict them.

## Follow the workflow

### 1. Simplify the question

- Choose the smallest contrast that could reveal the effect.
- Hold nuisance dimensions fixed where possible.
- Avoid broad parameter grids, regression suites, and population claims initially.
- State the mechanism, the observable map-level prediction, and what would count against it.

### 2. Show the input or manipulation

- Plot the concrete inputs needed to interpret the contrast: for example an image patch, transformation or retinal path, framewise speed, spatial-frequency power, or the predicted landing in a unit's preferred range.
- Put observed quantities and derived proxies in separate, clearly labeled panels.
- Save the figure and the values used to produce it.
- Stop at the input/mechanism checkpoint.

### 3. Compare multiple raw maps

- Plot several units, conditions, and timepoints rather than relying on one attractive example.
- Use matched spatial extents and comparable color scaling; state when a shared scale would obscure structure and a per-unit scale is used.
- Keep the stimulus, path, unit identity, condition, time, rate, and relevant instantaneous metric adjacent to the map.
- Add direct condition-difference maps when subtraction is meaningful.
- Describe what is visibly changing without generalizing to the population.
- Stop at the multi-map checkpoint.

### 4. Select examples audibly

- Define selection roles before choosing units when practical.
- Include at least one positive example, one dissociation or failure case, and one control when the available units permit it.
- Useful roles include largest map change, largest metric change, strong proxy with weak observed change, weak proxy with strong observed change, low-preference control, high-preference control, and an off-axis or otherwise mechanistically distinct control.
- Save a CSV or equivalent table with unit identifier, selection role, criterion name, criterion value, reference condition/timepoint, and relevant metadata.
- Distinguish algorithmic selection from user-requested examples. Never present post hoc hand-picking as automatic selection.

### 5. Drill into selected units

- Produce complete time-resolved map sheets, difference maps, and response or metric timecourses for the selected units.
- Keep metric values on or immediately beside the maps from which they arise.
- Check whether temporal averaging, normalization, clipping, or a near-zero denominator changes the interpretation.
- Refine, qualify, or reject the mechanism based on these examples.
- Stop at the drill-down checkpoint.

### 6. Summarize last

- Compute group curves, uncertainty intervals, and population summaries only after the map-level behavior is understood.
- Test whether the proposed proxy explains both map change and metric change.
- Report dissociations and failures alongside positive results.
- Prefer paired differences or absolute quantities over ratios when the reference can be near zero.
- Make every aggregate claim traceable to the concrete examples and saved selection table.

## Preserve an audit trail

- Save commands, configuration, cache identity or provenance, source data identifiers, and output paths.
- Record whether artifacts are smoke tests, targeted visualization renders, or production summaries.
- Preserve raw and normalized quantities needed to diagnose an apparent effect.
- Do not overwrite a prior interpretable checkpoint merely to make a polished final figure.

## Report each checkpoint

Report:

1. artifacts produced;
2. what is visibly happening;
3. surprising, contradictory, or ambiguous cases;
4. what remains unsupported;
5. the smallest useful next step;
6. the specific question requiring human judgment, if any.

## Load the SSI precedent when relevant

Read [references/ssi-debugging-case-study.md](references/ssi-debugging-case-study.md) when the task involves SSI, BackImage, retinal motion, activation-map sharpness, unit-map debugging, power redistribution, or adapting the prior SSI workflow. Apply its transferable method and its metric/provenance caveats; do not copy its scientific conclusions into a different analysis.

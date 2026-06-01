# fixRSVP trajectory implementation readiness

## What is now prepared

1. A pilot scaffold script exists at scripts/run_fixrsvp_trajectory_pilot.py.
2. The script uses eval/fixrsvp.py:get_fixrsvp_data as the canonical loader.
3. The required output directory tree from the revised analysis plan is created automatically.
4. Required machine-readable output files are created so each stage can append real metrics incrementally.
5. A Stage 0 readiness table is emitted to session_qc.csv to gate whether to proceed.

## Why this is the right integration point

1. eval/fixrsvp.py already handles trial collation, duplicate removal, fixation thresholding, and image-id alignment.
2. scripts/fixrsvp_eye_conventions.py provides explicit eye convention transforms, which should be used for all px/deg conversions in downstream stages.
3. Existing Jacobian and covariance scripts under scripts/jacobian_predictive_framework and scripts/figure_fixrsvp_mcfarland_covariance_*.py already establish naming conventions for session outputs in outputs/.

## Immediate implementation backlog by stage

### Stage 0 QC

1. Extend session_qc.csv from one-row readiness to a richer long-form schema (metric, value, split).
2. Save basic plots:
   - eye position histogram,
   - eye velocity distribution,
   - drift segment duration distribution (after segmentation lands).
3. Add a hard fail criterion JSON in summaries/ with pass/fail reasons.

### Stage 1 perisaccadic transient

1. Add microsaccade detector module with calibrated velocity thresholds and refractory period.
2. Save event table to microsaccades.csv with onset/peak/offset and validity flags.
3. Build peri-event population tensor and transient-axis SVD outputs.

### Stage 2 drift segmentation and trajectory visualization

1. Define exclusion windows around microsaccades.
2. Create drift segment table in drift_segments.csv.
3. Build residual PCA trajectories per image-window and export representative plots.

### Stage 3 distance-distance geometry

1. Implement pair-class generator for the exact classes in the revised plan.
2. Compute neural and eye distances in residual PCA space.
3. Write class-wise corr and robust slope with bootstrap CIs to distance_distance_metrics.csv.

### Stage 4 increment geometry

1. Compute latent increments dz and eye increments dp from drift-only bins.
2. Fit ridge/reduced-rank local maps dz = A dp with held-out evaluation.
3. Emit magnitude coupling and directional alignment metrics to increment_metrics.csv.

### Stage 5 image-conditioned generalization

1. Implement split strategy by image-window repeats.
2. Compare within-image vs cross-image vs shuffle performance.
3. Emit shared-vs-image-specific variance fractions to image_conditioning_metrics.csv.

### Stage 6 microsaccade boundaries

1. Only unlock if Stage 1 transient duration is stable.
2. Compare pre/post local maps with and without transient subspace subtraction.
3. Save angles and cross-prediction deltas in microsaccade_boundaries/.

### Stage 7 covariance bridge

1. Compute covariance for drift-only, peri-event, and recovery windows.
2. Align with prior FEM covariance bases where available.
3. Write covariance alignment metrics to covariance_alignment_metrics.csv.

## Data contract for stage modules

Each stage function should consume a shared session payload:

1. robs: (trial, time, unit)
2. eyepos: (trial, time, 2) in visual degrees
3. image_ids: (trial, time), -1 for invalid
4. trial_ids: stable trial identity vector
5. cids: unit ids aligned to robs last axis
6. valid masks derived from finite robs, finite eyepos, and valid image id

Each stage should return:

1. metrics rows for a csv file,
2. optional artifacts dict with arrays for reuse by later stages,
3. a summary dict for summaries/.

## Suggested first pilot command

python scripts/run_fixrsvp_trajectory_pilot.py \
  --subject Allen \
  --date 2022-02-16 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_120.yaml \
  --use-cached-data

## Exit criteria before full implementation

1. session_qc.csv shows enough drift_step_candidates.
2. frac_image_time_cells_ge_2 is high enough for leave-one-out residuals.
3. Valid bin fraction is acceptable after fixation and image-id masking.
4. If any fail, use reduced-scope fallback (distance-distance + covariance only).

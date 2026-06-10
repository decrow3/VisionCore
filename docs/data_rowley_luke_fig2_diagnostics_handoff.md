# DataRowleyV1V2 Handoff: Luke/Figure 2 Export Diagnostics

This note lists export-side diagnostics that would be best implemented in
`/home/declan/DataRowleyV1V2`. VisionCore can diagnose inclusion after export,
but several failure modes should be caught where the Rowley datasets are built.

## 1. Unit Identity And Gate Provenance Manifest

Add a per-session manifest that joins unit identity across FixRSVP, dots
calibration, YAML metadata, `qccontam`, `sortercontam`, and any V1/V2 region
labels. The goal is to make it impossible for a figure analysis to silently use
column indices as cluster IDs or compare gates in mismatched CID spaces.

Suggested output columns:

- `session`, `eye`, `cluster_id`, `fixrsvp_col`, `dots_col`
- `region`, `depth`, `shank`, `kilosort_label`
- `qccontam_pass`, `sortercontam_pass`, `yaml_cids`, `yaml_visual`
- `dots_snr`, `dots_n_spikes`, `dots_gate_pass`
- `fixrsvp_n_spikes`, `fixrsvp_nan_frac`, `fixrsvp_exported`

Suggested checks:

- Assert that FixRSVP and dots exports expose stable `cluster_ids` or `cids`
  metadata, not only positional columns.
- Assert that the V1 unit pool used for export matches the documented gate
  source.
- Write a compact `export_unit_manifest.tsv` next to each exported dataset.

## 2. Gaze Center, PRL, And PPD Geometry Audit

Add an export diagnostic that records the coordinate origin and conversion used
for each eye and each stimulus block. This should compare the center used for
FixRSVP fixation gating, dots RF localization, PRL/localization outputs, and
the session-specific pixels-per-degree calibration.

Suggested output columns:

- `session`, `eye`, `block`, `ppd_i`, `ppd_j`, `ppd_source`
- `gaze_center_i_px`, `gaze_center_j_px`, `center_source`
- `prl_i_px`, `prl_j_px`, `prl_source`
- `rf_x_deg`, `rf_y_deg`, `rf_diameter_deg`, `rf_eccentricity_deg`
- `rf_x_px`, `rf_y_px`, `rf_sigma_px`, `rf_geometry_source`

Suggested checks:

- Assert that RF diameter/eccentricity in degrees are computed with the same
  session/eye-specific PPD used by the dots calibration.
- Assert plausible ranges for V1 RFs near the trained fixation/PRL location.
- Save before/after coordinate examples for a few units so VisionCore can
  compare exported dots RF geometry against step07-style RF geometry.

## 3. FixRSVP Timing And Missingness Audit

Add a per-trial timing/missingness manifest so figure code can distinguish poor
neural reliability from export alignment problems, invalid eye samples, or
NaN-heavy trial segments.

Suggested output columns:

- `session`, `eye`, `trial_id`, `trial_start_t_ephys`
- `n_psth_bins`, `first_psth_bin`, `last_psth_bin`
- `n_fix_bins`, `n_dpi_valid_bins`, `n_finite_eye_bins`
- `fraction_eye_valid`, `fraction_robs_finite`
- `stim_onset_t_ephys`, `first_flip_t_ephys`, `timing_source`

Suggested checks:

- Assert that `trial_inds`, `psth_inds`, `t_bins`, and binned `robs` have
  consistent lengths and monotonically increasing time within each trial.
- Explicitly document any 240 Hz to 120 Hz downsampling and verify that
  stimulus/frame timing survives it.
- Export the DPI-valid mask and any truncation/invalid-bin masks used during
  dataset construction, so downstream analyses can reproduce whole-trial and
  bin-level eye-position gates.

VisionCore-side script:
`/home/declan/VisionCore/scripts/diagnose_luke_fig2_inclusion.py` now performs
post-export checks for gate overlap, reliability versus trial count, and
lagged split-half PSTH reliability.

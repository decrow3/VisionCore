# Neuron inclusion criteria — current test_rowley10.py behavior

## Context and constraints

The binocular fixrsvp DictDataset has 290 units, 293 trials, psth_inds 0–108 (454 ms at
240 Hz, starting at the first image flip — no pre-stimulus baseline within the trial).

### Relationship to the session YAML

The session YAML (`Luke_2026-03-16_binocular_V1.yaml`) already contains pre-computed
quality lists, written by `scripts/build_dataset_configs_from_base.py`.

**All YAML QC lists are PASSING lists — they contain units that passed the criterion,
not units that failed.** This is easy to misread.

| YAML field  | Meaning (units that PASS)                          | n (binocular) | n (right-eye) |
|-------------|-----------------------------------------------------|---------------|---------------|
| `visual`    | SNR > 5.0 from gaborium STE                         | 1 (cid 1053)  | 14            |
| `qcmissing` | med_missing_pct < 45% (passing spike completeness)  | 85            | 85            |
| `qccontam`  | contam_pct < 50% (passing contamination check)      | ~all          | ~all          |
| `cids`      | intersect(visual, qccontam) — for digital twin      | 1             | 14            |

So 290 − 85 = 205 units have ≥ 45% missing spikes in this session. Only 85 have
reasonably complete spike recordings.

`qccontam` passes nearly all units because the Rowley contamination check is lenient at
50% threshold (or because kilosort QC files aren't available and dummy zeros are used —
see `build_dataset_configs_from_base.py` line 113).

**test_rowley9.py must NOT read from the YAML.** All criteria are computed independently
from raw data. The YAML values are cross-checked at the end to validate the pipeline:
discrepancies flag either a bug in this script or an issue with how the YAML was
generated. The binocular YAML giving only 1 visual unit at SNR > 5.0 needs independent
verification — it may be correct (binocular gaborium has noisier eye tracking → fewer
valid frames → lower STE SNR) but should be confirmed at multiple thresholds.

---

## Criterion 0 — Fixation eccentricity gate (currently missing everywhere)

**Status**: NOT enforced in test_rowley3–8. The `apply_radius_filter = False` flag has
existed since test_rowley6 with `radius_deg = 7.0`, but is always disabled and uses the
wrong radius anyway.

**What the experiment enforces**: the `faceRadius` parameter in `FixRsvpTrial` is
**1.5°** for this session (Luke_2026-03-16, confirmed from trial parameters). The
experiment control software aborts the trial when the eye leaves this window — but only
at the trial level. Within-trial bins where the eye has drifted to e.g. 1.2° are never
caught by the trial-end mechanism.

**What `dpi_valid` captures**: hardware DPI tracking validity only. It catches 1.8% of
bins where the tracker lost the eye, but misses 456 bins where the tracker was fine but
the eye exceeded 1.5° (eccentricity > 1.5° AND dpi_valid = 1).

**Impact on the data**:
- 5.7% of all bins have eccentricity > 1.5°
- 94 / 293 trials (32%) have at least one bin outside the fixation window
- Max eccentricity 47.7° — clearly a tracking artifact

**Fix for test_rowley9**: add an eccentricity mask to `dfs_serial` immediately after
loading the eye trace and before trial-align:

```python
fixation_radius_deg = 1.5   # matches faceRadius from experiment parameters
ecc_valid = np.hypot(eyepos_serial[:, 0], eyepos_serial[:, 1]) <= fixation_radius_deg
dfs_serial = dfs_serial & ecc_valid
```

This should be applied BEFORE the trial-align step so that trials truncated by a
fixation break get shorter `dur_trial` values and may be dropped by the
`dur > min_fix_dur_bins` gate. Note that the `valid_eyepos_radius = 7` in the dataset
metadata is for the shifter training ROI, not for this behavioural validity check.

This filter belongs in the dataset generation pipeline eventually (write back to
`dpi_valid` during `04_generate_shifted_datasets.py`), but for now apply it in script.

---

## Two unit pools

Because some analyses are per-neuron (FEM fraction, PSTH, rasters) and others require
matched observations across neurons (McFarland covariance), define two overlapping pools:

**Pool A — per-neuron analyses**  
Broader inclusion. Each unit carries its own valid-trial subset; matching is not required.

**Pool B — covariance analysis**  
Subset of Pool A. All units must share a common set of valid trials. Pool A units that
would suppress too many trials for everyone are downgraded to Pool A only.

---

## Criterion 1 — Visual responsiveness (dots RF SNR)

The current script no longer uses gaborium STE SNR as the hard visual gate. It uses the
step-01 dots-calibration path instead, matching the Rowley preprocessing pipeline more
closely for legacy `processed_mvp` sessions.

**Data source**:
- `dpi_calibration/dots_binned_data.dset`
- `dpi_calibration/{eye}_eye/calibrated_dpi.csv`
- `dpi_calibration/{eye}_eye/calibration_params.npz`
- cached output: `dpi_calibration/{eye}_eye/dots_rf_snr.npz`

**Method**:
- interpolate calibrated gaze to dots frames
- bin dots stimulus in retinal coordinates with `bin_dots_to_stimulus(...)`
- compute dots STA over valid gaze samples
- compute `max_snr` with `calculate_rf_snr(...)`
- use the primary-eye dots SNR for `visual_mask`

The script reports both `SNR >= 5.0` and `SNR >= 10.0`, but the actual gate is currently
`SNR >= 10.0` on the primary eye:

```python
visual_mask = max_snr_visual >= snr_threshold_primary   # default 10.0
```

The YAML `visual` list remains a cross-check only. It still reflects the older gaborium
criterion and should not be treated as the source of truth for `test_rowley10.py`.

### Empirical finding from the current runs

For `Luke_2026-03-01` after switching to dots RF:
- `SNR >= 5.0`: 260 / 279
- `SNR >= 10.0`: 112 / 279
- `visual_mask & spikes_ok`: 43 / 279
- `visual_mask & spikes_ok & reliability_ok`: 1 / 279

For `Luke_2026-03-02` and `Luke_2026-03-16`, dots RF similarly removes the gaborium
bottleneck: many more units pass the visual gate than survive the downstream FixRSVP
reliability check. In the current code path, split-half reliability is the dominant
filter, not dots RF SNR.

---

## Criterion 2 — FixRSVP stimulus responsiveness

**d-prime is not directly computable** from the binocular fixrsvp DictDataset: the trial
window starts at the first image flip (psth_inds = 0) with no pre-stimulus baseline.

Options in ascending complexity:

**Option A — split-half PSTH reliability** (recommended)  
Does the neuron have a consistent, repeatable response pattern across trials? A neuron
with low split-half r is either silent or noise-dominated in the FixRSVP context.

```python
rng = np.random.default_rng(42)
r2_per_unit = np.zeros(n_units)
for _ in range(20):
    perm = rng.permutation(n_trials)
    half = n_trials // 2
    psth_a = np.nanmean(robs_mc[perm[:half]], axis=0)   # (T, n_units)
    psth_b = np.nanmean(robs_mc[perm[half:2*half]], axis=0)
    for j in range(n_units):
        a, b = psth_a[:, j], psth_b[:, j]
        fin = np.isfinite(a) & np.isfinite(b)
        if fin.sum() > 2 and np.std(a[fin]) > 0 and np.std(b[fin]) > 0:
            r2_per_unit[j] += np.corrcoef(a[fin], b[fin])[0, 1] ** 2
reliability_mask = (r2_per_unit / 20) >= min_reliability   # default 0.10
```
Directly parallels test_rowley5 and figure2_inclusion.py. Applies per-neuron (Pool A).

**Option B — pseudo-baseline d-prime within trial**  
Use the very first bins (psth_inds ≤ 4, ~17 ms, before neural response onset) as a
crude baseline vs bins 12–72 (50–300 ms) as response. Noisy for sparse units.
Previous use: test_rowley5, `min_dprime = 0.05`.

**Option C — raw spike times (future work)**  
Load kilosort spike times; use inter-trial intervals as baseline. Closest to the Yates
implementation in `figure2_inclusion.py`. Adds ephys file dependency.

Use **Option A** now. If Option B is added, combine with AND: both must pass.

### Empirical finding from the current Luke_2026-03-16 run

The FixRSVP split-half reliability gate is currently the main limiter among visually
responsive units. Using the current `r² >= 0.10` criterion, only 13 of the 33 visual
units at SNR >= 3.5 pass reliability, and 11 pass both reliability and spike count.

This means the present script is effectively selecting units that are both visually
responsive in Gaborium and strongly repeatable in FixRSVP. That is a stricter criterion
than visual responsiveness alone.

---

## Criterion 3 — Missing data and truncation masking

There are now two distinct missing-data steps:

### Stage 0 — runtime truncation datafilter (`missing_pct < 45`)

Before trial alignment, `test_rowley10.py` now applies the same amplitude-truncation QC
mask used by the VisionCore runtime datafilter pipeline. This comes from the Rowley
registry via `sess.get_missing_pct_interp(cids)` and is evaluated at each time bin.

```python
missing_pct_fun = sess.get_missing_pct_interp(cids_all)
missing_pct = missing_pct_fun(t_bins_serial)          # (T, n_units)
missing_pct_mask = missing_pct < 45

# Match VisionCore's runtime behavior: chronically high-missing units are treated
# as multi-units and not zeroed out by the filter.
chronic_multi_units = np.nanmedian(missing_pct, axis=0) >= 45
missing_pct_mask[:, chronic_multi_units] = True

robs_serial_masked = robs_serial.copy()
robs_serial_masked[~missing_pct_mask] = np.nan
```

This happens before the spike-count and split-half reliability criteria, so those two
criteria are now computed on truncation-masked spike trains rather than the raw `.dset`
values.

This is important because the session YAML and Rowley config generation expect missing
spikes to be handled at runtime by a datafilter, not by a hard upstream exclusion from
`cids`.

The McFarland estimator computes pairwise covariances over time bins valid for ALL
included units. A unit with many NaN bins suppresses those bins for everyone.

`qcmissing` in the YAML flags units where > 45% of (lag-windowed) bins are invalid.
This is a useful signal but not a hard exclusion. In the current script it is best read
as a configuration cross-check, while the actual runtime mask comes from
`missing_pct < 45` and the post-alignment `nan_frac` gate below.

### Stage A — per-unit NaN fraction gate (Pool A → Pool B)

```python
# After trial-align and dur_trial gate
nan_frac_per_unit = (
  (np.isnan(robs_mc) & dfs_mc[:, :, None]).sum(axis=(0, 1))
  / max(int(dfs_mc.sum()), 1)
)                                                     # (n_all_units,)
nan_ok = nan_frac_per_unit <= max_unit_nan_frac            # default 0.20
```

Units in `qcmissing` but below `max_unit_nan_frac` within the fixrsvp window are
allowed into Pool B. Units above the threshold are Pool A only.

### Stage B — trial quality gate (for Pool B)

After Pool B units are selected, drop trials that would be invalid for too many of them:

```python
robs_b = robs_mc[:, :, pool_b_mask]                        # (n_trials, T, n_b)
unit_missing_per_trial = (np.isnan(robs_b) & dfs_mc[:, :, None]).any(axis=1)  # (n_trials, n_b)
bad_unit_frac = unit_missing_per_trial.mean(axis=1)         # (n_trials,)
good_trials_b = good_trials & (bad_unit_frac <= max_bad_trial_frac)  # default 0.10
```

Drop trials where >10% of Pool B units have any NaN in that trial. This is on top of
the existing `dur_trial > min_fix_dur_bins` gate.

After this, recompute `valid_used` from Pool B units + `good_trials_b`. The existing
all-or-nothing per-bin mask inside `run_covariance_decomposition` handles the rest.

### Empirical finding from the current runs

For `Luke_2026-03-01`, the newly restored runtime truncation filter reports:
- valid bins after `missing_pct < 45`: 12,615,950 / 13,045,203
- units with median missing percentage >= 45 but retained by the multi-unit fallback:
  230 / 279

So the truncation QC is active and materially affects the spike matrix, but it still does
not explain the major final loss. Even after truncation masking, the dominant bottleneck
remains split-half reliability.

---

## Combined filter chain

```
Load fixrsvp dset
  ↓
Eccentricity gate: dfs_serial &= hypot(eyepos) <= 1.5°    [NEW — Criterion 0]
  ↓
Trial-align → good_trials (dur > min_fix_dur_bins = 20 bins)
  ↓
Load runtime truncation QC via `sess.get_missing_pct_interp(...)`
  ↓
Mask `robs_serial` where missing_pct >= 45                    [NEW in test_rowley10]
  ↓
Load primary-eye dots RF SNR cache / recompute → visual_mask
  ↓
total_spikes_ok:   nansum(robs_mc, axis=(0,1)) > 200          [existing]
  ↓
nan_frac_ok:       valid-bin-only NaN fraction <= 0.20        [new, Stage A]
  ↓
reliability_ok:    split-half PSTH r² ≥ 0.10 across all bins  [new, Option A]
  ↓
POOL A = visual_mask & total_spikes_ok & reliability_ok
         (nan_frac_ok failure → Pool A only, not excluded)
  ↓
POOL B = POOL A & nan_frac_ok
  ↓
good_trials_b:     bad_unit_frac ≤ 0.10 on Pool B             [new, Stage B]
  ↓
Per-neuron analyses → POOL A, per-unit good_trials
Covariance analysis → POOL B, good_trials_b
```

---

## Parameters (config block)

```python
# Fixation eccentricity gate  [NEW]
fixation_radius_deg    = 1.5   # matches faceRadius from FixRsvpTrial parameters

# Visual responsiveness
snr_threshold_primary  = 10.0  # dots RF SNR on the primary eye
snr_threshold_report   = 5.0   # report only; not the hard gate

# Runtime truncation filter
missing_pct_threshold  = 45.0  # matches VisionCore missing_pct datafilter

# FixRSVP responsiveness
min_reliability        = 0.10  # split-half PSTH r², averaged over 20 splits

# Missing data — Pool B gate
max_unit_nan_frac      = 0.20  # units above this → Pool A only
max_bad_trial_frac     = 0.10  # trials with >10% Pool B units NaN → dropped for covariance

# Existing
total_spikes_threshold = 200
min_fix_dur_bins       = 20
```

---

## Waterfall report (print to stdout)

```
All units:               279
After spike threshold:   129
After dots RF SNR:       43   (right-eye, threshold=10.0, after spike threshold)
After reliability:       1
─────────────────────────────
POOL A (per-neuron):     1 / 279
After NaN-frac gate:     1
─────────────────────────────
POOL B (covariance):     1 / 279

Trials (good_trials):       286 / 295
After bad-unit-frac gate:   286 / 295   (Pool B covariance trials)
```

---

## YAML cross-check (end of script)

After all criteria are computed independently, compare against the YAML and print a
reconciliation table:

```
=== YAML cross-check ===
visual (YAML):          [1053]             (binocular, SNR≥5.0)
visual (this script):   [...]              (right-eye, SNR≥5.0)
  agreement:  X / Y  —  in YAML not script: [...]  in script not YAML: [...]

qcmissing (YAML):       85 units
nan_frac > 20% (this):  X units
  overlap:    X / 85

Pool A cluster IDs:     [...]
Pool B cluster IDs:     [...]
```

If independently computed `visual` substantially disagrees with the YAML, report it
prominently. Do not silently fall back to the YAML value.

## YAML write-back (future pipeline step)

Once thresholds are validated across sessions, add a pipeline step
`08_compute_inclusion_criteria.py` that writes back:

```yaml
visual:      [...]   # dots RF SNR ≥ threshold, primary eye
qcmissing:   [...]   # units with nan_frac > max_unit_nan_frac
reliability: [...]   # units failing split-half PSTH r threshold
pool_a:      [...]   # final Pool A cluster IDs
pool_b:      [...]   # final Pool B cluster IDs
```

---

## Notes vs previous scripts

| Script      | Visual check | d-prime / reliability | NaN / missing gate  | Trial drop   |
|-------------|-------------|----------------------|---------------------|--------------|
| test_rowley3 | None        | None                 | None                | None         |
| test_rowley4 | None        | None                 | None                | None         |
| test_rowley5 | None        | d' + split-half r    | truncation QC (45%) | None         |
| test_rowley6–8 | None      | None                 | None                | None         |
| **test_rowley10** | primary-eye dots RF SNR (independent) | split-half $r^2$ over 20 random splits | runtime `missing_pct` mask + `nan_frac` gate | bad-unit-frac |
| figure2_inclusion | **Implicit via `prepare_data → cids`** (= YAML-driven session selection, with split-half $r^2$ used only as a diagnostic metric) | split-half $r^2$ + other metrics explored, not hard-gated | runtime `missing_pct` datafilter via `prepare_data` | `min_total_spikes = 500` via `align_fixrsvp_trials` |

---

## Reliability calibration against Figure 2 / Yates

Using the current `test_rowley10.py` logic with:
- runtime truncation masking from `missing_pct < 45`
- primary-eye dots RF gate at `SNR >= 10`
- spike threshold `> 200`
- split-half reliability computed as mean $r^2$ over 20 random 50/50 splits

the aggregate candidate-unit reliability distribution across runnable Rowley
`processed_mvp` dataset entries is:

- `n = 273`
- median $r^2 = 0.0143`
- Q1 = `0.0052`
- Q3 = `0.0480`
- fraction with $r^2 >= 0.10` = `0.1465`

The corresponding Figure 2 / Yates distribution, computed with the same 20-split $r^2$
form on the Figure 2 analysis sessions, is:

- `n = 1102`
- median $r^2 = 0.2100`
- Q1 = `0.0945`
- Q3 = `0.3795`
- fraction with $r^2 >= 0.10` = `0.7332`

Interpretation: the current hard cutoff `r² >= 0.10` is not calibrated to the Rowley
legacy sessions in the same way it would be for the Figure 2 / Yates population. Under
the present Rowley candidate-unit definition, only about 14.7% of units clear that
threshold, versus about 73.3% in the Figure 2 / Yates set.

This supports the current working conclusion:
- the metric itself matches Figure 2 (20 random split-half PSTH $r^2$)
- the hard threshold is a Rowley-side script decision, not an inherited Figure 2 gate
- for legacy Rowley sessions, `0.10` is currently a very strict cutoff

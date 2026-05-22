# Pipeline Hypotheses: Why Unit Counts Are Low

*Written 2026-04-27. Based on analysis of trow10.log across all 9 sessions (Luke_2025-07-30 through Luke_2026-03-16).*

---

## The Facts That Need Explaining

| Session | Units | Pass spike | Pass SNR | Pass reliability | Pool A | Pool B | Notes |
|---------|-------|-----------|---------|-----------------|--------|--------|-------|
| 2025-07-30 | 160 | 56 | 56 | 13 | 13 | 11 | |
| 2025-08-04 | 129 | 43–44 | 43–44 | 3 | 3 | 3 | tiny trial count (65) |
| 2025-08-05 | — | — | — | — | — | — | Kilosort dir missing |
| 2026-03-01 | 279 | 155 | 154 | 4 | 3 | 2 | 295 trials, still only 4 reliable |
| 2026-03-02 (L) | 278 | 152 | 124 | 15 | 10 | 10 | |
| 2026-03-02 (R) | 278 | 153 | 124 | 17 | 13 | 13 | |
| 2026-03-08 | 151 | 62 | 62 | 6 | 5 | 3 | fixation center (−5.5°, 2.1°) |
| 2026-03-09 | 120 | 45 | 45 | **0** | 0 | 0 | fixation center (−0.1°, 4.9°) |
| 2026-03-11 | 99 | 57 | 57 | 12 | 12 | 10 | |
| 2026-03-13 | 179 | 64 | 64 | 2 | 1 | 1 | 81.9% DPI valid; fixation center (0.1°, 5.0°); 67% trials pass duration |
| 2026-03-16 | 119 | 95 | 94 | 23 | 23 | 19 | **best session** |

The cross-cutting alarming number is that **67–88% of units have median missing_pct ≥ 45%** in every single session. The scripts apply `missing_pct_threshold = 45%` to exclude these units from Pool B and to NaN their bins before the reliability calculation.

---

## Hypothesis 1 — The Kilosort amplitude threshold is systematically too high, creating artificial truncation across all units

**Where in the pipeline:** Kilosort sorting step, before any pipeline script runs.

The `amp_truncation/truncation_qc.npz` file measures the fraction of spikes estimated to fall below the sorter's amplitude detection threshold. A `median_missing_pct ≥ 45%` in 67–88% of units means the vast majority of neurons are chronically below threshold for nearly half their spike production. This is not plausible for well-isolated single units unless the detection threshold is too high.

Two likely causes: (a) the Neuropixels probe gain settings in the `patched_pipeline_results` differed from defaults, raising the effective threshold; or (b) the `patched` curation pass accepted units that Kilosort had already assigned poor amplitude, causing good units to inherit poor truncation scores.

**Test:** Check the Kilosort config params.py for `Th`, `ThPre`, and compare to the Yates pipeline. If Rowley uses higher thresholds, that explains the universal truncation pattern.

---

## Hypothesis 2 — The FixRSVP image sequence is randomised per trial, making the PSTH alignment uninformative

**Where in the pipeline:** `DataRowleyV1V2/exp/fix_rsvp.py`, `generate_fixrsvp_dataset`.

The code finds `start_idx = np.where(image_ids == 2)[0][0]` — the first frame where image ID 2 appears — and aligns all trials to that point. `psth_inds` then counts forward from there. If the images after image 2 are drawn randomly from the 65-image library on each trial (only image 1 is the "different" one per the code comment), the PSTH at time T is averaging over different images across trials. The cross-half correlation would be near zero in expectation regardless of how tuned a neuron is.

The comment "The first image is different on every trial so we skip it" could mean only image 1 varies, OR it could mean image 1 is special and everything after is also random. The image_id==2 alignment trick only makes the PSTH meaningful if subsequent images follow a fixed sequence.

**Test:** Load a raw session `exp['D']` and print `trial.image_ids` for several trials. If the sequence after position 1 is identical across trials, split-half PSTH reliability is valid. If it varies, the entire reliability criterion is measuring noise.

---

## Hypothesis 3 — The `patched_pipeline_results` directories were re-sorted after the DPI calibration was finalised, creating a CID mismatch

**Where in the pipeline:** Step 04 (`generate_shifted_datasets.py`) — the CIDs used to generate `fixrsvp.dset` may differ from those in the dots RF SNR cache, the gaborium STE SNR, or the YAML QC lists.

If a re-sort changed cluster IDs, the dots RF SNR cache (computed on the old sort) would match by shape but map SNR values to the wrong units. The cache validity check is `np.array_equal(cached_cids, target_cids)` — if both sorts happened to produce the same number of units with different IDs that happen to sort the same way, the check would pass while the SNR values are misassigned.

The evidence: 2026-03-01 has 154 units passing dots RF SNR ≥ 5 and 295 trials, but only 4 pass reliability. If the SNR values are attributed to the wrong units, `visual_mask` would pass the wrong 154 units, and the reliably responding ones (correct identities) would be excluded before reliability is even computed.

**Test:** For one session, compare `dset.metadata['cluster_ids']` from `fixrsvp.dset` against the `cids` stored in `dots_rf_snr.npz`. Check whether the spike counts in `fixrsvp.dset` for units flagged as high-SNR actually have visible responses in the raster plots.

---

## Hypothesis 4 — The DPI calibration is unreliable for several sessions, causing the eccentricity gate to exclude most neural signal even after median-subtraction

**Where in the pipeline:** Step 01 (`calibrate_ddpi_new.py`), then carried through step 04.

Sessions 2026-03-09 and 2026-03-13 both show fixation centers at ~5° elevation (−0.1°, 4.9° and 0.1°, 5.0°). The median-subtraction fix centres the eccentricity gate on the median, which is correct for systematic offset — but only if the DPI noise is low. If the calibration is noisy (large within-session scatter), the 1.5° fixation window will exclude many bins where the animal was actually fixating. 

This is compounded for 2026-03-13 which has 81.9% DPI valid (all other sessions: 96.7–99.9%) and only 135/201 trials passing the minimum duration filter (67% vs 90–97% for good sessions). Both facts are consistent with a DPI tracking failure: the tracker loses signal frequently, and when it does report valid positions, they're noisy.

The 5° elevation offset in calibrated coordinates is also suspicious for sessions where the animal uses an online eye (the right eye) for the task but the calibration was done on the left eye.

**Test:** Inspect the DPI calibration QC figures in `processed_path/dpi_calibration/{eye}_eye/` for the problematic sessions. If the dots-calibration scatter plot shows poor alignment or the PRL correction diverged, the calibration failed.

---

## Hypothesis 5 — The foveal depth band filter is cutting out most of the recording for sessions where probe placement varies

**Where in the pipeline:** Step 04, `use_foveal_depth_band = True`, `foveal_depth_band_um = 1500`.

The 1500μm band centred on the foveal representation should capture a large portion of the 3840μm Neuropixels probe. But if the probe was inserted shallower than usual, or if the recorded region of cortex is displaced from the expected foveal representation, the depth filter could exclude most units. The unit counts vary from 99 (2026-03-11) to 279 (2026-03-01), a 2.8× range across sessions with similar array sizes — this is consistent with variable depth filter impact.

The PIPELINE_GUIDE explicitly warns: "if fixrsvp.dset is tiny, depth filter too aggressive — check `foveal_depth_band_um`."

**Test:** Re-run step 04 for a low-unit session with `use_foveal_depth_band = False` and check how many units appear. If the count jumps substantially, the band is cutting signal. Also check whether the units in Pool A cluster tightly within the band or are spread through it.

---

## Hypothesis 6 — The analysis window of 240 bins (1 second) is longer than individual image presentations, mixing responses to multiple images in each PSTH

**Where in the pipeline:** `test_rowley10.py`, `valid_time_bins = 240`, but rooted in how `generate_fixrsvp_dataset` structures the data.

If the FixRSVP task shows images at 4–8 Hz (125–250 ms per image), a 1-second window captures 4–8 image transitions. The PSTH at time bin T aggregates the response to whatever image was on-screen at time T. If the sequence is fixed (hypothesis 2 resolution: sequence IS fixed), this is fine — but only the bins corresponding to image-onset transients will carry strong signal. Most bins (during steady-state viewing of the same image) will look similar to baseline. This dilutes the split-half correlation.

Worse: units with sharp onset transients (common in V1) fire briefly at image onset and are silent otherwise. Their signal is concentrated in 4–8 bins per image transition, but the correlation is computed over all 240 bins including 220+ near-silent ones. Noise dominates the correlation.

**Test:** Truncate the analysis window to 60 bins (250ms) — the duration of the first image after image 2 — and recompute reliability. If reliability jumps substantially, the window is the problem. Alternatively, identify the image transition times in `psth_inds` and compute reliability only on onset-aligned windows.

---

## Hypothesis 7 — The `missing_pct` interpolation uses spike-index-based time windows that may be misaligned with the fixrsvp trial time bins

**Where in the pipeline:** `DataRowleyV1V2/data/registry.py`, `get_missing_pct_qc` — conversion of `window_blocks` from spike indices to seconds.

The truncation QC windows are stored as spike indices (`window_blocks`) and converted to seconds by indexing into the spike time array: `time_windows[mask] = shank_st[shank_clu == cid][wb]`. If the spike times in `shank_st` are on a different clock than the `t_bins_serial` values in the fixrsvp dataset, the `missing_pct_interp` function would be interpolating at the wrong times — reporting, e.g., a unit as 60% missing during a period where it is actually clean, causing unnecessary NaN masking.

The timing chain (nidaq → ptb → ephys) is complex and involves separate `.mat` timing files for each shank. A silent bug in the clock alignment for one shank's spike times would affect the entire QC interpolation.

**Test:** For one unit with high median missing_pct, plot `missing_pct_values` over time alongside the actual spike rate. If the reported truncation peaks don't correspond to periods of reduced firing, the timing is misaligned.

---

## Hypothesis 8 — The `2026-03-09` session has a fundamental data integrity issue: robs in fixrsvp.dset is near-zero or corrupted for most units

**Where in the pipeline:** Step 04, `generate_fixrsvp_dataset` call.

The session has 45 units passing spike threshold (>200 spikes total) and dots RF SNR ≥ 5, plus 112 valid trials. Getting exactly 0 reliable units is statistically implausible under any reasonable noise model — even with a 0.05 threshold, chance fluctuations should produce some false positives. This strongly suggests a pathological data state rather than genuine biology.

Candidate causes: the `ptb2ephys` clock mapping failed silently for this session, placing all spike times outside the fixrsvp trial windows. `bin_spikes` would then return near-zero counts for all units across all trials. Units would appear to have "spikes" from a different data epoch that accidentally passed the total count threshold, but the trial-aligned robs would be near-uniform noise.

**Test:** Load `fixrsvp.dset` for 2026-03-09 directly and inspect `robs.mean(0)` (mean firing rate per unit) and `robs.var(0)`. Also plot a few trial rasters for the 45 units. If all units show flat rasters with no peri-stimulus structure, the spike times are not aligned to the stimulus.

---

## Hypothesis 9 — The FixRSVP stimuli are face images that do not strongly drive V1 neurons, making the dots RF SNR an unreliable proxy for FixRSVP responsiveness

**Where in the pipeline:** Conceptual mismatch between the visual criterion (dots RF SNR, measuring simple feature tuning) and the reliability criterion (FixRSVP, using complex face-like images).

The image library contains 65 images named `im01–im65`, loaded from `rsvpFixStim.mat`. The presence of `faceRadius`, `get_face_library()`, and `get_facecal_library()` functions alongside this task strongly suggests the images are marmoset faces or face-like stimuli. V1 neurons are tuned to local orientation, spatial frequency, and contrast — not to face identity or holistic face features. They should still give reliable responses to the edges and textures within faces, but the responses will be weaker and harder to detect than responses to optimised Gabor patches or drifting gratings.

The dots RF SNR measures direction/motion selectivity, which is also not a direct predictor of face-image responsiveness. Units that pass both filters but fail reliability may simply not be strongly driven by the specific natural images in the FixRSVP set.

**Test:** Compute the STE/STA from the fixrsvp `stim` array directly (the same way the gaborium STA is computed in pipeline/05). If units that pass dots RF SNR show low fixrsvp STE SNR, the stimuli aren't driving them. The fixrsvp gaborium STE approach would be a much better visual responsiveness criterion than dots RF SNR.

---

## Hypothesis 10 — The `missing_pct_threshold = 45%` is far too aggressive for this recording setup and is removing the majority of real signal before reliability can be assessed

**Where in the pipeline:** `test_rowley10.py` parameter; originating in the QC design for this probe/sorter combination.

The 45% threshold was presumably chosen to match the VisionCore training pipeline. But 45% missing in these data means the sorter is routinely missing nearly half the spikes from a unit — which, for the sorter's perspective, is plausible if the unit's amplitude is close to the detection threshold. For Rowley data with `patched_pipeline_results` (manually curated), many borderline units were deliberately kept because they are real neurons, even though their amplitudes are low.

Setting `missing_pct_threshold = 45%` then treats these legitimate low-amplitude units as pathological and NaN-masks half their spike bins. Even with the unmasked reliability fix (H8 in the analysis code), the NaN masking does not affect the raw robs in the .dset — it's applied in the script. So the unmasked fix correctly bypasses this. But the fact that 67–88% of units fail the median test means the threshold needs to be revisited for this data type.

**Test:** Re-run test_rowley10 with `missing_pct_threshold = 70%` (or disable it entirely for Pool A reliability). If Pool A unit counts increase substantially (especially for sessions like 2026-03-01 where 230/279 units exceed the current threshold), the threshold is the binding constraint rather than true biological unresponsiveness.

---

## Priority Order for Investigation

1. **H2 (RSVP sequence randomisation)** — zero-cost to check, fundamental to whether reliability makes sense at all
2. **H8 (2026-03-09 data integrity)** — load the .dset and inspect robs directly; rules out or confirms catastrophic data failure
3. **H1 (Kilosort threshold)** — check params.py in the patched pipeline; explains the universal truncation pattern
4. **H10 (missing_pct threshold too strict)** — run test_rowley10 with relaxed threshold; can be done immediately
5. **H9 (fixrsvp STE SNR vs dots RF SNR)** — compute fixrsvp STE for one good session; if it identifies a different unit set, switch criteria
6. **H4 (DPI calibration quality)** — inspect calibration QC figures for 2026-03-09 and 2026-03-13
7. **H3 (CID mismatch after re-sort)** — compare cluster_ids across dset files and caches
8. **H6 (analysis window too long)** — truncate to 60 bins and recompute reliability
9. **H5 (depth filter too aggressive)** — re-run step 04 with band disabled for a low-unit session
10. **H7 (missing_pct timing misalignment)** — plot per-unit missing_pct time series vs spike rate

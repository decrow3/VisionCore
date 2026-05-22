#%%
"""
test_rowley9.py — neuron inclusion criteria + McFarland covariance decomposition

Changes from v8:

1. Criterion 0 — Eccentricity gate (NEW)
   Applies hypot(eyepos_cyclopean) <= 1.5° to dfs_serial for ALL eye sources.
   Matches faceRadius=1.5° from FixRsvpTrial experiment parameters.
   Previously dfs_serial used only dpi_valid (hardware validity); ~5.7% of bins
   with eccentricity > 1.5° and valid DPI tracking were silently included.

2. Criterion 1 — Visual responsiveness via gaborium STE SNR (NEW)
   SNR computed independently from gaborium_sta_ste.npy for binocular and
   right-eye datasets using sigma=[0,0,4,4] Gaussian smoothing (pipeline/05
   convention). Right-eye SNR used for visual_mask. YAML used as cross-check
   only, never as input.

3. Criterion 2 — FixRSVP split-half PSTH reliability (NEW)
   20-split average r². No d-prime: binocular fixrsvp has no pre-stimulus
   baseline (psth_inds 0–108 start at first image flip).

4. Two unit pools
   Pool A — visual_mask & spikes_ok & reliability_ok  (per-neuron analyses)
   Pool B — Pool A & nan_frac_ok                      (covariance, matched trials)

5. Stage B trial gate (NEW)
   For Pool B: drops trials where > max_bad_trial_frac of Pool B units have
   any NaN. Applied before covariance decomposition.

6. YAML cross-check (NEW)
   Independent criteria compared against binocular and right-eye YAML QC lists.
   Discrepancies reported prominently; script never falls back to YAML values.

7. Waterfall report printed to stdout.
"""

#%%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import yaml
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

import torch

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
from VisionCore.covariance import (run_covariance_decomposition,
                                    extract_valid_segments, extract_windows,
                                    estimate_vergence_conditional_on_cyclopean,
                                    conservative_cvergence_from_2d)
from VisionCore.subspace import project_to_psd
from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR

#%% ---------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

subject = 'Luke'
date    = '2026-03-16'
primary_eye = 'binocular'

legacy_processed_root = Path('/mnt/ssd2/RowleyMarmoV1V2/processed')
dataset_dir = Path('datasets_binocular')
fix_name    = 'fixrsvp.dset'

windows_ms             = [5, 10, 20, 40, 80]
total_spikes_threshold = 200
valid_time_bins        = 240
dt                     = 1 / 240.0
t_hist_ms              = 50
n_bins                 = 15

min_fix_dur_bins = 20

# ── Fixation eccentricity gate [Criterion 0]
fixation_radius_deg  = 1.5   # matches faceRadius from FixRsvpTrial parameters

# ── Gaborium STE SNR [Criterion 1]
snr_threshold_primary = 1.5#5.0   # a low bar for right-eye SNR for visual_mask
snr_threshold_report  = 5   # also reported for comparison; matches YAML

gaborium_bino_npy  = (legacy_processed_root / f'{subject}_{date}'
                      / 'datasets_binocular' / 'gaborium_sta_ste.npy')
gaborium_right_npy = (legacy_processed_root / f'{subject}_{date}'
                      / 'datasets_gaussian' / 'right_eye' / 'gaborium_sta_ste.npy')

yaml_bino_path  = (VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'sessions'
                   / f'{subject}_{date}_binocular_V1.yaml')
yaml_right_path = (VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'sessions'
                   / f'{subject}_{date}_right_V1.yaml')

# ── FixRSVP reliability [Criterion 2]
min_reliability      = 0.10
n_reliability_splits = 20

# ── NaN / missing data [Criterion 3]
max_unit_nan_frac  = 0.20   # units above this → Pool A only
max_bad_trial_frac = 0.10   # Stage B: trials with > this frac of Pool B NaN units dropped

# ── 2D analysis
n_bins_c_2d  = 5
n_bins_v_2d  = 5
min_pairs_2d = 30
n_shuff_2d   = 1

BASE_FIGURES_DIR = FIGURES_DIR / 'mcfarland' / f'{subject}_{date}_v9'

#%% ---------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)

def _get_required(dset, key):
    if key not in dset.keys():
        raise KeyError(f"Missing required key '{key}'. Available: {list(dset.keys())}")
    return _to_numpy(dset[key])

def _get_optional(dset, key, default=None):
    if key not in dset.keys():
        return default
    return _to_numpy(dset[key])

def _as_bool_1d(x, n_expected=None):
    x = np.asarray(x).reshape(-1)
    x = x > 0.5 if x.dtype != bool else x
    if n_expected is not None:
        assert len(x) == n_expected
    return x.astype(bool)

def _ensure_2d_eyepos(eyepos):
    eyepos = np.asarray(eyepos, dtype=np.float32)
    if eyepos.ndim == 1:
        return np.stack([eyepos, np.zeros_like(eyepos)], axis=1)
    if eyepos.shape[1] == 1:
        return np.concatenate([eyepos, np.zeros((len(eyepos), 1), dtype=np.float32)], axis=1)
    return eyepos

def get_eye_trace_and_valid(dset, eye_source='default'):
    keys = set(dset.keys())
    eyepos_default = _get_required(dset, 'eyepos')
    n = eyepos_default.shape[0]
    dpi_valid_default = _get_optional(dset, 'dpi_valid', np.ones(n, dtype=bool))
    dpi_valid_default = _as_bool_1d(dpi_valid_default, n)

    if eye_source in ('default', 'cyclopean'):
        return eyepos_default, dpi_valid_default, 'eyepos'

    if eye_source == 'left':
        for k in ['eyepos_left', 'eyepos_dpi_left']:
            if k in keys:
                eye   = _get_required(dset, k)
                valid = _get_optional(dset, 'dpi_valid_left', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("No eyepos_left / eyepos_dpi_left found.")

    if eye_source == 'right':
        for k in ['eyepos_right', 'eyepos_dpi_right']:
            if k in keys:
                eye   = _get_required(dset, k)
                valid = _get_optional(dset, 'dpi_valid_right', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("No eyepos_right / eyepos_dpi_right found.")

    if eye_source == 'binocular_diff':
        left = right = None
        for k in ['eyepos_left', 'eyepos_dpi_left']:
            if k in keys:
                left = _get_required(dset, k); break
        for k in ['eyepos_right', 'eyepos_dpi_right']:
            if k in keys:
                right = _get_required(dset, k); break
        if left is None or right is None:
            raise KeyError("No left/right eye traces for binocular_diff.")
        valid_l = _as_bool_1d(_get_optional(dset, 'dpi_valid_left',  np.ones(len(left),  dtype=bool)), len(left))
        valid_r = _as_bool_1d(_get_optional(dset, 'dpi_valid_right', np.ones(len(right), dtype=bool)), len(right))
        return left - right, (valid_l & valid_r), 'eyepos_left_minus_right'

    if eye_source == 'pupil_left':
        for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil']:
            if k in keys:
                eye   = _get_required(dset, k)
                valid = _get_optional(dset, 'pupil_valid_left', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("No left pupil trace found.")

    if eye_source == 'pupil_right':
        for k in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil']:
            if k in keys:
                eye   = _get_required(dset, k)
                valid = _get_optional(dset, 'pupil_valid_right', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("No right pupil trace found.")

    raise ValueError(f"Unknown eye_source: {eye_source!r}")


def _nearest_resample_bool(sample_times, values, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    values = _as_bool_1d(values, len(sample_times))
    target_times = np.asarray(target_times, dtype=np.float64)
    right_idx = np.searchsorted(sample_times, target_times, side='left')
    right_idx = np.clip(right_idx, 0, len(sample_times) - 1)
    left_idx  = np.clip(right_idx - 1, 0, len(sample_times) - 1)
    choose_left = (np.abs(target_times - sample_times[left_idx])
                   <= np.abs(sample_times[right_idx] - target_times))
    return values[np.where(choose_left, left_idx, right_idx)]


def _interp_xy(sample_times, xy, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float64)
    return np.column_stack([
        np.interp(target_times, sample_times, xy[:, 0]),
        np.interp(target_times, sample_times, xy[:, 1]),
    ]).astype(np.float32)


def add_calibrated_pupil_traces(dset, aux_processed_path):
    keys  = set(dset.keys())
    t_bins = _get_required(dset, 't_bins').astype(np.float64)
    for eye in ('left', 'right'):
        pupil_key = f'pupil_{eye}'
        valid_key = f'pupil_valid_{eye}'
        if pupil_key in keys and valid_key in keys:
            continue
        csv_path = Path(aux_processed_path) / 'dpi_calibration' / f'{eye}_eye' / 'calibrated_dpi.csv'
        if not csv_path.exists():
            print(f"No calibrated pupil CSV for {eye} eye: {csv_path}")
            continue
        try:
            pupil_df = pd.read_csv(csv_path, usecols=['t_ephys', 'pupil_i', 'pupil_j', 'pupil_valid'])
        except ValueError as exc:
            print(f"Skipping {eye} pupil import: {exc}")
            continue
        sample_times = pupil_df['t_ephys'].to_numpy(dtype=np.float64)
        pupil_xy     = pupil_df[['pupil_i', 'pupil_j']].to_numpy(dtype=np.float32)
        pupil_valid  = pupil_df['pupil_valid'].to_numpy()
        valid_samples = (np.isfinite(sample_times)
                         & _as_bool_1d(pupil_valid, len(sample_times))
                         & np.all(np.isfinite(pupil_xy), axis=1))
        if valid_samples.sum() < 2:
            continue
        dset[pupil_key] = _interp_xy(sample_times[valid_samples], pupil_xy[valid_samples], t_bins)
        dset[valid_key] = _nearest_resample_bool(sample_times, pupil_valid, t_bins)
        keys.update([pupil_key, valid_key])
        print(f"Loaded calibrated {pupil_key}")


def get_enabled_eye_configs(dset, primary_eye_name):
    keys = set(dset.keys())
    enabled_configs = [('primary-dpi', 'default')]
    has_left        = any(k in keys for k in ['eyepos_left', 'eyepos_dpi_left'])
    has_right       = any(k in keys for k in ['eyepos_right', 'eyepos_dpi_right'])
    has_left_pupil  = any(k in keys for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil'])
    has_right_pupil = any(k in keys for k in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil'])
    if has_left and has_right:
        enabled_configs.append(('binocular', 'binocular_diff'))
    if has_left:
        enabled_configs.append(('left-dpi', 'left'))
    if has_left_pupil:
        enabled_configs.append(('left-pupil', 'pupil_left'))
    if has_right:
        enabled_configs.append(('right-dpi', 'right'))
    if has_right_pupil:
        enabled_configs.append(('right-pupil', 'pupil_right'))
    primary_subdir = f'{primary_eye_name}-dpi'
    return [(primary_subdir if subdir == 'primary-dpi' else subdir, source)
            for subdir, source in enabled_configs]


def serial_to_trial_aligned(robs, eyepos, dfs, trial_inds, time_inds):
    unique_trials = np.unique(trial_inds)
    n_trials = len(unique_trials)
    n_time   = np.max(time_inds).item() + 1
    n_units  = robs.shape[1]
    robs_trial   = np.nan * np.zeros((n_trials, n_time, n_units), dtype=np.float32)
    eyepos_trial = np.nan * np.zeros((n_trials, n_time, 2),       dtype=np.float32)
    dfs_trial    = np.zeros((n_trials, n_time), dtype=bool)
    dur_trial    = np.zeros(n_trials, dtype=int)
    for itrial in range(n_trials):
        idx = np.where(trial_inds == unique_trials[itrial])[0]
        if not len(idx):
            continue
        tt = time_inds[idx]
        valid_tt = _as_bool_1d(dfs[idx], len(idx))
        if valid_tt.any():
            robs_trial[itrial, tt[valid_tt]]   = robs[idx][valid_tt]
            eyepos_trial[itrial, tt[valid_tt]] = eyepos[idx][valid_tt]
        dfs_trial[itrial, tt] = valid_tt
        dur_trial[itrial]     = valid_tt.sum()
    return robs_trial, eyepos_trial, dfs_trial, dur_trial, unique_trials


def plot_raster(t, trial_idx, height=1, ax=None, **kwargs):
    t   = np.stack([t, t, np.nan * np.ones_like(t)], axis=1).flatten()
    tri = np.stack([trial_idx, trial_idx + height, np.nan * np.ones_like(trial_idx)], axis=1).flatten()
    if ax is None:
        ax = plt.gca()
    ax.plot(t, tri, 'k', lw=0.5, **kwargs)


def plot_cov_vs_distance(mats, i, j, win_ms, ax=None):
    Ceye        = mats['Ceye']
    bin_centers = mats['bin_centers']
    count_e     = mats['count_e']
    Crate       = mats['Intercept']
    Cpsth       = mats['PSTH']
    valid = count_e > 0
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers[valid], Ceye[:, i, j][valid], 'o', alpha=0.6, label='Measured cov')
    ax.axhline(Crate[i, j], linestyle=':',  lw=2, label='Intercept (Crate)')
    ax.axhline(Cpsth[i, j], linestyle='--', lw=2, label='PSTH cov')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_xlabel('Δ Eye Trajectory (a.u.)')
    ax.set_ylabel('Covariance')
    ax.set_title(f'Neuron {i} | {win_ms} ms')
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    return ax


def _compute_snr(npy_path):
    sta_ste = np.load(npy_path)                                           # (2, n_units, n_lags, H, W)
    stes    = sta_ste[1]                                                  # (n_units, n_lags, H, W)
    signal  = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
    signal  = gaussian_filter(signal, sigma=[0, 0, 4, 4])
    noise   = np.median(signal[:, 0], axis=(1, 2))
    snr     = signal.max(axis=(2, 3)) / (noise[:, None] + 1e-8)
    return snr.max(axis=1)                                                # (n_units,)


#%% ---------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------

sess = get_session(subject, date)
print(f"Session: {sess.name}")
aux_processed_path = Path(sess.processed_path)
sess.processed_path = legacy_processed_root / f'{subject}_{date}'

fix_path = Path(sess.processed_path) / dataset_dir / fix_name
print(f"Loading fixrsvp from: {fix_path}")
assert fix_path.exists(), f"fixrsvp.dset not found at: {fix_path}"

dset_fix = DictDataset.load(fix_path)
add_calibrated_pupil_traces(dset_fix, aux_processed_path)

eye_configs = get_enabled_eye_configs(dset_fix, primary_eye)

print("Loaded fixrsvp DictDataset:")
print(f"  robs:       {dset_fix['robs'].shape}")
print(f"  keys:       {list(dset_fix.keys())}")
print(f"  enabled eye configs: {eye_configs}")

robs_serial  = _get_required(dset_fix, 'robs').astype(np.float32)
trial_inds_s = _get_required(dset_fix, 'trial_inds').astype(int)
time_inds_s  = _get_required(dset_fix, 'psth_inds').astype(int)
n_all_units  = robs_serial.shape[1]
cids_all     = np.array(dset_fix.metadata.get('cluster_ids', np.arange(n_all_units)))

n_time_full    = int(np.max(time_inds_s)) + 1
time_bins_full = np.arange(n_time_full) * dt
session_label  = f'{subject}_{date}'
frac_fem_summary = []

# Load vergence signal (left − right) once for all conditions
_keys      = set(dset_fix.keys())
_left_key  = next((k for k in ['eyepos_left', 'eyepos_dpi_left']  if k in _keys), None)
_right_key = next((k for k in ['eyepos_right', 'eyepos_dpi_right'] if k in _keys), None)
if _left_key and _right_key:
    _left_serial  = _ensure_2d_eyepos(_get_required(dset_fix, _left_key).astype(np.float32))
    _right_serial = _ensure_2d_eyepos(_get_required(dset_fix, _right_key).astype(np.float32))
    eyepos_verg_serial = _left_serial - _right_serial
    print(f"Vergence loaded ({_left_key} − {_right_key}), shape {eyepos_verg_serial.shape}")
else:
    eyepos_verg_serial = None
    print("No separate left/right traces — vergence analysis will be skipped.")

#%% ---------------------------------------------------------------------------
# Pre-loop inclusion criteria (computed once, independent of eye source)
# -----------------------------------------------------------------------------

print("\n" + "="*70)
print("INCLUSION CRITERIA")
print("="*70)

# ── Criterion 0: Eccentricity gate ──────────────────────────────────────────
# Always uses cyclopean eye position (the behavioural validity window).
eyepos_default_raw = _get_required(dset_fix, 'eyepos')
n_bins_serial = eyepos_default_raw.shape[0]
dpi_valid_default = _get_optional(dset_fix, 'dpi_valid', np.ones(n_bins_serial, dtype=bool))
dpi_valid_default = _as_bool_1d(dpi_valid_default, n_bins_serial)

eyepos_cyclopean = _ensure_2d_eyepos(eyepos_default_raw.astype(np.float32))
ecc = np.hypot(eyepos_cyclopean[:, 0], eyepos_cyclopean[:, 1])
ecc_mask = ecc <= fixation_radius_deg

n_total_bins   = len(ecc_mask)
n_dpi_valid    = dpi_valid_default.sum()
n_ecc_valid    = ecc_mask.sum()
n_newly_excl   = (~ecc_mask & dpi_valid_default).sum()

print(f"\nCriterion 0 — Eccentricity gate (radius={fixation_radius_deg}°)")
print(f"  Total serial bins:                 {n_total_bins}")
print(f"  dpi_valid bins:                    {n_dpi_valid} ({100*n_dpi_valid/n_total_bins:.1f}%)")
print(f"  ecc <= {fixation_radius_deg}° bins:              {n_ecc_valid} ({100*n_ecc_valid/n_total_bins:.1f}%)")
print(f"  Valid DPI but ecc > radius:        {n_newly_excl}  (newly excluded by this gate)")
print(f"  Max eccentricity:                  {ecc.max():.2f}°")

# dfs used for all pre-loop inclusion criteria computations
dfs_incl = dpi_valid_default & ecc_mask

# Trial-align with eccentricity-gated dfs
robs_trial_incl, _, dfs_trial_incl, dur_trial_incl, _ = serial_to_trial_aligned(
    robs_serial, eyepos_cyclopean, dfs_incl, trial_inds_s, time_inds_s)

good_trials   = dur_trial_incl > min_fix_dur_bins
n_all_trials  = len(good_trials)
n_good_trials = good_trials.sum()
print(f"\n  Trials total:          {n_all_trials}")
print(f"  Trials dur > {min_fix_dur_bins} bins:  {n_good_trials}")

# Restrict to analysis window
robs_mc = robs_trial_incl[good_trials]
dfs_mc_incl = dfs_trial_incl[good_trials]
n_time_analysis = min(valid_time_bins, robs_mc.shape[1])
iix = np.arange(n_time_analysis)
robs_mc = robs_mc[:, iix]                              # (n_good, T, n_units)
dfs_mc_incl = dfs_mc_incl[:, iix]

# ── Criterion 1: Gaborium STE SNR ────────────────────────────────────────────
print(f"\nCriterion 1 — Gaborium STE SNR (sigma=[0,0,4,4])")

max_snr_bino  = _compute_snr(gaborium_bino_npy)    # (n_all_units,)
max_snr_right = _compute_snr(gaborium_right_npy)   # (n_all_units,)

for label, snr_arr in [('binocular ', max_snr_bino), ('right-eye ', max_snr_right)]:
    n35 = (snr_arr >= snr_threshold_report).sum()
    n50 = (snr_arr >= snr_threshold_primary).sum()
    print(f"  {label}: SNR >= {snr_threshold_report}: {n35:3d}  |  SNR >= {snr_threshold_primary}: {n50:3d}")

# Scatter: binocular vs right-eye SNR
BASE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig_snr, ax_snr = plt.subplots(figsize=(5, 5))
ax_snr.scatter(max_snr_bino, max_snr_right, s=10, alpha=0.5, edgecolors='none')
for thr, ls in [(snr_threshold_report, ':'), (snr_threshold_primary, '--')]:
    ax_snr.axvline(thr, color='r', lw=0.8, linestyle=ls)
    ax_snr.axhline(thr, color='r', lw=0.8, linestyle=ls)
ax_snr.set_xlabel('Binocular SNR')
ax_snr.set_ylabel('Right-eye SNR')
ax_snr.set_title(f'Gaborium STE SNR comparison\n{session_label}')
fig_snr.tight_layout()
fig_snr.savefig(BASE_FIGURES_DIR / 'gaborium_snr_comparison.pdf')
plt.close(fig_snr)

# Histograms
fig_sh, axs_sh = plt.subplots(1, 2, figsize=(10, 4))
for ax_h, (label, snr_arr) in zip(axs_sh, [('binocular', max_snr_bino), ('right-eye', max_snr_right)]):
    ax_h.hist(snr_arr, bins=50, color='steelblue', alpha=0.7)
    for thr, ls in [(snr_threshold_report, ':'), (snr_threshold_primary, '--')]:
        ax_h.axvline(thr, color='r', lw=1.2, linestyle=ls, label=f'{thr}')
    ax_h.set_xlabel('Max SNR')
    ax_h.set_ylabel('Count')
    ax_h.set_title(f'{label} SNR')
    ax_h.legend(title='threshold', frameon=False)
fig_sh.suptitle(session_label)
fig_sh.tight_layout()
fig_sh.savefig(BASE_FIGURES_DIR / 'gaborium_snr_histograms.pdf')
plt.close(fig_sh)

visual_mask      = max_snr_right >= snr_threshold_primary   # primary: right-eye
visual_mask_bino = max_snr_bino  >= snr_threshold_primary   # comparison only
print(f"  visual_mask (right-eye, >= {snr_threshold_primary}): {visual_mask.sum()} / {n_all_units}")

# ── Criterion 2: Total spike count ───────────────────────────────────────────
spikes_per_unit = np.nansum(robs_mc, axis=(0, 1))
spikes_ok = spikes_per_unit > total_spikes_threshold
print(f"\nCriterion 2 — Spike threshold (> {total_spikes_threshold})")
print(f"  Passing: {spikes_ok.sum()} / {n_all_units}")

# ── Criterion 3: Split-half PSTH reliability ─────────────────────────────────
print(f"\nCriterion 3 — Split-half PSTH reliability (r², {n_reliability_splits} splits, >= {min_reliability})")
rng_rel    = np.random.default_rng(42)
n_t_rel    = robs_mc.shape[0]
r2_accum   = np.zeros(n_all_units)
for _split in range(n_reliability_splits):
    perm   = rng_rel.permutation(n_t_rel)
    half   = n_t_rel // 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        psth_a = np.nanmean(robs_mc[perm[:half]], axis=0)      # (T, n_units)
        psth_b = np.nanmean(robs_mc[perm[half:2*half]], axis=0)
    for j in range(n_all_units):
        a, b = psth_a[:, j], psth_b[:, j]
        fin  = np.isfinite(a) & np.isfinite(b)
        if fin.sum() > 2 and np.std(a[fin]) > 0 and np.std(b[fin]) > 0:
            r2_accum[j] += np.corrcoef(a[fin], b[fin])[0, 1] ** 2
mean_reliability = r2_accum / n_reliability_splits
reliability_ok   = mean_reliability >= min_reliability
print(f"  Passing: {reliability_ok.sum()} / {n_all_units}")

# ── NaN fraction gate (Pool A → Pool B) ──────────────────────────────────────
valid_counts_incl = max(int(dfs_mc_incl.sum()), 1)
nan_frac_per_unit = ((np.isnan(robs_mc) & dfs_mc_incl[:, :, None]).sum(axis=(0, 1))
                     / valid_counts_incl)
nan_ok = nan_frac_per_unit <= max_unit_nan_frac
print(f"\nNaN fraction gate (<= {max_unit_nan_frac})")
print(f"  Passing: {nan_ok.sum()} / {n_all_units}")

# ── Pool definitions ─────────────────────────────────────────────────────────
pool_a_mask = visual_mask & spikes_ok & reliability_ok
pool_b_mask = pool_a_mask & nan_ok

pool_a_inds = np.where(pool_a_mask)[0]
pool_b_inds = np.where(pool_b_mask)[0]
cids_pool_a = cids_all[pool_a_mask]
cids_pool_b = cids_all[pool_b_mask]

# ── Stage B baseline: trial gate for Pool B covariance ───────────────────────
if pool_b_mask.sum() > 0:
    robs_b           = robs_mc[:, :, pool_b_mask]             # (n_good, T, n_b)
    unit_missing_b   = (np.isnan(robs_b) & dfs_mc_incl[:, :, None]).any(axis=1)
    bad_unit_frac    = unit_missing_b.mean(axis=1)
    good_trials_b_base = bad_unit_frac <= max_bad_trial_frac
    n_b_trials_base  = good_trials_b_base.sum()
else:
    good_trials_b_base = np.zeros(n_good_trials, dtype=bool)
    n_b_trials_base = 0

# ── Waterfall report ─────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("WATERFALL REPORT")
print("─"*50)
print(f"All units:               {n_all_units}")
print(f"After spike threshold:   {spikes_ok.sum()}")
print(f"After gaborium SNR:      {(visual_mask & spikes_ok).sum()}"
      f"   (right-eye, >= {snr_threshold_primary})")
print(f"After reliability:       {pool_a_mask.sum()}")
print("─"*50)
print(f"POOL A (per-neuron):     {pool_a_mask.sum()} / {n_all_units}")
print(f"After NaN-frac gate:     {pool_b_mask.sum()}")
print("─"*50)
print(f"POOL B (covariance):     {pool_b_mask.sum()} / {n_all_units}")
print(f"\nTrials (good_trials):          {n_good_trials} / {n_all_trials}")
print(f"After bad-unit-frac gate:      {n_b_trials_base} / {n_all_trials}"
      f"   (Pool B covariance trials)")

# ── YAML cross-check ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("YAML CROSS-CHECK")
print("="*50)

def _load_yaml(path):
    if not path.exists():
        print(f"  [WARN] YAML not found: {path}")
        return None
    with open(path) as f:
        return yaml.safe_load(f)

yaml_bino  = _load_yaml(yaml_bino_path)
yaml_right = _load_yaml(yaml_right_path)

def _xcheck(label, yaml_set, script_set):
    only_yaml   = yaml_set - script_set
    only_script = script_set - yaml_set
    print(f"\n{label}")
    print(f"  YAML:   {len(yaml_set)} units  {sorted(yaml_set)}")
    print(f"  Script: {len(script_set)} units  {sorted(script_set)}")
    print(f"  Agreement: {len(yaml_set & script_set)} / {len(yaml_set | script_set)}")
    if only_yaml:
        print(f"  ** In YAML not script: {sorted(only_yaml)}")
    if only_script:
        print(f"  ** In script not YAML: {sorted(only_script)}")

if yaml_bino is not None:
    _xcheck(
        "visual — binocular (YAML snr>=5.0 via sigma=[0,2,2,2]) vs script bino (sigma=[0,0,4,4])",
        set(yaml_bino.get('visual', [])),
        set(cids_all[visual_mask_bino].tolist()),
    )

if yaml_right is not None:
    _xcheck(
        "visual — right-eye (YAML snr>=5.0) vs script right-eye (sigma=[0,0,4,4])",
        set(yaml_right.get('visual', [])),
        set(cids_all[visual_mask].tolist()),
    )

if yaml_bino is not None:
    yaml_qcm     = set(yaml_bino.get('qcmissing', []))   # YAML: units that PASS missing check
    script_pass  = set(cids_all[nan_ok].tolist())
    print(f"\nqcmissing (YAML = passing nan check):  {len(yaml_qcm)} units")
    print(f"nan_frac <= {max_unit_nan_frac} (script passing):  {len(script_pass)} units")
    print(f"  Both pass:          {len(yaml_qcm & script_pass)}")
    print(f"  YAML pass, script fail: {len(yaml_qcm - script_pass)}")
    print(f"  Script pass, YAML fail: {len(script_pass - yaml_qcm)}")
    note = ("Note: YAML threshold is med_missing_pct < 45% (different window/metric "
            f"from nan_frac <= {max_unit_nan_frac} in FixRSVP window)")
    print(f"  {note}")

    yaml_cids = set(yaml_bino.get('cids', []))
    print(f"\ncids (YAML binocular):   {sorted(yaml_cids)}")
    print(f"Pool B cluster IDs:      {sorted(cids_pool_b.tolist())}")
    only_yaml_cids   = yaml_cids - set(cids_pool_b.tolist())
    only_script_cids = set(cids_pool_b.tolist()) - yaml_cids
    if only_yaml_cids:
        print(f"  ** In YAML cids not Pool B: {sorted(only_yaml_cids)}")
    if only_script_cids:
        print(f"  ** In Pool B not YAML cids: {sorted(only_script_cids)}")

print(f"\nPool A cluster IDs: {sorted(cids_pool_a.tolist())}")
print(f"Pool B cluster IDs: {sorted(cids_pool_b.tolist())}")

#%% ---------------------------------------------------------------------------
# Per-eye-source analysis loop
# -----------------------------------------------------------------------------

for subdir_name, eye_source in eye_configs:
    figures_dir = BASE_FIGURES_DIR / subdir_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Eye source: {eye_source}  →  {figures_dir}")
    print(f"{'='*70}")

    # ── 1. Eye trace
    try:
        eyepos_serial, dfs_serial, eye_source_name = get_eye_trace_and_valid(
            dset_fix, eye_source=eye_source)
    except (KeyError, ValueError) as e:
        print(f"  Skipping: {e}")
        continue

    eyepos_serial = _ensure_2d_eyepos(eyepos_serial)
    dfs_serial    = _as_bool_1d(dfs_serial, robs_serial.shape[0])

    _eye_label_map = {
        'eyepos':                  'cyclopean',
        'eyepos_left_minus_right': 'vergence (L−R)',
    }
    display_eye_name = _eye_label_map.get(eye_source_name, eye_source_name)

    # Apply eccentricity gate using cyclopean eye for ALL sources
    dfs_serial = dfs_serial & ecc_mask

    # ── 2. Trial-align
    robs_trial, eyepos_trial, dfs_trial, dur_trial, _ = serial_to_trial_aligned(
        robs_serial, eyepos_serial, dfs_serial, trial_inds_s, time_inds_s)

    # ── 3. Trial filter
    good_trials_src = dur_trial > min_fix_dur_bins
    robs_mc_all   = robs_trial[good_trials_src]
    eyepos_mc_all = eyepos_trial[good_trials_src]
    dfs_mc_all    = dfs_trial[good_trials_src]
    dur_mc_all    = dur_trial[good_trials_src]
    n_good_trials_src = good_trials_src.sum()

    # ── 4. Guard: skip if pools are empty
    if pool_a_mask.sum() == 0:
        print("  Pool A is empty — skipping.")
        continue
    if pool_b_mask.sum() == 0:
        print(f"  {n_good_trials_src} / {len(good_trials_src)} trials kept  (Pool B: 0)")
        print("  Pool B is empty — skipping covariance.")
        _run_cov = False
    else:
        robs_b_src        = robs_mc_all[:, :, pool_b_mask]
        unit_missing_src  = (np.isnan(robs_b_src) & dfs_mc_all[:, :, None]).any(axis=1)
        bad_unit_frac_src = unit_missing_src.mean(axis=1)
        good_trials_b_within = bad_unit_frac_src <= max_bad_trial_frac
        n_b_trials = good_trials_b_within.sum()
        print(f"  {n_good_trials_src} / {len(good_trials_src)} trials kept"
              f"  (Pool B: {n_b_trials})")
        if n_b_trials == 0:
            print("  No valid covariance trials for Pool B — skipping covariance.")
            _run_cov = False
        else:
            _run_cov = True

    if pool_b_mask.sum() == 0:
        good_trials_b_within = np.zeros(n_good_trials_src, dtype=bool)
        n_b_trials = 0

    # ── 5. Slice for Pool B covariance (good_trials_b_within within good trials)
    if _run_cov:
        robs_mc_b   = robs_mc_all[good_trials_b_within]
        eyepos_mc_b = eyepos_mc_all[good_trials_b_within]
        dfs_mc_b    = dfs_mc_all[good_trials_b_within]
        dur_mc_b    = dur_mc_all[good_trials_b_within]

        robs_cov   = robs_mc_b[:, iix][:, :, pool_b_mask]
        eyepos_cov = eyepos_mc_b[:, iix]
        dfs_cov    = dfs_mc_b[:, iix]
        valid_cov  = (dfs_cov
                      & np.isfinite(robs_cov.sum(axis=2))
                      & np.isfinite(eyepos_cov.sum(axis=2)))

    # Trial-align vergence for covariance trials
    if eyepos_verg_serial is not None and _run_cov:
        _, eyepos_verg_trial, _, _, _ = serial_to_trial_aligned(
            robs_serial, eyepos_verg_serial, dfs_serial, trial_inds_s, time_inds_s)
        eyepos_verg_b    = eyepos_verg_trial[good_trials_src][good_trials_b_within]
        eyepos_verg_used = eyepos_verg_b[:, iix]
    else:
        eyepos_verg_used = None

    time_full = time_bins_full[:robs_mc_all.shape[1]]

    # ── 6. Covariance decomposition on Pool B
    if _run_cov:
        print(f"  Running run_covariance_decomposition "
              f"({robs_cov.shape[0]} trials, {robs_cov.shape[2]} Pool B units)...")
        results, last_mats = run_covariance_decomposition(
            robs_cov, eyepos_cov, valid_cov,
            window_sizes_ms=windows_ms, t_hist_ms=t_hist_ms,
            n_bins=n_bins, dt=dt,
            eyepos_vergence=eyepos_verg_used,
        )
    else:
        results, last_mats = [], []

    if _run_cov:
        window_idx = windows_ms.index(20) if 20 in windows_ms else 0
        win_label  = windows_ms[window_idx]

        Ctotal  = project_to_psd(last_mats[window_idx]['Total'])
        Cpsth   = project_to_psd(last_mats[window_idx]['PSTH'])
        Crate   = project_to_psd(last_mats[window_idx]['Intercept'])
        Cfem    = project_to_psd(last_mats[window_idx]['FEM'])
        CnoiseC = project_to_psd(Ctotal - Crate)
        MeanRates = results[window_idx]['Erates']

        denom        = np.diag(Crate)
        fem_fraction = np.where(denom > 0, 1.0 - np.diag(Cpsth) / denom, np.nan)
        fem_fraction = fem_fraction[np.isfinite(fem_fraction)]

        sweep_windows_ms  = [r['window_ms']       for r in results]
        sweep_ff_corr     = [r['ff_corr_mean']     for r in results]
        sweep_ff_corr_sem = [r['ff_corr_sem']      for r in results]
    else:
        win_label = windows_ms[0]
        fem_fraction = np.array([])
        sweep_windows_ms = windows_ms
        sweep_ff_corr = [np.nan] * len(windows_ms)
        sweep_ff_corr_sem = [np.nan] * len(windows_ms)
        Crate = Cpsth = CnoiseC = MeanRates = None

    # ── 7. 2D vergence analysis — ONLY for cyclopean eye source
    _run_2d          = _run_cov and (eye_source_name == 'eyepos') and (eyepos_verg_used is not None)
    _result_2d       = None
    _result_2d_shuff = None
    Cverg2d          = None

    if _run_2d:
        _device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _win_bins_2d = max(1, int(20 / (dt * 1000)))
        _t_hist_bins = int(t_hist_ms / (dt * 1000))
        _t_hist_used = max(_t_hist_bins, _win_bins_2d)

        _segs = extract_valid_segments(valid_cov, min_len_bins=36)
        _robs = torch.tensor(np.nan_to_num(robs_cov,          nan=0.0), dtype=torch.float32, device=_device)
        _eyec = torch.tensor(np.nan_to_num(eyepos_cov,        nan=0.0), dtype=torch.float32, device=_device)
        _eyev = torch.tensor(np.nan_to_num(eyepos_verg_used,  nan=0.0), dtype=torch.float32, device=_device)

        _SC, _EyeC, _Tidx, _ = extract_windows(_robs, _eyec, _segs, _win_bins_2d, _t_hist_used, device=str(_device))
        _,   _EyeV,  _,   _  = extract_windows(_robs, _eyev, _segs, _win_bins_2d, _t_hist_used, device=str(_device))

        if _SC is not None and _EyeV is not None and _SC.shape[0] >= 20:
            print(f"  Running 2D conditional estimator ({_SC.shape[0]} windows, 20 ms)...")
            _result_2d = estimate_vergence_conditional_on_cyclopean(
                _SC, _EyeC, _EyeV, _Tidx,
                n_bins_c=n_bins_c_2d, n_bins_v=n_bins_v_2d,
                min_pairs_per_bin=min_pairs_2d,
            )
            print(f"  Pair counts (d_c rows × d_v cols):\n{_result_2d['count2d']}")

            _EyeV_shuff = _EyeV.clone()
            _T_np       = _Tidx.detach().cpu().numpy()
            _rng_shuff  = np.random.default_rng(42)
            for _t_val in np.unique(_T_np):
                _idx_t = np.where(_T_np == _t_val)[0]
                if len(_idx_t) > 1:
                    _perm = _rng_shuff.permutation(len(_idx_t))
                    _EyeV_shuff[_idx_t] = _EyeV[_idx_t[_perm]]

            _result_2d_shuff = estimate_vergence_conditional_on_cyclopean(
                _SC, _EyeC, _EyeV_shuff, _Tidx,
                n_bins_c=n_bins_c_2d, n_bins_v=n_bins_v_2d,
                min_pairs_per_bin=min_pairs_2d,
            )

            _slope_real  = np.nanmean(np.diag(_result_2d['C_near_slope']))
            _slope_shuff = np.nanmean(np.diag(_result_2d_shuff['C_near_slope']))
            print(f"  Near-dc slope: real={_slope_real:.4f}  shuffle={_slope_shuff:.4f}")

            Cverg2d = conservative_cvergence_from_2d(_result_2d, CnoiseC, min_pairs=min_pairs_2d)
            if Cverg2d is not None:
                print(f"  Cverg2d diag: mean={np.nanmean(np.diag(Cverg2d)):.4f}, "
                      f"max={np.nanmax(np.diag(Cverg2d)):.4f}")
            else:
                print("  conservative_cvergence_from_2d: insufficient data.")
        else:
            print("  2D analysis skipped: insufficient windows.")

    # ── 8. FF before/after conservative vergence
    ff_before_verg2d      = None
    ff_after_verg2d       = None
    verg2d_frac_resid     = None
    fem_fraction_adj_verg = None

    if Cverg2d is not None and MeanRates is not None:
        CnoiseCV         = project_to_psd(CnoiseC - Cverg2d)
        ff_before_verg2d = np.where(MeanRates > 0, np.diag(CnoiseC)  / MeanRates, np.nan)
        ff_after_verg2d  = np.where(MeanRates > 0, np.diag(CnoiseCV) / MeanRates, np.nan)
        resid_diag       = np.diag(CnoiseC)
        verg2d_frac_resid = np.where(resid_diag > 0, np.diag(Cverg2d) / resid_diag, np.nan)
        _crate_d = np.diag(Crate)
        with np.errstate(invalid='ignore'):
            fem_fraction_adj_verg = np.where(
                _crate_d > 0,
                (np.diag(Crate - Cpsth) + np.diag(Cverg2d)) / _crate_d,
                np.nan,
            )

    # ── 9. Collect summary entry
    frac_fem_summary.append({
        'subdir_name':      subdir_name,
        'eye_source':       eye_source,
        'eye_source_name':  eye_source_name,
        'display_eye_name': display_eye_name,
        'win_label_ms':     win_label,
        'fem_fraction':     fem_fraction.copy(),
        'sweep_windows_ms':     sweep_windows_ms,
        'sweep_ff_corr':        sweep_ff_corr,
        'sweep_ff_corr_sem':    sweep_ff_corr_sem,
        'result_2d':            _result_2d,
        'result_2d_shuff':      _result_2d_shuff,
        'Cverg2d':              Cverg2d,
        'ff_before_verg2d':     ff_before_verg2d,
        'ff_after_verg2d':      ff_after_verg2d,
        'verg2d_frac_resid':    verg2d_frac_resid,
        'fem_fraction_adj_verg': fem_fraction_adj_verg,
        'pool_a_n':   pool_a_mask.sum(),
        'pool_b_n':   pool_b_mask.sum(),
        'n_b_trials': n_b_trials,
    })

    # ── 10. Per-condition figure (4 panels)
    fig_c, axs_c = plt.subplots(1, 4, figsize=(20, 4))

    ax = axs_c[0]
    if fem_fraction.size:
        ax.hist(fem_fraction, bins=np.linspace(0, 1, 31), color='steelblue', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('1 − α  (C_FEM / C_rate)')
    ax.set_ylabel('Count')
    ax.set_title(f'FEM fraction [Pool B, n={pool_b_mask.sum()}]\n{display_eye_name}')

    ax = axs_c[1]
    if _result_2d is not None:
        _C2d_c      = _result_2d['C2d']
        _dv_means_c = _result_2d['dv_cell_means']
        _count2d_c  = _result_2d['count2d']
        _x_real, _y_real = [], []
        for _bv in range(n_bins_v_2d):
            if _count2d_c[0, _bv] >= min_pairs_2d and np.isfinite(_dv_means_c[0, _bv]):
                _x_real.append(_dv_means_c[0, _bv])
                _y_real.append(np.nanmean(np.diag(_C2d_c[0, _bv])))
        if _x_real:
            ax.plot(_x_real, _y_real, 'o-', color='steelblue', lw=1.5, ms=6, label='near-dc (real)')
        if _result_2d_shuff is not None:
            _C2d_sh = _result_2d_shuff['C2d']
            _dv_sh  = _result_2d_shuff['dv_cell_means']
            _cnt_sh = _result_2d_shuff['count2d']
            _x_sh, _y_sh = [], []
            for _bv in range(n_bins_v_2d):
                if _cnt_sh[0, _bv] >= min_pairs_2d and np.isfinite(_dv_sh[0, _bv]):
                    _x_sh.append(_dv_sh[0, _bv])
                    _y_sh.append(np.nanmean(np.diag(_C2d_sh[0, _bv])))
            if _x_sh:
                ax.plot(_x_sh, _y_sh, 's--', color='gray', lw=1.2, ms=5, alpha=0.7, label='shuffle null')
        ax.set_xlabel('Vergence distance d_v (RMS)')
        ax.set_ylabel('mean diag(C2d)  [near-cyclopean]')
        ax.set_title(f'2D: near-dc C2d vs dv\n{display_eye_name}')
        ax.legend(frameon=False, fontsize=8)
    else:
        if any(np.isfinite(v) for v in sweep_ff_corr):
            ax.errorbar(sweep_windows_ms, sweep_ff_corr, yerr=sweep_ff_corr_sem,
                        fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3)
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('FF (McFarland corrected)')
        ax.set_title(f'FF vs window size\n{display_eye_name}')

    ax = axs_c[2]
    if verg2d_frac_resid is not None:
        _vf = verg2d_frac_resid[np.isfinite(verg2d_frac_resid)]
        if _vf.size:
            ax.hist(_vf, bins=np.linspace(0, 1, 31), color='mediumpurple', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Cverg2d / diag(CnoiseC)')
        ax.set_ylabel('Count')
        mn = np.nanmean(_vf) if _vf.size else float('nan')
        ax.set_title(f'Conservative verg frac of residual\nmean={mn:.3f} | {display_eye_name}')
    else:
        ax.text(0.5, 0.5, '2D not run\n(non-cyclopean source)',
                ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=10)
        ax.set_title(f'Conservative verg fraction\n{display_eye_name}')

    ax = axs_c[3]
    if ff_before_verg2d is not None and ff_after_verg2d is not None:
        _ok = np.isfinite(ff_before_verg2d) & np.isfinite(ff_after_verg2d)
        if _ok.any():
            _fb = ff_before_verg2d[_ok]; _fa = ff_after_verg2d[_ok]
            _lim = max(_fb.max(), _fa.max(), 0.1) * 1.05
            ax.scatter(_fb, _fa, s=20, alpha=0.7, color='darkorange', edgecolors='none')
            ax.plot([0, _lim], [0, _lim], 'k--', lw=0.8, alpha=0.5)
            ax.set_xlim(0, _lim); ax.set_ylim(0, _lim)
            ax.set_aspect('equal')
        ax.set_xlabel('FF before vergence (McFarland residual)')
        ax.set_ylabel('FF after conservative Cverg2d')
        ax.set_title(f'FF before/after Cverg2d ({win_label} ms)\n{display_eye_name}')
    else:
        ax.text(0.5, 0.5, '2D not run\n(non-cyclopean source)',
                ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=10)
        ax.set_title(f'FF before/after\n{display_eye_name}')

    fig_c.suptitle(f'{session_label} | {win_label} ms | Pool A={pool_a_mask.sum()} Pool B={pool_b_mask.sum()}')
    fig_c.tight_layout()
    fig_c.savefig(figures_dir / 'frac_fem_hist.pdf')
    plt.close(fig_c)

    # ── 11. Eye position heatmap (all Pool A good trials)
    ind_sorted = np.argsort(dur_mc_all)
    fig, ax = plt.subplots()
    ax.imshow(eyepos_mc_all[ind_sorted, :, 0], vmin=-0.5, vmax=0.5,
              aspect='auto', cmap='coolwarm', interpolation='none', origin='lower',
              extent=[time_full[0], time_full[-1], 0, robs_mc_all.shape[0]])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Trial')
    ax.set_title(f'Eye X | {display_eye_name}')
    fig.savefig(figures_dir / 'eyepos_heatmap.pdf')
    plt.close(fig)

    # ── 12. Per-unit raster PDF — Pool A units
    unit_pdf_path = figures_dir / f'unit_rasters_{session_label}.pdf'
    with PdfPages(unit_pdf_path) as pdf:
        for sel_a, cid in enumerate(cids_pool_a):
            unit_robs   = robs_mc_all[:, :, pool_a_inds[sel_a]]
            ind_u       = np.argsort(dur_mc_all)
            unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
            trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

            fig, ax_r = plt.subplots(figsize=(8, 4))
            ax_r.set_title(f'Neuron {cid} | {display_eye_name} [Pool A]')
            plot_raster(time_full[t_idx_r], trial_idx_r, height=1, ax=ax_r)
            ax_r.axvline(0, color='r', lw=0.8)
            ax_r.set_xlim(time_full[0], time_full[-1])
            ax_r.set_xlabel('Time (s)'); ax_r.set_ylabel('Trial')
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"  Saved {unit_pdf_path}")

    # ── 13. Per-unit covariance PDF — Pool B units only
    if _run_cov:
        cov_pdf_path = figures_dir / f'unit_cov_{session_label}.pdf'
        with PdfPages(cov_pdf_path) as pdf:
            for sel_b, cid in enumerate(cids_pool_b):
                unit_robs   = robs_mc_all[:, :, pool_b_inds[sel_b]]
                ind_u       = np.argsort(dur_mc_all)
                unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
                trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

                fig, axs_u = plt.subplots(1, 2, figsize=(12, 4))
                axs_u[0].set_title(f'Neuron {cid} | {display_eye_name} [Pool B]')
                plot_raster(time_full[t_idx_r], trial_idx_r, height=1, ax=axs_u[0])
                axs_u[0].axvline(0, color='r', lw=0.8)
                axs_u[0].set_xlim(time_full[0], time_full[-1])
                axs_u[0].set_xlabel('Time (s)'); axs_u[0].set_ylabel('Trial')
                plot_cov_vs_distance(last_mats[window_idx], sel_b, sel_b, win_label, ax=axs_u[1])
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f"  Saved {cov_pdf_path}")

    # Per-unit 2D PDF — cyclopean Pool B only
    if _result_2d is not None:
        _C2d_u      = _result_2d['C2d']
        _count2d_u  = _result_2d['count2d']
        _dv_means_u = _result_2d['dv_cell_means']
        _C_slope_u  = _result_2d['C_near_slope']
        _C_int_u    = _result_2d['C_near_intercept']
        _n_bc = _C2d_u.shape[0]; _n_bv = _C2d_u.shape[1]
        _colors_bc_u = ['steelblue', 'darkorange', 'teal']
        _labels_bc_u = ['d_c bin 0 [NEAR]', 'd_c bin 1 [mid]', 'd_c bin 2 [FAR]']

        unit_2d_path = figures_dir / f'unit_2d_cov_{session_label}.pdf'
        with PdfPages(unit_2d_path) as pdf:
            for sel_b, cid in enumerate(cids_pool_b):
                unit_robs   = robs_mc_all[:, :, pool_b_inds[sel_b]]
                ind_u       = np.argsort(dur_mc_all)
                unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
                trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

                fig_u, axs_u = plt.subplots(1, 3, figsize=(15, 4))

                axs_u[0].set_title(f'Neuron {cid} | {display_eye_name} [Pool B]')
                plot_raster(time_full[t_idx_r], trial_idx_r, height=1, ax=axs_u[0])
                axs_u[0].axvline(0, color='r', lw=0.8)
                axs_u[0].set_xlim(time_full[0], time_full[-1])
                axs_u[0].set_xlabel('Time (s)'); axs_u[0].set_ylabel('Trial')

                plot_cov_vs_distance(last_mats[window_idx], sel_b, sel_b, win_label, ax=axs_u[1])

                _ax2 = axs_u[2]
                _crate_ii = last_mats[window_idx]['Intercept'][sel_b, sel_b]
                for _bc_k in range(_n_bc):
                    _x2, _y2 = [], []
                    for _bv_k in range(_n_bv):
                        if (_count2d_u[_bc_k, _bv_k] >= 5
                                and np.isfinite(_dv_means_u[_bc_k, _bv_k])
                                and np.isfinite(_C2d_u[_bc_k, _bv_k, sel_b, sel_b])):
                            _x2.append(_dv_means_u[_bc_k, _bv_k])
                            _y2.append(_C2d_u[_bc_k, _bv_k, sel_b, sel_b])
                    if _x2:
                        _lbl2 = _labels_bc_u[_bc_k] if _bc_k < len(_labels_bc_u) else f'dc bin {_bc_k}'
                        _ax2.plot(_x2, _y2, 'o-',
                                  color=_colors_bc_u[_bc_k % len(_colors_bc_u)],
                                  lw=1.5, ms=5, label=_lbl2)
                _ax2.axhline(_crate_ii, color='k', lw=0.8, linestyle='--',
                             label=f'1D Crate ({win_label} ms)')
                _ax2.axhline(0, color='k', lw=0.4, alpha=0.3)
                if (np.isfinite(_C_slope_u[sel_b, sel_b])
                        and np.isfinite(_C_int_u[sel_b, sel_b])):
                    _ax2.annotate(
                        f'slope={_C_slope_u[sel_b,sel_b]:.3f}\n'
                        f'int@dv=0={_C_int_u[sel_b,sel_b]:.3f}',
                        xy=(0.02, 0.97), xycoords='axes fraction',
                        va='top', ha='left', fontsize=7, color='gray')
                _ax2.set_xlabel('Vergence dist d_v (RMS)')
                _ax2.set_ylabel('C2d[bc,bv,i,i]')
                _ax2.set_title(f'2D conditional var | Neuron {cid}')
                _ax2.legend(frameon=False, fontsize=7)

                fig_u.tight_layout()
                pdf.savefig(fig_u, bbox_inches='tight')
                plt.close(fig_u)
        print(f"  Saved {unit_2d_path}")

    # ── 14. Pickle output
    output = {
        'sess':              session_label,
        'cids_all':          cids_all,
        'cids_pool_a':       cids_pool_a,
        'cids_pool_b':       cids_pool_b,
        'pool_a_mask':       pool_a_mask,
        'pool_b_mask':       pool_b_mask,
        # inclusion diagnostics
        'visual_mask':       visual_mask,
        'visual_mask_bino':  visual_mask_bino,
        'spikes_ok':         spikes_ok,
        'reliability_ok':    reliability_ok,
        'nan_ok':            nan_ok,
        'max_snr_bino':      max_snr_bino,
        'max_snr_right':     max_snr_right,
        'mean_reliability':  mean_reliability,
        'nan_frac_per_unit': nan_frac_per_unit,
        'good_trials':       good_trials,
        'good_trials_b_within': good_trials_b_within,
        # covariance results
        'windows':           windows_ms,
        'results':           results,
        'last_mats':         last_mats,
        'result_2d':         _result_2d,
        'result_2d_shuff':   _result_2d_shuff,
        'Cverg2d':           Cverg2d,
        'meta': {
            'eye_source':         eye_source,
            'eye_source_name':    eye_source_name,
            'dataset_path':       str(fix_path),
            'dt':                 dt,
            't_hist_ms':          t_hist_ms,
            'n_bins':             n_bins,
            'valid_time_bins':    n_time_analysis,
            'fixation_radius_deg':   fixation_radius_deg,
            'snr_threshold':         snr_threshold_primary,
            'min_reliability':       min_reliability,
            'max_unit_nan_frac':     max_unit_nan_frac,
            'max_bad_trial_frac':    max_bad_trial_frac,
            'total_spikes_threshold': total_spikes_threshold,
            'min_fix_dur_bins':      min_fix_dur_bins,
            'n_bins_c_2d':  n_bins_c_2d,
            'n_bins_v_2d':  n_bins_v_2d,
            'min_pairs_2d': min_pairs_2d,
            'estimator':    'VisionCore.covariance v9',
        },
    }
    pkl_path = figures_dir / f'mcfarland_fixrsvp_{session_label}_{eye_source}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"  Saved {pkl_path}")

#%% ---------------------------------------------------------------------------
# Summary figure across eye sources
# -----------------------------------------------------------------------------

_fem_summaries = [s for s in frac_fem_summary if s['eye_source'] != 'binocular_diff']

if _fem_summaries:
    n_conds = len(_fem_summaries)
    n_cols  = 5
    fig_sum, axs_sum = plt.subplots(
        n_conds, n_cols,
        figsize=(5 * n_cols, 3.5 * n_conds),
        squeeze=False,
    )

    for row, summary in enumerate(_fem_summaries):
        row_title = f"{summary['subdir_name']} | {summary['display_eye_name']}"
        has_2d    = summary['result_2d'] is not None
        has_cverg = summary['Cverg2d'] is not None

        ax = axs_sum[row, 0]
        if summary['fem_fraction'].size:
            ax.hist(summary['fem_fraction'], bins=np.linspace(0, 1, 31),
                    color='steelblue', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('1 − α  (C_FEM / C_rate)')
        ax.set_ylabel('Count')
        ax.set_title(f"FEM fraction [Pool B={summary['pool_b_n']}]\n{row_title}")

        ax = axs_sum[row, 1]
        if has_2d:
            _c2d = summary['result_2d']
            im = ax.imshow(_c2d['count2d'], origin='lower', cmap='Blues', aspect='auto')
            ax.set_xlabel('d_v bin'); ax.set_ylabel('d_c bin')
            fig_sum.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f'2D pair counts\n{row_title}')
        else:
            ax.text(0.5, 0.5, 'cyclopean only',
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray', fontsize=9)
            ax.set_title(f'2D pair counts\n{row_title}')

        ax = axs_sum[row, 2]
        if has_2d:
            _r2d   = summary['result_2d']
            _r2d_s = summary['result_2d_shuff']
            _C2d_s = _r2d['C2d']
            _dv_s  = _r2d['dv_cell_means']
            _cnt_s = _r2d['count2d']
            _x_r, _y_r = [], []
            for _bv in range(_C2d_s.shape[1]):
                if _cnt_s[0, _bv] >= min_pairs_2d and np.isfinite(_dv_s[0, _bv]):
                    _x_r.append(_dv_s[0, _bv])
                    _y_r.append(np.nanmean(np.diag(_C2d_s[0, _bv])))
            if _x_r:
                ax.plot(_x_r, _y_r, 'o-', color='steelblue', lw=1.5, ms=6, label='real')
            if _r2d_s is not None:
                _C2d_sh = _r2d_s['C2d']; _dv_sh = _r2d_s['dv_cell_means']; _cnt_sh = _r2d_s['count2d']
                _x_sh, _y_sh = [], []
                for _bv in range(_C2d_sh.shape[1]):
                    if _cnt_sh[0, _bv] >= min_pairs_2d and np.isfinite(_dv_sh[0, _bv]):
                        _x_sh.append(_dv_sh[0, _bv])
                        _y_sh.append(np.nanmean(np.diag(_C2d_sh[0, _bv])))
                if _x_sh:
                    ax.plot(_x_sh, _y_sh, 's--', color='gray', lw=1.2, ms=5, alpha=0.7, label='shuffle')
            ax.set_xlabel('d_v (RMS)'); ax.set_ylabel('mean diag(C2d) [bc=0]')
            ax.set_title(f'Near-dc C2d vs vergence dist\n{row_title}')
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(0.5, 0.5, 'cyclopean only',
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray', fontsize=9)
            ax.set_title(f'Near-dc C2d vs dv\n{row_title}')

        ax = axs_sum[row, 3]
        if has_cverg and summary['verg2d_frac_resid'] is not None:
            _vf = summary['verg2d_frac_resid']
            _vf = _vf[np.isfinite(_vf)]
            if _vf.size:
                ax.hist(_vf, bins=np.linspace(0, 1, 31), color='mediumpurple', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Cverg2d / diag(CnoiseC)')
            ax.set_ylabel('Count')
            mn = np.nanmean(_vf) if _vf.size else float('nan')
            ax.set_title(f'Verg frac of residual (mean={mn:.3f})\n{row_title}')
        else:
            ax.text(0.5, 0.5, 'cyclopean only',
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray', fontsize=9)
            ax.set_title(f'Conservative verg fraction\n{row_title}')

        ax = axs_sum[row, 4]
        if has_cverg and summary['ff_before_verg2d'] is not None:
            _fb = summary['ff_before_verg2d']; _fa = summary['ff_after_verg2d']
            _ok = np.isfinite(_fb) & np.isfinite(_fa)
            if _ok.any():
                _lim = max(_fb[_ok].max(), _fa[_ok].max(), 0.1) * 1.05
                ax.scatter(_fb[_ok], _fa[_ok], s=20, alpha=0.7,
                           color='darkorange', edgecolors='none')
                ax.plot([0, _lim], [0, _lim], 'k--', lw=0.8, alpha=0.5)
                ax.set_xlim(0, _lim); ax.set_ylim(0, _lim)
                ax.set_aspect('equal')
            ax.set_xlabel('FF before Cverg2d'); ax.set_ylabel('FF after Cverg2d')
            ax.set_title(f'FF before/after ({summary["win_label_ms"]} ms)\n{row_title}')
        else:
            ws = np.array(summary['sweep_windows_ms'])
            _ff_vals = np.array(summary['sweep_ff_corr'], dtype=float)
            _ff_sems = np.array(summary['sweep_ff_corr_sem'], dtype=float)
            if np.isfinite(_ff_vals).any():
                ax.errorbar(ws, _ff_vals, yerr=_ff_sems,
                            fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3,
                            label='McFarland corrected')
            ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
            ax.set_xscale('log')
            ax.set_xlabel('Window size (ms)'); ax.set_ylabel('Mean FF ± SEM')
            ax.set_title(f'FF vs window size\n{row_title}')
            ax.legend(frameon=False, fontsize=7)

    fig_sum.suptitle(f'Covariance decomposition summary | {session_label} (v9)')
    fig_sum.tight_layout(rect=(0, 0, 1, 0.97))
    summary_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.pdf'
    fig_sum.savefig(summary_path, bbox_inches='tight')
    plt.close(fig_sum)
    print(f"\nSaved {summary_path}")

    # FEM fraction comparison across conditions
    n_conds_fem = len(_fem_summaries)
    _xmax_fem   = 1.2
    _bins_fem   = np.linspace(0, _xmax_fem, 37)

    fig_fem, axs_fem = plt.subplots(
        1, n_conds_fem, figsize=(4 * n_conds_fem, 4),
        sharey=True, sharex=True, squeeze=False,
    )
    for col, summary in enumerate(_fem_summaries):
        ax = axs_fem[0, col]
        _xfm = ax.get_xaxis_transform()
        _title_lines = [summary['subdir_name'], summary['display_eye_name']]
        _xlabel = 'FEM fraction  (1 − α = C_FEM / C_rate)'

        ff = summary['fem_fraction']
        if ff.size:
            ax.hist(ff, bins=_bins_fem, color='steelblue', alpha=0.5,
                    histtype='stepfilled', label='1 − α  (FEM)')
            ax.plot(np.median(ff), 1.03, 'v', transform=_xfm,
                    color='steelblue', markersize=7, clip_on=False)
            _title_lines.append(f"med FEM={np.median(ff):.3f}")

        _adj = summary.get('fem_fraction_adj_verg')
        if _adj is not None:
            _adj_fin = _adj[np.isfinite(_adj)]
            if _adj_fin.size:
                ax.hist(_adj_fin, bins=_bins_fem, color='darkorange',
                        histtype='step', linewidth=1.5,
                        label='(FEM + verg) / C_rate  [cross-space]')
                ax.plot(np.median(_adj_fin), 1.03, 'v', transform=_xfm,
                        color='darkorange', markersize=7, clip_on=False)
            _title_lines.append(f"med +verg={np.median(_adj_fin):.3f}")
            _xlabel = 'Fraction of C_rate  (orange crosses decomposition spaces)'
            ax.legend(frameon=False, fontsize=7)

        ax.axvline(1.0, color='k', lw=0.8, linestyle='--', alpha=0.45)
        ax.set_xlim(0, _xmax_fem)
        ax.set_xlabel(_xlabel)
        ax.set_title('\n'.join(_title_lines), fontsize=8)
        if col == 0:
            ax.set_ylabel('Count')

    fig_fem.suptitle(f'FEM fraction comparison | {session_label} (v9)', fontsize=11)
    fig_fem.tight_layout(rect=(0, 0, 1, 0.93))
    fem_cmp_path = BASE_FIGURES_DIR / f'fem_fraction_comparison_{session_label}.pdf'
    fig_fem.savefig(fem_cmp_path, bbox_inches='tight')
    plt.close(fig_fem)
    print(f"Saved {fem_cmp_path}")

# Text summary
_bin_edges      = np.linspace(0, 1.0, 31)
_bin_edges_wide = np.linspace(0, 1.2, 37)

txt_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.txt'
with open(txt_path, 'w') as f:
    f.write(f"Covariance decomposition summary — v9\n")
    f.write(f"Session: {session_label}\n")
    f.write(f"Estimator: VisionCore.covariance.run_covariance_decomposition\n")
    f.write(f"\nInclusion criteria:\n")
    f.write(f"  fixation_radius_deg={fixation_radius_deg}  snr_threshold={snr_threshold_primary}\n")
    f.write(f"  min_reliability={min_reliability}  max_unit_nan_frac={max_unit_nan_frac}\n")
    f.write(f"  max_bad_trial_frac={max_bad_trial_frac}  total_spikes_threshold={total_spikes_threshold}\n")
    f.write(f"\nPool A: {pool_a_mask.sum()} / {n_all_units}  cids={sorted(cids_pool_a.tolist())}\n")
    f.write(f"Pool B: {pool_b_mask.sum()} / {n_all_units}  cids={sorted(cids_pool_b.tolist())}\n")
    f.write(f"Good trials: {n_good_trials} / {n_all_trials}\n")
    f.write(f"Pool B covariance trials: {n_b_trials_base} / {n_all_trials}\n")
    f.write("\n")

    def _write_metric(f, label, values, edges=None):
        if edges is None:
            edges = _bin_edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        f.write(f"  --- {label} ---\n")
        f.write(f"  n units (finite): {values.size}\n")
        if values.size:
            f.write(f"  mean={values.mean():.4f}  median={np.median(values):.4f}  "
                    f"std={values.std():.4f}\n")
            f.write(f"  [min, max]: [{values.min():.4f}, {values.max():.4f}]\n")
            counts, _ = np.histogram(values, bins=edges)
            for center, count in zip(centers, counts):
                bar = '#' * count
                f.write(f"    {center:.2f}  {count:4d}  {bar}\n")
        f.write("\n")

    def _write_ff(f, label, vals):
        if vals is None or not np.isfinite(vals).any():
            f.write(f"  [{label}]: no data\n\n")
            return
        vals = vals[np.isfinite(vals)]
        f.write(f"  [{label}]\n")
        f.write(f"  n={vals.size}  mean={vals.mean():.4f}  median={np.median(vals):.4f}  "
                f"std={vals.std():.4f}\n")
        f.write(f"  [min, max]: [{vals.min():.4f}, {vals.max():.4f}]\n\n")

    for summary in frac_fem_summary:
        name = f"{summary['subdir_name']} ({summary['display_eye_name']})"
        f.write(f"{'='*60}\n")
        f.write(f"Eye source: {name}\n")
        if summary['eye_source'] == 'binocular_diff':
            f.write("  NOTE: excluded from FEM summary figure.\n")
        f.write(f"  Pool A={summary['pool_a_n']}  Pool B={summary['pool_b_n']}"
                f"  covariance trials={summary['n_b_trials']}\n\n")

        _write_metric(f, "FEM fraction (1 - alpha)  [McFarland pairwise, Pool B]",
                      summary['fem_fraction'])

        _adj_raw = summary.get('fem_fraction_adj_verg')
        if _adj_raw is not None:
            _adj_fin = _adj_raw[np.isfinite(_adj_raw)]
            if _adj_fin.size:
                _write_metric(
                    f,
                    "(C_FEM + Cverg2d) / C_rate  [cross-space; values >1 mean vergence noise > C_rate]",
                    _adj_fin,
                    edges=_bin_edges_wide,
                )
                f.write(f"  frac > 1.0: {(_adj_fin > 1.0).mean():.3f}\n\n")

        if summary['result_2d'] is not None:
            _r2d = summary['result_2d']
            f.write("  --- 2D vergence analysis [cyclopean only] ---\n")
            f.write(f"  Pair counts (d_c × d_v):\n{_r2d['count2d']}\n")
            _slope_diag = np.diag(_r2d['C_near_slope'])
            _finite_s = _slope_diag[np.isfinite(_slope_diag)]
            if _finite_s.size:
                f.write(f"  Near-dc slope (diag): mean={_finite_s.mean():.4f}, "
                        f"frac<0={(_finite_s<0).mean():.2f}\n")
            if summary['result_2d_shuff'] is not None:
                _slope_sh = np.diag(summary['result_2d_shuff']['C_near_slope'])
                _fs = _slope_sh[np.isfinite(_slope_sh)]
                if _fs.size:
                    f.write(f"  Shuffle slope (diag):  mean={_fs.mean():.4f}\n")
            f.write("\n")

        if summary['verg2d_frac_resid'] is not None:
            _vf = summary['verg2d_frac_resid']
            _write_metric(f, "Conservative verg frac of McFarland residual  [Cverg2d / diag(CnoiseC)]",
                          _vf[np.isfinite(_vf)])

        _write_ff(f, "FF before conservative Cverg2d (McFarland residual)", summary['ff_before_verg2d'])
        _write_ff(f, "FF after  conservative Cverg2d", summary['ff_after_verg2d'])
        f.write("\n")

print(f"Saved {txt_path}")
print("\nAll conditions complete.")

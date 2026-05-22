#%%
"""
test_rowleyv8.py — conservative vergence extension

Changes from v7:

1. 2D vergence analysis restricted to the cyclopean eye source.
   For other conditions the 2D columns in the summary figure are blank.
   This avoids the ambiguity of "cyclopean-matched" when the control
   trajectory is monocular or a difference signal.

2. Conservative Cverg2d replaces OLS C_vergence everywhere a subtractable
   component is claimed.  Cverg2d is derived from the near-cyclopean slice
   of the 2D conditional estimator, projected to PSD, and capped by the
   McFarland residual.  OLS estimates are retained in the pickle but removed
   from the main summary figure.

3. Within-time-bin vergence shuffle provides a null for the 2D near-cyclopean
   slope, overlaid on the C2d vs dv diagnostic panel.

4. _result_2d and Cverg2d are stored in the pickle output.

5. Summary figure (5 columns):
     FEM fraction | pair-count heatmap | near-cyclopean C2d vs dv |
     conservative verg frac of residual | FF before/after Cverg2d
   Columns 2-5 are blank for non-cyclopean rows.

Removed from main figures (demoted to pickle only):
  - OLS C_eye / C_rate fraction
  - OLS vergence fraction
  - binocular_diff in FEM summary figure
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
from matplotlib.backends.backend_pdf import PdfPages

import torch

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
from VisionCore.covariance import (run_covariance_decomposition,
                                    extract_valid_segments, extract_windows,
                                    estimate_vergence_conditional_on_cyclopean,
                                    conservative_cvergence_from_2d)
from VisionCore.subspace import project_to_psd

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

min_fix_dur_bins   = 20
apply_radius_filter = False
radius_deg          = 7.0

# 2D analysis
n_bins_c_2d  = 5    # near bin = lowest 1/n_bins_c of cyclopean pair distances
n_bins_v_2d  = 5
min_pairs_2d = 30   # min pairs per 2D cell for conservative estimator
n_shuff_2d   = 1    # number of within-time-bin vergence shuffles

BASE_FIGURES_DIR = Path('../figures/mcfarland') / f'{subject}_{date}_v8'

#%% ---------------------------------------------------------------------------
# Helpers (unchanged from v7)
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
        robs_trial[itrial, tt]   = robs[idx]
        eyepos_trial[itrial, tt] = eyepos[idx]
        dfs_trial[itrial, tt]    = dfs[idx]
        dur_trial[itrial]        = len(idx)
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

    if apply_radius_filter:
        radius_mask = np.hypot(eyepos_serial[:, 0], eyepos_serial[:, 1]) < radius_deg
        dfs_serial  = dfs_serial & radius_mask

    # ── 2. Trial-align
    robs_trial, eyepos_trial, dfs_trial, dur_trial, _ = serial_to_trial_aligned(
        robs_serial, eyepos_serial, dfs_serial, trial_inds_s, time_inds_s)

    # ── 3. Trial filter
    good_trials = dur_trial > min_fix_dur_bins
    robs_mc     = robs_trial[good_trials]
    eyepos_mc   = eyepos_trial[good_trials]
    dfs_mc      = dfs_trial[good_trials]
    dur_mc      = dur_trial[good_trials]
    print(f"  {good_trials.sum()} / {len(good_trials)} trials kept")

    # ── 4. Neuron gate
    spike_ok    = np.nansum(robs_mc, axis=(0, 1)) > total_spikes_threshold
    neuron_mask = np.where(spike_ok)[0]
    cids_used   = cids_all[neuron_mask]
    print(f"  {len(neuron_mask)} / {n_all_units} neurons pass spike threshold")
    if len(neuron_mask) == 0:
        print("  Skipping.")
        continue

    # ── 5. Analysis window
    n_time_analysis = min(valid_time_bins, robs_mc.shape[1])
    iix = np.arange(n_time_analysis)
    robs_used   = robs_mc[:, iix][:, :, neuron_mask]
    eyepos_used = eyepos_mc[:, iix]
    valid_used  = (dfs_mc[:, iix]
                   & np.isfinite(robs_mc[:, iix][:, :, neuron_mask].sum(axis=2))
                   & np.isfinite(eyepos_mc[:, iix].sum(axis=2)))

    # Trial-align vergence
    if eyepos_verg_serial is not None:
        _, eyepos_verg_trial, _, _, _ = serial_to_trial_aligned(
            robs_serial, eyepos_verg_serial, dfs_serial, trial_inds_s, time_inds_s)
        eyepos_verg_used = eyepos_verg_trial[good_trials][:, iix]
    else:
        eyepos_verg_used = None

    time_full = time_bins_full[:robs_mc.shape[1]]

    # ── 6. Covariance decomposition sweep
    print(f"  Running run_covariance_decomposition "
          f"({robs_used.shape[0]} trials, {robs_used.shape[2]} units)...")
    results, last_mats = run_covariance_decomposition(
        robs_used, eyepos_used, valid_used,
        window_sizes_ms=windows_ms, t_hist_ms=t_hist_ms,
        n_bins=n_bins, dt=dt,
        eyepos_vergence=eyepos_verg_used,
    )

    # ── Select 20 ms window index
    window_idx = windows_ms.index(20) if 20 in windows_ms else 0
    win_label  = windows_ms[window_idx]

    Ctotal    = project_to_psd(last_mats[window_idx]['Total'])
    Cpsth     = project_to_psd(last_mats[window_idx]['PSTH'])
    Crate     = project_to_psd(last_mats[window_idx]['Intercept'])
    Cfem      = project_to_psd(last_mats[window_idx]['FEM'])
    CnoiseC   = project_to_psd(Ctotal - Crate)   # McFarland residual
    MeanRates = results[window_idx]['Erates']

    denom = np.diag(Crate)
    fem_fraction = np.where(denom > 0, 1.0 - np.diag(Cpsth) / denom, np.nan)
    fem_fraction = fem_fraction[np.isfinite(fem_fraction)]

    sweep_windows_ms  = [r['window_ms']       for r in results]
    sweep_ff_corr     = [r['ff_corr_mean']     for r in results]
    sweep_ff_corr_sem = [r['ff_corr_sem']      for r in results]

    # ── 7. 2D vergence analysis — ONLY for cyclopean eye source
    # For other sources the "cyclopean-matched" control is ambiguous: monocular
    # position or a difference signal do not give the same stratification.
    _run_2d     = (eye_source_name == 'eyepos') and (eyepos_verg_used is not None)
    _result_2d       = None
    _result_2d_shuff = None
    Cverg2d          = None

    if _run_2d:
        _device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _win_bins_2d = max(1, int(20 / (dt * 1000)))
        _t_hist_bins = int(t_hist_ms / (dt * 1000))
        _t_hist_used = max(_t_hist_bins, _win_bins_2d)

        _segs = extract_valid_segments(valid_used, min_len_bins=36)
        _robs = torch.tensor(np.nan_to_num(robs_used,        nan=0.0), dtype=torch.float32, device=_device)
        _eyec = torch.tensor(np.nan_to_num(eyepos_used,      nan=0.0), dtype=torch.float32, device=_device)
        _eyev = torch.tensor(np.nan_to_num(eyepos_verg_used, nan=0.0), dtype=torch.float32, device=_device)

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

            # Within-time-bin vergence shuffle (null distribution)
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

            # Slope comparison: real vs shuffle
            _slope_real  = np.nanmean(np.diag(_result_2d['C_near_slope']))
            _slope_shuff = np.nanmean(np.diag(_result_2d_shuff['C_near_slope']))
            print(f"  Near-dc slope (diag mean): real={_slope_real:.4f}  shuffle={_slope_shuff:.4f}")

            # Conservative vergence covariance
            Cverg2d = conservative_cvergence_from_2d(_result_2d, CnoiseC, min_pairs=min_pairs_2d)
            if Cverg2d is not None:
                print(f"  Cverg2d diagonal: mean={np.nanmean(np.diag(Cverg2d)):.4f}, "
                      f"max={np.nanmax(np.diag(Cverg2d)):.4f}")
            else:
                print("  conservative_cvergence_from_2d: insufficient data.")
        else:
            print("  2D analysis skipped: insufficient windows.")

    # ── 8. Compute FF before/after conservative vergence subtraction
    ff_before_verg2d      = None
    ff_after_verg2d       = None
    verg2d_frac_resid     = None
    fem_fraction_adj_verg = None

    if Cverg2d is not None:
        CnoiseCV         = project_to_psd(CnoiseC - Cverg2d)
        ff_before_verg2d = np.where(MeanRates > 0, np.diag(CnoiseC)  / MeanRates, np.nan)
        ff_after_verg2d  = np.where(MeanRates > 0, np.diag(CnoiseCV) / MeanRates, np.nan)
        resid_diag       = np.diag(CnoiseC)
        verg2d_frac_resid = np.where(resid_diag > 0, np.diag(Cverg2d) / resid_diag, np.nan)
        # Cross-space metric: (C_FEM + Cverg2d) / C_rate.
        #
        # WHY THIS CROSSES DECOMPOSITION SPACES:
        #   McFarland pairwise partitions variance as:
        #       C_total = C_rate + C_noise
        #   where C_rate = C_psth + C_FEM is rate-locked variance (shared
        #   across trials with the same stimulus trajectory) and C_noise is
        #   the residual.  The standard FEM fraction (1-alpha = C_FEM/C_rate)
        #   lives entirely within the rate component.
        #
        #   Cverg2d is estimated from the near-cyclopean slice of the 2D
        #   conditional second-moment estimator, which operates on raw spike
        #   cross-products.  The slope of C2d vs vergence distance captures
        #   trial-to-trial covariance that varies with vergence mismatch — but
        #   because McFarland groups trials by *cyclopean* distance, vergence-
        #   driven rate variation ends up in C_noise, not C_rate.  Cverg2d is
        #   therefore a noise-space quantity (capped by diag(C_noise)).
        #
        #   Adding Cverg2d to C_FEM and dividing by C_rate mixes the two
        #   spaces.  When FF is large (C_noise >> C_rate, here ~11x on
        #   average), this ratio can substantially exceed 1 for individual
        #   neurons.  That is not a numerical error — it reflects that the
        #   vergence-driven *noise* is genuinely larger than the total rate
        #   variance for those cells.  The metric should be read as:
        #     "if vergence-driven noise were re-attributed to the rate
        #      component, what fraction of C_rate would eye movements explain?"
        #   Values > 1 mean vergence noise alone exceeds C_rate.
        #
        #   Use alongside Cverg2d/diag(C_noise) (noise-space fraction) to
        #   keep the two components in their native spaces.
        _crate_d = np.diag(Crate)
        with np.errstate(invalid='ignore'):
            fem_fraction_adj_verg = np.where(
                _crate_d > 0,
                (np.diag(Crate - Cpsth) + np.diag(Cverg2d)) / _crate_d,
                np.nan,
            )

    # ── 9. Collect summary entry
    frac_fem_summary.append({
        'subdir_name':     subdir_name,
        'eye_source':      eye_source,
        'eye_source_name': eye_source_name,
        'display_eye_name': display_eye_name,
        'win_label_ms':    win_label,
        # McFarland primary
        'fem_fraction':    fem_fraction.copy(),
        # FF vs window size (McFarland corrected)
        'sweep_windows_ms':   sweep_windows_ms,
        'sweep_ff_corr':      sweep_ff_corr,
        'sweep_ff_corr_sem':  sweep_ff_corr_sem,
        # 2D vergence (cyclopean only, else None)
        'result_2d':          _result_2d,
        'result_2d_shuff':    _result_2d_shuff,
        'Cverg2d':            Cverg2d,
        'ff_before_verg2d':      ff_before_verg2d,
        'ff_after_verg2d':       ff_after_verg2d,
        'verg2d_frac_resid':     verg2d_frac_resid,
        'fem_fraction_adj_verg': fem_fraction_adj_verg,
    })

    # ── 10. Per-condition figure (4 panels)
    fig_c, axs_c = plt.subplots(1, 4, figsize=(20, 4))

    # Panel 1: FEM fraction histogram [primary]
    ax = axs_c[0]
    if fem_fraction.size:
        ax.hist(fem_fraction, bins=np.linspace(0, 1, 31), color='steelblue', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('1 − α  (C_FEM / C_rate)')
    ax.set_ylabel('Count')
    ax.set_title(f'FEM fraction [PRIMARY]\n{display_eye_name}')

    # Panel 2: 2D near-cyclopean C2d vs dv (if available) else FF vs window size
    ax = axs_c[1]
    if _result_2d is not None:
        _C2d_c         = _result_2d['C2d']
        _dv_means_c    = _result_2d['dv_cell_means']
        _count2d_c     = _result_2d['count2d']
        _colors_bv     = plt.cm.viridis(np.linspace(0.1, 0.9, n_bins_v_2d))
        _x_real, _y_real = [], []
        for _bv in range(n_bins_v_2d):
            if _count2d_c[0, _bv] >= min_pairs_2d and np.isfinite(_dv_means_c[0, _bv]):
                _x_real.append(_dv_means_c[0, _bv])
                _y_real.append(np.nanmean(np.diag(_C2d_c[0, _bv])))
        if _x_real:
            ax.plot(_x_real, _y_real, 'o-', color='steelblue', lw=1.5, ms=6, label='near-dc (real)')
        # Shuffle overlay
        if _result_2d_shuff is not None:
            _C2d_sh     = _result_2d_shuff['C2d']
            _dv_sh      = _result_2d_shuff['dv_cell_means']
            _cnt_sh     = _result_2d_shuff['count2d']
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
        ax.errorbar(sweep_windows_ms, sweep_ff_corr, yerr=sweep_ff_corr_sem,
                    fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3)
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('FF (McFarland corrected)')
        ax.set_title(f'FF vs window size\n{display_eye_name}')

    # Panel 3: conservative vergence fraction of McFarland residual (if available)
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

    # Panel 4: FF scatter before/after conservative vergence (if available)
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

    fig_c.suptitle(f'{session_label} | {win_label} ms')
    fig_c.tight_layout()
    fig_c.savefig(figures_dir / 'frac_fem_hist.pdf')
    plt.close(fig_c)

    # ── 11. Eye position heatmap
    ind_sorted = np.argsort(dur_mc)
    fig, ax = plt.subplots()
    ax.imshow(eyepos_mc[ind_sorted, :, 0], vmin=-0.5, vmax=0.5,
              aspect='auto', cmap='coolwarm', interpolation='none', origin='lower',
              extent=[time_full[0], time_full[-1], 0, robs_mc.shape[0]])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Trial'); ax.set_title(f'Eye X | {display_eye_name}')
    fig.savefig(figures_dir / 'eyepos_heatmap.pdf')
    plt.close(fig)

    # ── 12. Per-unit PDFs (rasters + 1D cov-vs-distance + 2D conditional)
    unit_pdf_path = figures_dir / f'unit_rasters_{session_label}.pdf'
    with PdfPages(unit_pdf_path) as pdf:
        for sel_i, cid in enumerate(cids_used):
            unit_robs   = robs_mc[:, :, neuron_mask[sel_i]]
            ind_u       = np.argsort(dur_mc)
            unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
            trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

            fig, axs_u = plt.subplots(1, 2, figsize=(12, 4))
            axs_u[0].set_title(f'Neuron {cid} | {display_eye_name}')
            plot_raster(time_full[t_idx_r], trial_idx_r, height=1, ax=axs_u[0])
            axs_u[0].axvline(0, color='r', lw=0.8)
            axs_u[0].set_xlim(time_full[0], time_full[-1])
            axs_u[0].set_xlabel('Time (s)'); axs_u[0].set_ylabel('Trial')
            plot_cov_vs_distance(last_mats[window_idx], sel_i, sel_i, win_label, ax=axs_u[1])
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"  Saved {unit_pdf_path}")

    # Per-unit 2D PDF (only for cyclopean where 2D ran)
    if _result_2d is not None:
        _C2d_u        = _result_2d['C2d']
        _count2d_u    = _result_2d['count2d']
        _dv_means_u   = _result_2d['dv_cell_means']
        _C_slope_u    = _result_2d['C_near_slope']
        _C_int_u      = _result_2d['C_near_intercept']
        _n_bc = _C2d_u.shape[0]; _n_bv = _C2d_u.shape[1]
        _colors_bc_u  = ['steelblue', 'darkorange', 'teal']
        _labels_bc_u  = ['d_c bin 0 [NEAR]', 'd_c bin 1 [mid]', 'd_c bin 2 [FAR]']

        unit_2d_path = figures_dir / f'unit_2d_cov_{session_label}.pdf'
        with PdfPages(unit_2d_path) as pdf:
            for sel_i, cid in enumerate(cids_used):
                unit_robs   = robs_mc[:, :, neuron_mask[sel_i]]
                ind_u       = np.argsort(dur_mc)
                unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
                trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

                fig_u, axs_u = plt.subplots(1, 3, figsize=(15, 4))

                # Raster
                axs_u[0].set_title(f'Neuron {cid} | {display_eye_name}')
                plot_raster(time_full[t_idx_r], trial_idx_r, height=1, ax=axs_u[0])
                axs_u[0].axvline(0, color='r', lw=0.8)
                axs_u[0].set_xlim(time_full[0], time_full[-1])
                axs_u[0].set_xlabel('Time (s)'); axs_u[0].set_ylabel('Trial')

                # 1D McFarland cov-vs-distance
                plot_cov_vs_distance(last_mats[window_idx], sel_i, sel_i, win_label, ax=axs_u[1])

                # 2D per-unit: C2d[bc, bv, i, i] vs dv for each dc bin
                _ax2 = axs_u[2]
                _crate_ii = last_mats[window_idx]['Intercept'][sel_i, sel_i]
                for _bc_k in range(_n_bc):
                    _x2, _y2 = [], []
                    for _bv_k in range(_n_bv):
                        if (_count2d_u[_bc_k, _bv_k] >= 5
                                and np.isfinite(_dv_means_u[_bc_k, _bv_k])
                                and np.isfinite(_C2d_u[_bc_k, _bv_k, sel_i, sel_i])):
                            _x2.append(_dv_means_u[_bc_k, _bv_k])
                            _y2.append(_C2d_u[_bc_k, _bv_k, sel_i, sel_i])
                    if _x2:
                        _lbl2 = _labels_bc_u[_bc_k] if _bc_k < len(_labels_bc_u) else f'dc bin {_bc_k}'
                        _ax2.plot(_x2, _y2, 'o-',
                                  color=_colors_bc_u[_bc_k % len(_colors_bc_u)],
                                  lw=1.5, ms=5, label=_lbl2)
                _ax2.axhline(_crate_ii, color='k', lw=0.8, linestyle='--',
                             label=f'1D Crate ({win_label} ms)')
                _ax2.axhline(0, color='k', lw=0.4, alpha=0.3)
                if (np.isfinite(_C_slope_u[sel_i, sel_i])
                        and np.isfinite(_C_int_u[sel_i, sel_i])):
                    _ax2.annotate(
                        f'slope={_C_slope_u[sel_i,sel_i]:.3f}\n'
                        f'int@dv=0={_C_int_u[sel_i,sel_i]:.3f}',
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

    # ── 13. Pickle output
    output = {
        'sess':         session_label,
        'cids':         cids_all,
        'neuron_mask':  neuron_mask,
        'windows':      windows_ms,
        'cids_used':    cids_used,
        'results':      results,
        'last_mats':    last_mats,
        'result_2d':    _result_2d,
        'result_2d_shuff': _result_2d_shuff,
        'Cverg2d':      Cverg2d,
        'meta': {
            'eye_source':      eye_source,
            'eye_source_name': eye_source_name,
            'dataset_path':    str(fix_path),
            'dt':              dt,
            't_hist_ms':       t_hist_ms,
            'n_bins':          n_bins,
            'valid_time_bins': n_time_analysis,
            'total_spikes_threshold': total_spikes_threshold,
            'min_fix_dur_bins': min_fix_dur_bins,
            'n_bins_c_2d': n_bins_c_2d,
            'n_bins_v_2d': n_bins_v_2d,
            'min_pairs_2d': min_pairs_2d,
            'estimator': 'VisionCore.covariance v8',
        },
    }
    pkl_path = figures_dir / f'mcfarland_fixrsvp_{session_label}_{eye_source}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"  Saved {pkl_path}")

# ── Summary figure across eye sources
# binocular_diff excluded: running McFarland pairwise on a vergence signal
# is not the same conceptual quantity as monocular 1-alpha.
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
        row_title  = f"{summary['subdir_name']} | {summary['display_eye_name']}"
        has_2d     = summary['result_2d'] is not None
        has_cverg  = summary['Cverg2d'] is not None

        # Col 1: FEM fraction histogram [always]
        ax = axs_sum[row, 0]
        if summary['fem_fraction'].size:
            ax.hist(summary['fem_fraction'], bins=np.linspace(0, 1, 31),
                    color='steelblue', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('1 − α  (C_FEM / C_rate)')
        ax.set_ylabel('Count')
        ax.set_title(f'FEM fraction [PRIMARY]\n{row_title}')

        # Col 2: 2D pair-count heatmap (cyclopean only)
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

        # Col 3: near-cyclopean C2d vs dv with shuffle overlay (cyclopean only)
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

        # Col 4: conservative vergence fraction of McFarland residual (cyclopean only)
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

        # Col 5: FF before/after (Cverg2d if cyclopean, else McFarland corrected vs window)
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
            ax.errorbar(ws, summary['sweep_ff_corr'], yerr=summary['sweep_ff_corr_sem'],
                        fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3,
                        label='McFarland corrected')
            ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
            ax.set_xscale('log')
            ax.set_xlabel('Window size (ms)'); ax.set_ylabel('Mean FF ± SEM')
            ax.set_title(f'FF vs window size\n{row_title}')
            ax.legend(frameon=False, fontsize=7)

    fig_sum.suptitle(f'Covariance decomposition summary | {session_label} (v8)')
    fig_sum.tight_layout(rect=(0, 0, 1, 0.97))
    summary_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.pdf'
    fig_sum.savefig(summary_path, bbox_inches='tight')
    plt.close(fig_sum)
    print(f"\nSaved {summary_path}")

    # ── FEM fraction comparison: all conditions on a common scale
    # Cyclopean row also shows the vergence-adjusted metric overlaid in orange.
    n_conds_fem = len(_fem_summaries)
    _xmax_fem   = 1.2   # extend past 1 to reveal adjusted values that overshoot
    _bins_fem   = np.linspace(0, _xmax_fem, 37)

    fig_fem, axs_fem = plt.subplots(
        1, n_conds_fem, figsize=(4 * n_conds_fem, 4),
        sharey=True, sharex=True, squeeze=False,
    )
    for col, summary in enumerate(_fem_summaries):
        ax = axs_fem[0, col]
        _xfm = ax.get_xaxis_transform()  # x=data coords, y=axes fraction

        ff = summary['fem_fraction']
        if ff.size:
            ax.hist(ff, bins=_bins_fem, color='steelblue', alpha=0.5,
                    histtype='stepfilled', label='1 − α  (FEM)')
            ax.plot(np.median(ff), 1.03, 'v', transform=_xfm,
                    color='steelblue', markersize=7, clip_on=False)

        _adj = summary.get('fem_fraction_adj_verg')
        if _adj is not None:
            _adj_fin = _adj[np.isfinite(_adj)]
            if _adj_fin.size:
                ax.hist(_adj_fin, bins=_bins_fem, color='darkorange',
                        histtype='step', linewidth=1.5, label='(FEM + verg) / C_rate  [cross-space]')
                ax.plot(np.median(_adj_fin), 1.03, 'v', transform=_xfm,
                        color='darkorange', markersize=7, clip_on=False)
            ax.legend(frameon=False, fontsize=7)

        ax.axvline(1.0, color='k', lw=0.8, linestyle='--', alpha=0.45)
        ax.set_xlim(0, _xmax_fem)
        ax.set_xlabel('Fraction of C_rate  (orange crosses decomposition spaces)')
        ax.set_title(f"{summary['subdir_name']}\n{summary['display_eye_name']}", fontsize=8)
        if col == 0:
            ax.set_ylabel('Count')

    fig_fem.suptitle(f'FEM fraction comparison | {session_label} (v8)', fontsize=11)
    fig_fem.tight_layout(rect=(0, 0, 1, 0.93))
    fem_cmp_path = BASE_FIGURES_DIR / f'fem_fraction_comparison_{session_label}.pdf'
    fig_fem.savefig(fem_cmp_path, bbox_inches='tight')
    plt.close(fig_fem)
    print(f"Saved {fem_cmp_path}")

# ── Text summary
_bin_edges      = np.linspace(0, 1.0, 31)
_bin_edges_wide = np.linspace(0, 1.2, 37)   # extended for (FEM+verg)/Crate which can exceed 1

txt_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.txt'
with open(txt_path, 'w') as f:
    f.write(f"Covariance decomposition summary — v8\n")
    f.write(f"Session: {session_label}\n")
    f.write(f"Estimator: VisionCore.covariance.run_covariance_decomposition\n")
    f.write(f"Window: {windows_ms[window_idx]} ms  |  n_bins={n_bins}  |  t_hist={t_hist_ms} ms\n")
    f.write(f"2D analysis: n_bins_c={n_bins_c_2d}, n_bins_v={n_bins_v_2d}, min_pairs={min_pairs_2d}\n")
    f.write("\n")
    f.write("PRIMARY:  McFarland pairwise (dual-window) for C_FEM and 1-alpha.\n")
    f.write("EXTENSION: 2D conditional estimator (cyclopean eye source only).\n")
    f.write("           C_vergence|cyclopean estimated from near-cyclopean slope,\n")
    f.write("           projected to PSD, capped by McFarland residual.\n")
    f.write("OLS estimates (C_eye, C_vergence) retained in pickle, not in main figure.\n")
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
        f.write(f"  [min, max]: [{vals.min():.4f}, {vals.max():.4f}]\n")
        f.write("\n")

    for summary in frac_fem_summary:
        name = f"{summary['subdir_name']} ({summary['display_eye_name']})"
        f.write(f"{'='*60}\n")
        f.write(f"Eye source: {name}\n")
        if summary['eye_source'] == 'binocular_diff':
            f.write("  NOTE: excluded from FEM summary figure.\n")
        f.write("\n")

        _write_metric(f, "FEM fraction (1 - alpha)  [PRIMARY: McFarland pairwise]",
                      summary['fem_fraction'])

        _adj_raw = summary.get('fem_fraction_adj_verg')
        if _adj_raw is not None:
            _adj_fin = _adj_raw[np.isfinite(_adj_raw)]
            if _adj_fin.size:
                _write_metric(
                    f,
                    "(C_FEM + Cverg2d) / C_rate  [cross-space: C_FEM is rate-component, "
                    "Cverg2d is noise-component; values >1 mean vergence noise > C_rate]",
                    _adj_fin,
                    edges=_bin_edges_wide,
                )
                f.write(f"  frac > 1.0: {(_adj_fin > 1.0).mean():.3f}\n")
                f.write(f"  NOTE: C_noise/C_rate ratio is ~11x here, so Cverg2d (capped\n")
                f.write(f"        at C_noise) can exceed C_rate for individual neurons.\n")
                f.write(f"        Read alongside Cverg2d/diag(C_noise) (noise-space frac)\n")
                f.write(f"        which stays in [0,1] by construction.\n\n")

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
            _write_metric(f, "Conservative verg frac of McFarland residual  "
                          "[Cverg2d / diag(CnoiseC)]",
                          _vf[np.isfinite(_vf)])

        _write_ff(f, "FF before conservative Cverg2d (McFarland residual)",
                  summary['ff_before_verg2d'])
        _write_ff(f, "FF after  conservative Cverg2d",
                  summary['ff_after_verg2d'])
        f.write("\n")

print(f"Saved {txt_path}")
print("\nAll conditions complete.")

#%%
"""
test_rowleyv7.py

Identical to test_rowleyv6 but uses VisionCore.covariance.run_covariance_decomposition
instead of mcfarland_sim.DualWindowAnalysis.run_sweep.

Two intentional numerical differences vs v6:
  1. Erate uses pair-count-weighted mean (fixes second-moment/mean mismatch bias).
  2. Cpsth uses pair-count time-bin weights (consistent with Crate estimator).
Both changes reduce a small systematic negative bias in the shuffle null.

inspect_neuron_pair is not available in the functional API; the unit PDF shows
rasters only.
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

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
import torch
from VisionCore.covariance import (run_covariance_decomposition,
                                    extract_valid_segments, extract_windows,
                                    estimate_vergence_conditional_on_cyclopean)
from VisionCore.subspace import project_to_psd

#%% ---------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

subject = 'Luke'
date = '2026-03-16'
primary_eye = 'binocular'

legacy_processed_root = Path('/mnt/ssd2/RowleyMarmoV1V2/processed')
dataset_dir = Path('datasets_binocular')
fix_name = 'fixrsvp.dset'

# analysis params
windows_ms = [5, 10, 20, 40, 80]
total_spikes_threshold = 200
valid_time_bins = 240
dt = 1 / 240.0
t_hist_ms = 50
n_bins = 15

min_fix_dur_bins = 20
apply_radius_filter = False
radius_deg = 7.0

BASE_FIGURES_DIR = Path('../figures/mcfarland') / f'{subject}_{date}_v7'

#%% ---------------------------------------------------------------------------
# Helpers (unchanged from v6)
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
        assert len(x) == n_expected, f"Length mismatch: got {len(x)}, expected {n_expected}"
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
                eye = _get_required(dset, k)
                valid = _get_optional(dset, 'dpi_valid_left', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("Requested left-eye trace, but no eyepos_left / eyepos_dpi_left found.")

    if eye_source == 'right':
        for k in ['eyepos_right', 'eyepos_dpi_right']:
            if k in keys:
                eye = _get_required(dset, k)
                valid = _get_optional(dset, 'dpi_valid_right', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("Requested right-eye trace, but no eyepos_right / eyepos_dpi_right found.")

    if eye_source == 'binocular_diff':
        left = right = None
        for k in ['eyepos_left', 'eyepos_dpi_left']:
            if k in keys:
                left = _get_required(dset, k); break
        for k in ['eyepos_right', 'eyepos_dpi_right']:
            if k in keys:
                right = _get_required(dset, k); break
        if left is None or right is None:
            raise KeyError("Requested binocular_diff, but left/right eye traces not found.")
        valid_l = _as_bool_1d(_get_optional(dset, 'dpi_valid_left',  np.ones(len(left),  dtype=bool)), len(left))
        valid_r = _as_bool_1d(_get_optional(dset, 'dpi_valid_right', np.ones(len(right), dtype=bool)), len(right))
        return left - right, (valid_l & valid_r), 'eyepos_left_minus_right'

    if eye_source == 'pupil_left':
        for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil']:
            if k in keys:
                eye = _get_required(dset, k)
                valid = _get_optional(dset, 'pupil_valid_left', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("Requested pupil_left, but no left pupil trace found.")

    if eye_source == 'pupil_right':
        for k in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil']:
            if k in keys:
                eye = _get_required(dset, k)
                valid = _get_optional(dset, 'pupil_valid_right', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), k
        raise KeyError("Requested pupil_right, but no right pupil trace found.")

    raise ValueError(f"Unknown eye_source: {eye_source}")


def _nearest_resample_bool(sample_times, values, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    values = _as_bool_1d(values, len(sample_times))
    target_times = np.asarray(target_times, dtype=np.float64)
    right_idx = np.searchsorted(sample_times, target_times, side='left')
    right_idx = np.clip(right_idx, 0, len(sample_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(sample_times) - 1)
    choose_left = np.abs(target_times - sample_times[left_idx]) <= np.abs(sample_times[right_idx] - target_times)
    nearest_idx = np.where(choose_left, left_idx, right_idx)
    return values[nearest_idx]


def _interp_xy(sample_times, xy, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float64)
    return np.column_stack([
        np.interp(target_times, sample_times, xy[:, 0]),
        np.interp(target_times, sample_times, xy[:, 1]),
    ]).astype(np.float32)


def add_calibrated_pupil_traces(dset, aux_processed_path):
    keys = set(dset.keys())
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
            print(f"Skipping {eye} pupil import; calibrated CSV is missing pupil columns: {exc}")
            continue

        sample_times = pupil_df['t_ephys'].to_numpy(dtype=np.float64)
        pupil_xy = pupil_df[['pupil_i', 'pupil_j']].to_numpy(dtype=np.float32)
        pupil_valid = pupil_df['pupil_valid'].to_numpy()

        valid_samples = (
            np.isfinite(sample_times)
            & _as_bool_1d(pupil_valid, len(sample_times))
            & np.all(np.isfinite(pupil_xy), axis=1)
        )
        if valid_samples.sum() < 2:
            print(f"Skipping {eye} pupil import; fewer than 2 valid calibrated samples in {csv_path}")
            continue

        dset[pupil_key] = _interp_xy(sample_times[valid_samples], pupil_xy[valid_samples], t_bins)
        dset[valid_key] = _nearest_resample_bool(sample_times, pupil_valid, t_bins)
        keys.update([pupil_key, valid_key])
        print(f"Loaded calibrated {pupil_key} from {csv_path}")


def get_hist_bins(values, n_bins=30):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        pad = 0.05 if np.isclose(lo, 0.0) else max(0.05, abs(lo) * 0.05)
        return np.linspace(lo - pad, hi + pad, n_bins).tolist()
    return np.linspace(lo, hi, n_bins).tolist()


def get_enabled_eye_configs(dset, primary_eye_name):
    keys = set(dset.keys())
    enabled_configs = [('primary-dpi', 'default')]

    has_left       = any(k in keys for k in ['eyepos_left', 'eyepos_dpi_left'])
    has_right      = any(k in keys for k in ['eyepos_right', 'eyepos_dpi_right'])
    has_left_pupil = any(k in keys for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil'])
    has_right_pupil= any(k in keys for k in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil'])

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
    return [
        (primary_subdir if subdir == 'primary-dpi' else subdir, source)
        for subdir, source in enabled_configs
    ]


def serial_to_trial_aligned(robs, eyepos, dfs, trial_inds, time_inds):
    unique_trials = np.unique(trial_inds)
    n_trials = len(unique_trials)
    n_time = np.max(time_inds).item() + 1
    n_units = robs.shape[1]

    robs_trial   = np.nan * np.zeros((n_trials, n_time, n_units), dtype=np.float32)
    eyepos_trial = np.nan * np.zeros((n_trials, n_time, 2), dtype=np.float32)
    dfs_trial    = np.zeros((n_trials, n_time), dtype=bool)
    dur_trial    = np.zeros(n_trials, dtype=int)

    for itrial in range(n_trials):
        idx = np.where(trial_inds == unique_trials[itrial])[0]
        if len(idx) == 0:
            continue
        tt = time_inds[idx]
        robs_trial[itrial, tt]   = robs[idx]
        eyepos_trial[itrial, tt] = eyepos[idx]
        dfs_trial[itrial, tt]    = dfs[idx]
        dur_trial[itrial]        = len(idx)

    return robs_trial, eyepos_trial, dfs_trial, dur_trial, unique_trials


def plot_raster(t, trial_idx, height=1, ax=None, **kwargs):
    t   = np.stack([t,         t,                 np.nan * np.ones_like(t)],         axis=1).flatten()
    tri = np.stack([trial_idx, trial_idx + height, np.nan * np.ones_like(trial_idx)], axis=1).flatten()
    if ax is None:
        ax = plt.gca()
    ax.plot(t, tri, 'k', lw=0.5, **kwargs)


def plot_cov_vs_distance(mats, i, j, win_ms, ax=None):
    """
    Replicate inspect_neuron_pair: plot Ceye[k, i, j] vs bin_centers,
    with Crate[i,j] and Cpsth[i,j] as horizontal reference lines.
    """
    Ceye       = mats['Ceye']          # (n_bins, n_cells, n_cells)
    bin_centers = mats['bin_centers']  # (n_bins,)
    count_e    = mats['count_e']       # (n_bins,)
    Crate      = mats['Intercept']
    Cpsth      = mats['PSTH']

    valid = count_e > 0
    covs  = Ceye[:, i, j]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(bin_centers[valid], covs[valid], 'o', alpha=0.6, label='Measured cov')
    ax.axhline(Crate[i, j],  linestyle=':',  linewidth=2, label='Intercept (Crate)')
    ax.axhline(Cpsth[i, j],  linestyle='--', linewidth=2, label='PSTH cov')
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
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
print(f"  trial_inds: {dset_fix['trial_inds'].shape}")
print(f"  psth_inds:  {dset_fix['psth_inds'].shape}")
print(f"  keys:       {list(dset_fix.keys())}")
print(f"  enabled eye configs: {eye_configs}")

robs_serial  = _get_required(dset_fix, 'robs').astype(np.float32)
trial_inds_s = _get_required(dset_fix, 'trial_inds').astype(int)
time_inds_s  = _get_required(dset_fix, 'psth_inds').astype(int)
n_all_units  = robs_serial.shape[1]

cids_all = np.array(dset_fix.metadata.get('cluster_ids', np.arange(n_all_units)))

n_time_full    = int(np.max(time_inds_s)) + 1
time_bins_full = np.arange(n_time_full) * dt

session_label = f'{subject}_{date}'
frac_fem_summary = []

# Load vergence signal (left − right) once, if both eye traces are present.
# This is passed to run_covariance_decomposition for every config so that
# C_vergence is always computed against the correct binocular difference,
# not a meaningless channel subtraction of (X, Y).
_keys = set(dset_fix.keys())
_left_key  = next((k for k in ['eyepos_left',  'eyepos_dpi_left']  if k in _keys), None)
_right_key = next((k for k in ['eyepos_right', 'eyepos_dpi_right'] if k in _keys), None)

if _left_key and _right_key:
    _left_serial  = _ensure_2d_eyepos(_get_required(dset_fix, _left_key).astype(np.float32))
    _right_serial = _ensure_2d_eyepos(_get_required(dset_fix, _right_key).astype(np.float32))
    eyepos_verg_serial = _left_serial - _right_serial   # (T, 2): binocular difference
    print(f"Vergence signal loaded ({_left_key} − {_right_key}), shape {eyepos_verg_serial.shape}")
else:
    eyepos_verg_serial = None
    print("No separate left/right eye traces — vergence OLS will be skipped.")

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
        eyepos_serial, dfs_serial, eye_source_name = get_eye_trace_and_valid(dset_fix, eye_source=eye_source)
    except (KeyError, ValueError) as e:
        print(f"  Skipping: {e}")
        continue

    eyepos_serial = _ensure_2d_eyepos(eyepos_serial)
    dfs_serial    = _as_bool_1d(dfs_serial, robs_serial.shape[0])

    # Human-readable label for figure titles
    _eye_label_map = {
        'eyepos':                   'cyclopean',
        'eyepos_left_minus_right':  'vergence (L−R)',
    }
    display_eye_name = _eye_label_map.get(eye_source_name, eye_source_name)

    if apply_radius_filter:
        radius_mask = np.hypot(eyepos_serial[:, 0], eyepos_serial[:, 1]) < radius_deg
        dfs_serial  = dfs_serial & radius_mask

    # ── 2. Trial-align
    robs_trial, eyepos_trial, dfs_trial, dur_trial, _ = serial_to_trial_aligned(
        robs_serial, eyepos_serial, dfs_serial, trial_inds_s, time_inds_s,
    )

    # ── 3. Trial filter
    good_trials = dur_trial > min_fix_dur_bins
    robs_mc     = robs_trial[good_trials]
    eyepos_mc   = eyepos_trial[good_trials]
    dfs_mc      = dfs_trial[good_trials]
    dur_mc      = dur_trial[good_trials]
    print(f"  {good_trials.sum()} / {len(good_trials)} trials kept (min_dur={min_fix_dur_bins} bins)")

    # ── 4. Neuron gate
    spike_ok    = np.nansum(robs_mc, axis=(0, 1)) > total_spikes_threshold
    neuron_mask = np.where(spike_ok)[0]
    cids_used   = cids_all[neuron_mask]
    print(f"  {len(neuron_mask)} / {n_all_units} neurons pass spike threshold ({total_spikes_threshold})")

    if len(neuron_mask) == 0:
        print("  No neurons passed — skipping.")
        continue

    # ── 5. Analysis window
    n_time_analysis = min(valid_time_bins, robs_mc.shape[1])
    iix = np.arange(n_time_analysis)

    robs_used   = robs_mc[:, iix][:, :, neuron_mask]
    eyepos_used = eyepos_mc[:, iix]
    valid_used  = (
        dfs_mc[:, iix]
        & np.isfinite(robs_mc[:, iix][:, :, neuron_mask].sum(axis=2))
        & np.isfinite(eyepos_mc[:, iix].sum(axis=2))
    )

    # Trial-align vergence using the same good_trials mask and time window.
    # Uses robs_serial only as a dummy so serial_to_trial_aligned can infer
    # trial/time structure; the returned robs_verg_trial is discarded.
    if eyepos_verg_serial is not None:
        _, eyepos_verg_trial, _, _, _ = serial_to_trial_aligned(
            robs_serial, eyepos_verg_serial, dfs_serial, trial_inds_s, time_inds_s
        )
        eyepos_verg_used = eyepos_verg_trial[good_trials][:, iix]
    else:
        eyepos_verg_used = None

    time_full = time_bins_full[:robs_mc.shape[1]]

    # ── 6. Covariance decomposition sweep (VisionCore.covariance)
    print(f"  Running run_covariance_decomposition ({robs_used.shape[0]} trials, {robs_used.shape[2]} units)...")
    results, last_mats = run_covariance_decomposition(
        robs_used, eyepos_used, valid_used,
        window_sizes_ms=windows_ms,
        t_hist_ms=t_hist_ms,
        n_bins=n_bins,
        dt=dt,
        eyepos_vergence=eyepos_verg_used,
    )
    print(f"  Done.")

    # ── Covariance matrices (20 ms window)
    window_idx = windows_ms.index(20) if 20 in windows_ms else 0
    win_label  = windows_ms[window_idx]

    Ctotal     = project_to_psd(last_mats[window_idx]['Total'])
    Cpsth      = project_to_psd(last_mats[window_idx]['PSTH'])
    Crate      = project_to_psd(last_mats[window_idx]['Intercept'])
    Cfem       = project_to_psd(last_mats[window_idx]['FEM'])
    C_eye      = project_to_psd(last_mats[window_idx]['C_eye'])
    C_vergence = project_to_psd(last_mats[window_idx]['C_vergence'])
    CnoiseU    = project_to_psd(Ctotal - Cpsth)
    CnoiseC    = project_to_psd(Ctotal - Crate)
    MeanRates  = results[window_idx]['Erates']

    cmap = plt.get_cmap('RdBu')
    v    = np.abs(Ctotal).max() * 0.5

    # ── Eye position heatmap
    ind_sorted = np.argsort(dur_mc)
    fig, ax = plt.subplots()
    ax.imshow(
        eyepos_mc[ind_sorted, :, 0],
        vmin=-0.5, vmax=0.5, aspect='auto', cmap='coolwarm',
        interpolation='none', origin='lower',
        extent=[time_full[0], time_full[-1], 0, robs_mc.shape[0]],
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial (sorted by duration)')
    ax.set_title(f'Eye X | {display_eye_name}')
    fig.savefig(figures_dir / 'eyepos_heatmap.pdf')
    plt.close(fig)

    # ── Per-unit mean-rate PSTH
    for sel_i, cid in enumerate(cids_used):
        psth = np.nanmean(robs_mc[:, :, neuron_mask[sel_i]], axis=0)
        fig, ax = plt.subplots()
        ax.plot(time_full, psth)
        ax.axvline(0, color='r', lw=0.8, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spk / bin')
        ax.set_title(f'Neuron {cid} | {display_eye_name}')
        fig.savefig(figures_dir / f'neuron_{cid}_meanrate.pdf')
        plt.close(fig)

    # ── Per-unit rasters
    for sel_i, cid in enumerate(cids_used):
        unit_robs   = robs_mc[:, :, neuron_mask[sel_i]]
        good_u      = np.isnan(unit_robs).sum(1) == 0
        ind_u       = np.argsort(dur_mc[good_u])
        unit_sorted = np.nan_to_num(unit_robs[good_u][ind_u], nan=0.0)
        trial_idx, t_idx = np.where(unit_sorted > 0)
        fig, ax = plt.subplots()
        plot_raster(time_full[t_idx], trial_idx, height=1, ax=ax)
        ax.axvline(0, color='r', lw=0.8, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial (sorted by duration)')
        ax.set_title(f'Neuron {cid} raster | {display_eye_name}')
        fig.savefig(figures_dir / f'neuron_{cid}_raster.pdf')
        plt.close(fig)

    # ── Per-trial rasters
    for itrial in range(robs_mc.shape[0]):
        robs_it = np.nan_to_num(robs_mc[itrial][:, neuron_mask], nan=0.0)
        t_idx, u_idx = np.where(robs_it > 0)
        fig, ax = plt.subplots()
        plot_raster(time_full[t_idx], u_idx, height=1, ax=ax)
        ax.axis('off')
        ax2 = ax.twinx()
        ax2.plot(time_full, eyepos_mc[itrial, :, 0], '-r', lw=0.5)
        ax2.plot(time_full, eyepos_mc[itrial, :, 1], '-g', lw=0.5)
        ax2.set_ylim(-10, 10)
        ax.set_title(f'Trial {itrial} | {display_eye_name}')
        ax.set_xlim(time_full[0], time_full[-1])
        fig.savefig(figures_dir / f'trial_{itrial}_raster.pdf')
        plt.close(fig)

    # ── Covariance decomposition — PSTH view (3-panel)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(Ctotal,  cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[0].set_title('Total');          axs[0].axis('off')
    axs[1].imshow(Cpsth,   cmap=cmap, vmin=-v/2, vmax=v/2, interpolation='nearest'); axs[1].set_title('PSTH');           axs[1].axis('off')
    axs[2].imshow(CnoiseU, cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[2].set_title('Noise (Uncorr)'); axs[2].axis('off')
    fig.suptitle(f'{display_eye_name} | {win_label} ms')
    fig.savefig(figures_dir / f'covariance_decomposition_{window_idx}_psth.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ── Covariance decomposition — full view (5-panel, includes eye-signal term)
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))
    axs[0].imshow(Ctotal,  cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[0].set_title('Total');        axs[0].axis('off')
    axs[1].imshow(Cfem,    cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[1].set_title('FEM');          axs[1].axis('off')
    axs[2].imshow(Cpsth,   cmap=cmap, vmin=-v/2, vmax=v/2, interpolation='nearest'); axs[2].set_title('PSTH');         axs[2].axis('off')
    axs[3].imshow(CnoiseC, cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[3].set_title('Noise (Corr)'); axs[3].axis('off')
    axs[4].imshow(C_eye,   cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[4].set_title('Eye OLS');      axs[4].axis('off')
    fig.suptitle(f'{display_eye_name} | {win_label} ms')
    fig.savefig(figures_dir / f'covariance_decomposition_{window_idx}_full.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ── FEM fraction histogram
    # Alpha recomputed from PSD-projected diagonals (same convention as v6).
    denom = np.diag(Crate)
    alpha = np.divide(
        np.diag(Cpsth),
        denom,
        out=np.full(len(denom), np.nan, dtype=np.float64),
        where=denom > 0,
    )
    fem_fraction = 1.0 - alpha
    fem_fraction = fem_fraction[np.isfinite(fem_fraction)]

    # Eye OLS fraction of rate variance (same denominator as FEM fraction / 1-alpha)
    total_var = np.diag(Ctotal)
    rate_var  = np.diag(Crate)
    eye_frac_plot = np.divide(
        np.diag(C_eye), rate_var,
        out=np.full(len(rate_var), np.nan, dtype=np.float64),
        where=rate_var > 0,
    )
    eye_frac_plot = eye_frac_plot[np.isfinite(eye_frac_plot)]

    # Vergence fraction of rate variance — analog of 1-alpha for the vergence signal
    rate_var = np.diag(Crate)
    vergence_frac_rate = np.divide(
        np.diag(C_vergence), rate_var,
        out=np.full(len(rate_var), np.nan, dtype=np.float64),
        where=rate_var > 0,
    )
    vergence_frac_rate = vergence_frac_rate[np.isfinite(vergence_frac_rate)]

    # Fano factor: baseline is the McFarland-corrected residual (Ctotal - Crate),
    # matching what the rest of the paper uses.  We then ask how much of that
    # residual is further explained by the vergence OLS estimate.
    noise_corr_diag = np.diag(CnoiseC)   # Ctotal − Crate (primary residual)
    vergence_diag   = np.diag(C_vergence)
    ff_before = np.divide(
        noise_corr_diag, MeanRates,
        out=np.full(len(MeanRates), np.nan, dtype=np.float64),
        where=MeanRates > 0,
    )
    ff_after = np.divide(
        np.maximum(noise_corr_diag - vergence_diag, 0.0), MeanRates,
        out=np.full(len(MeanRates), np.nan, dtype=np.float64),
        where=MeanRates > 0,
    )
    ff_before_finite = ff_before[np.isfinite(ff_before)]
    ff_after_finite  = ff_after[np.isfinite(ff_after)]

    # Window-size sweep: collect mean ± SEM FF at each window for the line plot
    sweep_windows_ms      = [r['window_ms']           for r in results]
    sweep_ff_corr         = [r['ff_corr_mean']         for r in results]
    sweep_ff_corr_sem     = [r['ff_corr_sem']          for r in results]
    sweep_ff_before_verg  = [r['ff_before_verg_mean']  for r in results]
    sweep_ff_before_sem   = [r['ff_before_verg_sem']   for r in results]
    sweep_ff_after_verg   = [r['ff_after_verg_mean']   for r in results]
    sweep_ff_after_sem    = [r['ff_after_verg_sem']    for r in results]

    frac_fem_summary.append({
        'subdir_name': subdir_name,
        'eye_source': eye_source,               # raw source key, used for filtering
        'eye_source_name': eye_source_name,
        'display_eye_name': display_eye_name,
        'win_label_ms': win_label,
        'fem_fraction': fem_fraction.copy(),
        'eye_frac': eye_frac_plot.copy(),
        'vergence_frac_rate': vergence_frac_rate.copy(),
        'ff_before': ff_before_finite.copy(),
        'ff_after': ff_after_finite.copy(),
        'ff_before_paired': ff_before.copy(),   # aligned per-neuron for scatter
        'ff_after_paired': ff_after.copy(),
        'sweep_windows_ms':     sweep_windows_ms,
        'sweep_ff_corr':        sweep_ff_corr,
        'sweep_ff_corr_sem':    sweep_ff_corr_sem,
        'sweep_ff_before_verg': sweep_ff_before_verg,
        'sweep_ff_before_sem':  sweep_ff_before_sem,
        'sweep_ff_after_verg':  sweep_ff_after_verg,
        'sweep_ff_after_sem':   sweep_ff_after_sem,
    })

    fig, axs_h = plt.subplots(1, 4, figsize=(20, 4))

    # Panel 1: FEM fraction (1 - alpha) — primary McFarland estimator
    if fem_fraction.size:
        axs_h[0].hist(fem_fraction, bins=np.linspace(0, 1, 31), color='steelblue', alpha=0.7)
    else:
        axs_h[0].text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axs_h[0].transAxes)
    axs_h[0].set_xlim(0, 1)
    axs_h[0].set_xlabel('1 − α  (C_FEM / C_rate, McFarland)')
    axs_h[0].set_ylabel('Count')
    axs_h[0].set_title(f'FEM fraction [PRIMARY] | {display_eye_name}')

    # Panel 2: Eye OLS fraction of rate variance — secondary check (same denom as col 1)
    if eye_frac_plot.size:
        axs_h[1].hist(eye_frac_plot, bins=np.linspace(0, 1, 31), color='coral', alpha=0.7)
    else:
        axs_h[1].text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axs_h[1].transAxes)
    axs_h[1].set_xlim(0, 1)
    axs_h[1].set_xlabel('C_eye_OLS / C_rate  (OLS check)')
    axs_h[1].set_ylabel('Count')
    axs_h[1].set_title(f'Eye OLS [CHECK] | {display_eye_name}')

    # Panel 3: Vergence fraction of rate variance
    if vergence_frac_rate.size:
        axs_h[2].hist(vergence_frac_rate, bins=np.linspace(0, 1, 31), color='mediumpurple', alpha=0.7)
    else:
        axs_h[2].text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axs_h[2].transAxes)
    axs_h[2].set_xlim(0, 1)
    axs_h[2].set_xlabel('C_vergence / C_rate  (OLS)')
    axs_h[2].set_ylabel('Count')
    axs_h[2].set_title(f'Vergence fraction of rate | {display_eye_name}')

    # Panel 4: Fano factor before/after removing vergence from McFarland residual.
    # x = FF of residual noise after McFarland FEM correction  (Ctotal - Crate)
    # y = FF of that residual minus vergence-driven variance    (OLS estimate)
    _ff_both = np.stack([ff_before, ff_after], axis=1)
    _ff_ok   = np.isfinite(_ff_both).all(axis=1)
    if _ff_ok.any():
        _fb = ff_before[_ff_ok]
        _fa = ff_after[_ff_ok]
        axs_h[3].scatter(_fb, _fa, s=20, alpha=0.7, color='darkorange', edgecolors='none')
        _lim = max(_fb.max(), _fa.max(), 0.1) * 1.05
        axs_h[3].plot([0, _lim], [0, _lim], 'k--', lw=0.8, alpha=0.5)
        axs_h[3].set_xlim(0, _lim)
        axs_h[3].set_ylim(0, _lim)
    else:
        axs_h[3].text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axs_h[3].transAxes)
    axs_h[3].set_xlabel('FF of McFarland residual')
    axs_h[3].set_ylabel('FF minus vergence (OLS)')
    axs_h[3].set_title(f'Fano factor ({win_label} ms) | {display_eye_name}')
    axs_h[3].set_aspect('equal')

    fig.suptitle(f'{session_label} | {win_label} ms')
    fig.tight_layout()
    fig.savefig(figures_dir / 'frac_fem_hist.pdf')
    plt.close(fig)

    # ── Mean rate vs noise variance
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(MeanRates, np.diag(CnoiseU), '.', ms=4)
    axs[0].plot(axs[0].get_xlim(), axs[0].get_xlim(), 'k', lw=0.5)
    axs[0].set_xlabel('Mean Rate')
    axs[0].set_ylabel('Variance')
    axs[0].set_title(f'Noise Var (uncorr) | {display_eye_name}')
    axs[1].plot(MeanRates, np.diag(CnoiseC), '.', ms=4)
    axs[1].plot(axs[1].get_xlim(), axs[1].get_xlim(), 'k', lw=0.5)
    axs[1].set_xlabel('Mean Rate')
    axs[1].set_ylabel('Variance')
    axs[1].set_title(f'Noise Var (corr) | {display_eye_name}')
    fig.tight_layout()
    fig.savefig(figures_dir / 'meanrate_vs_variance.pdf', bbox_inches='tight')
    plt.close(fig)

    # ── Multi-page PDF: unit rasters + cov-vs-distance (20 ms window)
    unit_pdf_path = figures_dir / f'unit_rasters_{session_label}.pdf'
    with PdfPages(unit_pdf_path) as pdf:
        for sel_i, cid in enumerate(cids_used):
            unit_robs   = robs_mc[:, :, neuron_mask[sel_i]]
            ind_u       = np.argsort(dur_mc)
            unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
            trial_idx, t_idx = np.where(unit_sorted > 0)

            fig, axs_u = plt.subplots(1, 2, figsize=(12, 4))

            axs_u[0].set_title(f'Neuron {cid} | {display_eye_name}')
            plot_raster(time_full[t_idx], trial_idx, height=1, ax=axs_u[0])
            axs_u[0].axvline(0, color='r', lw=0.8)
            axs_u[0].set_xlim(time_full[0], time_full[-1])
            axs_u[0].set_xlabel('Time (s)')
            axs_u[0].set_ylabel('Trial')

            plot_cov_vs_distance(last_mats[window_idx], sel_i, sel_i, win_label, ax=axs_u[1])

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"  Saved {unit_pdf_path}")

    # ── Multi-page PDF: trial rasters
    trial_pdf_path = figures_dir / f'trial_rasters_{session_label}.pdf'
    with PdfPages(trial_pdf_path) as pdf:
        for itrial in range(robs_mc.shape[0]):
            robs_it = np.nan_to_num(robs_mc[itrial][:, neuron_mask], nan=0.0)
            t_idx, u_idx = np.where(robs_it > 0)
            fig, ax = plt.subplots()
            plot_raster(time_full[t_idx], u_idx, height=1, ax=ax)
            ax.axis('off')
            ax2 = ax.twinx()
            ax2.plot(time_full, eyepos_mc[itrial, :, 0], '.r', ms=1)
            ax2.plot(time_full, eyepos_mc[itrial, :, 1], '.g', ms=1)
            ax2.set_ylim(-10, 10)
            ax.set_title(f'Trial {itrial} | {display_eye_name}')
            ax.set_xlim(time_full[0], time_full[-1])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"  Saved {trial_pdf_path}")

    # ── 2D conditional second moment: vergence stratified by cyclopean distance
    # The near-cyclopean slice (dc bin 0) tests whether vergence similarity
    # predicts additional covariance after cyclopean position is matched.
    # Expected signal: C2d[0, :] decreases with dv (less covariance when
    # vergence is mismatched, same logic as 1D McFarland).
    _result_2d = None
    if eyepos_verg_used is not None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _win_bins_2d  = max(1, int(20 / (dt * 1000)))          # 20 ms window
        _t_hist_bins  = int(t_hist_ms / (dt * 1000))
        _t_hist_used  = max(_t_hist_bins, _win_bins_2d)

        _segs  = extract_valid_segments(valid_used, min_len_bins=36)
        _robs  = torch.tensor(np.nan_to_num(robs_used, nan=0.0),
                              dtype=torch.float32, device=_device)
        _eyec  = torch.tensor(np.nan_to_num(eyepos_used, nan=0.0),
                              dtype=torch.float32, device=_device)
        _eyev  = torch.tensor(np.nan_to_num(eyepos_verg_used, nan=0.0),
                              dtype=torch.float32, device=_device)

        _SC, _EyeC, _Tidx, _ = extract_windows(_robs, _eyec, _segs,
                                                 _win_bins_2d, _t_hist_used,
                                                 device=str(_device))
        _,    _EyeV,  _,    _ = extract_windows(_robs, _eyev, _segs,
                                                 _win_bins_2d, _t_hist_used,
                                                 device=str(_device))

        if _SC is not None and _EyeV is not None and _SC.shape[0] >= 20:
            print(f"  Running 2D conditional estimator ({_SC.shape[0]} windows, 20 ms)...")
            _result_2d = estimate_vergence_conditional_on_cyclopean(
                _SC, _EyeC, _EyeV, _Tidx, n_bins_c=3, n_bins_v=5
            )

            _count2d = _result_2d['count2d']
            print(f"  Pair counts (d_c rows × d_v cols):\n{_count2d}")

            _C2d          = _result_2d['C2d']
            _dv_cell_means = _result_2d['dv_cell_means']
            _dc_cell_means = _result_2d['dc_cell_means']
            _Erate_2d     = _result_2d['Erate']
            _C_intercept  = _result_2d['C_near_intercept']
            _C_slope      = _result_2d['C_near_slope']

            n_bc, n_bv = _count2d.shape
            _colors_bc = ['steelblue', 'darkorange', 'teal']
            _labels_bc = ['d_c bin 0 [NEAR]', 'd_c bin 1 [mid]', 'd_c bin 2 [FAR]']

            fig_2d, axs_2d = plt.subplots(1, 3, figsize=(15, 4))

            # Panel 1: pair counts heatmap — always plot first
            ax = axs_2d[0]
            im = ax.imshow(_count2d, origin='lower', cmap='Blues', aspect='auto')
            ax.set_xlabel('d_v bin'); ax.set_ylabel('d_c bin')
            ax.set_title(f'Pair counts\n{display_eye_name} | 20 ms')
            fig_2d.colorbar(im, ax=ax)

            # Panel 2: mean diag(C2d) vs d_v for each d_c bin (Fano-like, in count² units)
            ax = axs_2d[1]
            _ff_ref = np.nanmean(np.diag(project_to_psd(last_mats[window_idx]['Intercept'])))
            for bc_k in range(n_bc):
                _x, _y = [], []
                for bv_k in range(n_bv):
                    if _count2d[bc_k, bv_k] >= 10 and np.isfinite(_dv_cell_means[bc_k, bv_k]):
                        _x.append(_dv_cell_means[bc_k, bv_k])
                        _y.append(np.nanmean(np.diag(_C2d[bc_k, bv_k])))
                if not _x:
                    continue
                _lbl = _labels_bc[bc_k] if bc_k < len(_labels_bc) else f'd_c bin {bc_k}'
                ax.plot(_x, _y, 'o-', color=_colors_bc[bc_k % len(_colors_bc)],
                        lw=1.5, ms=6, label=f'{_lbl}  (n={_count2d[bc_k].sum()})')
            ax.axhline(_ff_ref, color='k', lw=0.8, alpha=0.4, linestyle='--',
                       label='1D Crate intercept (diag mean)')
            ax.set_xlabel('Vergence distance d_v (RMS)'); ax.set_ylabel('mean diag(C2d)')
            ax.set_title(f'Conditional covariance vs vergence dist\n{display_eye_name} | 20 ms')
            ax.legend(frameon=False, fontsize=8)

            # Panel 3: near-cyclopean vergence slope on diagonal (per neuron)
            ax = axs_2d[2]
            _slope_diag = np.diag(_C_slope) if np.isfinite(_C_slope).any() else None
            if _slope_diag is not None and np.isfinite(_slope_diag).any():
                _slope_finite = _slope_diag[np.isfinite(_slope_diag)]
                ax.hist(_slope_finite, bins=30, color='mediumpurple', alpha=0.7)
                ax.axvline(0, color='k', lw=0.8, linestyle='--')
                ax.set_xlabel('dC/d(d_v)  near-cyclopean slope')
                ax.set_ylabel('Count')
                mn = np.nanmean(_slope_finite)
                ax.set_title(f'Vergence slope (diag, near-dc bin)\nmean={mn:.4f} | {display_eye_name}')
            else:
                ax.text(0.5, 0.5, 'Insufficient data for slope fit',
                        ha='center', va='center', transform=ax.transAxes)

            fig_2d.suptitle(f'{session_label}  —  2D conditional second moment')
            fig_2d.tight_layout()
            _path_2d = figures_dir / '2d_conditional_second_moment.pdf'
            fig_2d.savefig(_path_2d)
            plt.close(fig_2d)
            print(f"  Saved {_path_2d}")

            # Summary stats
            if _slope_diag is not None and np.isfinite(_slope_diag).any():
                _s = _slope_diag[np.isfinite(_slope_diag)]
                print(f"  Vergence slope (diag, near-dc): mean={_s.mean():.4f}, "
                      f"median={np.median(_s):.4f}, "
                      f"frac<0={(_s<0).mean():.2f}")

            # ── Per-unit 2D conditional variance PDF
            # compute_conditional_second_moments_2d operates on all C neurons jointly
            # (outer product Si.T @ Sj is C×C per bin).  The per-neuron view is the
            # diagonal C2d[:, :, i, i] — each unit's conditional variance as a function
            # of vergence distance, sliced by cyclopean distance bin.
            _unit_2d_path = figures_dir / f'unit_2d_cov_{session_label}.pdf'
            with PdfPages(_unit_2d_path) as _pdf_2d:
                for _sel_i, _cid in enumerate(cids_used):
                    _fig_u, _axs_u = plt.subplots(1, 3, figsize=(15, 4))

                    # Left: unit raster (same style as unit_rasters PDF)
                    _unit_robs   = robs_mc[:, :, neuron_mask[_sel_i]]
                    _ind_u       = np.argsort(dur_mc)
                    _unit_sorted = np.nan_to_num(_unit_robs[_ind_u], nan=0.0)
                    _trial_r, _t_r = np.where(_unit_sorted > 0)
                    _axs_u[0].set_title(f'Neuron {_cid} | {display_eye_name}')
                    plot_raster(time_full[_t_r], _trial_r, height=1, ax=_axs_u[0])
                    _axs_u[0].axvline(0, color='r', lw=0.8)
                    _axs_u[0].set_xlim(time_full[0], time_full[-1])
                    _axs_u[0].set_xlabel('Time (s)')
                    _axs_u[0].set_ylabel('Trial (sorted by dur)')

                    # Middle: 1D McFarland cov-vs-distance at 20 ms
                    plot_cov_vs_distance(last_mats[window_idx], _sel_i, _sel_i,
                                         win_label, ax=_axs_u[1])

                    # Right: 2D conditional variance C2d[bc, bv, i, i] vs dv
                    _ax2d = _axs_u[2]
                    _crate_ii = last_mats[window_idx]['Intercept'][_sel_i, _sel_i]
                    for _bc_k in range(n_bc):
                        _x2, _y2 = [], []
                        for _bv_k in range(n_bv):
                            if (_count2d[_bc_k, _bv_k] >= 5
                                    and np.isfinite(_dv_cell_means[_bc_k, _bv_k])
                                    and np.isfinite(_C2d[_bc_k, _bv_k, _sel_i, _sel_i])):
                                _x2.append(_dv_cell_means[_bc_k, _bv_k])
                                _y2.append(_C2d[_bc_k, _bv_k, _sel_i, _sel_i])
                        if _x2:
                            _lbl2 = (_labels_bc[_bc_k] if _bc_k < len(_labels_bc)
                                     else f'dc bin {_bc_k}')
                            _ax2d.plot(_x2, _y2, 'o-',
                                       color=_colors_bc[_bc_k % len(_colors_bc)],
                                       lw=1.5, ms=5, label=_lbl2)
                    _ax2d.axhline(_crate_ii, color='k', lw=0.8, linestyle='--',
                                  label=f'1D Crate ({win_label} ms)')
                    _ax2d.axhline(0, color='k', lw=0.4, alpha=0.3)
                    # Annotate fitted slope for this neuron
                    if (np.isfinite(_C_slope[_sel_i, _sel_i])
                            and np.isfinite(_C_intercept[_sel_i, _sel_i])):
                        _ax2d.annotate(
                            f'slope={_C_slope[_sel_i,_sel_i]:.3f}\n'
                            f'int@dv=0={_C_intercept[_sel_i,_sel_i]:.3f}',
                            xy=(0.02, 0.97), xycoords='axes fraction',
                            va='top', ha='left', fontsize=7,
                            color='gray'
                        )
                    _ax2d.set_xlabel('Vergence dist d_v (RMS)')
                    _ax2d.set_ylabel('C2d[bc,bv,i,i]  (unit variance)')
                    _ax2d.set_title(f'2D conditional var | Neuron {_cid}')
                    _ax2d.legend(frameon=False, fontsize=7)

                    _fig_u.tight_layout()
                    _pdf_2d.savefig(_fig_u, bbox_inches='tight')
                    plt.close(_fig_u)
            print(f"  Saved {_unit_2d_path}")
        else:
            print("  2D analysis skipped: insufficient windows.")

    # ── Pickle output
    output = {
        'sess': session_label,
        'cids': cids_all,
        'neuron_mask': neuron_mask,
        'windows': windows_ms,
        'cids_used': cids_used,
        'results': results,
        'last_mats': last_mats,
        'meta': {
            'eye_source': eye_source,
            'eye_source_name': eye_source_name,
            'dataset_path': str(fix_path),
            'dataset_dir': str(dataset_dir),
            'dt': dt,
            't_hist_ms': t_hist_ms,
            'n_bins': n_bins,
            'valid_time_bins': n_time_analysis,
            'total_spikes_threshold': total_spikes_threshold,
            'min_fix_dur_bins': min_fix_dur_bins,
            'apply_radius_filter': apply_radius_filter,
            'radius_deg': radius_deg if apply_radius_filter else None,
            'estimator': 'VisionCore.covariance.run_covariance_decomposition',
        },
    }
    pkl_path = figures_dir / f'mcfarland_fixrsvp_{session_label}_{eye_source}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"  Saved {pkl_path}")

# ── Summary figures across all eye sources (one row per eye config, 4 panels)
# binocular_diff is excluded: running McFarland pairwise on a vergence signal
# doesn't produce the same conceptual quantity as monocular 1-alpha.
# Its per-condition page is still saved; it just shouldn't sit alongside the
# monocular FEM estimates in this comparison.
_fem_summaries = [s for s in frac_fem_summary if s['eye_source'] != 'binocular_diff']

if _fem_summaries:
    n_conds = len(_fem_summaries)
    n_cols  = 5  # FEM fraction | Eye OLS | Vergence | Fano scatter | FF vs window size
    fig, axs = plt.subplots(n_conds, n_cols, figsize=(5 * n_cols, 3.5 * n_conds), squeeze=False)

    for row, summary in enumerate(_fem_summaries):
        row_title = f"{summary['subdir_name']} | {summary['display_eye_name']}"

        # FEM fraction
        ax = axs[row, 0]
        if summary['fem_fraction'].size:
            ax.hist(summary['fem_fraction'], bins=np.linspace(0, 1, 31), color='steelblue', alpha=0.7)
        ax.set_xlim(0, 1); ax.set_xlabel('1 − α  (FEM / rate)'); ax.set_ylabel('Count')
        ax.set_title(f'FEM fraction\n{row_title}')

        # Eye OLS fraction
        ax = axs[row, 1]
        if summary['eye_frac'].size:
            ax.hist(summary['eye_frac'], bins=np.linspace(0, 1, 31), color='coral', alpha=0.7)
        ax.set_xlim(0, 1); ax.set_xlabel('C_eye_OLS / C_rate'); ax.set_ylabel('Count')
        ax.set_title(f'Eye OLS fraction\n{row_title}')

        # Vergence fraction of rate variance
        ax = axs[row, 2]
        if summary['vergence_frac_rate'].size:
            ax.hist(summary['vergence_frac_rate'], bins=np.linspace(0, 1, 31), color='mediumpurple', alpha=0.7)
        ax.set_xlim(0, 1); ax.set_xlabel('Vergence / rate var'); ax.set_ylabel('Count')
        ax.set_title(f'Vergence fraction of rate\n{row_title}')

        # Fano factor scatter: before vs after vergence correction
        ax = axs[row, 3]
        fb = summary['ff_before_paired']
        fa = summary['ff_after_paired']
        _ok = np.isfinite(fb) & np.isfinite(fa)
        if _ok.any():
            ax.scatter(fb[_ok], fa[_ok], s=20, alpha=0.7, color='darkorange', edgecolors='none')
            _lim = max(fb[_ok].max(), fa[_ok].max(), 0.1) * 1.05
            ax.plot([0, _lim], [0, _lim], 'k--', lw=0.8, alpha=0.5)
            ax.set_xlim(0, _lim); ax.set_ylim(0, _lim)
        ax.set_xlabel('FF (before vergence)'); ax.set_ylabel('FF (after vergence)')
        ax.set_title(f'Fano factor ({summary["win_label_ms"]} ms)\n{row_title}')
        ax.set_aspect('equal')

        # FF vs window size (mean ± SEM across neurons)
        ax = axs[row, 4]
        ws = np.array(summary['sweep_windows_ms'])
        ax.errorbar(ws, summary['sweep_ff_corr'],        yerr=summary['sweep_ff_corr_sem'],
                    fmt='o-',  color='steelblue',  lw=1.5, ms=5, capsize=3, label='McFarland corrected')
        ax.errorbar(ws, summary['sweep_ff_before_verg'], yerr=summary['sweep_ff_before_sem'],
                    fmt='s--', color='darkorange',  lw=1.5, ms=5, capsize=3, label='Residual (before vergence)')
        ax.errorbar(ws, summary['sweep_ff_after_verg'],  yerr=summary['sweep_ff_after_sem'],
                    fmt='s-',  color='teal',        lw=1.5, ms=5, capsize=3, label='Residual (after vergence)')
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('Mean FF ± SEM')
        ax.set_title(f'FF vs window size\n{row_title}')
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle(f'Covariance decomposition summary | {session_label} (v7 / covariance.py)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    summary_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.pdf'
    fig.savefig(summary_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {summary_path}")

# ── Text summary
_hist_bins = np.linspace(0, 1, 31)
_bin_edges = _hist_bins
_bin_centers = 0.5 * (_bin_edges[:-1] + _bin_edges[1:])

txt_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.txt'
with open(txt_path, 'w') as f:
    f.write(f"Covariance decomposition summary\n")
    f.write(f"Session: {session_label}\n")
    f.write(f"Estimator: VisionCore.covariance.run_covariance_decomposition\n")
    f.write(f"Window: {windows_ms[window_idx]} ms  |  n_bins={n_bins}  |  t_hist={t_hist_ms} ms\n")
    f.write(f"\n")
    f.write(f"PRIMARY:  McFarland pairwise (dual-window, cross-trial histograms) for C_FEM and 1-alpha.\n")
    f.write(f"CHECK:    OLS regression of spike counts on mean eye position (C_FEM_linear).\n")
    f.write(f"VERGENCE: OLS regression on (left - right) eye position (C_vergence_linear).\n")
    f.write(f"Residual covariance / Fano factor baseline: Ctotal - Crate (McFarland), matching rest of paper.\n")
    f.write(f"Histogram bins: 0.00 to 1.00 in steps of {_bin_edges[1]-_bin_edges[0]:.3f}\n")
    f.write("\n")

    def _write_metric(f, label, values):
        f.write(f"  --- {label} ---\n")
        f.write(f"  n units (finite): {values.size}\n")
        if values.size:
            f.write(f"  mean:   {values.mean():.4f}\n")
            f.write(f"  median: {np.median(values):.4f}\n")
            f.write(f"  std:    {values.std():.4f}\n")
            f.write(f"  [min, max]: [{values.min():.4f}, {values.max():.4f}]\n")
            counts, _ = np.histogram(values, bins=_bin_edges)
            f.write(f"  Histogram (bin_center, count):\n")
            for center, count in zip(_bin_centers, counts):
                bar = '#' * count
                f.write(f"    {center:.2f}  {count:4d}  {bar}\n")
        f.write("\n")

    for summary in frac_fem_summary:
        name = f"{summary['subdir_name']} ({summary['display_eye_name']})"
        f.write(f"{'='*60}\n")
        f.write(f"Eye source: {name}\n")
        if summary['eye_source'] == 'binocular_diff':
            f.write(f"  NOTE: binocular_diff excluded from FEM summary figure.\n")
            f.write(f"  McFarland pairwise on a vergence signal is not the same quantity as monocular 1-alpha.\n")
        f.write("\n")
        _write_metric(f, "FEM fraction (1 - alpha)  [PRIMARY: McFarland pairwise]", summary['fem_fraction'])
        _write_metric(f, "Eye OLS / C_rate  [CHECK: OLS monocular, same denom as 1-alpha]", summary['eye_frac'])
        _write_metric(f, "Vergence / rate variance  [OLS on left - right]", summary['vergence_frac_rate'])
        # Fano factor: separate bins (not 0-1)
        ff_all = np.concatenate([summary['ff_before'], summary['ff_after']])
        if ff_all.size:
            ff_lo, ff_hi = 0.0, max(float(ff_all.max()), 0.1)
            _ff_edges  = np.linspace(ff_lo, ff_hi, 31)
            _ff_centers = 0.5 * (_ff_edges[:-1] + _ff_edges[1:])
            f.write(f"  --- Fano factor baseline: Ctotal - Crate (McFarland residual, matching rest of paper) ---\n")
            for label, vals in [("Before vergence removal (McFarland residual)", summary['ff_before']),
                                 ("After  vergence removal (minus C_vergence OLS)", summary['ff_after'])]:
                f.write(f"  [{label}]\n")
                if vals.size:
                    f.write(f"  n units (finite): {vals.size}\n")
                    f.write(f"  mean:   {vals.mean():.4f}\n")
                    f.write(f"  median: {np.median(vals):.4f}\n")
                    f.write(f"  std:    {vals.std():.4f}\n")
                    f.write(f"  [min, max]: [{vals.min():.4f}, {vals.max():.4f}]\n")
                    counts, _ = np.histogram(vals, bins=_ff_edges)
                    for center, count in zip(_ff_centers, counts):
                        bar = '#' * count
                        f.write(f"    {center:.3f}  {count:4d}  {bar}\n")
                f.write("\n")
        f.write("\n")

print(f"Saved {txt_path}")

print("\nAll conditions complete.")

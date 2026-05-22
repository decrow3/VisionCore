#%%
"""
test_rowleyv6.py

Runs DualWindowAnalysis for the available eye-trace sources from a fixRSVP dataset
and saves all figures (matching test_rowley5 output) into per-source subdirectories.

For testing, this script currently loads the shared binocular export from:

    processed/<SUBJECT_DATE>/datasets_binocular/fixrsvp.dset

If the dataset also includes binocular and pupil traces, those analyses are enabled
automatically.
"""

#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
from mcfarland_sim import DualWindowAnalysis, project_to_psd

#%% ---------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

subject = 'Luke'
date = '2026-03-16'
primary_eye = 'binocular'  # Determines default eyepos source and subdir naming.

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

BASE_FIGURES_DIR = Path('../figures/mcfarland') / f'{subject}_{date}'

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
        assert len(x) == n_expected, f"Length mismatch: got {len(x)}, expected {n_expected}"
    return x.astype(bool)

def _ensure_2d_eyepos(eyepos):
    """Ensure eye position is (N, 2); pad 1-channel traces with a zero column."""
    eyepos = np.asarray(eyepos, dtype=np.float32)
    if eyepos.ndim == 1:
        return np.stack([eyepos, np.zeros_like(eyepos)], axis=1)
    if eyepos.shape[1] == 1:
        return np.concatenate([eyepos, np.zeros((len(eyepos), 1), dtype=np.float32)], axis=1)
    return eyepos

def get_eye_trace_and_valid(dset, eye_source='default'):
    """
    Return:
        eyepos_serial: (N, 2)
        valid_serial:  (N,) bool
        source_name:   string used for bookkeeping
    """
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
        left = None
        right = None
        for k in ['eyepos_left', 'eyepos_dpi_left']:
            if k in keys:
                left = _get_required(dset, k)
                break
        for k in ['eyepos_right', 'eyepos_dpi_right']:
            if k in keys:
                right = _get_required(dset, k)
                break
        if left is None or right is None:
            raise KeyError("Requested binocular_diff, but left/right eye traces not found.")
        valid_l = _as_bool_1d(_get_optional(dset, 'dpi_valid_left', np.ones(len(left), dtype=bool)), len(left))
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

    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        pad = 0.05 if np.isclose(lo, 0.0) else max(0.05, abs(lo) * 0.05)
        return np.linspace(lo - pad, hi + pad, n_bins).tolist()
    return np.linspace(lo, hi, n_bins).tolist()


def get_enabled_eye_configs(dset, primary_eye_name):
    keys = set(dset.keys())
    enabled_configs = [('primary-dpi', 'default')]

    has_left = any(k in keys for k in ['eyepos_left', 'eyepos_dpi_left'])
    has_right = any(k in keys for k in ['eyepos_right', 'eyepos_dpi_right'])
    has_left_pupil = any(k in keys for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil'])
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
    enabled_configs = [
        (primary_subdir if subdir == 'primary-dpi' else subdir, source)
        for subdir, source in enabled_configs
    ]
    return enabled_configs


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
    """Vertical tick raster: t = spike times (x), trial_idx = trial/unit index (y)."""
    t   = np.stack([t,         t,                 np.nan * np.ones_like(t)],         axis=1).flatten()
    tri = np.stack([trial_idx, trial_idx + height, np.nan * np.ones_like(trial_idx)], axis=1).flatten()
    if ax is None:
        ax = plt.gca()
    ax.plot(t, tri, 'k', lw=0.5, **kwargs)


#%% ---------------------------------------------------------------------------
# Load dataset (once — robs and trial structure are shared across all conditions)
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
time_bins_full = np.arange(n_time_full) * dt   # seconds, relative to trial onset

session_label = f'{subject}_{date}'
frac_fem_summary = []

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

    if apply_radius_filter:
        radius_mask = np.hypot(eyepos_serial[:, 0], eyepos_serial[:, 1]) < radius_deg
        dfs_serial  = dfs_serial & radius_mask

    # ── 2. Trial-align
    robs_trial, eyepos_trial, dfs_trial, dur_trial, _ = serial_to_trial_aligned(
        robs_serial, eyepos_serial, dfs_serial, trial_inds_s, time_inds_s,
    )

    # ── 3. Trial filter
    good_trials = dur_trial > min_fix_dur_bins
    robs_mc     = robs_trial[good_trials]    # (n_trials, n_time, n_all_units)
    eyepos_mc   = eyepos_trial[good_trials]  # (n_trials, n_time, 2)
    dfs_mc      = dfs_trial[good_trials]     # (n_trials, n_time)
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

    robs_used   = robs_mc[:, iix][:, :, neuron_mask]   # (n_trials, T, n_sel)
    eyepos_used = eyepos_mc[:, iix]                     # (n_trials, T, 2)
    valid_used  = (
        dfs_mc[:, iix]
        & np.isfinite(robs_mc[:, iix][:, :, neuron_mask].sum(axis=2))
        & np.isfinite(eyepos_mc[:, iix].sum(axis=2))
    )

    time_full = time_bins_full[:robs_mc.shape[1]]

    # ── 6. McFarland sweep
    print(f"  Running DualWindowAnalysis ({robs_used.shape[0]} trials, {robs_used.shape[2]} units)...")
    analyzer = DualWindowAnalysis(robs_used, eyepos_used, valid_used, dt=dt)
    results, last_mats = analyzer.run_sweep(windows_ms, t_hist_ms=t_hist_ms, n_bins=n_bins)
    print(f"  Done.")

    # ── Covariance matrices (20 ms window) ───────────────────────────────────
    window_idx = windows_ms.index(20) if 20 in windows_ms else 0
    win_label  = windows_ms[window_idx]

    Ctotal  = project_to_psd(last_mats[window_idx]['Total'])
    Cpsth   = project_to_psd(last_mats[window_idx]['PSTH'])
    Crate   = project_to_psd(last_mats[window_idx]['Intercept'])
    Cfem    = project_to_psd(last_mats[window_idx]['FEM'])
    CnoiseU = project_to_psd(Ctotal - Cpsth)
    CnoiseC = project_to_psd(Ctotal - Crate)
    MeanRates = results[window_idx]['Erates']

    cmap = plt.get_cmap('RdBu')
    v    = np.abs(Ctotal).max() * 0.5

    # ── Eye position heatmap ─────────────────────────────────────────────────
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
    ax.set_title(f'Eye X | {eye_source_name}')
    fig.savefig(figures_dir / 'eyepos_heatmap.pdf')
    plt.close(fig)

    # ── Per-unit mean-rate PSTH (individual files) ───────────────────────────
    for sel_i, cid in enumerate(cids_used):
        psth = np.nanmean(robs_mc[:, :, neuron_mask[sel_i]], axis=0)
        fig, ax = plt.subplots()
        ax.plot(time_full, psth)
        ax.axvline(0, color='r', lw=0.8, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spk / bin')
        ax.set_title(f'Neuron {cid} | {eye_source_name}')
        fig.savefig(figures_dir / f'neuron_{cid}_meanrate.pdf')
        plt.close(fig)

    # ── Per-unit rasters (individual files) ──────────────────────────────────
    for sel_i, cid in enumerate(cids_used):
        unit_robs   = robs_mc[:, :, neuron_mask[sel_i]]          # (n_trials, n_time)
        good_u      = np.isnan(unit_robs).sum(1) == 0
        ind_u       = np.argsort(dur_mc[good_u])
        unit_sorted = np.nan_to_num(unit_robs[good_u][ind_u], nan=0.0)
        trial_idx, t_idx = np.where(unit_sorted > 0)
        fig, ax = plt.subplots()
        plot_raster(time_full[t_idx], trial_idx, height=1, ax=ax)
        ax.axvline(0, color='r', lw=0.8, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial (sorted by duration)')
        ax.set_title(f'Neuron {cid} raster | {eye_source_name}')
        fig.savefig(figures_dir / f'neuron_{cid}_raster.pdf')
        plt.close(fig)

    # ── Per-trial rasters (individual files) ─────────────────────────────────
    for itrial in range(robs_mc.shape[0]):
        # robs_mc[itrial][:, neuron_mask] avoids NumPy's advanced-index transposition
        # (arr[scalar, :, int_array] puts the advanced dim first; chaining avoids it)
        robs_it = np.nan_to_num(robs_mc[itrial][:, neuron_mask], nan=0.0)  # (n_time, n_sel)
        t_idx, u_idx = np.where(robs_it > 0)
        fig, ax = plt.subplots()
        plot_raster(time_full[t_idx], u_idx, height=1, ax=ax)
        ax.axis('off')
        ax2 = ax.twinx()
        ax2.plot(time_full, eyepos_mc[itrial, :, 0], '-r', lw=0.5)
        ax2.plot(time_full, eyepos_mc[itrial, :, 1], '-g', lw=0.5)
        ax2.set_ylim(-10, 10)
        ax.set_title(f'Trial {itrial} | {eye_source_name}')
        ax.set_xlim(time_full[0], time_full[-1])
        fig.savefig(figures_dir / f'trial_{itrial}_raster.pdf')
        plt.close(fig)

    # ── Covariance decomposition — PSTH view (3-panel) ───────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(Ctotal,  cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[0].set_title('Total');          axs[0].axis('off')
    axs[1].imshow(Cpsth,   cmap=cmap, vmin=-v/2, vmax=v/2, interpolation='nearest'); axs[1].set_title('PSTH');           axs[1].axis('off')
    axs[2].imshow(CnoiseU, cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[2].set_title('Noise (Uncorr)'); axs[2].axis('off')
    fig.suptitle(f'{eye_source_name} | {win_label} ms')
    fig.savefig(figures_dir / f'covariance_decomposition_{window_idx}_psth.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ── Covariance decomposition — full / FEM view (4-panel) ─────────────────
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    axs[0].imshow(Ctotal,  cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[0].set_title('Total');         axs[0].axis('off')
    axs[1].imshow(Cfem,    cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[1].set_title('FEM');           axs[1].axis('off')
    axs[2].imshow(Cpsth,   cmap=cmap, vmin=-v/2, vmax=v/2, interpolation='nearest'); axs[2].set_title('PSTH');          axs[2].axis('off')
    axs[3].imshow(CnoiseC, cmap=cmap, vmin=-v,   vmax=v,   interpolation='nearest'); axs[3].set_title('Noise (Corr)');  axs[3].axis('off')
    fig.suptitle(f'{eye_source_name} | {win_label} ms')
    fig.savefig(figures_dir / f'covariance_decomposition_{window_idx}_full.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ── FEM fraction histogram ────────────────────────────────────────────────
    # Match the older test_rowley3 plotting convention: compute alpha from the
    # PSD-projected PSTH and Intercept diagonals, then plot the FEM fraction as
    # (1 - alpha). This keeps the histogram aligned with the PSD-adjusted panels
    # above while using the bounded alpha definition from the earlier script.
    denom = np.diag(Crate)
    alpha = np.divide(
        np.diag(Cpsth),
        denom,
        out=np.full(len(denom), np.nan, dtype=np.float64),
        where=denom > 0,
    )
    fem_fraction = 1.0 - alpha
    fem_fraction = fem_fraction[np.isfinite(fem_fraction)]
    frac_fem_summary.append({
        'subdir_name': subdir_name,
        'eye_source_name': eye_source_name,
        'fem_fraction': fem_fraction.copy(),
    })

    fig, ax = plt.subplots()
    if fem_fraction.size:
        hist_bins = get_hist_bins(fem_fraction)
        ax.hist(fem_fraction, bins=hist_bins, color='steelblue', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No finite FEM fractions', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Frac. FEM (1 − α)')
    ax.set_ylabel('Count')
    ax.set_title(f'FEM fraction | {session_label} | {eye_source_name}')
    fig.savefig(figures_dir / 'frac_fem_hist.pdf')
    plt.close(fig)

    # ── Mean rate vs noise variance ───────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(MeanRates, np.diag(CnoiseU), '.', ms=4)
    axs[0].plot(axs[0].get_xlim(), axs[0].get_xlim(), 'k', lw=0.5)
    axs[0].set_xlabel('Mean Rate')
    axs[0].set_ylabel('Variance')
    axs[0].set_title(f'Noise Var (uncorr) | {eye_source_name}')
    axs[1].plot(MeanRates, np.diag(CnoiseC), '.', ms=4)
    axs[1].plot(axs[1].get_xlim(), axs[1].get_xlim(), 'k', lw=0.5)
    axs[1].set_xlabel('Mean Rate')
    axs[1].set_ylabel('Variance')
    axs[1].set_title(f'Noise Var (corr) | {eye_source_name}')
    fig.tight_layout()
    fig.savefig(figures_dir / 'meanrate_vs_variance.pdf', bbox_inches='tight')
    plt.close(fig)

    # ── Multi-page PDF: unit rasters + inspect_neuron_pair ───────────────────
    unit_pdf_path = figures_dir / f'unit_rasters_{session_label}.pdf'
    with PdfPages(unit_pdf_path) as pdf:
        for sel_i, cid in enumerate(cids_used):
            unit_robs   = robs_mc[:, :, neuron_mask[sel_i]]
            ind_u       = np.argsort(dur_mc)
            unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
            trial_idx, t_idx = np.where(unit_sorted > 0)

            fig, axs_u = plt.subplots(1, 2, figsize=(10, 4))
            axs_u[0].set_title(f'Neuron {cid} | {eye_source_name}')
            plot_raster(time_full[t_idx], trial_idx, height=1, ax=axs_u[0])
            axs_u[0].axvline(0, color='r', lw=0.8)
            axs_u[0].set_xlim(time_full[0], time_full[-1])
            axs_u[0].set_xlabel('Time (s)')
            axs_u[0].set_ylabel('Trial')
            analyzer.inspect_neuron_pair(sel_i, sel_i, win_label, ax=axs_u[1])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"  Saved {unit_pdf_path}")

    # ── Multi-page PDF: trial rasters ────────────────────────────────────────
    trial_pdf_path = figures_dir / f'trial_rasters_{session_label}.pdf'
    with PdfPages(trial_pdf_path) as pdf:
        for itrial in range(robs_mc.shape[0]):
            robs_it = np.nan_to_num(robs_mc[itrial][:, neuron_mask], nan=0.0)  # (n_time, n_sel)
            t_idx, u_idx = np.where(robs_it > 0)
            fig, ax = plt.subplots()
            plot_raster(time_full[t_idx], u_idx, height=1, ax=ax)
            ax.axis('off')
            ax2 = ax.twinx()
            ax2.plot(time_full, eyepos_mc[itrial, :, 0], '.r', ms=1)
            ax2.plot(time_full, eyepos_mc[itrial, :, 1], '.g', ms=1)
            ax2.set_ylim(-10, 10)
            ax.set_title(f'Trial {itrial} | {eye_source_name}')
            ax.set_xlim(time_full[0], time_full[-1])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"  Saved {trial_pdf_path}")

    # ── Pickle output ─────────────────────────────────────────────────────────
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
        },
    }
    pkl_path = figures_dir / f'mcfarland_fixrsvp_{session_label}_{eye_source}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"  Saved {pkl_path}")

if frac_fem_summary:
    n_panels = len(frac_fem_summary)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    axs_flat = axs.ravel()

    for ax, summary in zip(axs_flat, frac_fem_summary):
        fem_fraction = summary['fem_fraction']
        if fem_fraction.size:
            hist_bins = get_hist_bins(fem_fraction)
            ax.hist(fem_fraction, bins=hist_bins, color='steelblue', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'No finite FEM fractions', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{summary['subdir_name']} | {summary['eye_source_name']}")
        ax.set_xlabel('Frac. FEM (1 − α)')
        ax.set_ylabel('Count')

    for ax in axs_flat[n_panels:]:
        ax.axis('off')

    fig.suptitle(f'FEM fraction summary | {session_label}')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    summary_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.pdf'
    fig.savefig(summary_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {summary_path}")

print("\nAll conditions complete.")

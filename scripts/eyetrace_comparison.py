import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
from VisionCore.paths import FIGURES_DIR


legacy_processed_root = Path('/mnt/ssd2/RowleyMarmoV1V2/processed_mvp')
fix_name = 'fixrsvp.dset'

default_subject = 'Luke'
default_date = '2026-03-01'
default_primary_eye = 'right'
default_dataset_dir = Path('datasets_gaussian') / 'right_eye'

fixation_radius_deg = 1.5
valid_time_bins = 240
min_fix_dur_bins = 20
dt = 1 / 240.0
deg_trace_ylim = (-1.5, 1.5)
ecc_ylim = (0.0, 2.0)


def _parse_cli_args(argv):
    parsed = {}
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token.startswith('--'):
            if idx + 1 >= len(argv):
                raise ValueError(f'Missing value for {token}')
            parsed[token[2:]] = argv[idx + 1]
            idx += 2
        else:
            idx += 1
    return parsed


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


def _nearest_resample_bool(sample_times, values, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    values = _as_bool_1d(values, len(sample_times))
    target_times = np.asarray(target_times, dtype=np.float64)
    right_idx = np.searchsorted(sample_times, target_times, side='left')
    right_idx = np.clip(right_idx, 0, len(sample_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(sample_times) - 1)
    choose_left = (
        np.abs(target_times - sample_times[left_idx])
        <= np.abs(sample_times[right_idx] - target_times)
    )
    return values[np.where(choose_left, left_idx, right_idx)]


def _interp_xy(sample_times, xy, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float64)
    return np.column_stack([
        np.interp(target_times, sample_times, xy[:, 0]),
        np.interp(target_times, sample_times, xy[:, 1]),
    ]).astype(np.float32)


def _fit_affine_xy(source_xy, target_xy):
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape[0] < 20:
        return None
    design = np.column_stack([source_xy, np.ones(source_xy.shape[0], dtype=np.float64)])
    coeff_x, _, _, _ = np.linalg.lstsq(design, target_xy[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(design, target_xy[:, 1], rcond=None)
    return coeff_x.astype(np.float32), coeff_y.astype(np.float32)


def _apply_affine_xy(xy, coeff_x, coeff_y):
    xy = np.asarray(xy, dtype=np.float64)
    design = np.column_stack([xy, np.ones(xy.shape[0], dtype=np.float64)])
    pred_x = design @ np.asarray(coeff_x, dtype=np.float64)
    pred_y = design @ np.asarray(coeff_y, dtype=np.float64)
    return np.column_stack([pred_x, pred_y]).astype(np.float32)


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
            print(f"Skipping {eye} pupil import: {exc}")
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
            continue
        dset[pupil_key] = _interp_xy(sample_times[valid_samples], pupil_xy[valid_samples], t_bins)
        dset[valid_key] = _nearest_resample_bool(sample_times, pupil_valid, t_bins)
        print(f"Loaded calibrated {pupil_key}")


def add_affine_transformed_pupil_traces(dset):
    keys = set(dset.keys())
    affine_info = {}
    for eye in ('left', 'right'):
        pupil_key = f'pupil_{eye}'
        pupil_valid_key = f'pupil_valid_{eye}'
        dpi_key = f'eyepos_{eye}'
        dpi_valid_key = f'dpi_valid_{eye}'
        if pupil_key not in keys or dpi_key not in keys:
            continue

        raw_key = f'{pupil_key}_img'
        if raw_key not in keys:
            dset[raw_key] = np.array(_get_required(dset, pupil_key), copy=True)
            keys.add(raw_key)

        pupil_xy = _ensure_2d_eyepos(_get_required(dset, raw_key).astype(np.float32))
        dpi_xy = _ensure_2d_eyepos(_get_required(dset, dpi_key).astype(np.float32))
        pupil_valid = _as_bool_1d(_get_optional(dset, pupil_valid_key, np.ones(len(pupil_xy), dtype=bool)), len(pupil_xy))
        dpi_valid = _as_bool_1d(_get_optional(dset, dpi_valid_key, np.ones(len(dpi_xy), dtype=bool)), len(dpi_xy))
        joint_valid = pupil_valid & dpi_valid & np.all(np.isfinite(pupil_xy), axis=1) & np.all(np.isfinite(dpi_xy), axis=1)

        fit = _fit_affine_xy(pupil_xy[joint_valid], dpi_xy[joint_valid])
        if fit is None:
            print(f"Skipping affine pupil transform for {eye} eye: insufficient overlapping valid bins ({joint_valid.sum()})")
            continue

        coeff_x, coeff_y = fit
        dset[pupil_key] = _apply_affine_xy(pupil_xy, coeff_x, coeff_y)
        affine_info[eye] = {
            'joint_valid_bins': int(joint_valid.sum()),
            'coeff_x': coeff_x,
            'coeff_y': coeff_y,
        }
        print(f"Applied affine pupil->{eye}-dpi transform using {joint_valid.sum()} bins")

    return affine_info


def get_enabled_eye_configs(dset):
    keys = set(dset.keys())
    enabled_configs = [('cyclopean-dpi', 'default')]
    has_left = any(k in keys for k in ['eyepos_left', 'eyepos_dpi_left'])
    has_right = any(k in keys for k in ['eyepos_right', 'eyepos_dpi_right'])
    has_left_pupil = any(k in keys for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil'])
    has_right_pupil = any(k in keys for k in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil'])
    has_left_pupil_raw = 'pupil_left_img' in keys
    has_right_pupil_raw = 'pupil_right_img' in keys
    if has_left and has_right:
        enabled_configs.append(('binocular', 'binocular_diff'))
    if has_left:
        enabled_configs.append(('left-dpi', 'left'))
    if has_left_pupil:
        enabled_configs.append(('left-pupil', 'pupil_left'))
    if has_left_pupil_raw:
        enabled_configs.append(('left-pupil-img', 'pupil_left_img'))
    if has_right:
        enabled_configs.append(('right-dpi', 'right'))
    if has_right_pupil:
        enabled_configs.append(('right-pupil', 'pupil_right'))
    if has_right_pupil_raw:
        enabled_configs.append(('right-pupil-img', 'pupil_right_img'))
    return enabled_configs


def get_eye_trace_and_valid(dset, eye_source='default'):
    keys = set(dset.keys())
    eyepos_default = _get_required(dset, 'eyepos')
    n = eyepos_default.shape[0]
    dpi_valid_default = _get_optional(dset, 'dpi_valid', np.ones(n, dtype=bool))
    dpi_valid_default = _as_bool_1d(dpi_valid_default, n)

    if eye_source in ('default', 'cyclopean'):
        return eyepos_default, dpi_valid_default, 'eyepos'

    if eye_source == 'left':
        for key in ['eyepos_left', 'eyepos_dpi_left']:
            if key in keys:
                eye = _get_required(dset, key)
                valid = _get_optional(dset, 'dpi_valid_left', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), key
        raise KeyError('No eyepos_left / eyepos_dpi_left found.')

    if eye_source == 'right':
        for key in ['eyepos_right', 'eyepos_dpi_right']:
            if key in keys:
                eye = _get_required(dset, key)
                valid = _get_optional(dset, 'dpi_valid_right', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), key
        raise KeyError('No eyepos_right / eyepos_dpi_right found.')

    if eye_source == 'binocular_diff':
        left = right = None
        for key in ['eyepos_left', 'eyepos_dpi_left']:
            if key in keys:
                left = _get_required(dset, key)
                break
        for key in ['eyepos_right', 'eyepos_dpi_right']:
            if key in keys:
                right = _get_required(dset, key)
                break
        if left is None or right is None:
            raise KeyError('No left/right eye traces for binocular_diff.')
        valid_l = _as_bool_1d(_get_optional(dset, 'dpi_valid_left', np.ones(len(left), dtype=bool)), len(left))
        valid_r = _as_bool_1d(_get_optional(dset, 'dpi_valid_right', np.ones(len(right), dtype=bool)), len(right))
        return left - right, (valid_l & valid_r), 'eyepos_left_minus_right'

    if eye_source == 'pupil_left':
        for key in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil']:
            if key in keys:
                eye = _get_required(dset, key)
                valid = _get_optional(dset, 'pupil_valid_left', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), key
        raise KeyError('No left pupil trace found.')

    if eye_source == 'pupil_right':
        for key in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil']:
            if key in keys:
                eye = _get_required(dset, key)
                valid = _get_optional(dset, 'pupil_valid_right', np.ones(len(eye), dtype=bool))
                return eye, _as_bool_1d(valid, len(eye)), key
        raise KeyError('No right pupil trace found.')

    if eye_source == 'pupil_left_img':
        if 'pupil_left_img' in keys:
            eye = _get_required(dset, 'pupil_left_img')
            valid = _get_optional(dset, 'pupil_valid_left', np.ones(len(eye), dtype=bool))
            return eye, _as_bool_1d(valid, len(eye)), 'pupil_left_img'
        raise KeyError('No raw left pupil trace found.')

    if eye_source == 'pupil_right_img':
        if 'pupil_right_img' in keys:
            eye = _get_required(dset, 'pupil_right_img')
            valid = _get_optional(dset, 'pupil_valid_right', np.ones(len(eye), dtype=bool))
            return eye, _as_bool_1d(valid, len(eye)), 'pupil_right_img'
        raise KeyError('No raw right pupil trace found.')

    raise ValueError(f'Unknown eye_source: {eye_source!r}')


def serial_to_trial_aligned(eyepos, dfs, trial_inds, time_inds):
    unique_trials = np.unique(trial_inds)
    n_trials = len(unique_trials)
    n_time = np.max(time_inds).item() + 1
    eyepos_trial = np.nan * np.zeros((n_trials, n_time, 2), dtype=np.float32)
    dfs_trial = np.zeros((n_trials, n_time), dtype=bool)
    dur_trial = np.zeros(n_trials, dtype=int)
    for itrial in range(n_trials):
        idx = np.where(trial_inds == unique_trials[itrial])[0]
        if not len(idx):
            continue
        tt = time_inds[idx]
        valid_tt = _as_bool_1d(dfs[idx], len(idx))
        if valid_tt.any():
            eyepos_trial[itrial, tt[valid_tt]] = eyepos[idx][valid_tt]
        dfs_trial[itrial, tt] = valid_tt
        dur_trial[itrial] = valid_tt.sum()
    return eyepos_trial, dfs_trial, dur_trial


def _robust_limits(values, pad=0.05):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lo = np.quantile(values, 0.01)
    hi = np.quantile(values, 0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(values)) if values.size else 0.0
        return (center - 1.0, center + 1.0)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    a = a[ok]
    b = b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(ok.sum())


def _fit_affine(source_xy, target_xy):
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape[0] < 3:
        return None
    design = np.column_stack([source_xy, np.ones(source_xy.shape[0], dtype=np.float64)])
    coeff_x, _, _, _ = np.linalg.lstsq(design, target_xy[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(design, target_xy[:, 1], rcond=None)
    pred_x = design @ coeff_x
    pred_y = design @ coeff_y
    pred = np.column_stack([pred_x, pred_y])
    resid = target_xy - pred
    r2_x = 1.0 - (np.sum((target_xy[:, 0] - pred_x) ** 2) / np.sum((target_xy[:, 0] - np.mean(target_xy[:, 0])) ** 2))
    r2_y = 1.0 - (np.sum((target_xy[:, 1] - pred_y) ** 2) / np.sum((target_xy[:, 1] - np.mean(target_xy[:, 1])) ** 2))
    return {
        'coeff_x': coeff_x,
        'coeff_y': coeff_y,
        'pred': pred,
        'resid': resid,
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'rmse': float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
    }


def _speed(trace):
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return np.array([], dtype=np.float64)
    delta = np.diff(trace, axis=0)
    return np.sqrt(np.sum(delta ** 2, axis=1))


def _eye_label(name):
    mapping = {
        'eyepos': 'cyclopean',
        'eyepos_left_minus_right': 'vergence (L-R)',
        'pupil_left': 'left pupil (affine -> deg)',
        'pupil_right': 'right pupil (affine -> deg)',
        'pupil_left_img': 'left pupil raw image',
        'pupil_right_img': 'right pupil raw image',
    }
    return mapping.get(name, name)


def _is_degree_trace(subdir_name):
    return subdir_name in {'cyclopean-dpi', 'binocular', 'left-dpi', 'right-dpi', 'left-pupil', 'right-pupil'}


def _plot_session_overview(pdf, time_seconds, cyclopean_trace, dpi_valid_default, ecc, ecc_mask, fixation_center, session_label):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    sample_slice = slice(0, min(len(time_seconds), 6000))
    axs[0].plot(time_seconds[sample_slice], cyclopean_trace[sample_slice, 0], label='x', lw=1.0)
    axs[0].plot(time_seconds[sample_slice], cyclopean_trace[sample_slice, 1], label='y', lw=1.0)
    axs[0].set_ylabel('Cyclopean position')
    axs[0].set_ylim(*deg_trace_ylim)
    axs[0].set_title(f'Eyetrace overview | {session_label}')
    axs[0].legend(frameon=False)

    axs[1].plot(time_seconds[sample_slice], dpi_valid_default[sample_slice].astype(float), color='black', lw=0.8)
    axs[1].plot(time_seconds[sample_slice], ecc_mask[sample_slice].astype(float), color='darkorange', lw=0.8)
    axs[1].set_ylabel('Validity mask')
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['off', 'on'])
    axs[1].legend(['dpi_valid', 'ecc_mask'], frameon=False)

    axs[2].plot(time_seconds[sample_slice], ecc[sample_slice], color='steelblue', lw=1.0)
    axs[2].axhline(fixation_radius_deg, color='r', linestyle='--', lw=1.0)
    axs[2].set_ylim(*ecc_ylim)
    axs[2].set_ylabel('Eccentricity (deg)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title(
        f'Fixation center=({fixation_center[0]:.3f}, {fixation_center[1]:.3f}) | '
        f'valid={int(dpi_valid_default.sum())} ecc-pass={int(ecc_mask.sum())}'
    )

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _plot_trace_debug_page(pdf, subdir_name, display_eye_name, serial_trace, serial_valid, trial_trace, time_seconds):
    valid_trace = serial_trace[serial_valid]
    xlim = _robust_limits(valid_trace[:, 0] if valid_trace.size else np.array([]))
    ylim = _robust_limits(valid_trace[:, 1] if valid_trace.size else np.array([]))
    speed = _speed(valid_trace)
    use_degree_limits = _is_degree_trace(subdir_name)
    if use_degree_limits:
        xlim = deg_trace_ylim
        ylim = deg_trace_ylim

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    sample_slice = slice(0, min(len(time_seconds), 6000))
    axs[0, 0].plot(time_seconds[sample_slice], serial_trace[sample_slice, 0], lw=0.9, label='x')
    axs[0, 0].plot(time_seconds[sample_slice], serial_trace[sample_slice, 1], lw=0.9, label='y')
    axs[0, 0].plot(time_seconds[sample_slice], serial_valid[sample_slice].astype(float), lw=0.8, color='black', alpha=0.6)
    axs[0, 0].set_title('Serial trace preview')
    axs[0, 0].set_xlabel('Time (s)')
    if use_degree_limits:
        axs[0, 0].set_ylim(*deg_trace_ylim)
    axs[0, 0].legend(frameon=False, fontsize=8)

    if valid_trace.size:
        axs[0, 1].hexbin(valid_trace[:, 0], valid_trace[:, 1], gridsize=70, cmap='viridis', mincnt=1)
    axs[0, 1].set_title('XY occupancy')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_xlim(*xlim)
    axs[0, 1].set_ylim(*ylim)

    if valid_trace.size:
        axs[0, 2].hist(valid_trace[:, 0], bins=60, alpha=0.65, label='x')
        axs[0, 2].hist(valid_trace[:, 1], bins=60, alpha=0.65, label='y')
    axs[0, 2].set_title('Axis histograms')
    axs[0, 2].set_xlabel('Position')
    axs[0, 2].legend(frameon=False, fontsize=8)

    if speed.size:
        axs[1, 0].hist(speed, bins=60, color='steelblue', alpha=0.8)
    axs[1, 0].set_title('Step-size distribution')
    axs[1, 0].set_xlabel('Frame-to-frame displacement')

    trial_x = trial_trace[:, :, 0]
    trial_y = trial_trace[:, :, 1]
    axs[1, 1].imshow(
        trial_x,
        aspect='auto',
        origin='lower',
        interpolation='none',
        cmap='coolwarm',
        extent=(0, trial_x.shape[1] * dt, 0, trial_x.shape[0]),
        vmin=xlim[0],
        vmax=xlim[1],
    )
    axs[1, 1].set_title('Trial-aligned x heatmap')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Trial')

    axs[1, 2].imshow(
        trial_y,
        aspect='auto',
        origin='lower',
        interpolation='none',
        cmap='coolwarm',
        extent=(0, trial_y.shape[1] * dt, 0, trial_y.shape[0]),
        vmin=ylim[0],
        vmax=ylim[1],
    )
    axs[1, 2].set_title('Trial-aligned y heatmap')
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Trial')

    fig.suptitle(f'{subdir_name} | {display_eye_name}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _plot_per_trial_compare_page(pdf, source_a, source_b, trial_trace_a, trial_trace_b):
    n_trials = min(trial_trace_a.shape[0], trial_trace_b.shape[0])
    n_time = min(trial_trace_a.shape[1], trial_trace_b.shape[1])
    trial_trace_a = trial_trace_a[:n_trials, :n_time]
    trial_trace_b = trial_trace_b[:n_trials, :n_time]

    mean_a = np.nanmean(trial_trace_a, axis=1)
    mean_b = np.nanmean(trial_trace_b, axis=1)
    var_a = np.nanvar(trial_trace_a, axis=1)
    var_b = np.nanvar(trial_trace_b, axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    axs[0, 0].scatter(mean_a[:, 0], mean_b[:, 0], s=16, alpha=0.7)
    axs[0, 0].set_title(f'Trial mean x: {source_a} vs {source_b}')
    axs[0, 0].set_xlabel(source_a)
    axs[0, 0].set_ylabel(source_b)

    axs[0, 1].scatter(mean_a[:, 1], mean_b[:, 1], s=16, alpha=0.7)
    axs[0, 1].set_title(f'Trial mean y: {source_a} vs {source_b}')
    axs[0, 1].set_xlabel(source_a)
    axs[0, 1].set_ylabel(source_b)

    axs[1, 0].scatter(var_a[:, 0], var_b[:, 0], s=16, alpha=0.7)
    axs[1, 0].set_title(f'Trial variance x: {source_a} vs {source_b}')
    axs[1, 0].set_xlabel(source_a)
    axs[1, 0].set_ylabel(source_b)

    axs[1, 1].scatter(var_a[:, 1], var_b[:, 1], s=16, alpha=0.7)
    axs[1, 1].set_title(f'Trial variance y: {source_a} vs {source_b}')
    axs[1, 1].set_xlabel(source_a)
    axs[1, 1].set_ylabel(source_b)

    fig.suptitle(f'Per-trial mean/variance comparison | {source_a} vs {source_b}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return {
        'mean_x_r': _pearson(mean_a[:, 0], mean_b[:, 0])[0],
        'mean_y_r': _pearson(mean_a[:, 1], mean_b[:, 1])[0],
        'var_x_r': _pearson(var_a[:, 0], var_b[:, 0])[0],
        'var_y_r': _pearson(var_a[:, 1], var_b[:, 1])[0],
        'n_trials': int(n_trials),
    }


def _plot_affine_fit_page(pdf, source_name, target_name, source_xy, target_xy, affine_fit):
    pred = affine_fit['pred']
    resid = affine_fit['resid']
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    axs[0, 0].hexbin(target_xy[:, 0], pred[:, 0], gridsize=80, cmap='viridis', mincnt=1)
    axs[0, 0].set_title(f'Affine fit x: predicted {target_name}')
    axs[0, 0].set_xlabel(f'{target_name} actual x')
    axs[0, 0].set_ylabel(f'{target_name} predicted x')

    axs[0, 1].hexbin(target_xy[:, 1], pred[:, 1], gridsize=80, cmap='viridis', mincnt=1)
    axs[0, 1].set_title(f'Affine fit y: predicted {target_name}')
    axs[0, 1].set_xlabel(f'{target_name} actual y')
    axs[0, 1].set_ylabel(f'{target_name} predicted y')

    axs[1, 0].hist(resid[:, 0], bins=60, alpha=0.7, label='dx')
    axs[1, 0].hist(resid[:, 1], bins=60, alpha=0.7, label='dy')
    axs[1, 0].set_title('Affine residuals')
    axs[1, 0].set_xlabel('Residual')
    axs[1, 0].legend(frameon=False, fontsize=8)

    text = (
        f'{source_name} -> {target_name}\n\n'
        f'x = {affine_fit["coeff_x"][0]:.4f} * sx + {affine_fit["coeff_x"][1]:.4f} * sy + {affine_fit["coeff_x"][2]:.4f}\n'
        f'y = {affine_fit["coeff_y"][0]:.4f} * sx + {affine_fit["coeff_y"][1]:.4f} * sy + {affine_fit["coeff_y"][2]:.4f}\n\n'
        f'R2_x = {affine_fit["r2_x"]:.4f}\n'
        f'R2_y = {affine_fit["r2_y"]:.4f}\n'
        f'RMSE = {affine_fit["rmse"]:.4f}'
    )
    axs[1, 1].axis('off')
    axs[1, 1].text(0.03, 0.95, text, va='top', ha='left', family='monospace')

    fig.suptitle(f'Affine fit diagnostic | {source_name} to {target_name}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _plot_pairwise_debug_page(pdf, source_a, source_b, trace_a, valid_a, trace_b, valid_b):
    joint = valid_a & valid_b & np.all(np.isfinite(trace_a), axis=1) & np.all(np.isfinite(trace_b), axis=1)
    xy_a = trace_a[joint]
    xy_b = trace_b[joint]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    if len(xy_a):
        axs[0, 0].hexbin(xy_a[:, 0], xy_b[:, 0], gridsize=80, cmap='magma', mincnt=1)
    axs[0, 0].set_title(f'x: {source_a} vs {source_b}')
    axs[0, 0].set_xlabel(source_a)
    axs[0, 0].set_ylabel(source_b)

    if len(xy_a):
        axs[0, 1].hexbin(xy_a[:, 1], xy_b[:, 1], gridsize=80, cmap='magma', mincnt=1)
    axs[0, 1].set_title(f'y: {source_a} vs {source_b}')
    axs[0, 1].set_xlabel(source_a)
    axs[0, 1].set_ylabel(source_b)

    ra_x, n_x = _pearson(xy_a[:, 0] if len(xy_a) else np.array([]), xy_b[:, 0] if len(xy_b) else np.array([]))
    ra_y, n_y = _pearson(xy_a[:, 1] if len(xy_a) else np.array([]), xy_b[:, 1] if len(xy_b) else np.array([]))
    ra_mag, n_mag = _pearson(
        np.linalg.norm(xy_a, axis=1) if len(xy_a) else np.array([]),
        np.linalg.norm(xy_b, axis=1) if len(xy_b) else np.array([]),
    )

    text = (
        f'joint valid bins: {int(joint.sum())}\n'
        f'r_x = {ra_x:.4f}  (n={n_x})\n'
        f'r_y = {ra_y:.4f}  (n={n_y})\n'
        f'r_mag = {ra_mag:.4f}  (n={n_mag})'
    )
    axs[1, 0].axis('off')
    axs[1, 0].text(0.03, 0.95, text, va='top', ha='left', family='monospace')

    if len(xy_a):
        diff = xy_b - xy_a
        axs[1, 1].hist(diff[:, 0], bins=60, alpha=0.65, label='dx')
        axs[1, 1].hist(diff[:, 1], bins=60, alpha=0.65, label='dy')
    axs[1, 1].set_title(f'{source_b} - {source_a}')
    axs[1, 1].set_xlabel('Difference')
    axs[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(f'Pairwise trace comparison | {source_a} vs {source_b}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return {
        'joint_valid_bins': int(joint.sum()),
        'r_x': ra_x,
        'r_y': ra_y,
        'r_mag': ra_mag,
        'joint_xy_a': xy_a,
        'joint_xy_b': xy_b,
    }


def main():
    cli_args = _parse_cli_args(sys.argv[1:])
    subject = cli_args.get('subject', os.environ.get('ROWLEY_SUBJECT', default_subject))
    date = cli_args.get('date', os.environ.get('ROWLEY_DATE', default_date))
    primary_eye = cli_args.get('primary-eye', os.environ.get('ROWLEY_PRIMARY_EYE', default_primary_eye))
    dataset_dir = Path(cli_args.get('dataset-dir', os.environ.get('ROWLEY_DATASET_DIR', str(default_dataset_dir))))

    sess = get_session(subject, date)
    aux_processed_path = Path(sess.processed_path)
    sess.processed_path = legacy_processed_root / f'{subject}_{date}'

    fix_path = Path(sess.processed_path) / dataset_dir / fix_name
    print(f'Loading fixrsvp from: {fix_path}')
    assert fix_path.exists(), f'fixrsvp.dset not found at: {fix_path}'

    dset_fix = DictDataset.load(fix_path)
    add_calibrated_pupil_traces(dset_fix, aux_processed_path)
    pupil_affine_info = add_affine_transformed_pupil_traces(dset_fix)
    eye_configs = get_enabled_eye_configs(dset_fix)

    eyepos_default_raw = _get_required(dset_fix, 'eyepos')
    n_bins_serial = eyepos_default_raw.shape[0]
    dpi_valid_default = _get_optional(dset_fix, 'dpi_valid', np.ones(n_bins_serial, dtype=bool))
    dpi_valid_default = _as_bool_1d(dpi_valid_default, n_bins_serial)
    eyepos_cyclopean = _ensure_2d_eyepos(eyepos_default_raw.astype(np.float32))
    valid_ep = eyepos_cyclopean[dpi_valid_default]
    fixation_center = np.median(valid_ep, axis=0) if len(valid_ep) > 0 else np.zeros(2, dtype=np.float32)
    ecc = np.hypot(
        eyepos_cyclopean[:, 0] - fixation_center[0],
        eyepos_cyclopean[:, 1] - fixation_center[1],
    )
    ecc_mask = ecc <= fixation_radius_deg

    trial_inds_s = _get_required(dset_fix, 'trial_inds').astype(int)
    time_inds_s = _get_required(dset_fix, 'psth_inds').astype(int)
    time_bins_serial = _get_required(dset_fix, 't_bins').astype(np.float64)
    time_seconds = time_bins_serial - time_bins_serial[0]
    session_label = f'{subject}_{date}'
    output_dir = FIGURES_DIR / 'mcfarland' / f'{session_label}_eyetrace_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f'eyetrace_comparison_{session_label}.pdf'
    txt_path = output_dir / f'eyetrace_comparison_{session_label}.txt'

    trace_data = {}
    pairwise_rows = []
    trialwise_rows = []
    affine_rows = []

    with PdfPages(pdf_path) as pdf:
        _plot_session_overview(
            pdf,
            time_seconds,
            eyepos_cyclopean,
            dpi_valid_default,
            ecc,
            ecc_mask,
            fixation_center,
            session_label,
        )

        for subdir_name, eye_source in eye_configs:
            eyepos_serial, dfs_serial, eye_source_name = get_eye_trace_and_valid(dset_fix, eye_source=eye_source)
            eyepos_serial = _ensure_2d_eyepos(eyepos_serial)
            dfs_serial = _as_bool_1d(dfs_serial, n_bins_serial)
            dfs_serial = dfs_serial & ecc_mask

            eyepos_trial, dfs_trial, dur_trial = serial_to_trial_aligned(
                eyepos_serial, dfs_serial, trial_inds_s, time_inds_s
            )
            good_trials = dur_trial > min_fix_dur_bins
            trial_trace = eyepos_trial[good_trials][:, :min(valid_time_bins, eyepos_trial.shape[1])]

            display_eye_name = _eye_label(eye_source_name)
            _plot_trace_debug_page(
                pdf,
                subdir_name,
                display_eye_name,
                eyepos_serial,
                dfs_serial,
                trial_trace,
                time_seconds,
            )

            trace_data[subdir_name] = {
                'display_eye_name': display_eye_name,
                'eye_source_name': eye_source_name,
                'serial_trace': eyepos_serial,
                'serial_valid': dfs_serial,
                'trial_trace': trial_trace,
                'good_trials': int(good_trials.sum()),
            }

        keys = list(trace_data.keys())
        for idx_a in range(len(keys)):
            for idx_b in range(idx_a + 1, len(keys)):
                key_a = keys[idx_a]
                key_b = keys[idx_b]
                row = _plot_pairwise_debug_page(
                    pdf,
                    key_a,
                    key_b,
                    trace_data[key_a]['serial_trace'],
                    trace_data[key_a]['serial_valid'],
                    trace_data[key_b]['serial_trace'],
                    trace_data[key_b]['serial_valid'],
                )
                row['source_a'] = key_a
                row['source_b'] = key_b
                pairwise_rows.append(row)

                trial_row = _plot_per_trial_compare_page(
                    pdf,
                    key_a,
                    key_b,
                    trace_data[key_a]['trial_trace'],
                    trace_data[key_b]['trial_trace'],
                )
                trial_row['source_a'] = key_a
                trial_row['source_b'] = key_b
                trialwise_rows.append(trial_row)

                is_pupil_pair = ('pupil' in key_a and 'dpi' in key_b) or ('pupil' in key_b and 'dpi' in key_a) or ('pupil' in key_a and 'cyclopean' in key_b) or ('pupil' in key_b and 'cyclopean' in key_a)
                if is_pupil_pair and row['joint_valid_bins'] >= 20:
                    fit_ab = _fit_affine(row['joint_xy_a'], row['joint_xy_b'])
                    if fit_ab is not None:
                        _plot_affine_fit_page(pdf, key_a, key_b, row['joint_xy_a'], row['joint_xy_b'], fit_ab)
                        affine_rows.append({
                            'source_name': key_a,
                            'target_name': key_b,
                            'r2_x': fit_ab['r2_x'],
                            'r2_y': fit_ab['r2_y'],
                            'rmse': fit_ab['rmse'],
                        })

                    fit_ba = _fit_affine(row['joint_xy_b'], row['joint_xy_a'])
                    if fit_ba is not None:
                        _plot_affine_fit_page(pdf, key_b, key_a, row['joint_xy_b'], row['joint_xy_a'], fit_ba)
                        affine_rows.append({
                            'source_name': key_b,
                            'target_name': key_a,
                            'r2_x': fit_ba['r2_x'],
                            'r2_y': fit_ba['r2_y'],
                            'rmse': fit_ba['rmse'],
                        })

    with open(txt_path, 'w') as f:
        f.write(f'Eyetrace comparison summary\n')
        f.write(f'Session: {session_label}\n')
        f.write(f'Dataset path: {fix_path}\n')
        f.write(f'Fixation center: ({fixation_center[0]:.4f}, {fixation_center[1]:.4f})\n')
        f.write(f'dpi_valid bins: {int(dpi_valid_default.sum())} / {len(dpi_valid_default)}\n')
        f.write(f'ecc-pass bins: {int(ecc_mask.sum())} / {len(ecc_mask)}\n\n')

        f.write('Pupil affine transforms\n')
        for eye, info in pupil_affine_info.items():
            coeff_x = np.asarray(info['coeff_x'])
            coeff_y = np.asarray(info['coeff_y'])
            f.write(f'  {eye}: joint_valid={info["joint_valid_bins"]}\n')
            f.write(f'    x = {coeff_x[0]:.6f} * sx + {coeff_x[1]:.6f} * sy + {coeff_x[2]:.6f}\n')
            f.write(f'    y = {coeff_y[0]:.6f} * sx + {coeff_y[1]:.6f} * sy + {coeff_y[2]:.6f}\n')
        f.write('\n')

        f.write('Per-source summary\n')
        for key, info in trace_data.items():
            valid = info['serial_valid'] & np.all(np.isfinite(info['serial_trace']), axis=1)
            trace = info['serial_trace'][valid]
            speed = _speed(trace)
            f.write(f'  {key} ({info["display_eye_name"]})\n')
            f.write(f'    valid bins: {int(valid.sum())}\n')
            f.write(f'    good trials: {info["good_trials"]}\n')
            if trace.size:
                mean_xy = np.mean(trace, axis=0)
                std_xy = np.std(trace, axis=0)
                f.write(f'    mean xy: ({mean_xy[0]:.4f}, {mean_xy[1]:.4f})\n')
                f.write(f'    std xy: ({std_xy[0]:.4f}, {std_xy[1]:.4f})\n')
            if speed.size:
                f.write(f'    median step size: {np.median(speed):.4f}\n')
                f.write(f'    q95 step size: {np.quantile(speed, 0.95):.4f}\n')
            f.write('\n')

        f.write('Pairwise correlations\n')
        for row in pairwise_rows:
            f.write(
                f'  {row["source_a"]} vs {row["source_b"]}: '
                f'joint={row["joint_valid_bins"]} '
                f'r_x={row["r_x"]:.4f} '
                f'r_y={row["r_y"]:.4f} '
                f'r_mag={row["r_mag"]:.4f}\n'
            )

        f.write('\nPer-trial mean/variance correlations\n')
        for row in trialwise_rows:
            f.write(
                f'  {row["source_a"]} vs {row["source_b"]}: '
                f'n_trials={row["n_trials"]} '
                f'mean_x_r={row["mean_x_r"]:.4f} '
                f'mean_y_r={row["mean_y_r"]:.4f} '
                f'var_x_r={row["var_x_r"]:.4f} '
                f'var_y_r={row["var_y_r"]:.4f}\n'
            )

        f.write('\nAffine fit diagnostics\n')
        for row in affine_rows:
            f.write(
                f'  {row["source_name"]} -> {row["target_name"]}: '
                f'R2_x={row["r2_x"]:.4f} '
                f'R2_y={row["r2_y"]:.4f} '
                f'RMSE={row["rmse"]:.4f}\n'
            )

    print(f'Saved {pdf_path}')
    print(f'Saved {txt_path}')


if __name__ == '__main__':
    main()
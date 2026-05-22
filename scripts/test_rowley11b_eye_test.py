"""
test_rowley11b_eye_test.py - focused pupil vs DPI eye-trace diagnostics

Loads a FixRSVP dataset using the same session-YAML resolution used by
test_rowley11, imports calibrated pupil traces from the DPI calibration CSVs,
fits the affine pupil->DPI transform per eye, and reports per-eye alignment
statistics.

Outputs:
- stdout summary with per-eye correlations and variances
- PDF figure with trace overlays and DPI-vs-pupil scatter plots
- NPZ file with the numeric summary
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
from VisionCore.paths import FIGURES_DIR, VISIONCORE_ROOT


session_yaml_root = VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'sessions'
fix_name = 'fixrsvp.dset'

default_subject = 'Luke'
default_date = '2026-03-02'
default_primary_eye = 'left'

max_scatter_points = 50000
trace_preview_bins = 3000


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


def _load_yaml_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _resolve_session_yaml_path(subject_name, session_date, eye_name, yaml_root, explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.is_absolute() else (yaml_root / path)
    return yaml_root / f'{subject_name}_{session_date}_{eye_name}_V1.yaml'


def _resolve_session_data_root(dataset_directory):
    dataset_directory = Path(dataset_directory)
    for candidate in [dataset_directory, *dataset_directory.parents]:
        if (candidate / 'dpi_calibration').exists() or (candidate / 'dots_calibration').exists():
            return candidate
    return dataset_directory.parent


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
        zeros = np.zeros((len(eyepos), 1), dtype=np.float32)
        return np.concatenate([eyepos, zeros], axis=1)
    return eyepos[:, :2]


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
            print(f'No calibrated pupil CSV for {eye} eye: {csv_path}')
            continue
        pupil_df = pd.read_csv(csv_path, usecols=['t_ephys', 'pupil_i', 'pupil_j', 'pupil_valid'])
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
        keys.update([pupil_key, valid_key])
        print(f'Loaded calibrated {pupil_key}')


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
        joint_valid = (
            pupil_valid
            & dpi_valid
            & np.all(np.isfinite(pupil_xy), axis=1)
            & np.all(np.isfinite(dpi_xy), axis=1)
        )

        fit = _fit_affine_xy(pupil_xy[joint_valid], dpi_xy[joint_valid])
        if fit is None:
            print(f'Skipping affine pupil transform for {eye} eye: insufficient overlapping valid bins ({joint_valid.sum()})')
            continue

        coeff_x, coeff_y = fit
        dset[pupil_key] = _apply_affine_xy(pupil_xy, coeff_x, coeff_y)
        affine_info[eye] = {
            'joint_valid_bins': int(joint_valid.sum()),
            'coeff_x': coeff_x,
            'coeff_y': coeff_y,
        }
        print(f'Applied affine pupil->{eye}-dpi transform using {joint_valid.sum()} bins')

    return affine_info


def _corrcoef_safe(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _variance_safe(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    return float(np.nanvar(x, ddof=0))


def _subsample_indices(n_points, max_points, seed=0):
    if n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def summarize_eye_pair(dset, eye):
    dpi_key = f'eyepos_{eye}'
    pupil_key = f'pupil_{eye}'
    dpi_valid_key = f'dpi_valid_{eye}'
    pupil_valid_key = f'pupil_valid_{eye}'

    keys = set(dset.keys())
    if dpi_key not in keys or pupil_key not in keys:
        return None

    dpi_xy = _ensure_2d_eyepos(_get_required(dset, dpi_key))
    pupil_xy = _ensure_2d_eyepos(_get_required(dset, pupil_key))
    if len(dpi_xy) != len(pupil_xy):
        raise ValueError(f'Length mismatch for {eye} eye: dpi={len(dpi_xy)} pupil={len(pupil_xy)}')

    dpi_valid = _as_bool_1d(_get_optional(dset, dpi_valid_key, np.ones(len(dpi_xy), dtype=bool)), len(dpi_xy))
    pupil_valid = _as_bool_1d(_get_optional(dset, pupil_valid_key, np.ones(len(pupil_xy), dtype=bool)), len(pupil_xy))
    finite_mask = np.all(np.isfinite(dpi_xy), axis=1) & np.all(np.isfinite(pupil_xy), axis=1)
    joint_valid = dpi_valid & pupil_valid & finite_mask

    dpi_valid_xy = dpi_xy[joint_valid]
    pupil_valid_xy = pupil_xy[joint_valid]
    if dpi_valid_xy.shape[0] == 0:
        return {
            'eye': eye,
            'n_total': int(len(dpi_xy)),
            'n_joint_valid': 0,
            'joint_valid_fraction': 0.0,
        }

    delta_xy = pupil_valid_xy - dpi_valid_xy
    radial_dpi = np.linalg.norm(dpi_valid_xy, axis=1)
    radial_pupil = np.linalg.norm(pupil_valid_xy, axis=1)

    return {
        'eye': eye,
        'n_total': int(len(dpi_xy)),
        'n_joint_valid': int(joint_valid.sum()),
        'joint_valid_fraction': float(joint_valid.mean()),
        'corr_x': _corrcoef_safe(dpi_valid_xy[:, 0], pupil_valid_xy[:, 0]),
        'corr_y': _corrcoef_safe(dpi_valid_xy[:, 1], pupil_valid_xy[:, 1]),
        'corr_radius': _corrcoef_safe(radial_dpi, radial_pupil),
        'var_dpi_x': _variance_safe(dpi_valid_xy[:, 0]),
        'var_dpi_y': _variance_safe(dpi_valid_xy[:, 1]),
        'var_pupil_x': _variance_safe(pupil_valid_xy[:, 0]),
        'var_pupil_y': _variance_safe(pupil_valid_xy[:, 1]),
        'var_delta_x': _variance_safe(delta_xy[:, 0]),
        'var_delta_y': _variance_safe(delta_xy[:, 1]),
        'rmse_x': float(np.sqrt(np.mean(delta_xy[:, 0] ** 2))),
        'rmse_y': float(np.sqrt(np.mean(delta_xy[:, 1] ** 2))),
        'mean_delta_x': float(np.mean(delta_xy[:, 0])),
        'mean_delta_y': float(np.mean(delta_xy[:, 1])),
        'dpi_xy': dpi_valid_xy,
        'pupil_xy': pupil_valid_xy,
    }


def _format_float(value):
    if value is None or not np.isfinite(value):
        return 'nan'
    return f'{value:.4f}'


def print_summary(subject, date, summaries):
    print('')
    print(f'Eye-trace diagnostics for {subject}_{date}')
    print('-' * 88)
    header = (
        f"{'eye':<8} {'n_joint':>8} {'frac':>8} {'corr_x':>10} {'corr_y':>10} {'corr_r':>10} "
        f"{'var_dpi_x':>12} {'var_pupil_x':>12} {'var_dpi_y':>12} {'var_pupil_y':>12}"
    )
    print(header)
    print('-' * len(header))
    for summary in summaries:
        print(
            f"{summary['eye']:<8} {summary['n_joint_valid']:>8d} {summary['joint_valid_fraction']:>8.3f} "
            f"{_format_float(summary.get('corr_x')):>10} {_format_float(summary.get('corr_y')):>10} "
            f"{_format_float(summary.get('corr_radius')):>10} {_format_float(summary.get('var_dpi_x')):>12} "
            f"{_format_float(summary.get('var_pupil_x')):>12} {_format_float(summary.get('var_dpi_y')):>12} "
            f"{_format_float(summary.get('var_pupil_y')):>12}"
        )


def save_diagnostic_figure(output_path, subject, date, summaries, affine_info):
    n_eyes = len(summaries)
    fig, axes = plt.subplots(n_eyes, 4, figsize=(18, 5 * n_eyes), squeeze=False)
    fig.suptitle(f'Pupil vs DPI diagnostics: {subject}_{date}', fontsize=14)

    for row_idx, summary in enumerate(summaries):
        eye = summary['eye']
        dpi_xy = summary['dpi_xy']
        pupil_xy = summary['pupil_xy']
        sample_idx = _subsample_indices(len(dpi_xy), max_scatter_points, seed=row_idx)
        dpi_sample = dpi_xy[sample_idx]
        pupil_sample = pupil_xy[sample_idx]
        preview_idx = slice(0, min(trace_preview_bins, len(dpi_xy)))

        ax = axes[row_idx, 0]
        ax.plot(dpi_xy[preview_idx, 0], label='dpi x', lw=1.0)
        ax.plot(pupil_xy[preview_idx, 0], label='pupil x', lw=1.0, alpha=0.8)
        ax.plot(dpi_xy[preview_idx, 1], label='dpi y', lw=1.0)
        ax.plot(pupil_xy[preview_idx, 1], label='pupil y', lw=1.0, alpha=0.8)
        ax.set_title(f'{eye} eye trace preview')
        ax.set_xlabel('Joint-valid time bin')
        ax.set_ylabel('Position')
        ax.legend(frameon=False, fontsize=8)

        ax = axes[row_idx, 1]
        lims_x = [np.nanmin([dpi_sample[:, 0].min(), pupil_sample[:, 0].min()]), np.nanmax([dpi_sample[:, 0].max(), pupil_sample[:, 0].max()])]
        ax.scatter(dpi_sample[:, 0], pupil_sample[:, 0], s=3, alpha=0.25, rasterized=True)
        ax.plot(lims_x, lims_x, color='k', linestyle='--', lw=1.0)
        ax.set_title(f'{eye} x scatter (r={summary["corr_x"]:.3f})')
        ax.set_xlabel('DPI x')
        ax.set_ylabel('Pupil x')

        ax = axes[row_idx, 2]
        lims_y = [np.nanmin([dpi_sample[:, 1].min(), pupil_sample[:, 1].min()]), np.nanmax([dpi_sample[:, 1].max(), pupil_sample[:, 1].max()])]
        ax.scatter(dpi_sample[:, 1], pupil_sample[:, 1], s=3, alpha=0.25, rasterized=True)
        ax.plot(lims_y, lims_y, color='k', linestyle='--', lw=1.0)
        ax.set_title(f'{eye} y scatter (r={summary["corr_y"]:.3f})')
        ax.set_xlabel('DPI y')
        ax.set_ylabel('Pupil y')

        ax = axes[row_idx, 3]
        delta_xy = pupil_xy - dpi_xy
        ax.scatter(delta_xy[:, 0], delta_xy[:, 1], s=3, alpha=0.2, rasterized=True)
        ax.axhline(0.0, color='k', linestyle='--', lw=1.0)
        ax.axvline(0.0, color='k', linestyle='--', lw=1.0)
        ax.set_title(f'{eye} residual cloud')
        ax.set_xlabel('Pupil - DPI x')
        ax.set_ylabel('Pupil - DPI y')

        affine_eye = affine_info.get(eye, {})
        summary_text = (
            f"joint bins: {summary['n_joint_valid']}\n"
            f"corr radius: {_format_float(summary['corr_radius'])}\n"
            f"var dx: {_format_float(summary['var_delta_x'])}\n"
            f"var dy: {_format_float(summary['var_delta_y'])}\n"
            f"rmse x/y: {_format_float(summary['rmse_x'])}, {_format_float(summary['rmse_y'])}\n"
            f"affine bins: {affine_eye.get('joint_valid_bins', 'n/a')}"
        )
        axes[row_idx, 3].text(
            0.02,
            0.98,
            summary_text,
            transform=axes[row_idx, 3].transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)


def save_summary_npz(output_path, summaries):
    payload = {}
    for summary in summaries:
        eye = summary['eye']
        for key, value in summary.items():
            if key in {'dpi_xy', 'pupil_xy', 'eye'}:
                continue
            payload[f'{eye}_{key}'] = value
    np.savez(output_path, **payload)


def main():
    cli_args = _parse_cli_args(sys.argv[1:])
    session_yaml_arg = cli_args.get('session-yaml', os.environ.get('ROWLEY_SESSION_YAML'))
    subject = cli_args.get('subject', os.environ.get('ROWLEY_SUBJECT', default_subject))
    date = cli_args.get('date', os.environ.get('ROWLEY_DATE', default_date))
    primary_eye = cli_args.get('primary-eye', os.environ.get('ROWLEY_PRIMARY_EYE', default_primary_eye))
    dataset_dir_override = cli_args.get('dataset-dir', os.environ.get('ROWLEY_DATASET_DIR'))

    session_yaml_path = _resolve_session_yaml_path(
        subject,
        date,
        primary_eye,
        session_yaml_root,
        explicit_path=session_yaml_arg,
    )
    if not session_yaml_path.exists():
        raise FileNotFoundError(f'Session YAML not found: {session_yaml_path}')

    session_yaml_config = _load_yaml_config(session_yaml_path)
    session_from_yaml = str(session_yaml_config.get('session', f'{subject}_{date}'))
    try:
        subject, date = session_from_yaml.split('_', 1)
    except ValueError as exc:
        raise ValueError(f'Unexpected session name in YAML: {session_from_yaml}') from exc

    yaml_eye = str(session_yaml_config.get('eye', primary_eye))
    if primary_eye not in {'left', 'right'} and yaml_eye in {'left', 'right'}:
        primary_eye = yaml_eye
    elif primary_eye not in {'left', 'right'}:
        primary_eye = default_primary_eye

    dataset_dir = Path(dataset_dir_override or session_yaml_config['directory'])
    if not dataset_dir.is_absolute():
        dataset_dir = (session_yaml_path.parent / dataset_dir).resolve()

    session_data_root = _resolve_session_data_root(dataset_dir)
    sess = get_session(subject, date)
    aux_processed_path = Path(sess.processed_path)
    fix_path = dataset_dir / fix_name

    print(f'Session: {sess.name}')
    print(f'Session YAML: {session_yaml_path}')
    print(f'Dataset dir: {dataset_dir}')
    print(f'Session data root: {session_data_root}')
    print(f'Loading fixrsvp from: {fix_path}')
    if not fix_path.exists():
        raise FileNotFoundError(f'fixrsvp.dset not found at: {fix_path}')

    dset_fix = DictDataset.load(fix_path)
    add_calibrated_pupil_traces(dset_fix, aux_processed_path)
    affine_info = add_affine_transformed_pupil_traces(dset_fix)

    summaries = []
    for eye in ('left', 'right'):
        summary = summarize_eye_pair(dset_fix, eye)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError('No eye pairs were available for DPI vs pupil diagnostics.')

    print_summary(subject, date, summaries)

    output_dir = FIGURES_DIR / 'mcfarland' / f'{subject}_{date}_v11b_eye_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f'eye_trace_compare_{subject}_{date}.pdf'
    summary_path = output_dir / f'eye_trace_compare_{subject}_{date}_summary.npz'

    save_diagnostic_figure(figure_path, subject, date, summaries, affine_info)
    save_summary_npz(summary_path, summaries)

    print('')
    print(f'Saved figure: {figure_path}')
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    main()
#%%
"""
test_rowley12.py — fixation-radius sweep for 1−alpha stability

Built on top of test_rowley11.  All unit selection (Pool A/B) is computed
once at the default fixation radius (1.5°).  The covariance decomposition is
then re-run at four radii [0.75°, 1.0°, 1.5°, 2.0°] using the cyclopean eye,
with Pool B units held fixed.

Purpose: if 1−alpha = 1 − diag(Cpsth)/diag(Crate) drifts systematically with
fixation radius, the ratio is confounded by RF-drive dilution (windows at more
eccentric positions have lower / flatter PSTHs, so Cpsth drops without a
corresponding drop in Crate).  Stability across radii is evidence against
this confound.

Output: per-radius PKL + PDF, plus a summary PDF with overlaid histograms and
median 1−alpha vs radius.
"""

#%%
import sys
import os
import subprocess
import shutil
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
from DataRowleyV1V2.dots_calibration.training import bin_dots_to_stimulus, calculate_rf_snr
from DataRowleyV1V2.shifter.preprocess import normalize_stimulus, create_valid_eyepos_mask
from DataRowleyV1V2.utils.rf import calc_sta
from DataYatesV1 import DictDataset
from VisionCore.covariance import (run_covariance_decomposition,
                                    extract_valid_segments, extract_windows,
                                    estimate_rate_covariance,
                                    fit_intercept_linear, fit_intercept_pava,
                                    estimate_vergence_conditional_on_cyclopean,
                                    conservative_cvergence_from_2d)
from VisionCore.subspace import project_to_psd
from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR

#%% ---------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

session_yaml_root = VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'sessions'
fix_name    = 'fixrsvp.dset'

default_subject = 'Luke'
default_date = '2026-03-01'
default_primary_eye = 'right'
default_dataset_dir = Path('datasets_gaussian') / 'right_eye'


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


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
    with open(path) as f:
        return yaml.safe_load(f)


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


def _discover_session_configs(yaml_root, fix_filename):
    configs = []
    for yaml_path in sorted(yaml_root.glob('*_V1.yaml')):
        config = _load_yaml_config(yaml_path)
        dataset_dir_candidate = Path(config.get('directory', ''))
        if not str(dataset_dir_candidate):
            print(f"[WARN] Skipping YAML without directory: {yaml_path.name}")
            continue
        if not dataset_dir_candidate.is_absolute():
            dataset_dir_candidate = (yaml_path.parent / dataset_dir_candidate).resolve()
        fix_path_candidate = dataset_dir_candidate / fix_filename
        if not fix_path_candidate.exists():
            print(f"[WARN] Skipping YAML with missing {fix_filename}: {yaml_path.name}")
            continue
        eye_candidate = config.get('eye')
        if eye_candidate not in {'left', 'right'}:
            continue
        session_name = str(config.get('session', yaml_path.stem.replace('_V1', '')))
        try:
            subject_name, session_date = session_name.split('_', 1)
        except ValueError:
            print(f"[WARN] Skipping unexpected session name in YAML: {yaml_path.name}")
            continue
        configs.append({
            'subject': subject_name,
            'date': session_date,
            'primary_eye': eye_candidate,
            'dataset_dir': str(dataset_dir_candidate),
            'session_yaml': str(yaml_path),
        })
    return configs


cli_args = _parse_cli_args(sys.argv[1:])
session_yaml_arg = cli_args.get('session-yaml', os.environ.get('ROWLEY_SESSION_YAML'))
subject = cli_args.get('subject', os.environ.get('ROWLEY_SUBJECT', default_subject))
date = cli_args.get('date', os.environ.get('ROWLEY_DATE', default_date))
primary_eye = cli_args.get('primary-eye', os.environ.get('ROWLEY_PRIMARY_EYE', default_primary_eye))
dataset_dir_override = cli_args.get('dataset-dir', os.environ.get('ROWLEY_DATASET_DIR'))
output_root_name = cli_args.get('output-root', os.environ.get('ROWLEY_OUTPUT_ROOT', f'{subject}_v12'))
fresh_output_root = _env_flag('ROWLEY_FRESH_OUTPUT', False)
visual_source_mode = cli_args.get('visual-source', os.environ.get('ROWLEY_VISUAL_SOURCE', 'dots_rf')).strip().lower()
if visual_source_mode not in {'dots_rf', 'yaml_visual'}:
    raise ValueError(f"Unsupported visual source: {visual_source_mode!r}. Expected one of: dots_rf, yaml_visual")
single_session_requested = any(key in cli_args for key in ('subject', 'date', 'primary-eye', 'dataset-dir', 'session-yaml'))
run_all_sessions = _env_flag('ROWLEY_RUN_ALL_DATASETS', not single_session_requested)
session_filter = os.environ.get('ROWLEY_SESSION_FILTER')
MCFARLAND_DIR = FIGURES_DIR / 'mcfarland'
OUTPUT_ROOT_DIR = MCFARLAND_DIR / output_root_name

if run_all_sessions and not _env_flag('ROWLEY_CHILD_RUN', False):
    if fresh_output_root and OUTPUT_ROOT_DIR.exists():
        print(f"Removing existing output root: {OUTPUT_ROOT_DIR}")
        shutil.rmtree(OUTPUT_ROOT_DIR)
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    session_configs = _discover_session_configs(session_yaml_root, fix_name)
    if session_filter:
        session_configs = [cfg for cfg in session_configs if session_filter in f"{cfg['subject']}_{cfg['date']}|{cfg['primary_eye']}|{Path(cfg['session_yaml']).name}"]
    print(f"Discovered {len(session_configs)} runnable session YAML entries")
    failures = []
    for cfg in session_configs:
        session_label = f"{cfg['subject']}_{cfg['date']}"
        print("\n" + "=" * 80)
        print(f"Dispatching {session_label} | session_yaml={Path(cfg['session_yaml']).name} | primary_eye={cfg['primary_eye']}")
        print("=" * 80)
        child_env = os.environ.copy()
        child_env['ROWLEY_CHILD_RUN'] = '1'
        cmd = [
            sys.executable,
            __file__,
            '--session-yaml', cfg['session_yaml'],
            '--subject', cfg['subject'],
            '--date', cfg['date'],
            '--primary-eye', cfg['primary_eye'],
            '--dataset-dir', cfg['dataset_dir'],
            '--output-root', output_root_name,
            '--visual-source', visual_source_mode,
        ]
        result = subprocess.run(cmd, env=child_env)
        if result.returncode != 0:
            failures.append((session_label, cfg['session_yaml'], result.returncode))
            print(f"[FAIL] {session_label} | {cfg['session_yaml']} exited with code {result.returncode}")
    print("\n" + "=" * 80)
    print(f"Processed {len(session_configs)} dataset entries")
    if failures:
        print(f"Failures: {len(failures)}")
        for session_label, session_yaml_failed, code in failures:
            print(f"  {session_label} | {session_yaml_failed} -> exit {code}")

    # ── Cross-session summary figure ──────────────────────────────────────────
    print("\nBuilding cross-session summary figure...")
    pkl_paths = sorted(OUTPUT_ROOT_DIR.glob('*/*/fixrad_*/mcfarland_fixrsvp_*.pkl'))
    if not pkl_paths:
        print("No per-eye PKL files found — skipping cross-session summary.")
    else:
        _cs_sessions   = []
        _cs_windows    = []
        _cs_ff_before  = []
        _cs_ff_bsem    = []
        _cs_ff_after   = []
        _cs_ff_asem    = []
        _cs_fem_fracs  = []
        _cs_n_units    = []

        for pkl_path in pkl_paths:
            try:
                with open(pkl_path, 'rb') as fh:
                    pdata = pickle.load(fh)
            except Exception as e:
                print(f"  [skip] {pkl_path.name}: {e}")
                continue

            results   = pdata.get('results', [])
            last_mats = pdata.get('last_mats')   # list of per-window dicts
            meta      = pdata.get('meta', {})
            pool_b_mask_pkl = pdata.get('pool_b_mask')
            eye_lbl   = pkl_path.parent.parent.name
            sess_lbl  = f"{pkl_path.parent.parent.parent.name.replace('_v12', '')} | {eye_lbl}"
            n_units   = int(pool_b_mask_pkl.sum()) if pool_b_mask_pkl is not None else 0

            if not results:
                continue

            ws        = [r['window_ms']              for r in results]
            ff_before = [r.get('ff_before_verg_mean', np.nan) for r in results]
            ff_bsem   = [r.get('ff_before_verg_sem',  np.nan) for r in results]
            ff_after  = [r.get('ff_after_verg_mean',  np.nan) for r in results]
            ff_asem   = [r.get('ff_after_verg_sem',   np.nan) for r in results]

            # FEM fraction from last_mats (list of per-window dicts; match per-session: 20ms window)
            # project_to_psd mirrors the per-session computation; prevents spurious values >1
            fem_frac = np.array([], dtype=float)
            if last_mats:
                _widx  = next((i for i, r in enumerate(results) if r['window_ms'] == 20), -1)
                _mats  = last_mats[_widx]
                _Crate = _mats.get('Intercept')
                _CPSTH = _mats.get('PSTH')
                if _Crate is not None and _CPSTH is not None:
                    _Crate = project_to_psd(_Crate)
                    _CPSTH = project_to_psd(_CPSTH)
                    _diag_rate = np.diag(_Crate)
                    _diag_psth = np.diag(_CPSTH)
                    _valid = _diag_rate > 0
                    if _valid.any():
                        fem_frac = np.where(
                            _valid,
                            1.0 - _diag_psth / np.where(_valid, _diag_rate, 1.0),
                            np.nan)
                        fem_frac = fem_frac[np.isfinite(fem_frac)]

            _cs_sessions.append(sess_lbl)
            _cs_windows.append(np.array(ws, dtype=float))
            _cs_ff_before.append(np.array(ff_before, dtype=float))
            _cs_ff_bsem.append(np.array(ff_bsem, dtype=float))
            _cs_ff_after.append(np.array(ff_after, dtype=float))
            _cs_ff_asem.append(np.array(ff_asem, dtype=float))
            _cs_fem_fracs.append(fem_frac)
            _cs_n_units.append(n_units)

        if not _cs_sessions:
            print("No valid PKL data found — skipping cross-session summary.")
        else:
            n_sess   = len(_cs_sessions)
            cmap     = plt.get_cmap('tab10')
            colors   = [cmap(i % 10) for i in range(n_sess)]

            fig_cs, axs_cs = plt.subplots(1, 3, figsize=(15, 4.5))

            # Panel 0: FF before vergence vs window size
            ax = axs_cs[0]
            for i, sess in enumerate(_cs_sessions):
                ws = _cs_windows[i]
                yb = _cs_ff_before[i]
                ys = _cs_ff_bsem[i]
                if np.isfinite(yb).any():
                    ax.errorbar(ws, yb, yerr=ys,
                                fmt='o-', color=colors[i], lw=1.4, ms=4,
                                capsize=2, label=f'{sess} (n={_cs_n_units[i]})',
                                alpha=0.85)
            ax.axhline(1.0, color='k', lw=0.6, linestyle=':', alpha=0.4)
            ax.set_xscale('log')
            ax.set_xlabel('Window size (ms)')
            ax.set_ylabel('Mean FF ± SEM')
            ax.set_title('FF before vergence correction')
            ax.legend(frameon=False, fontsize=6.5, loc='upper right')

            # Panel 1: FF after vergence vs window size
            ax = axs_cs[1]
            for i, sess in enumerate(_cs_sessions):
                ws = _cs_windows[i]
                ya = _cs_ff_after[i]
                ys = _cs_ff_asem[i]
                if np.isfinite(ya).any():
                    ax.errorbar(ws, ya, yerr=ys,
                                fmt='s--', color=colors[i], lw=1.4, ms=4,
                                capsize=2, label=f'{sess} (n={_cs_n_units[i]})',
                                alpha=0.85)
            ax.axhline(1.0, color='k', lw=0.6, linestyle=':', alpha=0.4)
            ax.set_xscale('log')
            ax.set_xlabel('Window size (ms)')
            ax.set_ylabel('Mean FF ± SEM')
            ax.set_title('FF after vergence correction')
            ax.legend(frameon=False, fontsize=6.5, loc='upper right')

            # Panel 2: FEM fraction histograms overlaid
            ax = axs_cs[2]
            _bins_cs = np.linspace(0, 1.2, 37)
            for i, sess in enumerate(_cs_sessions):
                ff = _cs_fem_fracs[i]
                if ff.size:
                    ax.hist(ff, bins=_bins_cs, histtype='step',
                            color=colors[i], lw=1.4, alpha=0.8,
                            label=f'{sess}  med={np.median(ff):.2f}')
            ax.axvline(1.0, color='k', lw=0.8, linestyle='--', alpha=0.4)
            ax.set_xlabel('FEM fraction  (1 − α)')
            ax.set_ylabel('Count')
            ax.set_title('FEM fraction — all sessions')
            ax.legend(frameon=False, fontsize=6.5)

            fig_cs.suptitle('Cross-session covariance summary (v12, fixation-radius sweep)', fontsize=11)
            fig_cs.tight_layout(rect=(0, 0, 1, 0.97))
            cs_path = OUTPUT_ROOT_DIR / 'cross_session_summary_v12.pdf'
            fig_cs.savefig(cs_path, bbox_inches='tight')
            plt.close(fig_cs)
            print(f"Saved {cs_path}")

    # ── Combined FEM histograms split by fixation radius ─────────────────────
    print("\nBuilding combined FEM histograms split by fixation radius...")
    from collections import defaultdict
    all_pkl_paths = sorted(OUTPUT_ROOT_DIR.glob('*/*/fixrad_*/mcfarland_fixrsvp_*.pkl'))
    _radius_eye_pkls = defaultdict(lambda: defaultdict(list))
    for _p in all_pkl_paths:
        _radius_name = _p.parent.name
        _eye_name = _p.parent.parent.name
        _radius_eye_pkls[_radius_name][_eye_name].append(_p)

    _radius_names = sorted(_radius_eye_pkls)
    _bins_comb   = np.linspace(0, 1.2, 37)
    _bctrs_comb  = 0.5 * (_bins_comb[:-1] + _bins_comb[1:])
    _bwidth_comb = _bins_comb[1] - _bins_comb[0]

    if _radius_names:
        comb_path = OUTPUT_ROOT_DIR / 'cross_session_fem_combined_v12.pdf'
        with PdfPages(comb_path) as pdf:
            for _radius_name in _radius_names:
                _eye_pkls = _radius_eye_pkls[_radius_name]
                _eye_names = sorted(k for k in _eye_pkls if k != 'binocular_diff')
                if not _eye_names:
                    continue

                fig_comb, axs_comb = plt.subplots(
                    1, len(_eye_names),
                    figsize=(4 * len(_eye_names), 4.5),
                    squeeze=False,
                )

                for _si, _eye_name in enumerate(_eye_names):
                    ax = axs_comb[0, _si]
                    _sess_counts = []
                    _all_fracs   = []
                    for _pkl_p in sorted(_eye_pkls[_eye_name]):
                        try:
                            with open(_pkl_p, 'rb') as _fh:
                                _pd = pickle.load(_fh)
                        except Exception:
                            continue
                        _res  = _pd.get('results', [])
                        _lm   = _pd.get('last_mats')
                        if not _res or not _lm:
                            continue
                        _wi   = next((i for i, r in enumerate(_res) if r['window_ms'] == 20), -1)
                        _mm   = _lm[_wi]
                        _Cr   = _mm.get('Intercept')
                        _Cp   = _mm.get('PSTH')
                        if _Cr is None or _Cp is None:
                            continue
                        _Cr = project_to_psd(_Cr)
                        _Cp = project_to_psd(_Cp)
                        _dr = np.diag(_Cr)
                        _dp = np.diag(_Cp)
                        _v  = _dr > 0
                        if not _v.any():
                            continue
                        _ff = np.where(_v, 1.0 - _dp / np.where(_v, _dr, 1.0), np.nan)
                        _ff = _ff[np.isfinite(_ff)]
                        if _ff.size == 0:
                            continue
                        _counts, _ = np.histogram(_ff, bins=_bins_comb)
                        _slbl = _pkl_p.parent.parent.parent.name.replace('_v12', '')
                        _sess_counts.append((_slbl, _counts))
                        _all_fracs.append(_ff)

                    if not _sess_counts:
                        ax.set_title(_eye_name, fontsize=8)
                        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
                        continue

                    _total_counts = np.zeros(len(_bctrs_comb))
                    for _slbl, _counts in _sess_counts:
                        _total_counts += _counts

                    ax.bar(_bctrs_comb, _total_counts, width=_bwidth_comb * 0.92,
                           color='steelblue', alpha=0.75)

                    _med = np.median(np.concatenate(_all_fracs))
                    _xfm = ax.get_xaxis_transform()
                    ax.plot(_med, 1.03, 'v', transform=_xfm,
                            color='steelblue', markersize=7, clip_on=False)
                    ax.text(_med, 1.08, f'{_med:.2f}', transform=_xfm,
                            ha='center', va='bottom', fontsize=7, clip_on=False)

                    ax.axvline(1.0, color='k', lw=0.8, linestyle='--', alpha=0.4)
                    ax.set_xlim(0, 1.2)
                    ax.set_xlabel('FEM fraction  (1 − α)')
                    _n_units_total = int(sum(len(_ff) for _ff in _all_fracs))
                    ax.set_title(
                        f'{_eye_name}\n(n sessions={len(_sess_counts)}, n units={_n_units_total})',
                        fontsize=8,
                    )
                    if _si == 0:
                        ax.set_ylabel('Count (all sessions)')

                fig_comb.suptitle(
                    f'FEM fraction — combined histogram by eye trace type | {_radius_name} (v12, 20 ms window)',
                    fontsize=11,
                )
                fig_comb.tight_layout(rect=(0, 0, 1, 0.97))
                pdf.savefig(fig_comb, bbox_inches='tight')
                plt.close(fig_comb)

        print(f"Saved {comb_path}")

    sys.exit(0)

windows_ms             = [5, 10, 20, 40, 80]
focal_window_ms        = 20   # window used for 1-alpha comparison across radii
total_spikes_threshold = 200
valid_time_bins        = 240
dt                     = 1 / 240.0
t_hist_ms              = 50
intercept_diag_base_edges_deg = np.array([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.2], dtype=np.float64)
intercept_diag_min_pairs_per_bin = 2000
n_bins                 = 15

min_fix_dur_bins = 20

# ── Fixation-radius sweep (v12) ───────────────────────────────────────────────
fixation_radii_deg   = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2]  # radii to test
# Pool B is always selected at the default radius below; only ecc_mask for the
# covariance windows changes across iterations.

# ── Fixation eccentricity gate [Criterion 0]
fixation_radius_deg  = 1.5   # matches faceRadius from FixRsvpTrial parameters

# ── Dots RF SNR [Criterion 1]
snr_threshold_primary = 5.0  # mirrors pipeline/01 dots calibration unit-selection threshold
snr_threshold_report  = 2.5   # report a looser comparison threshold for debugging

dots_roi_deg  = np.array([[-5, 5], [-5, 5]], dtype=np.float32)
dots_dxy_deg  = 0.2
dots_sta_lags = np.arange(2, 8)

session_yaml_path = _resolve_session_yaml_path(
    subject,
    date,
    primary_eye,
    session_yaml_root,
    explicit_path=session_yaml_arg,
)
assert session_yaml_path.exists(), f"Session YAML not found: {session_yaml_path}"
session_yaml_config = _load_yaml_config(session_yaml_path)

session_from_yaml = str(session_yaml_config.get('session', f'{subject}_{date}'))
try:
    subject, date = session_from_yaml.split('_', 1)
except ValueError as exc:
    raise ValueError(f"Unexpected session name in YAML: {session_from_yaml}") from exc

yaml_eye = str(session_yaml_config.get('eye', primary_eye))
if primary_eye not in {'left', 'right'} and yaml_eye in {'left', 'right'}:
    primary_eye = yaml_eye
elif primary_eye not in {'left', 'right'}:
    primary_eye = str(session_yaml_config.get('right_eye', default_primary_eye))

dataset_dir = Path(dataset_dir_override or session_yaml_config['directory'])
if not dataset_dir.is_absolute():
    dataset_dir = (session_yaml_path.parent / dataset_dir).resolve()
session_data_root = _resolve_session_data_root(dataset_dir)

dots_binned_path = session_data_root / 'dpi_calibration' / 'dots_binned_data.dset'
dots_primary_eye_dir = session_data_root / 'dpi_calibration' / f'{primary_eye}_eye'
dots_primary_csv = dots_primary_eye_dir / 'calibrated_dpi.csv'
dots_primary_params = dots_primary_eye_dir / 'calibration_params.npz'
dots_primary_snr_cache = dots_primary_eye_dir / 'dots_rf_snr.npz'

yaml_bino_path  = (VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'sessions'
                   / f'{subject}_{date}_binocular_V1.yaml')
yaml_right_path = (VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'sessions'
                   / f'{subject}_{date}_right_V1.yaml')

# ── FixRSVP reliability [Criterion 2]
min_reliability      = 0.025 # leaving out, too strict. 0.05 #previously 0.1,
n_reliability_splits = 20

# ── Runtime truncation datafilter
missing_pct_threshold = 45.0  # matches VisionCore missing_pct datafilter threshold

# ── NaN / missing data [Criterion 3]
max_unit_nan_frac  = 0.20   # units above this → Pool A only
max_bad_trial_frac = 0.10   # Stage B: trials with > this frac of Pool B NaN units dropped

# ── 2D analysis
n_bins_c_2d  = 5
n_bins_v_2d  = 5
min_pairs_2d = 30
n_shuff_2d   = 1

import datetime as _dt
SESSION_OUTPUT_DIR = OUTPUT_ROOT_DIR / f'{subject}_{date}_v12'
BASE_FIGURES_DIR = SESSION_OUTPUT_DIR

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


def _robust_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return 0.67448975 * (x - med) / mad


def _compute_split_half_reliability(
    robs_trials,
    n_splits,
    seed=42,
    min_valid_bins=10,
    min_trials_per_half=2,
):
    n_trials, _, n_units = robs_trials.shape
    rng_rel = np.random.default_rng(seed)
    r2_accum = np.zeros(n_units, dtype=np.float64)
    r2_count = np.zeros(n_units, dtype=np.int64)
    if n_trials < 2:
        return r2_accum

    for _split in range(n_splits):
        perm = rng_rel.permutation(n_trials)
        half = n_trials // 2
        if half < min_trials_per_half:
            break
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            psth_a = np.nanmean(robs_trials[idx_a], axis=0)
            psth_b = np.nanmean(robs_trials[idx_b], axis=0)

        counts_a = np.sum(np.isfinite(robs_trials[idx_a]), axis=0)
        counts_b = np.sum(np.isfinite(robs_trials[idx_b]), axis=0)

        for unit_idx in range(n_units):
            a = psth_a[:, unit_idx]
            b = psth_b[:, unit_idx]
            fin = (
                np.isfinite(a)
                & np.isfinite(b)
                & (counts_a[:, unit_idx] >= min_trials_per_half)
                & (counts_b[:, unit_idx] >= min_trials_per_half)
            )
            if fin.sum() < min_valid_bins:
                continue
            if np.std(a[fin]) <= 0 or np.std(b[fin]) <= 0:
                continue
            r2_accum[unit_idx] += np.corrcoef(a[fin], b[fin])[0, 1] ** 2
            r2_count[unit_idx] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        return np.divide(
            r2_accum,
            r2_count,
            out=np.zeros_like(r2_accum),
            where=r2_count > 0,
        )


def _compute_missing_pct_mask(session, t_bins, cids, threshold):
    missing_pct_fun = session.get_missing_pct_interp(cids)
    pct = _to_numpy(missing_pct_fun(t_bins)).astype(np.float32)
    valid_mask = pct < threshold
    chronic_multi_units = np.nanmedian(pct, axis=0) >= threshold
    valid_mask[:, chronic_multi_units] = True
    return valid_mask, pct, chronic_multi_units


def _save_reliability_histogram(output_path, all_values, candidate_values, threshold, session_label,
                                masked_values=None, candidate_masked_values=None):
    fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    bins = np.linspace(0.0, 0.4, 161).tolist()

    all_values = np.asarray(all_values, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    candidate_values = candidate_values[np.isfinite(candidate_values)]
    masked_values_fin = np.array([], dtype=np.float64)
    if masked_values is not None:
        masked_values_fin = np.asarray(masked_values, dtype=np.float64)
        masked_values_fin = masked_values_fin[np.isfinite(masked_values_fin)]
    candidate_masked_fin = np.array([], dtype=np.float64)
    if candidate_masked_values is not None:
        candidate_masked_fin = np.asarray(candidate_masked_values, dtype=np.float64)
        candidate_masked_fin = candidate_masked_fin[np.isfinite(candidate_masked_fin)]

    ax = axs[0]
    if all_values.size:
        ax.hist(all_values, bins=bins, color='lightgray', alpha=0.9, label=f'all units unmasked (n={all_values.size})')
    if masked_values_fin.size:
        ax.hist(masked_values_fin, bins=bins, color='salmon', alpha=0.6,
                label=f'all units masked (n={masked_values_fin.size})')
    if candidate_values.size:
        ax.hist(candidate_values, bins=bins, color='steelblue', alpha=0.7,
                label=f'visual & spikes unmasked (n={candidate_values.size})')
    else:
        ax.text(0.5, 0.5, 'No finite candidate-unit reliability values',
                ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.axvline(threshold, color='r', linestyle='--', lw=1.0, label=f'threshold={threshold:.2f}')
    ax.set_xlim(0, 0.4)
    ax.set_xlabel('Split-half PSTH reliability (r²)')
    ax.set_ylabel('Unit count')
    ax.set_title(f'Reliability distribution — all units\n{session_label}')
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1]
    if candidate_values.size:
        ax.hist(candidate_values, bins=bins, color='steelblue', alpha=0.75,
                label=f'visual & spikes unmasked (n={candidate_values.size})')
        if candidate_masked_fin.size:
            ax.hist(candidate_masked_fin, bins=bins, color='darkorange', alpha=0.45,
                    label=f'visual & spikes masked (n={candidate_masked_fin.size})')
    else:
        ax.text(0.5, 0.5, 'No finite candidate-unit reliability values',
                ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.axvline(threshold, color='r', linestyle='--', lw=1.0, label=f'threshold={threshold:.2f}')
    ax.set_xlim(0, 0.4)
    ax.set_xlabel('Split-half PSTH reliability (r²)')
    ax.set_ylabel('Candidate unit count')
    ax.set_title('Reliability distribution — visual & spikes candidates only')
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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
    enabled_configs = [('cyclopean-dpi', 'default')]
    has_left        = any(k in keys for k in ['eyepos_left', 'eyepos_dpi_left'])
    has_right       = any(k in keys for k in ['eyepos_right', 'eyepos_dpi_right'])
    has_left_pupil  = any(k in keys for k in ['eyepos_pupil_left', 'pupil_left', 'eyepos_left_pupil'])
    has_right_pupil = any(k in keys for k in ['eyepos_pupil_right', 'pupil_right', 'eyepos_right_pupil'])
    if has_left and has_right:
        enabled_configs.append(('binocular_diff', 'binocular_diff'))
    if has_left:
        enabled_configs.append(('left-dpi', 'left'))
    if has_left_pupil:
        enabled_configs.append(('left-pupil', 'pupil_left'))
    if has_right:
        enabled_configs.append(('right-dpi', 'right'))
    if has_right_pupil:
        enabled_configs.append(('right-pupil', 'pupil_right'))
    return enabled_configs


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
    bin_edges   = mats.get('bin_edges')
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
    ax.set_xlabel('Delta eye trajectory RMS (deg)')
    ax.set_ylabel('Covariance')
    if bin_edges is not None and len(bin_edges) >= 2:
        ax.set_title(
            f'Neuron {i} | {win_ms} ms\n'
            f'd range {bin_edges[0]:.3f}-{bin_edges[-1]:.3f} deg'
        )
    else:
        ax.set_title(f'Neuron {i} | {win_ms} ms')
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_pairwise_distance_bin_counts(mats, win_ms, ax=None):
    bin_edges = mats.get('bin_edges')
    count_e = np.asarray(mats.get('count_e', []))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if bin_edges is None or count_e.size == 0:
        ax.text(0.5, 0.5, 'No pairwise-distance bin data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Pooled pairwise-distance bins | {win_ms} ms')
        return ax

    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    widths = np.diff(bin_edges)
    valid = np.isfinite(widths) & (widths > 0) & np.isfinite(count_e)
    if np.any(valid):
        ax.bar(
            bin_edges[:-1][valid],
            count_e[valid],
            width=widths[valid],
            align='edge',
            color='steelblue',
            alpha=0.75,
            edgecolor='white',
            linewidth=0.8,
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(centers[valid], count_e[valid], color='navy', lw=1.1, marker='o', ms=3)
    else:
        ax.text(0.5, 0.5, 'No finite pairwise-distance bins', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Delta eye trajectory RMS (deg)')
    ax.set_ylabel('Pair count')
    ax.set_title(f'Pooled pairwise-distance bins | {win_ms} ms')
    ax.grid(True, axis='y', alpha=0.2)
    return ax


def compute_pooled_pairwise_distances(eyepos, valid_mask, t_count_bins, t_hist_bins, min_seg_len=36):
    eyepos = np.asarray(eyepos, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    segments = extract_valid_segments(valid_mask, min_len_bins=min_seg_len)
    if not segments:
        return np.array([], dtype=np.float32)

    robs_dummy = torch.zeros((eyepos.shape[0], eyepos.shape[1], 1), dtype=torch.float32)
    eyepos_t = torch.tensor(np.nan_to_num(eyepos, nan=0.0), dtype=torch.float32)
    _, eye_traj, t_idx, _ = extract_windows(
        robs_dummy,
        eyepos_t,
        segments,
        t_count_bins,
        t_hist_bins,
        device='cpu',
    )
    if eye_traj is None or t_idx is None:
        return np.array([], dtype=np.float32)

    t_np = t_idx.detach().cpu().numpy()
    total_len = eye_traj.shape[1]
    inv_sqrt_t = 1.0 / np.sqrt(float(total_len))
    distances = []
    for t_val in np.unique(t_np):
        idx = np.where(t_np == t_val)[0]
        if len(idx) < 2:
            continue
        traj_t = eye_traj[idx].reshape(len(idx), -1).float()
        ii, jj = torch.triu_indices(len(idx), len(idx), offset=1)
        d_t = (torch.cdist(traj_t, traj_t)[ii, jj] * inv_sqrt_t).detach().cpu().numpy()
        if d_t.size:
            distances.append(d_t.astype(np.float32, copy=False))

    if not distances:
        return np.array([], dtype=np.float32)
    return np.concatenate(distances, axis=0)


def plot_pairwise_distance_distribution_page(pdf, mats, result_row, distances):
    win_ms = result_row['window_ms']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    plot_pairwise_distance_bin_counts(mats, win_ms, ax=axs[0, 0])

    if distances.size == 0:
        for ax in [axs[0, 1], axs[1, 0], axs[1, 1]]:
            ax.text(0.5, 0.5, 'No pooled pairwise distances', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
        fig.suptitle(f'Pairwise-distance diagnostics | {win_ms:.2f} ms')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return

    finite = np.asarray(distances, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.array([], dtype=np.float64)

    if finite.size:
        p995 = float(np.quantile(finite, 0.995))
        hist_max = max(p995, float(np.max(finite)) * 0.25, 0.2)
        full_edges = np.linspace(0.0, hist_max, 80)
        axs[0, 1].hist(finite, bins=full_edges, color='steelblue', alpha=0.8)
        axs[0, 1].set_xlim(0.0, hist_max)
        axs[0, 1].set_xlabel('Delta eye trajectory RMS (deg)')
        axs[0, 1].set_ylabel('Pair count')
        axs[0, 1].set_title('Raw pooled-distance histogram')

        near_max = min(hist_max, max(0.2, float(np.quantile(finite, 0.10))))
        near_edges = np.linspace(0.0, near_max, 80)
        axs[1, 0].hist(finite, bins=near_edges, color='darkorange', alpha=0.8)
        axs[1, 0].set_xlim(0.0, near_max)
        axs[1, 0].set_xlabel('Delta eye trajectory RMS (deg)')
        axs[1, 0].set_ylabel('Pair count')
        axs[1, 0].set_title('Near-zero histogram zoom')

        finite_sorted = np.sort(finite)
        cdf_y = np.arange(1, finite_sorted.size + 1, dtype=np.float64) / finite_sorted.size
        axs[1, 1].plot(finite_sorted, cdf_y, color='black', lw=1.5)
        axs[1, 1].axvline(near_max, color='0.6', lw=0.8, linestyle=':')
        axs[1, 1].set_xlim(0.0, hist_max)
        axs[1, 1].set_ylim(0.0, 1.0)
        axs[1, 1].set_xlabel('Delta eye trajectory RMS (deg)')
        axs[1, 1].set_ylabel('CDF')
        axs[1, 1].set_title('Empirical CDF')

        text = (
            f'n_pairs={finite.size}\n'
            f'min={np.min(finite):.4f} deg\n'
            f'p1={np.quantile(finite, 0.01):.4f} deg\n'
            f'p5={np.quantile(finite, 0.05):.4f} deg\n'
            f'p10={np.quantile(finite, 0.10):.4f} deg\n'
            f'median={np.quantile(finite, 0.50):.4f} deg\n'
            f'p95={np.quantile(finite, 0.95):.4f} deg\n'
            f'max={np.max(finite):.4f} deg'
        )
        axs[0, 0].text(0.98, 0.97, text, transform=axs[0, 0].transAxes,
                       ha='right', va='top', fontsize=7, color='gray')
    else:
        for ax in [axs[0, 1], axs[1, 0], axs[1, 1]]:
            ax.text(0.5, 0.5, 'No finite pooled pairwise distances', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()

    fig.suptitle(f'Pairwise-distance diagnostics | {win_ms:.2f} ms')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _apply_rate_covariance_limit(Crate, Ceye, Ctotal):
    Crate = np.array(Crate, copy=True)
    Ceye = np.array(Ceye, copy=True)
    if Ctotal is None:
        return Crate, Ceye
    bad_mask = np.diag(Crate) > 0.99 * np.diag(Ctotal)
    Crate[bad_mask, :] = np.nan
    Crate[:, bad_mask] = np.nan
    Ceye[:, bad_mask, :] = np.nan
    Ceye[:, :, bad_mask] = np.nan
    return Crate, Ceye


def _compute_diag_linear_slopes(Ceye, bin_centers, count_e, d_max=0.4, min_bins=3):
    Ceye = np.asarray(Ceye, dtype=np.float64)
    x = np.asarray(bin_centers, dtype=np.float64)
    w_all = np.asarray(count_e, dtype=np.float64)
    use_mask = np.isfinite(x) & (x > 0) & (x <= d_max) & np.isfinite(w_all) & (w_all > 0)
    idx = np.where(use_mask)[0]
    n_cells = Ceye.shape[1]
    slopes = np.full(n_cells, np.nan, dtype=np.float64)
    if idx.size < min_bins:
        return slopes
    x_loc = x[idx]
    w_loc = w_all[idx]
    s0 = np.sum(w_loc)
    sx = np.sum(w_loc * x_loc)
    sxx = np.sum(w_loc * x_loc ** 2)
    det = s0 * sxx - sx ** 2
    if det <= 0:
        return slopes
    for cell_idx in range(n_cells):
        y = Ceye[idx, cell_idx, cell_idx]
        valid = np.isfinite(y)
        if np.sum(valid) < min_bins:
            continue
        xv = x_loc[valid]
        wv = w_loc[valid]
        yv = y[valid]
        s0v = np.sum(wv)
        sxv = np.sum(wv * xv)
        sxxv = np.sum(wv * xv ** 2)
        detv = s0v * sxxv - sxv ** 2
        if detv <= 0:
            continue
        syv = np.sum(wv * yv)
        sxyv = np.sum(wv * xv * yv)
        slopes[cell_idx] = (s0v * sxyv - sxv * syv) / detv
    return slopes


def _compute_fem_fraction_from_crate(Crate, Cpsth):
    denom = np.diag(Crate)
    vals = np.where(denom > 0, 1.0 - np.diag(Cpsth) / denom, np.nan)
    return vals[np.isfinite(vals)]


def _finite_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def _merge_sparse_distance_edges(base_edges, distances, min_pairs_per_bin):
    base_edges = np.asarray(base_edges, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    if base_edges.size < 2 or distances.size == 0:
        return base_edges

    counts, _ = np.histogram(distances, bins=base_edges)
    merged = [float(base_edges[0])]
    acc = 0
    for idx, count in enumerate(counts):
        acc += int(count)
        is_last = (idx == len(counts) - 1)
        if acc >= min_pairs_per_bin or is_last:
            merged.append(float(base_edges[idx + 1]))
            acc = 0

    merged = np.asarray(merged, dtype=np.float64)
    if merged[-1] < base_edges[-1]:
        merged = np.append(merged, base_edges[-1])
    merged = np.unique(merged)
    if merged.size < 2:
        return np.asarray([base_edges[0], base_edges[-1]], dtype=np.float64)
    return merged


def _compute_intercept_diagnostics(robs_cov, eyepos_cov, valid_cov, Cpsth, Ctotal, t_count_bins, t_hist_bins):
    robs_t = torch.tensor(np.nan_to_num(robs_cov, nan=0.0), dtype=torch.float32)
    eyepos_t = torch.tensor(np.nan_to_num(eyepos_cov, nan=0.0), dtype=torch.float32)
    segments = extract_valid_segments(valid_cov, min_len_bins=36)
    if not segments:
        return None
    spike_counts, eye_traj, t_idx, _ = extract_windows(
        robs_t,
        eyepos_t,
        segments,
        t_count_bins,
        max(t_hist_bins, t_count_bins),
        device='cpu',
    )
    if spike_counts is None or eye_traj is None or t_idx is None:
        return None

    pooled_distances = compute_pooled_pairwise_distances(
        eyepos_cov,
        valid_cov,
        t_count_bins=t_count_bins,
        t_hist_bins=max(t_hist_bins, t_count_bins),
        min_seg_len=36,
    )
    adaptive_edges = _merge_sparse_distance_edges(
        intercept_diag_base_edges_deg,
        pooled_distances,
        min_pairs_per_bin=intercept_diag_min_pairs_per_bin,
    )

    Crate_fixed_first, _, Ceye_fixed, bin_centers_fixed, count_e_fixed, bin_edges_fixed = estimate_rate_covariance(
        spike_counts,
        eye_traj,
        t_idx,
        n_bins=adaptive_edges,
        Ctotal=Ctotal,
        intercept_mode='linear',
    )
    Crate_fixed_zero = fit_intercept_linear(Ceye_fixed, bin_centers_fixed, count_e_fixed, eval_at_first_bin=False)
    Crate_fixed_iso = fit_intercept_pava(Ceye_fixed, count_e_fixed)
    Crate_fixed_low = Ceye_fixed[0].copy() if Ceye_fixed.shape[0] else np.full_like(Crate_fixed_first, np.nan)

    Crate_fixed_zero, Ceye_fixed_zero = _apply_rate_covariance_limit(Crate_fixed_zero, Ceye_fixed, Ctotal)
    Crate_fixed_iso, Ceye_fixed_iso = _apply_rate_covariance_limit(Crate_fixed_iso, Ceye_fixed, Ctotal)
    Crate_fixed_low, Ceye_fixed_low = _apply_rate_covariance_limit(Crate_fixed_low, Ceye_fixed, Ctotal)
    Crate_fixed_first, Ceye_fixed_first = _apply_rate_covariance_limit(Crate_fixed_first, Ceye_fixed, Ctotal)

    return {
        'pooled_distances': pooled_distances,
        'adaptive_bin_edges': np.asarray(bin_edges_fixed, dtype=np.float64),
        'adaptive_bin_centers': np.asarray(bin_centers_fixed, dtype=np.float64),
        'adaptive_count_e': np.asarray(count_e_fixed, dtype=np.float64),
        'adaptive_Ceye': Ceye_fixed_first,
        'adaptive_lowest_bin': Crate_fixed_low,
        'adaptive_linear_first': Crate_fixed_first,
        'adaptive_linear_zero': Crate_fixed_zero,
        'adaptive_isotonic': Crate_fixed_iso,
        'fem_adaptive_lowest_bin': _compute_fem_fraction_from_crate(Crate_fixed_low, Cpsth),
        'fem_adaptive_linear_first': _compute_fem_fraction_from_crate(Crate_fixed_first, Cpsth),
        'fem_adaptive_linear_zero': _compute_fem_fraction_from_crate(Crate_fixed_zero, Cpsth),
        'fem_adaptive_isotonic': _compute_fem_fraction_from_crate(Crate_fixed_iso, Cpsth),
        'diag_slopes_adaptive': _compute_diag_linear_slopes(Ceye_fixed_first, bin_centers_fixed, count_e_fixed),
        'adaptive_min_pairs_per_bin': int(intercept_diag_min_pairs_per_bin),
        'base_bin_edges': np.asarray(intercept_diag_base_edges_deg, dtype=np.float64),
    }


def plot_intercept_diagnostic_page(pdf, diag, mats_pct, Cpsth, fem_fraction_current, diag_slopes_current, display_eye_name, win_ms):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    current_centers = np.asarray(mats_pct['bin_centers'], dtype=np.float64)
    current_counts = np.asarray(mats_pct['count_e'], dtype=np.float64)
    current_edges = np.asarray(mats_pct.get('bin_edges', []), dtype=np.float64)
    current_ceye = np.asarray(mats_pct['Ceye'], dtype=np.float64)
    fixed_centers = np.asarray(diag['adaptive_bin_centers'], dtype=np.float64)
    fixed_counts = np.asarray(diag['adaptive_count_e'], dtype=np.float64)
    fixed_edges = np.asarray(diag['adaptive_bin_edges'], dtype=np.float64)
    fixed_ceye = np.asarray(diag['adaptive_Ceye'], dtype=np.float64)

    current_diag = np.diagonal(current_ceye, axis1=1, axis2=2)
    fixed_diag = np.diagonal(fixed_ceye, axis1=1, axis2=2)
    current_med = np.array([_finite_median(row) for row in current_diag], dtype=np.float64)
    fixed_med = np.array([_finite_median(row) for row in fixed_diag], dtype=np.float64)
    if np.isfinite(current_med).any():
        axs[0, 0].plot(current_centers[np.isfinite(current_med)], current_med[np.isfinite(current_med)], 'o-', color='steelblue', label='percentile bins')
    if np.isfinite(fixed_med).any():
        axs[0, 0].plot(fixed_centers[np.isfinite(fixed_med)], fixed_med[np.isfinite(fixed_med)], 's-', color='darkorange', label='adaptive degree bins')
    else:
        axs[0, 0].text(0.98, 0.08, 'adaptive-bin diag(Ceye): no finite bins', transform=axs[0, 0].transAxes,
                       ha='right', va='bottom', fontsize=7, color='darkorange')
    axs[0, 0].axhline(np.nanmedian(np.diag(Cpsth)), color='0.4', linestyle='--', lw=1.0, label='median diag(Cpsth)')
    axs[0, 0].set_xlabel('Delta eye trajectory RMS (deg)')
    axs[0, 0].set_ylabel('Median diag(Ceye)')
    axs[0, 0].set_title('Raw Ceye(d) diagonal summary')
    axs[0, 0].legend(frameon=False, fontsize=7)

    distances = np.asarray(diag['pooled_distances'], dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    near_max = max(0.08, min(0.2, float(np.quantile(distances, 0.25)) if distances.size else 0.2))
    if distances.size:
        axs[0, 1].hist(distances, bins=np.linspace(0.0, near_max, 80), color='lightgray', alpha=0.95)
    for edge in current_edges[1:-1]:
        if edge <= near_max:
            axs[0, 1].axvline(edge, color='steelblue', lw=0.8, alpha=0.6)
    for edge in fixed_edges[1:-1]:
        if edge <= near_max:
            axs[0, 1].axvline(edge, color='darkorange', lw=0.8, alpha=0.5, linestyle=':')
    axs[0, 1].set_xlim(0.0, near_max)
    axs[0, 1].set_xlabel('Delta eye trajectory RMS (deg)')
    axs[0, 1].set_ylabel('Pair count')
    axs[0, 1].set_title('Near-zero pooled distance density')

    bins_fem = np.linspace(0, 1.2, 37)
    axs[1, 0].hist(fem_fraction_current, bins=bins_fem, histtype='step', color='steelblue', lw=1.6, label='current percentile first-bin')
    axs[1, 0].hist(diag['fem_adaptive_linear_first'], bins=bins_fem, histtype='step', color='darkorange', lw=1.4, label='adaptive first-bin')
    axs[1, 0].hist(diag['fem_adaptive_linear_zero'], bins=bins_fem, histtype='step', color='firebrick', lw=1.4, label='adaptive linear-zero')
    axs[1, 0].hist(diag['fem_adaptive_isotonic'], bins=bins_fem, histtype='step', color='teal', lw=1.4, label='adaptive isotonic')
    axs[1, 0].hist(diag['fem_adaptive_lowest_bin'], bins=bins_fem, histtype='step', color='gray', lw=1.2, label='adaptive lowest-bin')
    axs[1, 0].set_xlabel('FEM fraction (1 - alpha)')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title('Crate variant comparison')
    axs[1, 0].legend(frameon=False, fontsize=7)

    text = (
        f'current median={_finite_median(fem_fraction_current):.4f}\n'
        f'adaptive first median={_finite_median(diag["fem_adaptive_linear_first"]):.4f}\n'
        f'adaptive zero median={_finite_median(diag["fem_adaptive_linear_zero"]):.4f}\n'
        f'adaptive isotonic median={_finite_median(diag["fem_adaptive_isotonic"]):.4f}\n'
        f'adaptive lowest median={_finite_median(diag["fem_adaptive_lowest_bin"]):.4f}\n\n'
        f'frac positive diag slope (current)={np.nanmean(diag_slopes_current > 0):.3f}\n'
        f'frac positive diag slope (adaptive)={np.nanmean(diag["diag_slopes_adaptive"] > 0):.3f}\n\n'
        f'current first bin hi={current_edges[1]:.4f} deg\n'
        f'adaptive first bin hi={fixed_edges[1]:.4f} deg\n'
        f'current first-bin pairs={int(current_counts[0]) if current_counts.size else 0}\n'
        f'adaptive first-bin pairs={int(fixed_counts[0]) if fixed_counts.size else 0}\n'
        f'min pairs target={int(diag["adaptive_min_pairs_per_bin"])}'
    )
    axs[1, 1].axis('off')
    axs[1, 1].text(0.03, 0.97, text, va='top', ha='left', family='monospace')

    fig.suptitle(f'Intercept diagnostics | {display_eye_name} | {win_ms:.2f} ms')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _compute_snr(npy_path):
    sta_ste = np.load(npy_path)                                           # (2, n_units, n_lags, H, W)
    stes    = sta_ste[1]                                                  # (n_units, n_lags, H, W)
    signal  = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
    signal  = gaussian_filter(signal, sigma=[0, 0, 4, 4])
    noise   = np.median(signal[:, 0], axis=(1, 2))
    snr     = signal.max(axis=(2, 3)) / (noise[:, None] + 1e-8)
    return snr.max(axis=1)                                                # (n_units,)


def _compute_sta_ste_cache(dset_path, npy_path, label, n_lags=20):
    print(f"  [INFO] {label} STA/STE cache missing; recomputing from: {dset_path}")
    dset = DictDataset.load(dset_path)
    stim = normalize_stimulus(dset['stim'])
    dfs  = create_valid_eyepos_mask(dset['eyepos'], dset['dpi_valid'], valid_radius=10).float()

    stas = calc_sta(
        stim, dset['robs'], n_lags, dfs,
        device='cpu', progress=True,
        stim_modifier=lambda x: x,
    ).cpu().numpy()
    stes = calc_sta(
        stim, dset['robs'], n_lags, dfs,
        device='cpu', progress=True,
        stim_modifier=lambda x: x**2,
    ).cpu().numpy()

    sta_ste = np.stack([stas, stes], axis=0).astype(np.float32, copy=False)
    np.save(npy_path, sta_ste)
    print(f"  [INFO] Saved {label} STA/STE cache: {npy_path}")
    return sta_ste


def _compute_snr_if_available(npy_path, dset_path, n_units, label):
    if npy_path.exists():
        return _compute_snr(npy_path), True, 'cache'
    if dset_path.exists():
        _compute_sta_ste_cache(dset_path, npy_path, label)
        return _compute_snr(npy_path), True, 'recomputed'
    print(f"  [WARN] {label} STA/STE not found: {npy_path}")
    print("         Falling back to no visual prefilter for this session.")
    return np.full(n_units, np.nan, dtype=np.float32), False, 'skipped'


def _compute_dots_snr_if_available(
    dots_dset_path,
    calibrated_dpi_csv,
    calibration_params_path,
    cache_path,
    target_cids,
    label,
):
    if not dots_dset_path.exists():
        print(f"  [WARN] {label} dots cache not found: {dots_dset_path}")
        print("         Falling back to no visual prefilter for this session.")
        return np.full(len(target_cids), np.nan, dtype=np.float32), False, 'skipped'
    if not calibrated_dpi_csv.exists():
        print(f"  [WARN] {label} calibrated DPI CSV not found: {calibrated_dpi_csv}")
        print("         Falling back to no visual prefilter for this session.")
        return np.full(len(target_cids), np.nan, dtype=np.float32), False, 'skipped'
    if not calibration_params_path.exists():
        print(f"  [WARN] {label} calibration params not found: {calibration_params_path}")
        print("         Falling back to no visual prefilter for this session.")
        return np.full(len(target_cids), np.nan, dtype=np.float32), False, 'skipped'

    target_cids = np.asarray(target_cids)

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_cids = np.asarray(cached['cids'])
        cached_max_snr = np.asarray(cached['max_snr'], dtype=np.float32)
        if cached_cids.shape == target_cids.shape and np.array_equal(cached_cids, target_cids):
            return cached_max_snr, True, 'cache'

    dots_dset = DictDataset.load(dots_dset_path)
    dots_cids = np.asarray(dots_dset.metadata.get('cids', np.arange(_to_numpy(dots_dset['robs']).shape[1])))
    dots_index = {int(cid): idx for idx, cid in enumerate(dots_cids.tolist())}
    matched = np.array([dots_index.get(int(cid), -1) for cid in target_cids], dtype=int)
    found_mask = matched >= 0
    if not found_mask.any():
        print(f"  [WARN] {label} dots cache has no matching cluster IDs")
        print("         Falling back to no visual prefilter for this session.")
        return np.full(len(target_cids), np.nan, dtype=np.float32), False, 'skipped'

    params = np.load(calibration_params_path, allow_pickle=True)
    ppd = float(np.asarray(params['ppd']).reshape(-1)[0])
    dpi_df = pd.read_csv(calibrated_dpi_csv, usecols=['t_ephys', 'i', 'j', 'valid'])

    sample_times = dpi_df['t_ephys'].to_numpy(dtype=np.float64)
    gaze_pix = dpi_df[['i', 'j']].to_numpy(dtype=np.float32)
    gaze_valid = dpi_df['valid'].to_numpy()
    valid_samples = (
        np.isfinite(sample_times)
        & _as_bool_1d(gaze_valid, len(sample_times))
        & np.all(np.isfinite(gaze_pix), axis=1)
    )
    if valid_samples.sum() < 2:
        print(f"  [WARN] {label} has too few valid calibrated gaze samples")
        print("         Falling back to no visual prefilter for this session.")
        return np.full(len(target_cids), np.nan, dtype=np.float32), False, 'skipped'

    t_bins = _to_numpy(dots_dset['t_bins']).astype(np.float64)
    dots_pix = _to_numpy(dots_dset['dots_pix']).astype(np.float32)
    robs = _to_numpy(dots_dset['robs']).astype(np.float32)

    gaze_interp = _interp_xy(sample_times[valid_samples], gaze_pix[valid_samples], t_bins)
    gaze_valid_interp = _nearest_resample_bool(sample_times, gaze_valid, t_bins)

    roi_pix = np.flipud(dots_roi_deg * ppd)
    dxy_pix = dots_dxy_deg * ppd
    i_edges = np.arange(roi_pix[0, 0], roi_pix[0, 1] + dxy_pix, dxy_pix)
    j_edges = np.arange(roi_pix[1, 0], roi_pix[1, 1] + dxy_pix, dxy_pix)

    stim = bin_dots_to_stimulus(dots_pix, gaze_interp, i_edges, j_edges)[gaze_valid_interp]
    robs_valid = robs[gaze_valid_interp]
    if stim.shape[0] == 0:
        print(f"  [WARN] {label} has no valid dots frames after gaze masking")
        print("         Falling back to no visual prefilter for this session.")
        return np.full(len(target_cids), np.nan, dtype=np.float32), False, 'skipped'

    stas = calc_sta(
        stim[..., None],
        robs_valid,
        dots_sta_lags,
        reverse_correlate=False,
        progress=True,
    ).squeeze().cpu().numpy()
    max_snr_all, _, _ = calculate_rf_snr(stas, dots_dxy_deg)

    max_snr_target = np.full(len(target_cids), np.nan, dtype=np.float32)
    max_snr_target[found_mask] = max_snr_all[matched[found_mask]].astype(np.float32, copy=False)
    np.savez(cache_path, cids=target_cids, max_snr=max_snr_target)
    return max_snr_target, True, 'recomputed'


#%% ---------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------

sess = get_session(subject, date)
print(f"Session: {sess.name}")
aux_processed_path = Path(sess.processed_path)

fix_path = dataset_dir / fix_name
print(f"Loading fixrsvp from: {fix_path}")
assert fix_path.exists(), f"fixrsvp.dset not found at: {fix_path}"

dset_fix = DictDataset.load(fix_path)
add_calibrated_pupil_traces(dset_fix, aux_processed_path)
pupil_affine_info = add_affine_transformed_pupil_traces(dset_fix)

eye_configs = get_enabled_eye_configs(dset_fix, primary_eye)

print("Loaded fixrsvp DictDataset:")
print(f"  robs:       {dset_fix['robs'].shape}")
print(f"  keys:       {list(dset_fix.keys())}")
print(f"  enabled eye configs: {eye_configs}")

robs_serial  = _get_required(dset_fix, 'robs').astype(np.float32)
t_bins_serial = _get_required(dset_fix, 't_bins').astype(np.float64)
trial_inds_s = _get_required(dset_fix, 'trial_inds').astype(int)
time_inds_s  = _get_required(dset_fix, 'psth_inds').astype(int)
n_all_units  = robs_serial.shape[1]
cids_all     = np.array(dset_fix.metadata.get('cluster_ids', np.arange(n_all_units)))

missing_pct_mask, missing_pct_values, chronic_missing_units = _compute_missing_pct_mask(
    sess,
    t_bins_serial,
    cids_all,
    missing_pct_threshold,
)
robs_serial_masked = robs_serial.copy()
robs_serial_masked[~missing_pct_mask] = np.nan

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

# The calibrated az/el origin is not always at the fixation cross: some sessions
# have a systematic offset of several degrees (confirmed for 2026-03-08, 03-09,
# 03-02 right-eye). Subtract the median valid-DPI eye position so the window is
# centred on where the animal was actually fixating, not on the calibration origin.
_valid_ep = eyepos_cyclopean[dpi_valid_default]
fixation_center = np.median(_valid_ep, axis=0) if len(_valid_ep) > 0 else np.zeros(2, dtype=np.float32)
ecc = np.hypot(eyepos_cyclopean[:, 0] - fixation_center[0],
               eyepos_cyclopean[:, 1] - fixation_center[1])
ecc_mask = ecc <= fixation_radius_deg

n_total_bins   = len(ecc_mask)
n_dpi_valid    = dpi_valid_default.sum()
n_ecc_valid    = ecc_mask.sum()
n_newly_excl   = (~ecc_mask & dpi_valid_default).sum()

print(f"\nCriterion 0 — Eccentricity gate (radius={fixation_radius_deg}°)")
print(f"  Fixation center (median valid eyepos): ({fixation_center[0]:.3f}°, {fixation_center[1]:.3f}°)")
print(f"  Total serial bins:                 {n_total_bins}")
print(f"  dpi_valid bins:                    {n_dpi_valid} ({100*n_dpi_valid/n_total_bins:.1f}%)")
print(f"  ecc <= {fixation_radius_deg}° bins:              {n_ecc_valid} ({100*n_ecc_valid/n_total_bins:.1f}%)")
print(f"  Valid DPI but ecc > radius:        {n_newly_excl}  (newly excluded by this gate)")
print(f"  Max eccentricity (from center):    {ecc.max():.2f}°")

# dfs used for all pre-loop inclusion criteria computations
dfs_incl = dpi_valid_default & ecc_mask

print(f"\nRuntime datafilter — missing_pct < {missing_pct_threshold}")
print(f"  Valid bins after truncation mask:      {missing_pct_mask.sum()} / {missing_pct_mask.size}")
print(f"  Units with median missing >= threshold (kept as multi-unit fallback): {chronic_missing_units.sum()} / {n_all_units}")

# Trial-align with eccentricity-gated dfs.
# Two versions: masked (missing_pct NaNs applied, used for spike counts / covariance)
# and raw (no truncation-QC NaNs, used only for reliability so that chronic truncation
# doesn't artificially suppress the split-half correlation for well-tuned units).
robs_trial_incl, _, dfs_trial_incl, dur_trial_incl, _ = serial_to_trial_aligned(
    robs_serial_masked, eyepos_cyclopean, dfs_incl, trial_inds_s, time_inds_s)
robs_trial_raw, _, _, _, _ = serial_to_trial_aligned(
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
robs_mc     = robs_mc[:, iix]                          # (n_good, T, n_units) — truncation-QC NaNs applied
dfs_mc_incl = dfs_mc_incl[:, iix]
robs_mc_raw = robs_trial_raw[good_trials][:, iix]      # (n_good, T, n_units) — NaN only outside fixation window

# ── Criterion 1: Dots RF SNR ─────────────────────────────────────────────────
print(f"\nCriterion 1 — Dots RF SNR (pipeline/01-style ForageDots STA)")

max_snr_visual, visual_snr_available, visual_snr_source = _compute_dots_snr_if_available(
    dots_binned_path,
    dots_primary_csv,
    dots_primary_params,
    dots_primary_snr_cache,
    cids_all,
    f'{primary_eye}-eye',
)

_snr_source_label_map = {
    'cache': 'existing dots RF SNR cache',
    'recomputed': 'recomputed from dots_binned_data.dset',
    'skipped': 'criterion skipped',
}
yaml_visual_cids = np.asarray(session_yaml_config.get('visual', []), dtype=int)
yaml_visual_mask = np.isin(cids_all, yaml_visual_cids)
visual_source_label = _snr_source_label_map[visual_snr_source]
max_snr_bino = np.full(n_all_units, np.nan, dtype=np.float32)
max_snr_right = max_snr_visual.copy() if primary_eye == 'right' else np.full(n_all_units, np.nan, dtype=np.float32)
bino_snr_available = False
right_snr_available = visual_snr_available if primary_eye == 'right' else False
bino_snr_source = 'not-used'
right_snr_source = visual_snr_source if primary_eye == 'right' else 'not-used'
visual_mask_bino = np.zeros(n_all_units, dtype=bool)

if visual_snr_available:
    n_lo = np.isfinite(max_snr_visual) & (max_snr_visual >= snr_threshold_report)
    n_hi = np.isfinite(max_snr_visual) & (max_snr_visual >= snr_threshold_primary)
    print(f"  {primary_eye}-eye: SNR >= {snr_threshold_report}: {n_lo.sum():3d}  |  SNR >= {snr_threshold_primary}: {n_hi.sum():3d}")
else:
    print(f"  {primary_eye}-eye: dots RF SNR unavailable; visual criterion skipped")
print(f"  SNR source summary: {primary_eye}-eye={visual_source_label}")
print(f"  YAML visual list: {yaml_visual_mask.sum()} / {n_all_units}")

BASE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig_snr, ax_snr = plt.subplots(figsize=(5, 4))
if visual_snr_available:
    finite_snr = max_snr_visual[np.isfinite(max_snr_visual)]
    ax_snr.hist(finite_snr, bins=50, color='steelblue', alpha=0.75)
    for thr, ls in [(snr_threshold_report, ':'), (snr_threshold_primary, '--')]:
        ax_snr.axvline(thr, color='r', lw=1.0, linestyle=ls)
else:
    ax_snr.text(0.5, 0.5, 'Dots RF SNR unavailable',
                ha='center', va='center', transform=ax_snr.transAxes, color='gray')
ax_snr.set_xlabel('Max dots RF SNR')
ax_snr.set_ylabel('Count')
ax_snr.set_title(f'Dots RF SNR | {primary_eye}-eye\n{session_label}')
fig_snr.tight_layout()
fig_snr.savefig(BASE_FIGURES_DIR / 'dots_rf_snr_histogram.pdf')
plt.close(fig_snr)

if visual_source_mode == 'yaml_visual':
    visual_mask = yaml_visual_mask
    print(f"  visual_mask (yaml visual list): {visual_mask.sum()} / {n_all_units}")
elif visual_snr_available:
    visual_mask = max_snr_visual >= snr_threshold_primary
    print(f"  visual_mask ({primary_eye}-eye, >= {snr_threshold_primary}; {visual_source_label}): {visual_mask.sum()} / {n_all_units}")
else:
    print(f"  [ERROR] dots_rf mode selected but dots RF SNR unavailable for {session_label}.")
    print(f"  Run the dots calibration pipeline first, or use --visual-source yaml_visual.")
    sys.exit(1)

# ── Criterion 2: Total spike count ───────────────────────────────────────────
spikes_per_unit = np.nansum(robs_mc, axis=(0, 1))
spikes_ok = spikes_per_unit > total_spikes_threshold
print(f"\nCriterion 2 — Spike threshold (> {total_spikes_threshold})")
print(f"  Passing: {spikes_ok.sum()} / {n_all_units}")

# ── Trial diagnostics before reliability (diagnostic only) ───────────────────
pre_rel_candidate_mask = visual_mask & spikes_ok
trial_valid_frac = dfs_mc_incl.mean(axis=1)
if pre_rel_candidate_mask.any():
    robs_pre_rel = robs_mc[:, :, pre_rel_candidate_mask]
    trial_pop_spikes = np.nansum(robs_pre_rel, axis=(1, 2))
    trial_nan_frac_pre_rel = (
        (np.isnan(robs_pre_rel) & dfs_mc_incl[:, :, None]).sum(axis=(1, 2))
        / np.maximum(dfs_mc_incl.sum(axis=1) * pre_rel_candidate_mask.sum(), 1)
    )
    valid_unit_counts = np.sum(np.isfinite(robs_pre_rel), axis=2)
    trial_mean_psth = np.divide(
        np.nansum(robs_pre_rel, axis=2),
        valid_unit_counts,
        out=np.full((robs_pre_rel.shape[0], robs_pre_rel.shape[1]), np.nan, dtype=np.float64),
        where=valid_unit_counts > 0,
    )
    valid_trial_counts = np.sum(np.isfinite(trial_mean_psth), axis=0)
    mean_psth_across_trials = np.divide(
        np.nansum(trial_mean_psth, axis=0),
        valid_trial_counts,
        out=np.full(trial_mean_psth.shape[1], np.nan, dtype=np.float64),
        where=valid_trial_counts > 0,
    )
    trial_psth_dev = np.sqrt(np.nanmean((trial_mean_psth - mean_psth_across_trials[None, :]) ** 2, axis=1))
else:
    trial_pop_spikes = np.zeros(robs_mc.shape[0], dtype=np.float64)
    trial_nan_frac_pre_rel = np.zeros(robs_mc.shape[0], dtype=np.float64)
    trial_mean_psth = np.zeros((robs_mc.shape[0], robs_mc.shape[1]), dtype=np.float64)
    mean_psth_across_trials = np.zeros(robs_mc.shape[1], dtype=np.float64)
    trial_psth_dev = np.zeros(robs_mc.shape[0], dtype=np.float64)

trial_valid_z = _robust_zscore(trial_valid_frac)
trial_pop_spikes_z = _robust_zscore(trial_pop_spikes)
trial_nan_frac_z = _robust_zscore(trial_nan_frac_pre_rel)
trial_psth_dev_z = _robust_zscore(trial_psth_dev)
diag_bad_trials = (
    (trial_valid_z < -3.5)
    | (np.abs(trial_pop_spikes_z) > 3.5)
    | (trial_nan_frac_z > 3.5)
    | (trial_psth_dev_z > 2.0)
)
diag_bad_trial_indices = np.flatnonzero(diag_bad_trials)

print("\nPre-reliability bad-trial screen (diagnostic only)")
print(f"  Candidate units for trial diagnostics: {pre_rel_candidate_mask.sum()} / {n_all_units}")
print(f"  Flagged suspicious trials:            {diag_bad_trials.sum()} / {n_good_trials}")
print(f"  Low valid-bin frac outliers:          {(trial_valid_z < -3.5).sum()} / {n_good_trials}")
print(f"  Population spike outliers:            {(np.abs(trial_pop_spikes_z) > 3.5).sum()} / {n_good_trials}")
print(f"  NaN-fraction outliers:                {(trial_nan_frac_z > 3.5).sum()} / {n_good_trials}")
print(f"  PSTH-deviation outliers (>2.0 robust z): {(trial_psth_dev_z > 2.0).sum()} / {n_good_trials}")
print(f"  Flagged trial indices (0-based):      {diag_bad_trial_indices.tolist()}")

fig_trial_diag, axs_trial_diag = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
axs_trial_diag[0].plot(np.arange(len(mean_psth_across_trials)) * dt * 1000.0, mean_psth_across_trials,
                       color='black', lw=1.5)
axs_trial_diag[0].set_title(f'Mean PSTH across good trials | candidate units={pre_rel_candidate_mask.sum()}')
axs_trial_diag[0].set_xlabel('Time (ms)')
axs_trial_diag[0].set_ylabel('Mean spike count')

axs_trial_diag[1].hist(trial_psth_dev_z[np.isfinite(trial_psth_dev_z)], bins=30, color='steelblue', alpha=0.75)
axs_trial_diag[1].axvline(2.0, color='r', linestyle='--', lw=1.0)
axs_trial_diag[1].set_title('Trial PSTH deviation from session median PSTH')
axs_trial_diag[1].set_xlabel('Deviation robust z-score (median/MAD)')
axs_trial_diag[1].set_ylabel('Trial count')

axs_trial_diag[2].scatter(trial_pop_spikes, trial_psth_dev_z, s=18, alpha=0.75, color='darkorange', edgecolors='none')
axs_trial_diag[2].axhline(2.0, color='r', linestyle='--', lw=1.0)
axs_trial_diag[2].set_title('Trial population spikes vs PSTH deviation')
axs_trial_diag[2].set_xlabel('Total spikes across candidate units')
axs_trial_diag[2].set_ylabel('PSTH deviation robust z-score')

fig_trial_diag.tight_layout()
fig_trial_diag.savefig(BASE_FIGURES_DIR / f'trial_inclusion_diagnostics_{session_label}.pdf')
plt.close(fig_trial_diag)

# ── Criterion 3: Split-half PSTH reliability ─────────────────────────────────
# Primary reliability is still computed on robs_mc_raw so we do not change unit
# selection yet. A masked, NaN-aware reliability is computed alongside it as a
# diagnostic to evaluate whether excluding lost timepoints improves the estimate.
print(f"\nCriterion 3 — Split-half PSTH reliability (r², {n_reliability_splits} splits, >= {min_reliability})")
mean_reliability = _compute_split_half_reliability(robs_mc_raw, n_reliability_splits, seed=42)
reliability_ok   = mean_reliability >= min_reliability
print(f"  Passing (unmasked primary): {reliability_ok.sum()} / {n_all_units}")

# Diagnostic comparison: masked, NaN-aware reliability.
mean_reliability_masked = _compute_split_half_reliability(robs_mc, n_reliability_splits, seed=42)
reliability_ok_masked   = mean_reliability_masked >= min_reliability
print(f"  Passing (masked, NaN-aware; diagnostic): {reliability_ok_masked.sum()} / {n_all_units}")
n_recovered = int((reliability_ok_masked & ~reliability_ok).sum())
print(f"  Units recovered by masked NaN-aware reliability: {n_recovered}")

if (~diag_bad_trials).sum() >= 20:
    mean_reliability_stable = _compute_split_half_reliability(
        robs_mc_raw[~diag_bad_trials], n_reliability_splits, seed=42)
    reliability_ok_stable = mean_reliability_stable >= min_reliability
    print(f"  Passing after excluding flagged trials: {reliability_ok_stable.sum()} / {n_all_units}")
else:
    mean_reliability_stable = None
    reliability_ok_stable = None
    print("  Stable-trial reliability check skipped: too few trials after diagnostics")

reliability_candidate_values = mean_reliability[pre_rel_candidate_mask]
reliability_candidate_masked_values = mean_reliability_masked[pre_rel_candidate_mask]
reliability_hist_path = BASE_FIGURES_DIR / f'reliability_r2_histogram_{session_label}.pdf'
_save_reliability_histogram(
    reliability_hist_path,
    mean_reliability,
    reliability_candidate_values,
    min_reliability,
    session_label,
    masked_values=mean_reliability_masked,
    candidate_masked_values=reliability_candidate_masked_values,
)
print(f"  Saved {reliability_hist_path}")

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
if visual_source_mode == 'yaml_visual':
    print(f"After visual filter:     {(visual_mask & spikes_ok).sum()}"
        f"   (yaml visual list)")
elif visual_snr_available:
    print(f"After dots RF SNR:       {(visual_mask & spikes_ok).sum()}"
        f"   ({primary_eye}-eye, >= {snr_threshold_primary}; {visual_source_label})")
else:
    print(f"After dots RF SNR:       {(visual_mask & spikes_ok).sum()}"
        f"   (criterion skipped; all spike-threshold units retained)")
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

print("\nvisual — binocular YAML cross-check skipped (script now uses dots RF SNR on the primary eye)")

if yaml_right is not None and primary_eye == 'right':
    _xcheck(
        f"visual — right-eye (YAML snr>=5.0 gaborium) vs script right-eye (dots RF SNR >= {snr_threshold_primary})",
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
# Fixation-radius sweep — conditioned by eye trace type, Pool B held fixed
# -----------------------------------------------------------------------------

_eye_label_map = {
    'eyepos': 'cyclopean',
    'eyepos_left_minus_right': 'vergence (L-R)',
}

for eye_subdir_name, eye_source in eye_configs:
    eye_base_dir = BASE_FIGURES_DIR / eye_subdir_name
    eye_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Eye source: {eye_source}  →  {eye_base_dir}")
    print(f"{'='*70}")

    try:
        eyepos_serial_base, dfs_serial_base, eye_source_name = get_eye_trace_and_valid(
            dset_fix, eye_source=eye_source)
    except (KeyError, ValueError) as e:
        print(f"  Skipping: {e}")
        continue

    eyepos_serial_base = _ensure_2d_eyepos(eyepos_serial_base)
    dfs_serial_base = _as_bool_1d(dfs_serial_base, robs_serial.shape[0])
    display_eye_base = _eye_label_map.get(eye_source_name, eye_source_name)

    for radius in fixation_radii_deg:
        subdir_name = f'fixrad_{radius:.2f}deg'
        figures_dir = eye_base_dir / subdir_name
        figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Fixation radius: {radius}° | eye source: {eye_source_name}  →  {figures_dir}")
        print(f"{'='*70}")

        eyepos_serial = eyepos_serial_base
        display_eye_name = f'{display_eye_base}  r={radius}°'

        # Radius-specific eccentricity gate always uses cyclopean eccentricity.
        ecc_mask_r = ecc <= radius
        dfs_serial = dfs_serial_base & dpi_valid_default & ecc_mask_r
        print(f"  Bins inside radius: {ecc_mask_r.sum()} / {n_bins_serial}  "
              f"({100*ecc_mask_r.mean():.1f}%)")

        # ── 2. Trial-align
        robs_trial, eyepos_trial, dfs_trial, dur_trial, _ = serial_to_trial_aligned(
            robs_serial_masked, eyepos_serial, dfs_serial, trial_inds_s, time_inds_s)

        # ── 3. Trial filter
        good_trials_src = dur_trial > min_fix_dur_bins
        robs_mc_all   = robs_trial[good_trials_src]
        eyepos_mc_all = eyepos_trial[good_trials_src]
        dfs_mc_all    = dfs_trial[good_trials_src]
        dur_mc_all    = dur_trial[good_trials_src]
        n_good_trials_src = good_trials_src.sum()
        good_trials_b_within = np.zeros(n_good_trials_src, dtype=bool)
        n_b_trials = 0

        # ── 4. Guard: skip if pools are empty
        if pool_b_mask.sum() == 0:
            print("  Pool B is empty — skipping.")
            continue
        if pool_b_mask.sum() < 2:
            good_trials_b_within = np.ones(n_good_trials_src, dtype=bool)
            n_b_trials = n_good_trials_src
            print(f"  {n_good_trials_src} / {len(good_trials_src)} trials kept  (Pool B: {pool_b_mask.sum()})")
            print("  Pool B has fewer than 2 units — skipping covariance.")
            _run_cov = False
        elif pool_b_mask.sum() >= 2:
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

        # ── 5. Slice for Pool B covariance
        if _run_cov:
            robs_mc_b   = robs_mc_all[good_trials_b_within]
            eyepos_mc_b = eyepos_mc_all[good_trials_b_within]
            dfs_mc_b    = dfs_mc_all[good_trials_b_within]

            robs_cov   = robs_mc_b[:, iix][:, :, pool_b_mask]
            eyepos_cov = eyepos_mc_b[:, iix]
            dfs_cov    = dfs_mc_b[:, iix]
            valid_cov  = (dfs_cov
                          & np.isfinite(robs_cov.sum(axis=2))
                          & np.isfinite(eyepos_cov.sum(axis=2)))

        # ── 6. Covariance decomposition
        if _run_cov:
            print(f"  Running run_covariance_decomposition "
                  f"({robs_cov.shape[0]} trials, {robs_cov.shape[2]} Pool B units)...")
            results, last_mats = run_covariance_decomposition(
                robs_cov, eyepos_cov, valid_cov,
                window_sizes_ms=windows_ms, t_hist_ms=t_hist_ms,
                n_bins=n_bins, dt=dt,
            )
        else:
            results, last_mats = [], []

        if _run_cov:
            window_idx = next(
                (i for i, r in enumerate(results) if r['window_ms'] == focal_window_ms),
                0)
            win_label = results[window_idx]['window_ms'] if results else focal_window_ms

            Ctotal  = project_to_psd(last_mats[window_idx]['Total'])
            Cpsth   = project_to_psd(last_mats[window_idx]['PSTH'])
            Crate   = project_to_psd(last_mats[window_idx]['Intercept'])
            CnoiseC = project_to_psd(Ctotal - Crate)

            denom        = np.diag(Crate)
            fem_fraction = np.where(denom > 0, 1.0 - np.diag(Cpsth) / denom, np.nan)
            fem_fraction = fem_fraction[np.isfinite(fem_fraction)]

            sweep_windows_ms         = [r['window_ms'] for r in results]
            sweep_ff_corr            = [r['ff_corr_mean'] for r in results]
            sweep_ff_corr_sem        = [r['ff_corr_sem'] for r in results]
            sweep_ff_before_verg     = [r['ff_before_verg_mean'] for r in results]
            sweep_ff_before_verg_sem = [r['ff_before_verg_sem'] for r in results]
            sweep_ff_after_verg      = [r['ff_after_verg_mean'] for r in results]
            sweep_ff_after_verg_sem  = [r['ff_after_verg_sem'] for r in results]
        else:
            win_label = focal_window_ms
            fem_fraction = np.array([])
            sweep_windows_ms         = windows_ms
            sweep_ff_corr            = [np.nan] * len(windows_ms)
            sweep_ff_corr_sem        = [np.nan] * len(windows_ms)
            sweep_ff_before_verg     = [np.nan] * len(windows_ms)
            sweep_ff_before_verg_sem = [np.nan] * len(windows_ms)
            sweep_ff_after_verg      = [np.nan] * len(windows_ms)
            sweep_ff_after_verg_sem  = [np.nan] * len(windows_ms)
            Cpsth = Crate = CnoiseC = None

        # ── 7. Collect sweep entry
        frac_fem_summary.append({
            'radius': radius,
            'eye_source': eye_source,
            'eye_source_name': eye_source_name,
            'eye_subdir_name': eye_subdir_name,
            'subdir_name': subdir_name,
            'display_eye_name': display_eye_name,
            'display_eye_base': display_eye_base,
            'fem_fraction': fem_fraction.copy(),
            'cpsth_diag': np.diag(Cpsth) if Cpsth is not None else np.array([]),
            'crate_diag': np.diag(Crate) if Crate is not None else np.array([]),
            'n_b_trials': n_b_trials,
            'pool_a_n': pool_a_mask.sum(),
            'pool_b_n': pool_b_mask.sum(),
            'sweep_windows_ms': sweep_windows_ms,
            'sweep_ff_corr': sweep_ff_corr,
            'sweep_ff_corr_sem': sweep_ff_corr_sem,
            'sweep_ff_before_verg': sweep_ff_before_verg,
            'sweep_ff_before_verg_sem': sweep_ff_before_verg_sem,
            'sweep_ff_after_verg': sweep_ff_after_verg,
            'sweep_ff_after_verg_sem': sweep_ff_after_verg_sem,
        })

        # ── 8. Per-radius figure: 1-alpha histogram + FF vs window
        fig_c, axs_c = plt.subplots(1, 2, figsize=(10, 4))

        ax = axs_c[0]
        if fem_fraction.size:
            ax.hist(fem_fraction, bins=np.linspace(0, 1.2, 37), color='steelblue', alpha=0.7)
            med = np.nanmedian(fem_fraction)
            ax.axvline(med, color='steelblue', lw=1.5, linestyle='--',
                       label=f'med={med:.3f}')
        ax.axvline(1.0, color='k', lw=0.8, linestyle='--', alpha=0.4)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('1 − α  (C_FEM / C_rate)')
        ax.set_ylabel('Count')
        ax.set_title(f'FEM fraction [Pool B, n={pool_b_mask.sum()}]\n{display_eye_name}')
        ax.legend(frameon=False, fontsize=8)

        ax = axs_c[1]
        ws = np.array(sweep_windows_ms)
        _ff = np.array(sweep_ff_corr, dtype=float)
        _ff_sem = np.array(sweep_ff_corr_sem, dtype=float)
        if np.isfinite(_ff).any():
            ax.errorbar(ws, _ff, yerr=_ff_sem,
                        fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3)
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('FF (McFarland corrected)')
        ax.set_title(f'FF vs window size\n{display_eye_name}')

        fig_c.suptitle(f'{session_label} | {win_label} ms | Pool A={pool_a_mask.sum()} Pool B={pool_b_mask.sum()}')
        fig_c.tight_layout()
        fig_c.savefig(figures_dir / 'frac_fem_hist.pdf')
        plt.close(fig_c)

        # ── 9. Eye position heatmap
        ind_sorted = np.argsort(dur_mc_all)
        time_full = time_bins_full[:robs_mc_all.shape[1]]
        fig, ax = plt.subplots()
        im = ax.imshow(eyepos_mc_all[ind_sorted, :, 0], vmin=-0.5, vmax=0.5,
                   aspect='auto', cmap='coolwarm', interpolation='none', origin='lower',
                   extent=(float(time_full[0]), float(time_full[-1]), 0.0, float(robs_mc_all.shape[0])))
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Trial')
        ax.set_title(f'Eye X | {display_eye_name}')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Eye X (deg)')
        fig.tight_layout()
        fig.savefig(figures_dir / 'eyepos_heatmap.pdf')
        plt.close(fig)

        if _run_cov:
            cov_pdf_path = figures_dir / f'unit_cov_{session_label}.pdf'
            with PdfPages(cov_pdf_path) as pdf:
                for sel_b, cid in enumerate(cids_pool_b):
                    unit_robs = robs_mc_all[:, :, pool_b_inds[sel_b]]
                    ind_u = np.argsort(dur_mc_all)
                    unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
                    trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

                    fig, axs_u = plt.subplots(1, 2, figsize=(12, 4))
                    axs_u[0].set_title(f'Neuron {cid} | {display_eye_name} [Pool B]')
                    plot_raster(time_full[t_idx_r], trial_idx_r, height=1, ax=axs_u[0])
                    axs_u[0].axvline(0, color='r', lw=0.8)
                    axs_u[0].set_xlim(time_full[0], time_full[-1])
                    axs_u[0].set_xlabel('Time (s)')
                    axs_u[0].set_ylabel('Trial')
                    plot_cov_vs_distance(last_mats[window_idx], sel_b, sel_b, win_label, ax=axs_u[1])
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f"  Saved {cov_pdf_path}")

        n_time_analysis = len(iix)
        output = {
            'pool_a_mask': pool_a_mask,
            'pool_b_mask': pool_b_mask,
            'visual_mask': visual_mask,
            'spikes_ok': spikes_ok,
            'nan_ok': nan_ok,
            'mean_reliability': mean_reliability,
            'nan_frac_per_unit': nan_frac_per_unit,
            'good_trials': good_trials,
            'results': results,
            'last_mats': last_mats,
            'meta': {
                'fixation_radius_deg': radius,
                'fixation_radii_sweep': fixation_radii_deg,
                'pool_b_fixed_at_deg': fixation_radius_deg,
                'eye_source': eye_source,
                'eye_source_name': eye_source_name,
                'eye_subdir_name': eye_subdir_name,
                'dataset_path': str(fix_path),
                'dt': dt,
                't_hist_ms': t_hist_ms,
                'n_bins': n_bins,
                'valid_time_bins': n_time_analysis,
                'snr_threshold': snr_threshold_primary,
                'missing_pct_threshold': missing_pct_threshold,
                'min_reliability': min_reliability,
                'max_unit_nan_frac': max_unit_nan_frac,
                'max_bad_trial_frac': max_bad_trial_frac,
                'total_spikes_threshold': total_spikes_threshold,
                'min_fix_dur_bins': min_fix_dur_bins,
                'estimator': 'VisionCore.covariance v12',
            },
        }
        pkl_path = figures_dir / f'mcfarland_fixrsvp_{session_label}_{eye_source}_r{radius:.2f}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(output, f)
        print(f"  Saved {pkl_path}")

#%% ---------------------------------------------------------------------------
# Radius sweep summary figure
# -----------------------------------------------------------------------------

from collections import defaultdict

summary_by_eye = defaultdict(list)
for summary in frac_fem_summary:
    summary_by_eye[summary['eye_subdir_name']].append(summary)

summary_pdf_path = BASE_FIGURES_DIR / f'fixrad_sweep_summary_{session_label}.pdf'
with PdfPages(summary_pdf_path) as pdf:
    for eye_subdir_name, summaries in summary_by_eye.items():
        summaries = sorted(summaries, key=lambda s: s['radius'])
        radii     = [s['radius'] for s in summaries]
        fem_fracs = [s['fem_fraction'] for s in summaries]
        cpsths    = [s['cpsth_diag'] for s in summaries]
        crates    = [s['crate_diag'] for s in summaries]
        n_trials  = [s['n_b_trials'] for s in summaries]
        display_eye_base = summaries[0]['display_eye_base']

        medians = [np.nanmedian(f) if f.size else np.nan for f in fem_fracs]
        q25s    = [np.nanpercentile(f, 25) if f.size else np.nan for f in fem_fracs]
        q75s    = [np.nanpercentile(f, 75) if f.size else np.nan for f in fem_fracs]
        med_cpsth = [np.nanmedian(c) if len(c) > 0 else np.nan for c in cpsths]
        med_crate = [np.nanmedian(c) if len(c) > 0 else np.nan for c in crates]

        cmap    = plt.get_cmap('plasma')
        colors  = [cmap(i / max(1, len(radii) - 1)) for i in range(len(radii))]
        bins_ff = np.linspace(-0.1, 1.5, 50).tolist()

        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (r, ff) in enumerate(zip(radii, fem_fracs)):
            if ff.size == 0:
                continue
            ax.hist(ff, bins=bins_ff, histtype='step', color=colors[i], lw=1.8,
                    label=f'{r}°  n={ff.size}  med={np.nanmedian(ff):.2f}')
        ax.axvline(1.0, color='k', lw=0.8, linestyle='--', alpha=0.4)
        ax.axvline(0.0, color='k', lw=0.5, linestyle=':', alpha=0.3)
        ax.set_xlabel('1 − alpha  (FEM fraction)')
        ax.set_ylabel('Unit count')
        ax.set_title(f'1−alpha by fixation radius\n{session_label} | {display_eye_base} | Pool B fixed at {fixation_radius_deg}°')
        ax.legend(frameon=False, fontsize=8, title='radius')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.fill_between(radii, q25s, q75s, alpha=0.18, color='steelblue', label='IQR')
        ax.plot(radii, medians, 'o-', color='steelblue', lw=2.0, ms=7, label='Median 1−alpha')
        for r, m in zip(radii, medians):
            if np.isfinite(m):
                ax.text(r, m + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=8)
        ax.axhline(0.0, color='k', lw=0.5, linestyle=':')
        ax.set_xlabel('Fixation radius (°)')
        ax.set_ylabel('1 − alpha')
        ax.set_title(f'Median 1−alpha vs fixation radius\n{session_label} | {display_eye_base} | {focal_window_ms} ms window')
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot(radii, med_cpsth, 's-', color='darkorange', lw=1.8, ms=7)
        for r, v in zip(radii, med_cpsth):
            if np.isfinite(v):
                axes[0].text(r, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        axes[0].set_xlabel('Fixation radius (°)')
        axes[0].set_ylabel('Median diag(Cpsth)')
        axes[0].set_title('Median diag(Cpsth) vs radius')

        axes[1].plot(radii, med_crate, 's-', color='steelblue', lw=1.8, ms=7)
        for r, v in zip(radii, med_crate):
            if np.isfinite(v):
                axes[1].text(r, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        axes[1].set_xlabel('Fixation radius (°)')
        axes[1].set_ylabel('Median diag(Crate)')
        axes[1].set_title('Median diag(Crate) vs radius')
        fig.suptitle(f'{session_label} | {display_eye_base} | {focal_window_ms} ms | Pool B fixed at {fixation_radius_deg}°', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].bar([str(r) for r in radii], n_trials, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Fixation radius (°)')
        axes[0].set_ylabel('Covariance trials')
        axes[0].set_title('Trial count vs radius')

        ratio = []
        for p, r_ in zip(cpsths, crates):
            if len(p) > 0 and len(r_) > 0:
                valid = r_ > 0
                ratio.append(np.nanmedian(p[valid] / r_[valid]) if valid.any() else np.nan)
            else:
                ratio.append(np.nan)
        axes[1].plot(radii, ratio, 'o-', color='darkorange', lw=1.8, ms=7,
                     label='median diag(Cpsth)/diag(Crate)')
        axes[1].plot(radii, [1 - m for m in medians], 's--', color='steelblue', lw=1.4, ms=6,
                     label='1 − (median 1−alpha)')
        axes[1].set_xlabel('Fixation radius (°)')
        axes[1].set_ylabel('Ratio')
        axes[1].set_title('Cpsth/Crate ratio vs radius')
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f'{session_label} | {display_eye_base}', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"\nSaved sweep summary: {summary_pdf_path}")
print("\nAll radii complete.")

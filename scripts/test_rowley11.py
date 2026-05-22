#%%
"""
test_rowley11.py — session-YAML-backed 03-02 loading test

Changes from v8:

0. Dataset resolution via session YAMLs (NEW)
    Uses `experiments/dataset_configs/sessions/*_V1.yaml` as the source of
    truth for the dataset directory instead of walking legacy processed_mvp
    folders. The YAML `directory:` field now resolves `fixrsvp.dset`.

1. Criterion 0 — Eccentricity gate (NEW)
   Applies hypot(eyepos_cyclopean) <= 1.5° to dfs_serial for ALL eye sources.
   Matches faceRadius=1.5° from FixRsvpTrial experiment parameters.
   Previously dfs_serial used only dpi_valid (hardware validity); ~5.7% of bins
   with eccentricity > 1.5° and valid DPI tracking were silently included.

2. Criterion 1 — Visual responsiveness via dots RF SNR (UPDATED)
    SNR computed from step-01 dots calibration inputs using ForageDots trials,
    calibrated eye traces, `bin_dots_to_stimulus(...)`, and
    `calculate_rf_snr(...)`. The primary-eye dots SNR is used for visual_mask.
    YAML can also be selected explicitly as an alternate visual source.

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
import subprocess
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
visual_source_mode = cli_args.get('visual-source', os.environ.get('ROWLEY_VISUAL_SOURCE', 'dots_rf')).strip().lower()
if visual_source_mode not in {'dots_rf', 'yaml_visual'}:
    raise ValueError(f"Unsupported visual source: {visual_source_mode!r}. Expected one of: dots_rf, yaml_visual")
single_session_requested = any(key in cli_args for key in ('subject', 'date', 'primary-eye', 'dataset-dir', 'session-yaml'))
run_all_sessions = _env_flag('ROWLEY_RUN_ALL_DATASETS', not single_session_requested)
session_filter = os.environ.get('ROWLEY_SESSION_FILTER')

if run_all_sessions and not _env_flag('ROWLEY_CHILD_RUN', False):
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
    mcfarland_dir = FIGURES_DIR / 'mcfarland'
    pkl_paths = sorted(mcfarland_dir.glob('*_v12/cyclopean-dpi/mcfarland_fixrsvp_*_default.pkl'))
    if not pkl_paths:
        print("No cyclopean PKL files found — skipping cross-session summary.")
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
            sess_lbl  = pkl_path.parent.parent.name.replace('_v12', '')
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

            fig_cs.suptitle('Cross-session covariance summary (v12, cyclopean DPI)', fontsize=11)
            fig_cs.tight_layout(rect=(0, 0, 1, 0.97))
            cs_path = mcfarland_dir / 'cross_session_summary_v12.pdf'
            fig_cs.savefig(cs_path, bbox_inches='tight')
            plt.close(fig_cs)
            print(f"Saved {cs_path}")

    # ── Combined FEM histogram per eye trace type ─────────────────────────────
    print("\nBuilding combined FEM histogram per eye trace type...")
    from collections import defaultdict
    all_pkl_paths = sorted(mcfarland_dir.glob('*_v12/*/mcfarland_fixrsvp_*.pkl'))
    _subdir_pkls = defaultdict(list)
    for _p in all_pkl_paths:
        _subdir_pkls[_p.parent.name].append(_p)

    _subdirs_ordered = sorted(k for k in _subdir_pkls if k != 'binocular')
    _bins_comb   = np.linspace(0, 1.2, 37)
    _bctrs_comb  = 0.5 * (_bins_comb[:-1] + _bins_comb[1:])
    _bwidth_comb = _bins_comb[1] - _bins_comb[0]
    _cmap_comb   = plt.get_cmap('tab10')

    if _subdirs_ordered:
        fig_comb, axs_comb = plt.subplots(
            1, len(_subdirs_ordered),
            figsize=(4 * len(_subdirs_ordered), 4.5),
            squeeze=False,
        )
        for _si, _subdir in enumerate(_subdirs_ordered):
            ax = axs_comb[0, _si]
            _sess_counts = []
            _all_fracs   = []
            for _pkl_p in sorted(_subdir_pkls[_subdir]):
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
                _slbl = _pkl_p.parent.parent.name.replace('_v12', '')
                _sess_counts.append((_slbl, _counts))
                _all_fracs.append(_ff)

            if not _sess_counts:
                ax.set_title(_subdir, fontsize=8)
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
            ax.set_title(f'{_subdir}\n(n sessions={len(_sess_counts)})', fontsize=8)
            if _si == 0:
                ax.set_ylabel('Count (all sessions)')

        fig_comb.suptitle('FEM fraction — combined histogram by eye trace type (v12, 20 ms window)',
                          fontsize=11)
        fig_comb.tight_layout(rect=(0, 0, 1, 0.97))
        comb_path = mcfarland_dir / 'cross_session_fem_combined_v12.pdf'
        fig_comb.savefig(comb_path, bbox_inches='tight')
        plt.close(fig_comb)
        print(f"Saved {comb_path}")

    sys.exit(0)

windows_ms             = [5, 10, 20, 40, 80]
total_spikes_threshold = 200
valid_time_bins        = 240
dt                     = 1 / 240.0
t_hist_ms              = 50
intercept_diag_base_edges_deg = np.array([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.2], dtype=np.float64)
intercept_diag_min_pairs_per_bin = 2000
n_bins                 = 15

min_fix_dur_bins = 20

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
min_reliability      = 0 # leaving out, too strict. 0.05 #previously 0.1,
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

BASE_FIGURES_DIR = FIGURES_DIR / 'mcfarland' / f'{subject}_{date}_v12'

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


def _compute_split_half_reliability(robs_trials, n_splits, seed=42):
    n_trials, _, n_units = robs_trials.shape
    rng_rel = np.random.default_rng(seed)
    r2_accum = np.zeros(n_units, dtype=np.float64)
    if n_trials < 2:
        return r2_accum

    for _split in range(n_splits):
        perm = rng_rel.permutation(n_trials)
        half = n_trials // 2
        if half == 0:
            break
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            psth_a = np.nanmean(robs_trials[perm[:half]], axis=0)
            psth_b = np.nanmean(robs_trials[perm[half:2 * half]], axis=0)
        for unit_idx in range(n_units):
            a = psth_a[:, unit_idx]
            b = psth_b[:, unit_idx]
            fin = np.isfinite(a) & np.isfinite(b)
            if fin.sum() > 2 and np.std(a[fin]) > 0 and np.std(b[fin]) > 0:
                r2_accum[unit_idx] += np.corrcoef(a[fin], b[fin])[0, 1] ** 2
    return r2_accum / n_splits


def _compute_missing_pct_mask(session, t_bins, cids, threshold):
    missing_pct_fun = session.get_missing_pct_interp(cids)
    pct = _to_numpy(missing_pct_fun(t_bins)).astype(np.float32)
    valid_mask = pct < threshold
    chronic_multi_units = np.nanmedian(pct, axis=0) >= threshold
    valid_mask[:, chronic_multi_units] = True
    return valid_mask, pct, chronic_multi_units


def _save_reliability_histogram(output_path, all_values, candidate_values, threshold, session_label,
                                masked_values=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, 1.0, 41).tolist()

    all_values = np.asarray(all_values, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    candidate_values = candidate_values[np.isfinite(candidate_values)]

    if all_values.size:
        ax.hist(all_values, bins=bins, color='lightgray', alpha=0.9, label=f'all units unmasked (n={all_values.size})')
    if masked_values is not None:
        masked_values = np.asarray(masked_values, dtype=np.float64)
        masked_values = masked_values[np.isfinite(masked_values)]
        if masked_values.size:
            ax.hist(masked_values, bins=bins, color='salmon', alpha=0.6,
                    label=f'all units masked (n={masked_values.size})')
    if candidate_values.size:
        ax.hist(candidate_values, bins=bins, color='steelblue', alpha=0.7,
                label=f'visual & spikes unmasked (n={candidate_values.size})')
    else:
        ax.text(0.5, 0.5, 'No finite candidate-unit reliability values',
                ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.axvline(threshold, color='r', linestyle='--', lw=1.0, label=f'threshold={threshold:.2f}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Split-half PSTH reliability (r²)')
    ax.set_ylabel('Unit count')
    ax.set_title(f'Reliability distribution\n{session_label}')
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
        enabled_configs.append(('binocular', 'binocular_diff'))
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
    visual_mask = np.ones(n_all_units, dtype=bool)
    print(f"  visual_mask fallback (all units kept): {visual_mask.sum()} / {n_all_units}")

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
# Reliability is computed on robs_mc_raw (truncation-QC NaNs NOT applied) so that
# units with high missing_pct are judged on their actual firing pattern, not on
# artificially sparse data.  robs_mc (masked) is still used for everything else.
print(f"\nCriterion 3 — Split-half PSTH reliability (r², {n_reliability_splits} splits, >= {min_reliability})")
mean_reliability = _compute_split_half_reliability(robs_mc_raw, n_reliability_splits, seed=42)
reliability_ok   = mean_reliability >= min_reliability
print(f"  Passing (unmasked spikes): {reliability_ok.sum()} / {n_all_units}")

# Diagnostic comparison: what would the masked version give?
mean_reliability_masked = _compute_split_half_reliability(robs_mc, n_reliability_splits, seed=42)
reliability_ok_masked   = mean_reliability_masked >= min_reliability
print(f"  Passing (truncation-QC masked, for comparison): {reliability_ok_masked.sum()} / {n_all_units}")
n_recovered = int((reliability_ok & ~reliability_ok_masked).sum())
print(f"  Units recovered by unmasking: {n_recovered}")

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
reliability_hist_path = BASE_FIGURES_DIR / f'reliability_r2_histogram_{session_label}.pdf'
_save_reliability_histogram(
    reliability_hist_path,
    mean_reliability,
    reliability_candidate_values,
    min_reliability,
    session_label,
    masked_values=mean_reliability_masked,
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
        print("  Pool A is empty — skipping.")
        continue
    if pool_b_mask.sum() < 2:
        good_trials_b_within = np.ones(n_good_trials_src, dtype=bool)
        n_b_trials = n_good_trials_src
        print(f"  {n_good_trials_src} / {len(good_trials_src)} trials kept  (Pool B: {pool_b_mask.sum()})")
        print("  Pool B has fewer than 2 units — skipping covariance.")
        _run_cov = False
    if pool_b_mask.sum() == 0:
        print(f"  {n_good_trials_src} / {len(good_trials_src)} trials kept  (Pool B: 0)")
        print("  Pool B is empty — skipping covariance.")
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
        win_bins_diag = int(results[window_idx]['window_bins']) if results else max(1, int(win_label / (dt * 1000)))
        t_hist_bins_cov = int(t_hist_ms / (dt * 1000))

        Ctotal  = project_to_psd(last_mats[window_idx]['Total'])
        Cpsth   = project_to_psd(last_mats[window_idx]['PSTH'])
        Crate   = project_to_psd(last_mats[window_idx]['Intercept'])
        Cfem    = project_to_psd(last_mats[window_idx]['FEM'])
        CnoiseC = project_to_psd(Ctotal - Crate)
        MeanRates = results[window_idx]['Erates']
        diag_slopes_current = _compute_diag_linear_slopes(
            last_mats[window_idx]['Ceye'],
            last_mats[window_idx]['bin_centers'],
            last_mats[window_idx]['count_e'],
        )
        intercept_diagnostics = _compute_intercept_diagnostics(
            robs_cov,
            eyepos_cov,
            valid_cov,
            Cpsth,
            Ctotal,
            t_count_bins=win_bins_diag,
            t_hist_bins=t_hist_bins_cov,
        )

        denom        = np.diag(Crate)
        fem_fraction = np.where(denom > 0, 1.0 - np.diag(Cpsth) / denom, np.nan)
        fem_fraction = fem_fraction[np.isfinite(fem_fraction)]

        sweep_windows_ms         = [r['window_ms']            for r in results]
        sweep_ff_corr            = [r['ff_corr_mean']          for r in results]
        sweep_ff_corr_sem        = [r['ff_corr_sem']           for r in results]
        sweep_ff_before_verg     = [r['ff_before_verg_mean']   for r in results]
        sweep_ff_before_verg_sem = [r['ff_before_verg_sem']    for r in results]
        sweep_ff_after_verg      = [r['ff_after_verg_mean']    for r in results]
        sweep_ff_after_verg_sem  = [r['ff_after_verg_sem']     for r in results]
    else:
        win_label = windows_ms[0]
        diag_slopes_current = np.array([], dtype=np.float64)
        intercept_diagnostics = None
        fem_fraction = np.array([])
        sweep_windows_ms         = windows_ms
        sweep_ff_corr            = [np.nan] * len(windows_ms)
        sweep_ff_corr_sem        = [np.nan] * len(windows_ms)
        sweep_ff_before_verg     = [np.nan] * len(windows_ms)
        sweep_ff_before_verg_sem = [np.nan] * len(windows_ms)
        sweep_ff_after_verg      = [np.nan] * len(windows_ms)
        sweep_ff_after_verg_sem  = [np.nan] * len(windows_ms)
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
        'sweep_windows_ms':          sweep_windows_ms,
        'sweep_ff_corr':             sweep_ff_corr,
        'sweep_ff_corr_sem':         sweep_ff_corr_sem,
        'sweep_ff_before_verg':      sweep_ff_before_verg,
        'sweep_ff_before_verg_sem':  sweep_ff_before_verg_sem,
        'sweep_ff_after_verg':       sweep_ff_after_verg,
        'sweep_ff_after_verg_sem':   sweep_ff_after_verg_sem,
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
        'diag_slopes_current': diag_slopes_current,
        'intercept_diagnostics': intercept_diagnostics,
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

    # ── 12. Per-unit raster PDF — spike-threshold units
    unit_pdf_path = figures_dir / f'unit_rasters_{session_label}.pdf'
    with PdfPages(unit_pdf_path) as pdf:
        spike_unit_inds = np.where(spikes_ok)[0]
        spike_unit_cids = cids_all[spikes_ok]
        for unit_idx, cid in zip(spike_unit_inds, spike_unit_cids):
            unit_robs   = robs_mc_all[:, :, unit_idx]
            ind_u       = np.argsort(dur_mc_all)
            unit_sorted = np.nan_to_num(unit_robs[ind_u], nan=0.0)
            trial_idx_r, t_idx_r = np.where(unit_sorted > 0)

            fig, ax_r = plt.subplots(figsize=(8, 4))
            ax_r.set_title(f'Neuron {cid} | {display_eye_name} [spikes > {total_spikes_threshold}]')
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

        pairdist_pdf_path = figures_dir / f'pairwise_distance_bins_{session_label}.pdf'
        t_hist_bins_cov = int(t_hist_ms / (dt * 1000))
        with PdfPages(pairdist_pdf_path) as pdf:
            for result_row, mats_row in zip(results, last_mats):
                t_count_bins = int(result_row['window_bins'])
                pooled_distances = compute_pooled_pairwise_distances(
                    eyepos_cov,
                    valid_cov,
                    t_count_bins=t_count_bins,
                    t_hist_bins=max(t_hist_bins_cov, t_count_bins),
                    min_seg_len=36,
                )
                plot_pairwise_distance_distribution_page(pdf, mats_row, result_row, pooled_distances)
        print(f"  Saved {pairdist_pdf_path}")

        if intercept_diagnostics is not None:
            intercept_pdf_path = figures_dir / f'intercept_diagnostics_{session_label}.pdf'
            with PdfPages(intercept_pdf_path) as pdf:
                plot_intercept_diagnostic_page(
                    pdf,
                    intercept_diagnostics,
                    last_mats[window_idx],
                    Cpsth,
                    fem_fraction,
                    diag_slopes_current,
                    display_eye_name,
                    win_label,
                )
            print(f"  Saved {intercept_pdf_path}")

    # Per-unit 2D PDF — cyclopean Pool B only
    if _result_2d is not None:
        _C2d_u      = _result_2d['C2d']
        _count2d_u  = _result_2d['count2d']
        _dv_means_u = _result_2d['dv_cell_means']
        _dc_edges_u = _result_2d['dc_edges']
        _dv_edges_u = _result_2d['dv_edges']
        _C_slope_u  = _result_2d['C_near_slope']
        _C_int_u    = _result_2d['C_near_intercept']
        _n_bc = _C2d_u.shape[0]; _n_bv = _C2d_u.shape[1]
        _colors_bc_u = ['steelblue', 'darkorange', 'teal']

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
                for _edge in _dv_edges_u[1:-1]:
                    _ax2.axvline(_edge, color='0.85', lw=0.8, linestyle=':', zorder=0)
                for _bc_k in range(_n_bc):
                    _x2, _y2 = [], []
                    for _bv_k in range(_n_bv):
                        if (_count2d_u[_bc_k, _bv_k] >= 5
                                and np.isfinite(_dv_means_u[_bc_k, _bv_k])
                                and np.isfinite(_C2d_u[_bc_k, _bv_k, sel_b, sel_b])):
                            _x2.append(_dv_means_u[_bc_k, _bv_k])
                            _y2.append(_C2d_u[_bc_k, _bv_k, sel_b, sel_b])
                    if _x2:
                        if _bc_k + 1 < len(_dc_edges_u):
                            _dc_lo = _dc_edges_u[_bc_k]
                            _dc_hi = _dc_edges_u[_bc_k + 1]
                            _dc_tag = 'NEAR' if _bc_k == 0 else ('FAR' if _bc_k == _n_bc - 1 else 'MID')
                            _lbl2 = f'd_c {_dc_lo:.3f}-{_dc_hi:.3f} deg [{_dc_tag}]'
                        else:
                            _lbl2 = f'd_c bin {_bc_k}'
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
                _ax2.set_xlabel('Vergence distance d_v RMS (deg)')
                _ax2.set_ylabel('C2d[bc,bv,i,i]')
                _ax2.set_title(
                    f'2D conditional var | Neuron {cid}\n'
                    f'd_v range {_dv_edges_u[0]:.3f}-{_dv_edges_u[-1]:.3f} deg'
                )
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
        'yaml_visual_mask':  yaml_visual_mask,
        'visual_mask_bino':  visual_mask_bino,
        'max_snr_visual':    max_snr_visual,
        'visual_snr_source': visual_snr_source,
        'visual_snr_mode':   visual_source_mode,
        'spikes_ok':         spikes_ok,
        'reliability_ok':    reliability_ok,
        'nan_ok':            nan_ok,
        'max_snr_bino':      max_snr_bino,
        'max_snr_right':     max_snr_right,
        'mean_reliability':  mean_reliability,
        'reliability_candidate_values': reliability_candidate_values,
        'missing_pct_threshold': missing_pct_threshold,
        'missing_pct_values': missing_pct_values,
        'chronic_missing_units': chronic_missing_units,
        'nan_frac_per_unit': nan_frac_per_unit,
        'good_trials':       good_trials,
        'good_trials_b_within': good_trials_b_within,
        'bino_snr_source':   bino_snr_source,
        'right_snr_source':  right_snr_source,
        'pupil_affine_info': pupil_affine_info,
        'diag_bad_trials':   diag_bad_trials,
        'diag_bad_trial_indices': diag_bad_trial_indices,
        'trial_valid_frac':  trial_valid_frac,
        'trial_pop_spikes':  trial_pop_spikes,
        'trial_nan_frac_pre_rel': trial_nan_frac_pre_rel,
        'trial_psth_dev':    trial_psth_dev,
        'trial_psth_dev_z':  trial_psth_dev_z,
        'mean_psth_across_trials': mean_psth_across_trials,
        # covariance results
        'windows':           windows_ms,
        'results':           results,
        'last_mats':         last_mats,
        'result_2d':         _result_2d,
        'result_2d_shuff':   _result_2d_shuff,
        'Cverg2d':           Cverg2d,
        'intercept_diagnostics': intercept_diagnostics,
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
            'visual_snr_mode':       visual_source_mode,
            'visual_snr_source':     visual_snr_source,
            'bino_snr_source':       bino_snr_source,
            'right_snr_source':      right_snr_source,
            'pupil_affine_info':     pupil_affine_info,
            'missing_pct_threshold': missing_pct_threshold,
            'min_reliability':       min_reliability,
            'max_unit_nan_frac':     max_unit_nan_frac,
            'max_bad_trial_frac':    max_bad_trial_frac,
            'total_spikes_threshold': total_spikes_threshold,
            'min_fix_dur_bins':      min_fix_dur_bins,
            'n_bins_c_2d':  n_bins_c_2d,
            'n_bins_v_2d':  n_bins_v_2d,
            'min_pairs_2d': min_pairs_2d,
            'estimator':    'VisionCore.covariance v12',
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
    n_cols  = 5
    fig_sum, axs_sum = plt.subplots(
        1, n_cols,
        figsize=(5 * n_cols, 3.5),
        squeeze=False,
    )

    for row, summary in enumerate(_fem_summaries[:1]):
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
            im = ax.imshow(
                _c2d['count2d'],
                origin='lower',
                cmap='Blues',
                aspect='auto',
                extent=(_c2d['dv_edges'][0], _c2d['dv_edges'][-1], _c2d['dc_edges'][0], _c2d['dc_edges'][-1]),
            )
            ax.set_xlabel('d_v RMS (deg)'); ax.set_ylabel('d_c RMS (deg)')
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
        ws       = np.array(summary['sweep_windows_ms'])
        _before  = np.array(summary['sweep_ff_before_verg'],     dtype=float)
        _b_sem   = np.array(summary['sweep_ff_before_verg_sem'], dtype=float)
        _after   = np.array(summary['sweep_ff_after_verg'],      dtype=float)
        _a_sem   = np.array(summary['sweep_ff_after_verg_sem'],  dtype=float)
        if np.isfinite(_before).any():
            ax.errorbar(ws, _before, yerr=_b_sem,
                        fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3,
                        label='before verg')
        if np.isfinite(_after).any():
            ax.errorbar(ws, _after, yerr=_a_sem,
                        fmt='s--', color='darkorange', lw=1.5, ms=5, capsize=3,
                        label='after verg')
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('Mean FF ± SEM')
        ax.set_title(f'FF vs window size\n{row_title}')
        ax.legend(frameon=False, fontsize=7)

    fig_sum.suptitle(f'Covariance decomposition summary | {session_label} (v12)')
    fig_sum.tight_layout(rect=(0, 0, 1, 0.97))
    summary_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.pdf'
    fig_sum.savefig(summary_path, bbox_inches='tight')
    plt.close(fig_sum)
    print(f"\nSaved {summary_path}")

    # FEM fraction comparison across conditions (2 rows: histogram + FF vs window)
    n_conds_fem = len(_fem_summaries)
    _xmax_fem   = 1.2
    _bins_fem   = np.linspace(0, _xmax_fem, 37)

    fig_fem, axs_fem = plt.subplots(
        2, n_conds_fem, figsize=(3 * n_conds_fem, 7),
        squeeze=False,
    )
    for col, summary in enumerate(_fem_summaries):
        # ── Row 0: FEM fraction histogram ────────────────────────────────────
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

        # ── Row 1: FF vs window size ──────────────────────────────────────────
        ax = axs_fem[1, col]
        ws = np.array(summary['sweep_windows_ms'])
        _ff_vals = np.array(summary['sweep_ff_corr'], dtype=float)
        _ff_sems = np.array(summary['sweep_ff_corr_sem'], dtype=float)
        if np.isfinite(_ff_vals).any():
            ax.errorbar(ws, _ff_vals, yerr=_ff_sems,
                        fmt='o-', color='steelblue', lw=1.5, ms=5, capsize=3)
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        if col == 0:
            ax.set_ylabel('Mean FF ± SEM')
        ax.set_title(f'FF vs window size\n{summary["subdir_name"]}', fontsize=8)

    fig_fem.suptitle(f'FEM fraction comparison | {session_label} (v12)', fontsize=11)
    fig_fem.tight_layout(rect=(0, 0, 1, 0.97))
    fem_cmp_path = BASE_FIGURES_DIR / f'fem_fraction_comparison_{session_label}.pdf'
    fig_fem.savefig(fem_cmp_path, bbox_inches='tight')
    plt.close(fig_fem)
    print(f"Saved {fem_cmp_path}")

# Text summary
_bin_edges      = np.linspace(0, 1.0, 31)
_bin_edges_wide = np.linspace(0, 1.2, 37)

txt_path = BASE_FIGURES_DIR / f'frac_fem_summary_{session_label}.txt'
with open(txt_path, 'w') as f:
    f.write(f"Covariance decomposition summary — v12\n")
    f.write(f"Session: {session_label}\n")
    f.write(f"Estimator: VisionCore.covariance.run_covariance_decomposition\n")
    f.write(f"\nInclusion criteria:\n")
    f.write(f"  fixation_radius_deg={fixation_radius_deg}  snr_threshold={snr_threshold_primary}\n")
    f.write(f"  visual_snr_mode={visual_source_mode}  visual_snr_source={visual_snr_source}\n")
    f.write(f"  missing_pct_threshold={missing_pct_threshold}\n")
    f.write(f"  min_reliability={min_reliability}  max_unit_nan_frac={max_unit_nan_frac}\n")
    f.write(f"  max_bad_trial_frac={max_bad_trial_frac}  total_spikes_threshold={total_spikes_threshold}\n")
    f.write(f"\nPool A: {pool_a_mask.sum()} / {n_all_units}  cids={sorted(cids_pool_a.tolist())}\n")
    f.write(f"Pool B: {pool_b_mask.sum()} / {n_all_units}  cids={sorted(cids_pool_b.tolist())}\n")
    f.write(f"Good trials: {n_good_trials} / {n_all_trials}\n")
    f.write(f"Pool B covariance trials: {n_b_trials_base} / {n_all_trials}\n")
    f.write(f"Flagged trial indices (0-based within good trials): {diag_bad_trial_indices.tolist()}\n")
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

        _diag_current = summary.get('diag_slopes_current')
        _diag_intercept = summary.get('intercept_diagnostics')
        if _diag_intercept is not None:
            f.write("  --- Intercept diagnostics [20 ms] ---\n")
            if _diag_current is not None and np.isfinite(_diag_current).any():
                _dc = _diag_current[np.isfinite(_diag_current)]
                f.write(f"  current percentile bins: frac positive diag slope={(_dc > 0).mean():.3f}, median slope={np.median(_dc):.4f}\n")
            _df = _diag_intercept.get('diag_slopes_adaptive')
            if _df is not None and np.isfinite(_df).any():
                _dfv = _df[np.isfinite(_df)]
                f.write(f"  adaptive degree bins:   frac positive diag slope={(_dfv > 0).mean():.3f}, median slope={np.median(_dfv):.4f}\n")
            _adaptive_edges = summary['intercept_diagnostics']['adaptive_bin_edges']
            _current_first_hi = _adaptive_edges[1] if len(_adaptive_edges) > 1 else np.nan
            _adaptive_counts = summary['intercept_diagnostics']['adaptive_count_e']
            _first_pairs = int(_adaptive_counts[0]) if len(_adaptive_counts) else 0
            f.write(f"  adaptive first edge upper bound: {_current_first_hi:.4f} deg\n")
            f.write(f"  adaptive first-bin pairs: {_first_pairs}  (target>={summary['intercept_diagnostics']['adaptive_min_pairs_per_bin']})\n")
            f.write(f"  FEM medians:\n")
            f.write(f"    current percentile first-bin: {_finite_median(summary['fem_fraction']):.4f}\n")
            f.write(f"    adaptive lowest-bin:          {_finite_median(_diag_intercept['fem_adaptive_lowest_bin']):.4f}\n")
            f.write(f"    adaptive linear first-bin:    {_finite_median(_diag_intercept['fem_adaptive_linear_first']):.4f}\n")
            f.write(f"    adaptive linear zero:         {_finite_median(_diag_intercept['fem_adaptive_linear_zero']):.4f}\n")
            f.write(f"    adaptive isotonic:            {_finite_median(_diag_intercept['fem_adaptive_isotonic']):.4f}\n")
            f.write("\n")

        _write_ff(f, "FF before conservative Cverg2d (McFarland residual)", summary['ff_before_verg2d'])
        _write_ff(f, "FF after  conservative Cverg2d", summary['ff_after_verg2d'])
        f.write("\n")

print(f"Saved {txt_path}")
print("\nAll conditions complete.")

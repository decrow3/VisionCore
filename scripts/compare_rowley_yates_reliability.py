import os
import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, os.path.abspath(REPO_ROOT))
sys.path.insert(0, os.path.abspath(WORKSPACE_ROOT / 'DataYatesV1'))
sys.path.insert(0, os.path.abspath(WORKSPACE_ROOT / 'DataRowleyV1V2'))

from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset
from VisionCore.covariance import align_fixrsvp_trials
from VisionCore.paths import FIGURES_DIR, VISIONCORE_ROOT
from models.config_loader import load_dataset_configs
from models.data import prepare_data


LEGACY_PROCESSED_ROOT = Path('/mnt/ssd2/RowleyMarmoV1V2/processed_mvp')
FIX_NAME = 'fixrsvp.dset'
FIXATION_RADIUS_DEG = 1.5
SNR_THRESHOLD_PRIMARY = 10.0
MISSING_PCT_THRESHOLD = 45.0
MIN_FIX_DUR_BINS = 20
VALID_TIME_BINS = 240
N_RELIABILITY_SPLITS = 20
TOTAL_SPIKES_THRESHOLD = 200
FIGURE2_MIN_TOTAL_SPIKES = 500
RELIABILITY_THRESHOLD = 0.05
OUTPUT_DIR = FIGURES_DIR / 'mcfarland' / 'reliability_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _get_required(dset, key):
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


def _discover_session_configs(processed_root, fix_filename):
    candidate_roots = [
        (Path('datasets') / 'left_eye', 'left'),
        (Path('datasets') / 'right_eye', 'right'),
        (Path('datasets_gaussian') / 'left_eye', 'left'),
        (Path('datasets_gaussian') / 'right_eye', 'right'),
    ]
    configs = []
    for session_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        for dataset_dir_candidate, eye_candidate in candidate_roots:
            fix_path_candidate = session_dir / dataset_dir_candidate / fix_filename
            if not fix_path_candidate.exists():
                continue
            try:
                subject_name, session_date = session_dir.name.split('_', 1)
            except ValueError:
                continue
            configs.append({
                'subject': subject_name,
                'date': session_date,
                'primary_eye': eye_candidate,
                'dataset_dir': dataset_dir_candidate,
                'session_dir': session_dir,
            })
    return configs


def _compute_missing_pct_mask(session, t_bins, cids, threshold):
    missing_pct_fun = session.get_missing_pct_interp(cids)
    pct = _to_numpy(missing_pct_fun(t_bins)).astype(np.float32)
    valid_mask = pct < threshold
    chronic_multi_units = np.nanmedian(pct, axis=0) >= threshold
    valid_mask[:, chronic_multi_units] = True
    return valid_mask


def _compute_split_half_reliability(robs_trials, n_splits, seed=42):
    n_trials, _, n_units = robs_trials.shape
    rng_rel = np.random.default_rng(seed)
    r2_accum = np.zeros(n_units, dtype=np.float64)
    if n_trials < 2:
        return r2_accum

    for _ in range(n_splits):
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


def _serial_to_trial_aligned(robs, eyepos, dfs, trial_inds, time_inds):
    unique_trials = np.unique(trial_inds)
    n_trials = len(unique_trials)
    n_time = np.max(time_inds).item() + 1
    n_units = robs.shape[1]
    robs_trial = np.nan * np.zeros((n_trials, n_time, n_units), dtype=np.float32)
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
            robs_trial[itrial, tt[valid_tt]] = robs[idx][valid_tt]
            eyepos_trial[itrial, tt[valid_tt]] = eyepos[idx][valid_tt]
        dfs_trial[itrial, tt] = valid_tt
        dur_trial[itrial] = valid_tt.sum()
    return robs_trial, eyepos_trial, dfs_trial, dur_trial, unique_trials


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        'n': int(values.size),
        'median': float(np.median(values)),
        'q1': float(np.quantile(values, 0.25)),
        'q3': float(np.quantile(values, 0.75)),
        'frac_ge_threshold': float(np.mean(values >= RELIABILITY_THRESHOLD)),
    }


def collect_rowley_values():
    values = []
    failures = []
    entries = []
    for cfg in _discover_session_configs(LEGACY_PROCESSED_ROOT, FIX_NAME):
        label = f"{cfg['subject']}_{cfg['date']}|{cfg['dataset_dir']}|{cfg['primary_eye']}"
        try:
            sess = get_session(cfg['subject'], cfg['date'])
            sess.processed_path = cfg['session_dir']
            fix_path = cfg['session_dir'] / cfg['dataset_dir'] / FIX_NAME
            dset_fix = DictDataset.load(fix_path)

            robs_serial = _get_required(dset_fix, 'robs').astype(np.float32)
            t_bins_serial = _get_required(dset_fix, 't_bins').astype(np.float64)
            trial_inds_s = _get_required(dset_fix, 'trial_inds').astype(int)
            time_inds_s = _get_required(dset_fix, 'psth_inds').astype(int)
            cids_all = np.array(dset_fix.metadata.get('cluster_ids', np.arange(robs_serial.shape[1])))
            eyepos_default_raw = _get_required(dset_fix, 'eyepos')

            missing_mask = _compute_missing_pct_mask(sess, t_bins_serial, cids_all, MISSING_PCT_THRESHOLD)
            robs_serial = robs_serial.copy()
            robs_serial[~missing_mask] = np.nan

            n_bins_serial = eyepos_default_raw.shape[0]
            dpi_valid_default = _get_optional(dset_fix, 'dpi_valid', np.ones(n_bins_serial, dtype=bool))
            dpi_valid_default = _as_bool_1d(dpi_valid_default, n_bins_serial)
            eyepos_cyclopean = _ensure_2d_eyepos(eyepos_default_raw.astype(np.float32))
            ecc = np.hypot(eyepos_cyclopean[:, 0], eyepos_cyclopean[:, 1])
            dfs_incl = dpi_valid_default & (ecc <= FIXATION_RADIUS_DEG)

            robs_trial_incl, _, _, dur_trial_incl, _ = _serial_to_trial_aligned(
                robs_serial, eyepos_cyclopean, dfs_incl, trial_inds_s, time_inds_s)
            good_trials = dur_trial_incl > MIN_FIX_DUR_BINS
            robs_mc = robs_trial_incl[good_trials]
            robs_mc = robs_mc[:, :min(VALID_TIME_BINS, robs_mc.shape[1])]

            dots_cache = cfg['session_dir'] / 'dpi_calibration' / f"{cfg['primary_eye']}_eye" / 'dots_rf_snr.npz'
            cached = np.load(dots_cache, allow_pickle=True)
            cached_cids = np.asarray(cached['cids'])
            cached_max_snr = np.asarray(cached['max_snr'], dtype=np.float32)
            if cached_cids.shape != cids_all.shape or not np.array_equal(cached_cids, cids_all):
                raise RuntimeError('dots cache cid mismatch')

            visual_mask = np.isfinite(cached_max_snr) & (cached_max_snr >= SNR_THRESHOLD_PRIMARY)
            spikes_ok = np.nansum(robs_mc, axis=(0, 1)) > TOTAL_SPIKES_THRESHOLD
            candidate_mask = visual_mask & spikes_ok
            mean_reliability = _compute_split_half_reliability(robs_mc, N_RELIABILITY_SPLITS, seed=42)
            candidate_values = mean_reliability[candidate_mask]
            candidate_values = candidate_values[np.isfinite(candidate_values)]
            entries.append((label, _summarize(candidate_values)))
            if candidate_values.size:
                values.append(candidate_values)
        except Exception as exc:
            failures.append((label, str(exc)))
    if values:
        values = np.concatenate(values)
    else:
        values = np.array([], dtype=np.float64)
    return values, entries, failures


def collect_yates_values():
    values = []
    entries = []
    dataset_configs = load_dataset_configs(str(VISIONCORE_ROOT / 'experiments' / 'dataset_configs' / 'multi_basic_120_long.yaml'))
    for cfg in dataset_configs:
        session_name = cfg['session']
        if session_name.split('_')[0] not in {'Allen', 'Logan'}:
            continue
        cfg = dict(cfg)
        if 'fixrsvp' not in cfg['types']:
            cfg['types'] = cfg['types'] + ['fixrsvp']
        try:
            train_data, _, _ = prepare_data(cfg, strict=False)
            dset_idx = train_data.get_dataset_index('fixrsvp')
            fixrsvp_dset = train_data.dsets[dset_idx]
            robs, _, _, _, meta = align_fixrsvp_trials(
                fixrsvp_dset,
                valid_time_bins=120,
                min_fix_dur=20,
                min_total_spikes=FIGURE2_MIN_TOTAL_SPIKES,
            )
            if robs is None or robs.shape[0] < 10:
                continue
            n_trials = robs.shape[0]
            mean_reliability = _compute_split_half_reliability(robs, N_RELIABILITY_SPLITS, seed=42)
            finite_values = mean_reliability[np.isfinite(mean_reliability)]
            entries.append((session_name, _summarize(finite_values)))
            if finite_values.size:
                values.append(finite_values)
        except Exception as exc:
            entries.append((session_name, {'error': str(exc)}))
    if values:
        values = np.concatenate(values)
    else:
        values = np.array([], dtype=np.float64)
    return values, entries


def plot_comparison(rowley_values, yates_values):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0.0, 1.0, 41)
    for ax, values, title, color in [
        (axs[0], rowley_values, 'Rowley candidate units', 'steelblue'),
        (axs[1], yates_values, 'Yates / Figure 2 units', 'darkorange'),
    ]:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=bins, color=color, alpha=0.8)
            median = np.median(values)
            q1 = np.quantile(values, 0.25)
            q3 = np.quantile(values, 0.75)
            ax.axvline(median, color='k', lw=1.0, linestyle='-')
            ax.axvline(q1, color='k', lw=0.8, linestyle=':')
            ax.axvline(q3, color='k', lw=0.8, linestyle=':')
            ax.set_title(f'{title}\nmed={median:.3f}, q1={q1:.3f}, q3={q3:.3f}')
        else:
            ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_title(title)
        ax.axvline(RELIABILITY_THRESHOLD, color='r', lw=1.0, linestyle='--', alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Split-half PSTH reliability (r²)')
    axs[0].set_ylabel('Unit count')
    fig.suptitle(f'Rowley vs Yates reliability distributions (threshold={RELIABILITY_THRESHOLD:.2f})')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = OUTPUT_DIR / 'rowley_vs_yates_reliability_r2_distributions.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path


def write_summary(summary_path, rowley_values, yates_values, rowley_entries, yates_entries, rowley_failures):
    rowley_summary = _summarize(rowley_values)
    yates_summary = _summarize(yates_values)
    with open(summary_path, 'w') as f:
        f.write('Reliability comparison: Rowley vs Yates\n')
        f.write(f'threshold={RELIABILITY_THRESHOLD:.2f}\n\n')
        f.write(f'Rowley summary: {rowley_summary}\n')
        f.write(f'Yates summary: {yates_summary}\n\n')
        f.write('Rowley failures:\n')
        for label, error in rowley_failures:
            f.write(f'  {label}: {error}\n')
        f.write('\nRowley entries:\n')
        for label, summary in rowley_entries:
            f.write(f'  {label}: {summary}\n')
        f.write('\nYates entries:\n')
        for label, summary in yates_entries:
            f.write(f'  {label}: {summary}\n')


def main():
    rowley_values, rowley_entries, rowley_failures = collect_rowley_values()
    yates_values, yates_entries = collect_yates_values()
    figure_path = plot_comparison(rowley_values, yates_values)
    summary_path = OUTPUT_DIR / 'rowley_vs_yates_reliability_r2_summary.txt'
    write_summary(summary_path, rowley_values, yates_values, rowley_entries, yates_entries, rowley_failures)

    print(f'Saved {figure_path}')
    print(f'Saved {summary_path}')
    print(f'Rowley summary: {_summarize(rowley_values)}')
    print(f'Yates summary: {_summarize(yates_values)}')
    print(f'Rowley failures: {len(rowley_failures)}')


if __name__ == '__main__':
    main()
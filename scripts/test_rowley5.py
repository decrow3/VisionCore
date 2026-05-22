
#%%
import sys
import os
from pathlib import Path
from types import MethodType
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

SCRIPT_T0 = perf_counter()
LAST_T = SCRIPT_T0


def log_time(msg):
    global LAST_T
    now = perf_counter()
    print(f"[{now - SCRIPT_T0:8.2f}s | +{now - LAST_T:7.2f}s] {msg}", flush=True)
    LAST_T = now


log_time('Script start')

def fill_small_nan_gaps_prev_rowwise(X, k):
    """
    Row-wise: fill NaN runs of length <= k with the value immediately preceding the run.
    Leading NaNs stay NaN. Rows are independent. Gaps are along columns.

    Parameters
    ----------
    X : array-like, shape (R, C)
    k : int >= 0

    Returns
    -------
    Y : np.ndarray, shape (R, C)
    """
    X = np.asarray(X, dtype=float)
    Y = X.copy()
    if k <= 0:
        return Y

    R, C = Y.shape
    nan = np.isnan(Y)

    # Identify starts of NaN runs per row: nan & not nan immediately to the left
    left_nan = np.concatenate([np.zeros((R, 1), dtype=bool), nan[:, :-1]], axis=1)
    run_start = nan & ~left_nan

    # Label runs within each row: 1,2,3,... (0 for non-NaNs)
    run_id = np.cumsum(run_start, axis=1)
    run_id = run_id * nan  # 0 where not NaN

    max_runs = int(run_id.max())
    if max_runs == 0:
        return Y

    # Count run lengths: counts[r, rid] = number of NaNs in that run
    counts = np.zeros((R, max_runs + 1), dtype=np.int32)
    rr, cc = np.nonzero(nan)
    rid = run_id[rr, cc].astype(np.int32)
    np.add.at(counts, (rr, rid), 1)

    # For each run, store whether it has a valid previous value, and that fill value
    prev_ok = np.zeros((R, max_runs + 1), dtype=bool)
    fill_val = np.full((R, max_runs + 1), np.nan, dtype=float)

    rs_r, rs_c = np.nonzero(run_start)
    rs_id = run_id[rs_r, rs_c].astype(np.int32)

    has_prev = rs_c > 0
    prev_is_finite = np.zeros_like(has_prev, dtype=bool)
    prev_is_finite[has_prev] = np.isfinite(Y[rs_r[has_prev], rs_c[has_prev] - 1])

    ok = has_prev & prev_is_finite
    prev_ok[rs_r[ok], rs_id[ok]] = True
    fill_val[rs_r[ok], rs_id[ok]] = Y[rs_r[ok], rs_c[ok] - 1]

    # Decide which NaN entries to fill: run length <= k and run has valid previous value
    run_len = counts[rr, rid]
    do_fill = (run_len <= k) & prev_ok[rr, rid]

    Y[rr[do_fill], cc[do_fill]] = fill_val[rr[do_fill], rid[do_fill]]
    return Y


def plot_raster(ii, jj, height=1, ax=None, **kwargs):

    ii = np.stack([ii, ii, np.nan*np.ones_like(ii)], 1).flatten()
    jj = np.stack([jj, jj+height, np.nan*np.ones_like(jj)], 1).flatten()

    if ax is None:
        ax = plt.gca()
    ax.plot(ii, jj, 'k', **kwargs)


def parse_rowley_session_from_dataset_path(dataset_path):
    dataset_path = Path(dataset_path).resolve()
    if 'processed' not in dataset_path.parts:
        raise ValueError(f"Could not infer Rowley session from dataset path: {dataset_path}")

    processed_idx = dataset_path.parts.index('processed')
    if processed_idx + 1 >= len(dataset_path.parts):
        raise ValueError(f"Could not infer Rowley session from dataset path: {dataset_path}")

    session_name = dataset_path.parts[processed_idx + 1]
    if '_' not in session_name:
        raise ValueError(f"Session folder does not look like SUBJECT_DATE: {session_name}")

    subject, date = session_name.split('_', 1)
    return subject, date, session_name


def patch_rowley_imec_dir_discovery(sess):
    existing = list(sess.get_imec_dirs())
    if existing:
        return sess

    def _fallback_get_imec_dirs(self):
        patterns = [
            'patched_pipeline_results_*_imec*',
            'dredge_pipeline_results_*_imec*',
        ]
        imec_dirs = []
        for pattern in patterns:
            imec_dirs.extend(self.raw_path.glob(pattern))

        unique_dirs = sorted({d for d in imec_dirs if d.is_dir()})
        return unique_dirs

    sess.get_imec_dirs = MethodType(_fallback_get_imec_dirs, sess)

    patched_dirs = list(sess.get_imec_dirs())
    if not patched_dirs:
        raise FileNotFoundError(f'No IMEC result directories found under {sess.raw_path}')

    print('Patched session IMEC discovery to include dredge pipeline directories:')
    for imec_dir in patched_dirs:
        print(f'  {imec_dir}')

    return sess


def load_rowley_depth_df(sess):
    rows = []
    for shank_num, info in sorted(sess._get_shank_cluster_offsets().items()):
        ks_dir = info['dir'] / 'cur' / 'cur_sorter_output'
        spike_clusters_path = ks_dir / 'spike_clusters.npy'
        spike_positions_path = ks_dir / 'spike_positions.npy'

        if not spike_clusters_path.exists() or not spike_positions_path.exists():
            raise FileNotFoundError(
                f"Missing Kilosort depth inputs for shank {shank_num}: "
                f"{spike_clusters_path} / {spike_positions_path}"
            )

        spike_clusters = np.load(spike_clusters_path).astype(np.int64, copy=False)
        spike_y = np.load(spike_positions_path, mmap_mode='r')[:, 1]
        counts = np.bincount(spike_clusters)
        sum_y = np.bincount(spike_clusters, weights=spike_y)
        local_cids = np.flatnonzero(counts)
        depth_um = sum_y[local_cids] / counts[local_cids]

        rows.append(pd.DataFrame({
            'cid': local_cids + info['offset'],
            'local_cid': local_cids,
            'shank_or_probe': shank_num,
            'depth_um': depth_um,
            'n_spikes_depth': counts[local_cids],
        }))

    if not rows:
        raise ValueError(f"No shanks found for session {sess.name}")

    depth_df = pd.concat(rows, ignore_index=True)
    depth_df['cid'] = depth_df['cid'].astype(int)
    depth_df['local_cid'] = depth_df['local_cid'].astype(int)
    depth_df['shank_or_probe'] = depth_df['shank_or_probe'].astype(int)
    return depth_df.sort_values('cid').reset_index(drop=True)


from scipy.sparse import coo_matrix
def bin_spikes(spike_times, spike_clusters, t_bins, n_units=None):
    time_bins = np.digitize(spike_times, t_bins) - 1
    spike_indices = spike_clusters  # Don't subtract 1 - use cluster IDs directly as indices

    if n_units is None:
        n_units = int(np.max(spike_clusters)) + 1  # +1 because 0-indexed

    # Filter out spikes outside valid bin range and invalid cluster indices
    n_time_bins = len(t_bins) - 1
    valid = (time_bins >= 0) & (time_bins < n_time_bins) & (spike_indices >= 0) & (spike_indices < n_units)
    time_bins = time_bins[valid]
    spike_indices = spike_indices[valid]
    n_spikes = len(time_bins)

    counts = coo_matrix(
            (np.ones(n_spikes, dtype=int), (time_bins.astype(int), spike_indices.astype(int))),
            shape=(n_time_bins, n_units),
        )
    return counts
#%%
# import dictdataset



# Update for Luke_2026-03-16
#fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2026-03-16/datasets/right_eye/fixrsvp.dset'
#fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/fixrsvp.dset'
fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2026-03-01/datasets/left_eye/fixrsvp.dset'
#fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2026-03-08/datasets/right_eye/fixrsvp.dset'
#fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2026-03-09/datasets/right_eye/fixrsvp.dset'
#fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2026-03-11/datasets/left_eye/fixrsvp.dset'

#Curently empty or missing datasets:
#fpath = '/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2026-03-13/datasets/right_eye/fixrsvp.dset'

use_rowley = True

from models.data import DictDataset
dset = DictDataset.load(fpath)
log_time('Loaded DictDataset')

if use_rowley:
    from DataRowleyV1V2.data.registry import get_session

    subject, date, session_name = parse_rowley_session_from_dataset_path(fpath)
    sess = get_session(subject, date)
    sess = patch_rowley_imec_dir_discovery(sess)
    log_time(f'Created Rowley session handle for {session_name}')

    figures_subdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'mcfarland', str(sess.name))
    os.makedirs(figures_subdir, exist_ok=True)
    log_time('Prepared figures output directory')

    depth_df = load_rowley_depth_df(sess)
    log_time('Built unit depth table from raw Kilosort outputs')
    print(f'Loaded raw unit depth table from {sess.raw_path}')
    print(depth_df.head())

    fig, ax = plt.subplots(figsize=(8,5))
    for shank in sorted(depth_df['shank_or_probe'].unique()):
        group = depth_df[depth_df['shank_or_probe'] == shank]
        ax.scatter(group['cid'], group['depth_um'], label=f'Shank {shank}', s=20)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Depth (um)')
    ax.set_title('Unit Depths by Shank')
    ax.legend()
    fig.savefig(os.path.join(figures_subdir, 'unit_depths_by_shank.pdf'))
    plt.close(fig)
else:
    depth_df = pd.DataFrame(columns=['cid', 'shank_or_probe', 'depth_um'])
log_time('Finished depth report setup')

# Pre-select candidate cids (shallow units from each shank) before spike loading.
# This limits binned_spikes to only the units we care about, avoiding a multi-GB array.
depth_threshold_um = 2250 #3820 - 1250.0
depth_convention = 'from_tip'
stop_after = None  # Set to 'psth' or 'raster' to exit early after those outputs.
min_valid_bins_per_trial = 20
min_good_trials_per_unit = 10
min_total_spikes_valid = 200
use_curated_cids = False
curated_cids = None  # Optional iterable of original cids to keep (e.g. np.array([...]))
apply_truncation_qc = True
truncation_threshold_pct = 45.0
min_dprime = 0.05  # min pre/post stimulus d' computed from trial-wise rates; set 0 to disable
min_reliability = 0.1 #0.2    # min split-half Pearson r of PSTH in response window; set -1 to disable
response_win = (0.0, 0.35)  # seconds post-onset used for deflection and reliability
baseline_win = (-0.2, -0.1)   # seconds pre-onset used for baseline
_v1_cand_df = depth_df[(depth_df['shank_or_probe'] == 1) & (depth_df['depth_um'] >= depth_threshold_um)]
_v2_cand_df = depth_df[(depth_df['shank_or_probe'] == 0) & (depth_df['depth_um'] >= depth_threshold_um)]
candidate_cids = sorted(set(_v1_cand_df['cid'].astype(int).tolist() + _v2_cand_df['cid'].astype(int).tolist()))
if use_curated_cids and curated_cids is not None:
    curated_set = set(np.asarray(curated_cids, dtype=int).tolist())
    candidate_cids = [cid for cid in candidate_cids if cid in curated_set]
    print(f"Curated cid gate: keeping {len(candidate_cids)} shallow units")
cid_to_idx = {cid: i for i, cid in enumerate(candidate_cids)}
n_cand = len(candidate_cids)
print(f"Pre-selected {len(_v1_cand_df)} V1 candidates (shank 1/imec1), {len(_v2_cand_df)} V2 candidates (shank 0/imec0), {n_cand} total")
log_time('Pre-selected candidate cids')

if use_rowley:
    spikes = sess.load_spikes()  
    log_time('Loaded spikes')
    st = spikes[0]      
    clu = spikes[1]
    # Filter spikes to candidate units only and remap to compact indices 0..n_cand-1
    _cand_mask = np.isin(clu, candidate_cids)
    _clu_masked = clu[_cand_mask]
    _lookup = np.full(int(clu.max()) + 1, -1, dtype=np.int64)
    for _cid, _idx in cid_to_idx.items():
        _lookup[_cid] = _idx
    st = st[_cand_mask]
    clu = _lookup[_clu_masked]
    log_time(f'Filtered spikes to {n_cand} candidate units ({len(st)} spikes)')
    exp = sess.load_exp()
    log_time('Loaded experiment metadata')
    ptb2ephys = sess.load_ptb2ephys()[0]
    log_time('Loaded PTB to Ephys mapping')
    if apply_truncation_qc and n_cand > 0:
        missing_pct_fun = sess.get_missing_pct_interp(np.asarray(candidate_cids, dtype=int))
        log_time(f'Loaded truncation QC interpolator (threshold={truncation_threshold_pct:.1f}%)')
    else:
        missing_pct_fun = None
else:
    from DataYatesV1 import get_session
    sess = get_session('Allen', '2022-02-18')
    st = sess.get_spike_times()
    clu = sess.get_spike_clusters()
    exp = sess.exp
    ptb2ephys = sess.ptb2ephys
    missing_pct_fun = None

#%%
num_trials = len(exp['D'])
trial_protocols = [exp['D'][i]['PR']['name'] for i in range(num_trials)]

fixrsvp_trials = [i for i in range(num_trials) if trial_protocols[i] == 'FixRsvpStim']

#%%

NT = len(fixrsvp_trials)
nclu = n_cand  # compact index space for candidate units only

dt = 1/240
time_bins = np.arange(-1, 2, dt)

binned_spikes = np.nan*np.zeros((NT, len(time_bins)-1, nclu))
state = np.nan*np.zeros((NT, len(time_bins)-1))
eyepos = np.nan*np.zeros((NT, len(time_bins)-1, 3))
image_id = np.nan*np.zeros((NT, len(time_bins)-1))
breakfix = np.nan*np.zeros(NT)
ctrfix = np.nan*np.zeros((NT, 2))
truncation_dfs = np.ones((NT, len(time_bins)-1, nclu), dtype=bool)


log_time(f'Starting trial loop (NT={NT})')
for i in range(NT):
    if i % 100 == 0:
        log_time(f'Trial loop progress: {i}/{NT}')
    itrial = fixrsvp_trials[i]
    NH = exp['D'][itrial]['PR']['NoiseHistory']
    if NH is None or NH.shape[0] < 10:
        continue

    eye_data = exp['D'][itrial]['eyeData']

    c = exp['D'][itrial]['c']
    dx = exp['D'][itrial]['dx']
    dy = exp['D'][itrial]['dy']

    eye_data[:,2] = (eye_data[:,2] - c[1])*dy
    eye_data[:,1] = (eye_data[:,1] - c[0])*dx

    t2_candidates = np.where(NH[:,3]==2)[0]
    if len(t2_candidates) == 0:
        continue  # Skip this trial if no event == 2
    t2 = t2_candidates[0]
    eye_time = ptb2ephys(eye_data[:,0])
    frame_times = ptb2ephys(NH[:,0])

    if missing_pct_fun is not None:
        trial_bin_times = frame_times[0] + time_bins[:-1]
        trial_missing_pct = missing_pct_fun(trial_bin_times).detach().cpu().numpy()
        truncation_dfs[i] = trial_missing_pct < truncation_threshold_pct

    eye_time = eye_time - frame_times[0]
    # digitize to time_bins
    eye_digi = np.digitize(eye_time, time_bins) - 1
    eye_iix = (eye_digi >= 0) & (eye_digi < len(time_bins)-1)
    nh_digi = np.digitize(frame_times-frame_times[0], time_bins) - 1
    nh_good = (nh_digi >= 0) & (nh_digi < len(time_bins)-1)
    
    image_id[i, nh_digi[nh_good]] = NH[nh_good,3]
    eyepos[i, eye_digi[eye_iix], :] = eye_data[eye_iix, 1:4]
    state[i, eye_digi[eye_iix]] = eye_data[eye_iix, 4]
    
    ctr_ = np.nanmean(eyepos[i][(time_bins[:-1]>0) & (time_bins[:-1]<.1)], 0)
    ctrfix[i] = ctr_[:2]
    st_ix = (st > frame_times[0]+time_bins[0]) & (st < frame_times[0]+time_bins[-1])

    trial_st = st[st_ix] - frame_times[0]
    trial_clu = clu[st_ix]
    st_binned = bin_spikes(trial_st, trial_clu, time_bins, n_units=nclu)
    binned_spikes[i] = st_binned.toarray()

    try:    
        breakfix[i] = np.where(state[i]==3)[0][0]
    except:
        breakfix[i] = np.nan

log_time('Finished trial loop')
ctr = np.nanmean(ctrfix, 0)
eyepos[:,:,0] = fill_small_nan_gaps_prev_rowwise(eyepos[:,:,0], 2) - ctr[0]
eyepos[:,:,1] = fill_small_nan_gaps_prev_rowwise(eyepos[:,:,1], 2) - ctr[1]

dfs = np.isfinite(image_id)[:,:,None].repeat(binned_spikes.shape[-1], axis=2)
dfs &= truncation_dfs
image_id = fill_small_nan_gaps_prev_rowwise(image_id, 2)
if missing_pct_fun is not None:
    trunc_valid_bins = int(truncation_dfs.sum())
    combined_valid_bins = int(dfs.sum())
    print(
        f"Truncation QC valid bins: {trunc_valid_bins} / {truncation_dfs.size} "
        f"({100.0 * trunc_valid_bins / max(truncation_dfs.size, 1):.2f}%) at missing_pct < {truncation_threshold_pct:.1f}"
    )
    print(
        f"Combined analysis valid bins after image/truncation masks: {combined_valid_bins} / {dfs.size} "
        f"({100.0 * combined_valid_bins / max(dfs.size, 1):.2f}%)"
    )

# get fixation duration
Y = image_id.copy()
R, C = Y.shape
nan = np.isnan(Y)

# Identify starts of NaN runs per row: nan & not nan immediately to the left
left_nan = np.concatenate([np.zeros((R, 1), dtype=bool), nan[:, :-1]], axis=1)
run_start = nan & ~left_nan

# Label runs within each row: 1,2,3,... (0 for non-NaNs)
run_id = np.cumsum(run_start, axis=1)
run_id = run_id * nan  # 0 where not NaN

fix_dur = np.zeros(R)
for i in range(R):
    try:
        fix_dur[i] = np.where(run_id[i]==2)[0][0]
    except:
        fix_dur[i] = np.nan

good_trials = (fix_dur-np.where(time_bins[:-1]>0)[0][0]) > min_valid_bins_per_trial

# remove bad trials
binned_spikes = binned_spikes[good_trials]
eyepos = eyepos[good_trials][:,:,[0,1]]
image_id = image_id[good_trials]
fix_dur = fix_dur[good_trials]
dfs = dfs[good_trials]

# Final unit gate based on valid bins only.
robs_valid = np.where(dfs, np.nan_to_num(binned_spikes, nan=0.0), 0.0)
unit_total_spikes = robs_valid.sum(axis=(0, 1))
unit_good_trials = (dfs.sum(axis=1) > 0).sum(axis=0)
unit_keep_mask_final = (
    (unit_total_spikes >= min_total_spikes_valid)
    & (unit_good_trials >= min_good_trials_per_unit)
)
print('Final unit gate:')
print(f"  spikes >= {min_total_spikes_valid}: {(unit_total_spikes >= min_total_spikes_valid).sum()} / {len(unit_total_spikes)}")
print(f"  good trials >= {min_good_trials_per_unit}: {(unit_good_trials >= min_good_trials_per_unit).sum()} / {len(unit_good_trials)}")
print(f"  final keep: {unit_keep_mask_final.sum()} / {len(unit_keep_mask_final)}")
keep_idx = np.where(unit_keep_mask_final)[0]
binned_spikes = binned_spikes[:, :, keep_idx]
dfs = dfs[:, :, keep_idx]
candidate_cids = [candidate_cids[i] for i in keep_idx]
cid_to_idx = {cid: i for i, cid in enumerate(candidate_cids)}
print(f"After final unit filtering, NC = {len(candidate_cids)}")

log_time('Finished interpolation and trial filtering')
ind = np.argsort(fix_dur)

fig, ax = plt.subplots()
im = ax.imshow(eyepos[ind][:,:,0],
           vmin=-.5, vmax=.5,
           aspect='auto',
           cmap='coolwarm', interpolation='none',
           origin='lower',
           extent=[time_bins[0], time_bins[-1], 0, eyepos.shape[0]])
ax.set_xlim(-.1, 1.0)
fig.savefig(os.path.join(figures_subdir, 'eyepos_heatmap.pdf'))
plt.close(fig)

fig, ax = plt.subplots()
im = ax.imshow(image_id[ind],
    aspect='auto',
    interpolation='none', origin='lower', vmin=0, vmax=20,
    extent=[time_bins[0], time_bins[-1], 0, eyepos.shape[0]])
ax.set_xlim(-.1, 1.0)
fig.savefig(os.path.join(figures_subdir, 'image_id_heatmap.pdf'))
plt.close(fig)
log_time('Saved eye position and image heatmaps')
    

# --- DO NOT Select top 50 V1/V2 units by spike count from pre-selected candidates, take all passing the checks ---
# v1cids/v2cids are compact indices into binned_spikes; use candidate_cids[idx] for original cid.
spike_counts = np.nansum(binned_spikes, axis=(0,1))  # shape (n_cand,)
_v1_cand_indices = [cid_to_idx[c] for c in _v1_cand_df['cid'].astype(int) if c in cid_to_idx]
_v2_cand_indices = [cid_to_idx[c] for c in _v2_cand_df['cid'].astype(int) if c in cid_to_idx]
v1cids = sorted(_v1_cand_indices, key=lambda i: spike_counts[i], reverse=True)#[:50]
v2cids = sorted(_v2_cand_indices, key=lambda i: spike_counts[i], reverse=True)#[:50]
v1_orig = [candidate_cids[i] for i in v1cids]
v2_orig = [candidate_cids[i] for i in v2cids]
print(f"V1 cids (n={len(v1cids)}):", v1_orig)
print(f"V2 cids (n={len(v2cids)}):", v2_orig)
cid_shank_df = depth_df.set_index('cid')[['shank_or_probe', 'depth_um']]
print("\nShank mapping for selected V1 cids (shank 1=imec1/V1):")
for idx, orig_cid in zip(v1cids, v1_orig):
    if orig_cid in cid_shank_df.index:
        row = cid_shank_df.loc[orig_cid]
        print(f"  cid={orig_cid:5d}  idx={idx}  shank={int(row['shank_or_probe'])}  depth={row['depth_um']:.0f} um")
    else:
        print(f"  cid={orig_cid:5d}  NOT FOUND in depth report")
print("\nShank mapping for selected V2 cids (shank 0=imec0/V2):")
for idx, orig_cid in zip(v2cids, v2_orig):
    if orig_cid in cid_shank_df.index:
        row = cid_shank_df.loc[orig_cid]
        print(f"  cid={orig_cid:5d}  idx={idx}  shank={int(row['shank_or_probe'])}  depth={row['depth_um']:.0f} um")
    else:
        print(f"  cid={orig_cid:5d}  NOT FOUND in depth report")

log_time('Finished V1/V2 CID selection')
# Use all cids for analysis (can change to v1cids or v2cids as needed)
cids = np.array(v1cids + v2cids)
if len(cids) == 0:
    cids = np.arange(binned_spikes.shape[-1])

# good_trials = np.isnan(binned_spikes[:,:,cids[0]]).sum(1)==0
if len(cids) > 0:
    good_trials = (np.isnan(binned_spikes[:,:,cids[0]]).sum(1)==0) & (np.sum(np.diff(np.nansum(binned_spikes, 1), 0),1)>0)
    ind = np.argsort(fix_dur[good_trials])
else:
    good_trials = np.array([], dtype=bool)
    ind = np.array([], dtype=int)



#%%
    # plt.figure()
    # # plt.plot(st[0][st_ix], st[1][st_ix], 'k.')
    # plot_raster(st[st_ix], clu[st_ix])
    # plt.axvline(frame_times[t2], color='r', linestyle='--')
    # plt.title(f'Trial {itrial}')
    # plt.show()

#%%
from scipy.signal import savgol_filter

#%%
from tejas.rsvp_util import remove_duplicate_trials, align_image_ids

log_time('Starting remove_duplicate_trials')
robs, dfs, eyepos, dur, image_ids = remove_duplicate_trials(binned_spikes, dfs, eyepos, fix_dur, image_id)
log_time('Finished remove_duplicate_trials')

# --- Response filter: pre/post d' + split-half reliability ---
_pre  = (time_bins[:-1] >= baseline_win[0]) & (time_bins[:-1] < baseline_win[1])
_post = (time_bins[:-1] >= response_win[0]) & (time_bins[:-1] < response_win[1])
_baseline_trial_rates = np.nanmean(robs[:, _pre, :], axis=1)   # (n_trials, n_cand)
_response_trial_rates = np.nanmean(robs[:, _post, :], axis=1)  # (n_trials, n_cand)
_baseline_mean = np.nanmean(_baseline_trial_rates, axis=0)
_response_mean = np.nanmean(_response_trial_rates, axis=0)
_baseline_var = np.nanvar(_baseline_trial_rates, axis=0)
_response_var = np.nanvar(_response_trial_rates, axis=0)
_pooled_std = np.sqrt(0.5 * (_baseline_var + _response_var))
_dprime = np.divide(
    np.abs(_response_mean - _baseline_mean),
    _pooled_std,
    out=np.zeros_like(_response_mean),
    where=_pooled_std > 0,
)

_h1 = robs[0::2];  _h2 = robs[1::2]
_p1 = np.nanmean(_h1[:, _post, :], axis=0)  # (T_post, n_cand)
_p2 = np.nanmean(_h2[:, _post, :], axis=0)
_reliability = np.array([
    np.corrcoef(_p1[:, j], _p2[:, j])[0, 1]
    if np.std(_p1[:, j]) > 0 and np.std(_p2[:, j]) > 0 else 0.0
    for j in range(_p1.shape[1])
])
_reliability = np.nan_to_num(_reliability, nan=0.0)

_response_mask = (_dprime > min_dprime) & (_reliability > min_reliability)
print(f"Response filter: {_response_mask.sum()} / {len(_response_mask)} units pass "
    f"(dprime>{min_dprime}, split-half r>{min_reliability})")
v1cids = [cc for cc in v1cids if _response_mask[cc]]
v2cids = [cc for cc in v2cids if _response_mask[cc]]
log_time('Applied response filter')

#%%
from scipy.signal import savgol_filter

log_time(f'Starting mean-rate plots for {len(v1cids)+len(v2cids)} units')
for cc in v1cids + v2cids:
    fig, ax = plt.subplots()
    ax.plot(time_bins[:-1], np.nanmean(robs, 0)[:, cc])
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'Neuron {candidate_cids[cc]}')
    fig.savefig(os.path.join(figures_subdir, f'neuron_{candidate_cids[cc]}_meanrate.pdf'))
    plt.close(fig)
log_time('Finished mean-rate plots')
if stop_after == 'psth':
    log_time('Stopping after PSTH export')
    sys.exit(0)

log_time(f'Starting raster plots for {len(v1cids)+len(v2cids)} units')
for cc in v1cids + v2cids:
    good_trials_cc = np.isnan(robs[:, :, cc]).sum(1) == 0
    ind = np.argsort(dur[good_trials_cc])
    fig, ax = plt.subplots()
    jj, ii = np.where(np.nan_to_num(robs[:, :, cc][good_trials_cc][ind], nan=0.0))
    plot_raster(time_bins[ii], jj, height=1, ax=ax)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'Neuron {candidate_cids[cc]}')
    fig.savefig(os.path.join(figures_subdir, f'neuron_{candidate_cids[cc]}_raster.pdf'))
    plt.close(fig)
log_time('Finished raster plots')
if stop_after == 'raster':
    log_time('Stopping after raster export')
    sys.exit(0)

# for itrial in range(40):(robs, dfs, eyepos, dur, image_ids, 
#     ii, jj = np.where(binned_spikes[good_trials][:,:,:][itrial])
#     plot_raster(ii, jj, height=1)
#     plt.title(f"Trial {itrial}")
#     plt.show()


#%%
NT = robs.shape[0]

log_time(f'Starting trial raster exports for {NT} trials')
for itrial in range(NT):
    fig, ax = plt.subplots()
    ii, jj = np.where(robs[itrial])
    plot_raster(time_bins[:-1][ii], jj, height=1, ax=ax)
    ax.axis('off')
    ax2 = ax.twinx()
    ax2.plot(time_bins[:-1], eyepos[itrial][:,0], '-r')
    ax2.plot(time_bins[:-1], eyepos[itrial][:,1], '-g')
    ax2.set_ylim(-50, 50)
    ax3 = ax2.twinx()
    ax3.plot(time_bins[:-1], image_ids[itrial], '-b')
    ax3.set_ylim(0, 20)
    ax.set_title(f"Trial {itrial}")
    ax.set_xlim(-.250, 0.50)
    fig.savefig(os.path.join(figures_subdir, f'trial_{itrial}_raster.pdf'))
    plt.close(fig)

log_time('Finished trial raster exports')
#%%

from mcfarland_sim import DualWindowAnalysis

iix = (time_bins[:-1] > 0) & (time_bins[:-1] < 1.5)
robs_used = robs[:,iix]
eyepos_used = eyepos[:,iix]
valid_mask = dfs[:,iix].sum(2)>0

cids = np.array(v1cids + v2cids)
log_time(f'Initializing DualWindowAnalysis with {len(cids)} units')
analyzer = DualWindowAnalysis(robs_used[:,:,cids], eyepos_used, valid_mask, dt=dt)
log_time('Initialized DualWindowAnalysis')

#%%
log_time('Starting analyzer.run_sweep')
results, last_mats = analyzer.run_sweep([10, 20, 40], t_hist_ms=50, n_bins=35)
log_time('Finished analyzer.run_sweep')

#%%

from mcfarland_sim import project_to_psd

# import cmasher as cmr
# cmap = plt.get_cmap('cmr.prinsenvlag')   # MPL
cmap = plt.get_cmap('RdBu')   # MPL
window_idx = 0


Ctotal = last_mats[window_idx]['Total']
Cpsth = last_mats[window_idx]['PSTH']
Crate = last_mats[window_idx]['Intercept']
Cfem = last_mats[window_idx]['FEM']

MeanRates = results[window_idx]['Erates']

# project to psd
Ctotal = project_to_psd(Ctotal)
Cpsth = project_to_psd(Cpsth)
Crate = project_to_psd(Crate)
Cfem = project_to_psd(Cfem)
CnoiseC = Ctotal - Crate
CnoiseU = Ctotal - Cpsth
CnoiseU = project_to_psd(CnoiseU)
CnoiseC = project_to_psd(CnoiseC)
plt.plot(np.diag(CnoiseU), np.diag(CnoiseC), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
#%%
v = np.max(Ctotal.flatten()) * .5


fig, axs = plt.subplots(1,3, figsize=(20,5))
ax = axs[0]
ax.imshow(Ctotal, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Total')
ax.axis('off')

ax = axs[1]
ax.imshow(Cpsth, cmap=cmap, interpolation='nearest', vmin=-v/2, vmax=v/2)
ax.set_title('PSTH')
ax.axis('off')

ax = axs[2]
ax.imshow(CnoiseU, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Noise (Uncorrected)')
ax.axis('off')
fig.savefig(os.path.join(figures_subdir, f'covariance_decomposition_{window_idx}_psth.pdf'), bbox_inches='tight', dpi=300)
plt.close(fig)




fig, axs = plt.subplots(1,4, figsize=(20,5))
ax = axs[0]
ax.imshow(Ctotal, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Total')
ax.axis('off')
ax = axs[1]
ax.imshow(Cfem, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('FEM')
ax.axis('off')

ax = axs[2]
ax.imshow(Cpsth, cmap=cmap, interpolation='nearest', vmin=-v/2, vmax=v/2)
ax.set_title('PSTH')
ax.axis('off')

ax = axs[3]
ax.imshow(CnoiseC, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Noise (Corrected)')
ax.axis('off')
fig.savefig(os.path.join(figures_subdir, f'covariance_decomposition_{window_idx}_full.pdf'), bbox_inches='tight', dpi=300)
plt.close(fig)

#%% Alpha
isv1 = np.isin(cids, v1cids)
alpha = np.diag(Cpsth) / np.diag(Crate)

fig, ax = plt.subplots()
ax.hist(1-alpha[isv1], bins=np.linspace(0, 1, 50), alpha=0.5, label='V1')
ax.hist(1-alpha[~isv1], bins=np.linspace(0, 1, 50), alpha=0.5, label='V2')
ax.set_xlabel('Frac. FEM (1-alpha)')
ax.set_ylabel('Count')
ax.legend()
fig.savefig(os.path.join(figures_subdir, 'frac_fem_hist.pdf'))
plt.close(fig)



fig, axs = plt.subplots(1,2, figsize=(8,2))
ix = isv1
axs[0].plot(MeanRates[ix], np.diag(CnoiseU)[ix], '.', label='V1')
axs[0].plot(MeanRates[~ix], np.diag(CnoiseU)[~ix], '.', label='V2')
axs[0].plot(axs[0].get_xlim(), axs[0].get_xlim(), 'k')
axs[0].set_xlabel('Mean Rate')
axs[0].set_ylabel('Variance')
axs[0].legend()

axs[1].plot(MeanRates[ix], np.diag(CnoiseC)[ix], '.', label='V1')
axs[1].plot(MeanRates[~ix], np.diag(CnoiseC)[~ix], '.', label='V2')
axs[1].plot(axs[1].get_xlim(), axs[1].get_xlim(), 'k')
axs[1].set_xlabel('Mean Rate')
axs[1].set_ylabel('Variance')
axs[1].legend()

fig.savefig(os.path.join(figures_subdir, 'meanrate_vs_variance.pdf'))
plt.close(fig)
log_time('Finished covariance decomposition and summary figures')

#%%
from matplotlib.backends.backend_pdf import PdfPages
import os

figures_subdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'mcfarland', str(sess.name))
os.makedirs(figures_subdir, exist_ok=True)

unit_pdf_path = os.path.join(figures_subdir, f'unit_rasters_{sess.name}.pdf')
log_time('Starting unit PDF export')
with PdfPages(unit_pdf_path) as pdf:
    for i, cc in enumerate(cids):
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].set_title(f"Neuron {candidate_cids[cc]}")
        ind = np.argsort(dur)
        jj, ii = np.where(np.nan_to_num(robs[:,:,cc][ind], nan=0.0))
        plot_raster(time_bins[ii], jj, height=1, ax=ax[0])
        ax[0].axvline(0, color='r', linestyle='--')
        ax[0].set_xlim(-0.1, 1.0)

        analyzer.inspect_neuron_pair(i, i, 10, ax=ax[1], show=True)
        pdf.savefig(fig)
        plt.close(fig)
log_time('Finished unit PDF export')
#%%

from matplotlib.backends.backend_pdf import PdfPages

trial_pdf_path = os.path.join(figures_subdir, f'trial_rasters_{sess.name}.pdf')
log_time('Starting trial PDF export')
with PdfPages(trial_pdf_path) as pdf:
    NT = min(binned_spikes.shape[0], eyepos.shape[0], image_id.shape[0])
    for itrial in range(NT):
        fig, ax = plt.subplots()
        ii, jj = np.where(robs[itrial])
        plot_raster(time_bins[:-1][ii], jj, height=1)
        plt.axis('off')
        plt.gca().twinx()
        # Robustly match lengths for plotting
        n_eye = min(len(time_bins) - 1, eyepos[itrial].shape[0])
        plt.plot(time_bins[:n_eye], eyepos[itrial][:n_eye,0], '.r')
        plt.plot(time_bins[:n_eye], eyepos[itrial][:n_eye,1], '.g')
        plt.ylim(-150, 150)
        plt.gca().twinx()
        n_img = min(len(time_bins) - 1, image_id[itrial].shape[0]) if hasattr(image_id[itrial], 'shape') else min(len(time_bins) - 1, len(image_id[itrial]))
        plt.plot(time_bins[:n_img], image_id[itrial][:n_img], '.b')
        plt.ylim(0, 20)
        plt.title(f"Trial {itrial}")
        plt.xlim(-.250, 0.50)
        pdf.savefig(fig)
        plt.close(fig)

log_time('Finished trial PDF export')
log_time('Script complete')

# %%

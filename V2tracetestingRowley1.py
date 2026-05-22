
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# Reduce Agg path memory pressure for very large rasters
mpl.rcParams['agg.path.chunksize'] = 20000
def plot_raster(ii, jj, height=1, ax=None, method='auto', **kwargs):
    # Efficient raster: use vlines for moderate sizes, scatter('|') for huge sizes
    ii = np.asarray(ii)
    jj = np.asarray(jj)
    if ax is None:
        ax = plt.gca()

    color = kwargs.pop('color', 'k')
    linewidth = kwargs.pop('linewidth', 0.5)
    s = kwargs.pop('s', 10)

    if method == 'auto':
        method = 'vlines' if ii.size <= 200000 else 'scatter'

    if method == 'vlines':
        ax.vlines(ii, jj, jj + height, colors=color, linewidth=linewidth, **kwargs)
    else:
        ax.scatter(ii, jj, marker='|', c=color, s=s, linewidths=linewidth, **kwargs)


def bin_spikes(spike_times, spike_clusters, t_bins, n_units=None):
    time_bins = np.digitize(spike_times, t_bins) - 1
    spike_indices = spike_clusters - 1
    n_spikes = len(spike_times)

    if n_units is None:
        n_units = np.max(spike_clusters)
    counts = torch.sparse_coo_tensor(
            np.asarray([time_bins, spike_indices]),
            np.ones(n_spikes),
            (len(t_bins) - 1, n_units),
            dtype=torch.float32
        ).to_dense().numpy()
    return counts



#%% import dictdataset

fpath = '/mnt/ssd/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/fixrsvp.dset'

from models.data import DictDataset
dset = DictDataset.load(fpath)

use_rowley = True
if use_rowley:
    from DataRowleyV1V2.data.registry import get_session
    from DataRowleyV1V2.exp.general import get_trial_protocols
    sess = get_session('Luke', '2025-08-04')
    spikes = sess.load_spikes()  
    st = spikes[0]      
    clu = spikes[1]
    exp = sess.load_exp()
    ptb2ephys = sess.load_ptb2ephys()[0]
else:
    from DataYatesV1 import get_session
    sess = get_session('Allen', '2022-04-13')
    st = sess.get_spike_times()
    clu = sess.get_spike_clusters()
    exp = sess.exp
    ptb2ephys = sess.ptb2ephys

#%%
num_trials = len(exp['D'])
trial_protocols = [exp['D'][i]['PR']['name'] for i in range(num_trials)]

fixrsvp_trials = [i for i in range(num_trials) if trial_protocols[i] == 'FixRsvpStim']

#%%

NT = len(fixrsvp_trials)
nclu = np.max(clu)

# Extend to 1.25s so plots in [-0.25, 1.25] have data
time_bins = np.arange(-1, 1.25, 0.001)

binned_spikes = np.nan*np.zeros((NT, len(time_bins)-1, nclu))

for i in range(NT):
    itrial = fixrsvp_trials[i]
    NH = exp['D'][itrial]['PR']['NoiseHistory']
    if NH is None or NH.shape[0] < 10:
        continue

    t2 = np.where(NH[:,3]==2)[0][0]

    frame_times = ptb2ephys(NH[:,0])

    st_ix = (st > frame_times[0]-1) & (st < frame_times[0]+1.25)
    # if use_rowley:
    #     st_ix = st_ix & (clu >800)

    trial_st = st[st_ix] - frame_times[0]
    trial_clu = clu[st_ix]
    st_binned = bin_spikes(trial_st, trial_clu, time_bins, n_units=nclu)
    binned_spikes[i] = st_binned

    # plt.figure()
    # # plt.plot(st[0][st_ix], st[1][st_ix], 'k.')
    # plot_raster(st[st_ix], clu[st_ix])
    # plt.axvline(frame_times[t2], color='r', linestyle='--')
    # plt.title(f'Trial {itrial}')
    # plt.show()



# %%
# --- Pretty figures: PSTH colormaps and trial heatmaps ---
# Keep only valid indices in [0, nclu)
V2_cids=[363,364,366,367,368,372,373,378,388,389,397,399,411,412,413,420,427,431,433,446,457,468,471,472,476,494,507,509,513,518,525,535,550,554]
V1_cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
selected_cids = np.unique(clu).astype(int)-1 #V2_cids  # Choose V2 neurons for visualization

from scipy.ndimage import gaussian_filter1d

def zscore_time(mat, axis=1, eps=1e-9):
    # Z-score along time for each row (row = neuron or trial)
    m = np.nanmean(mat, axis=axis, keepdims=True)
    s = np.nanstd(mat, axis=axis, keepdims=True)
    return (mat - m) / (s + eps)

def get_time_centers(t_bins):
    return 0.5 * (t_bins[:-1] + t_bins[1:])

# Choose a neuron subset (e.g., V2). Fall back to all if not defined.
try:
    selected_cids = np.asarray(selected_cids, dtype=int).ravel()
except NameError:
    selected_cids = np.arange(nclu, dtype=int)

# Ensure numpy array type for downstream indexing and .size usage
selected_cids = np.asarray(selected_cids, dtype=int).ravel()

# Build a clean trial mask: non-empty trials and a minimum length
min_trial_length = 10
if 'trial_lengths' not in globals():
    # If not previously computed
    trial_lengths = np.zeros(NT, dtype=int)
good_trials_len = trial_lengths >= min_trial_length

# 1) Neuron × time PSTH colormap (mean across trials)
t_centers = get_time_centers(time_bins)

# Mean across trials for each neuron, then optional smoothing along time
psth_by_neuron = np.nanmean(binned_spikes[good_trials_len][:, :, selected_cids], axis=0)  # (T, Nc)
# Smooth each neuron’s PSTH along time (sigma in bins)
smooth_sigma_bins = 2
if smooth_sigma_bins and smooth_sigma_bins > 0:
    psth_by_neuron = gaussian_filter1d(psth_by_neuron, smooth_sigma_bins, axis=0, mode='nearest')

# Optionally sort neurons by peak latency (makes a nicer ridge visualization)
valid_cols = ~np.all(np.isnan(psth_by_neuron), axis=0)
if not np.any(valid_cols):
    print("Warning: All selected neurons have NaN PSTHs; skipping PSTH colormap.")
else:
    psth_by_neuron = psth_by_neuron[:, valid_cols]
    selected_cids = selected_cids[valid_cols]
    peak_idx = np.nanargmax(psth_by_neuron, axis=0)
    sort_order = np.argsort(peak_idx)
    psth_sorted = psth_by_neuron[:, sort_order].T  # (Nc, T)

# Z-score per neuron to normalize scale across units
    psth_sorted_z = zscore_time(psth_sorted, axis=1)

    plt.figure(figsize=(10, max(4, psth_sorted_z.shape[0] / 40)))
    im = plt.imshow(
        psth_sorted_z,
        aspect='auto',
        origin='lower',
        extent=[t_centers[0], t_centers[-1], 0, psth_sorted_z.shape[0]],
        cmap='magma',
        vmin=-1.5, vmax=3.0
    )
    plt.colorbar(im, label='Z-scored firing rate')
    plt.axvline(0, color='w', lw=1, ls='--', alpha=0.6)
    plt.xlim(-0.25, 1.25)
    plt.xlabel('Time (s)')
    plt.ylabel('Neurons (sorted by peak latency)')
    plt.title(f'PSTH colormap (mean across trials), {psth_sorted_z.shape[0]} neurons')
    plt.tight_layout()

# 2) Trial × time population colormap sorted by trial length
# Average across selected neurons to get population activity per trial
pop_by_trial = np.nanmean(binned_spikes[:, :, selected_cids], axis=2)  # (NT, T)

# Keep only good-length trials, sort by length (descending)
valid_trials = np.where(good_trials_len)[0]
sorted_trials = valid_trials[np.argsort(trial_lengths[good_trials_len])[::-1]]

pop_sorted = pop_by_trial[sorted_trials]  # (Nvalid, T)

# Optional smoothing along time per trial, then z-score per trial
if smooth_sigma_bins and smooth_sigma_bins > 0:
    pop_sorted = gaussian_filter1d(pop_sorted, smooth_sigma_bins, axis=1, mode='nearest')
pop_sorted_z = zscore_time(pop_sorted, axis=1)

plt.figure(figsize=(10, max(4, pop_sorted_z.shape[0] / 20)))
im = plt.imshow(
    pop_sorted_z,
    aspect='auto',
    origin='lower',
    extent=[t_centers[0], t_centers[-1], 0, pop_sorted_z.shape[0]],
    cmap='viridis',
    vmin=-1.5, vmax=3.0
)
plt.colorbar(im, label='Z-scored population rate')
plt.axvline(0, color='w', lw=1, ls='--', alpha=0.6)
plt.xlim(-0.25, 1.25)
plt.xlabel('Time (s)')
plt.ylabel('Trials (sorted by length)')
plt.title(f'Population activity across trials (N={pop_sorted_z.shape[0]}), neurons={selected_cids.size}')
plt.tight_layout()


# %% Forage trial-start triggered population responses
# Helper: get Forage (Dots or Gaborium pregen) trial start times in ephys
def _get_forage_starts_ephys(session, ptb2ephys, draw_latency=0,
							 include=('ForagePregenRepeatingNoise', 'ForageProceduralNoise')):
	"""
	Returns ephys-aligned trial start times for selected Forage protocols.

	Protocols included by default:
	- 'ForageDots': uses first `PR.NoiseHistory[:,0] + draw_latency`
	- 'ForagePregenGabor': uses first `PR.ProbeHistory[:,3] + draw_latency`
	"""
	exp = session.load_exp()
	protocols = get_trial_protocols(exp)

	starts_ptb = []
	for iT in range(len(exp['D'])):
		proto = protocols[iT]
		if proto not in include:
			continue
		pr = exp['D'][iT]['PR']
		if proto == 'ForageDots':
			flip_times = pr['NoiseHistory'][:, 0] + draw_latency
		elif proto == 'ForagePregenGabor':
			flip_times = pr['ProbeHistory'][:, 3] + draw_latency
		else:
			continue
		if flip_times.size == 0:
			continue
		starts_ptb.append(flip_times[0])

	if len(starts_ptb) == 0:
		raise RuntimeError("No selected Forage trials found in exp structure.")

	starts_ptb = np.asarray(starts_ptb, dtype=float)
	starts_ptb = starts_ptb[np.isfinite(starts_ptb)]
	starts_ephys = ptb2ephys(starts_ptb)
	return np.sort(starts_ephys)


def compute_sta_population(st, clu, cids, event_times,
                           t_pre=0.200, t_post=0.500, bin_ms=10.0):
    bin_sec = bin_ms / 1000.0
    rel_edges = np.arange(-t_pre, t_post + 1e-9, bin_sec).astype(np.float64)
    n_bins = len(rel_edges) - 1
    # Pre-index spikes per neuron
    cids = np.asarray(cids, dtype=int)
    st_min = float(np.min(st))
    st_max = float(np.max(st))
    st_by_unit = {cid: np.sort(np.asarray(st[clu == cid], dtype=np.float64)) for cid in cids}
    counts_sum = np.zeros(n_bins, dtype=np.float64)
    n_used = 0
    for et in np.asarray(event_times, dtype=np.float64):
        w0, w1 = et - t_pre, et + t_post
        if w1 < st_min or w0 > st_max:
            continue
        counts = np.zeros(n_bins, dtype=np.float64)
        for cid in cids:
            s = st_by_unit[cid]
            i0 = np.searchsorted(s, w0, side='left')
            i1 = np.searchsorted(s, w1, side='right')
            s_rel = s[i0:i1] - et
            if s_rel.size == 0:
                continue
            idx = np.searchsorted(rel_edges, s_rel, side='right') - 1
            valid = (idx >= 0) & (idx < n_bins)
            if np.any(valid):
                counts += np.bincount(idx[valid], minlength=n_bins)
        counts_sum += counts / max(1, cids.size)
        n_used += 1
    if n_used == 0:
        return rel_edges, np.zeros(n_bins, dtype=np.float64)
    psth = counts_sum / n_used / bin_sec
    return rel_edges, psth

def build_event_triggered_population_matrix(st, clu, cids, event_times,
                                            t_pre=0.200, t_post=0.500, bin_ms=10.0,
                                            smooth_sigma_bins=0):
    bin_sec = bin_ms / 1000.0
    rel_edges = np.arange(-t_pre, t_post + 1e-9, bin_sec).astype(np.float64)
    n_bins = len(rel_edges) - 1
    cids = np.asarray(cids, dtype=int)
    st_min = float(np.min(st))
    st_max = float(np.max(st))
    st_by_unit = {cid: np.sort(np.asarray(st[clu == cid], dtype=np.float64)) for cid in cids}
    rows = []
    for et in np.asarray(event_times, dtype=np.float64):
        w0, w1 = et - t_pre, et + t_post
        if w1 < st_min or w0 > st_max:
            continue
        counts = np.zeros(n_bins, dtype=np.float64)
        for cid in cids:
            s = st_by_unit[cid]
            i0 = np.searchsorted(s, w0, side='left')
            i1 = np.searchsorted(s, w1, side='right')
            s_rel = s[i0:i1] - et
            if s_rel.size == 0:
                continue
            idx = np.searchsorted(rel_edges, s_rel, side='right') - 1
            valid = (idx >= 0) & (idx < n_bins)
            if np.any(valid):
                counts += np.bincount(idx[valid], minlength=n_bins)
        rate = counts / max(1, cids.size) / bin_sec
        rows.append(rate)
    if len(rows) == 0:
        return rel_edges, np.zeros((0, n_bins), dtype=np.float64)
    mat = np.stack(rows, axis=0)
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        from scipy.ndimage import gaussian_filter1d as _gf1d
        mat = _gf1d(mat, smooth_sigma_bins, axis=1, mode='nearest')
    return rel_edges, mat

def plot_event_heatmap(mat, rel_edges, title, sort_by='peak', cmap='magma', vmin=None, vmax=None):
    if mat.shape[0] == 0:
        print(f"{title}: no events to plot.")
        return
    centers = 0.5 * (rel_edges[:-1] + rel_edges[1:])
    if sort_by == 'peak':
        order = np.argsort(np.nanargmax(mat, axis=1))
    elif sort_by == 'max':
        order = np.argsort(np.nanmax(mat, axis=1))[::-1]
    else:
        order = np.arange(mat.shape[0])
    mat_sorted = mat[order]
    plt.figure(figsize=(10, max(4, mat_sorted.shape[0] / 40)))
    im = plt.imshow(
        mat_sorted,
        aspect='auto',
        origin='lower',
        extent=[centers[0], centers[-1], 0, mat_sorted.shape[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.axvline(0, color='w', lw=1, ls='--', alpha=0.7)
    plt.colorbar(im, label='Population firing rate (Hz)')
    plt.xlabel('Time from Forage start (s)')
    plt.ylabel('Events (sorted)')
    plt.title(title)
    plt.tight_layout()

# Collect Forage starts → ephys
forage_ephys = _get_forage_starts_ephys(sess, ptb2ephys, draw_latency=0,
                                       include=('ForageDots', 'ForagePregenGabor', 'ForageProceduralNoise'))

if forage_ephys.size > 0:
    # Window and plotting
    edges_frg, psth_frg = compute_sta_population(
        st, clu, selected_cids,
        forage_ephys,
        t_pre=0.200, t_post=0.500, bin_ms=10.0
    )
    centers_frg = 0.5 * (edges_frg[:-1] + edges_frg[1:])
    plt.figure(figsize=(8, 4))
    plt.plot(centers_frg, psth_frg, color='C4')
    plt.axvline(0, color='k', lw=1, ls='--', alpha=0.7)
    plt.xlabel('Time from Forage start (s)')
    plt.ylabel('Population firing rate (Hz)')
    plt.title(f'Forage-start-triggered average, events={forage_ephys.size}, neurons={selected_cids.size}')
    plt.tight_layout()
    # 2D heatmaps: per-event population responses (raw and z-scored)
    edges_evt, mat_evt = build_event_triggered_population_matrix(
        st, clu, selected_cids, forage_ephys,
        t_pre=0.200, t_post=0.500, bin_ms=10.0, smooth_sigma_bins=2
    )
    if mat_evt.shape[0] > 0:
        plot_event_heatmap(
            mat_evt, edges_evt,
            f'Forage-start population heatmap (raw), events={mat_evt.shape[0]}',
            sort_by='none', cmap='magma'
        )
        mat_evt_z = zscore_time(mat_evt, axis=1)
        plot_event_heatmap(
            mat_evt_z, edges_evt,
            f'Forage-start population heatmap (Z), events={mat_evt.shape[0]}',
            sort_by='none', cmap='magma', vmin=-1.5, vmax=3.0
        )
else:
    print('No Forage* starts found for plotting.')


# %% Saccade-triggered response heatmaps (V1 vs V2)
from scipy.ndimage import gaussian_filter1d

def detect_saccades_from_eyepos(
    eyepos_xy,             # (NT, T, 2)
    time_bins,             # (T+1,) edges
    smooth_sigma_bins=2,   # smooth eye traces (bins)
    speed_percentile=99.5, # adaptive speed threshold (percentile)
    amp_thresh=0.3,        # deg; displacement across ±10 ms
    refrac_s=0.02,         # refractory period between saccades
    pre_s=0.2,             # window before saccade (s)
    post_s=0.3,            # window after saccade (s)
):
    NT, T, _ = eyepos_xy.shape
    dt = float(time_bins[1] - time_bins[0])
    pre_bins = int(round(pre_s / dt))
    post_bins = int(round(post_s / dt))
    refrac_bins = int(round(refrac_s / dt))
    amp_win_bins = max(1, int(round(0.01 / dt)))  # ±10 ms

    ex = eyepos_xy[:, :, 0].astype(float).copy()
    ey = eyepos_xy[:, :, 1].astype(float).copy()

    # NaN-aware Gaussian smoothing along time axis
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        def nan_gauss(x, sigma):
            x0 = np.where(np.isnan(x), 0.0, x)
            w = (~np.isnan(x)).astype(float)
            fx = gaussian_filter1d(x0, sigma, axis=1, mode='nearest')
            fw = gaussian_filter1d(w,  sigma, axis=1, mode='nearest')
            out = np.divide(fx, fw, out=np.full_like(fx, np.nan), where=(fw > 0))
            return out
        ex = nan_gauss(ex, smooth_sigma_bins)
        ey = nan_gauss(ey, smooth_sigma_bins)

    # Velocity magnitude
    dx = np.diff(ex, axis=1)
    dy = np.diff(ey, axis=1)
    speed = np.sqrt(dx**2 + dy**2) / dt

    # Adaptive speed threshold
    flat_speed = speed[~np.isnan(speed)]
    if flat_speed.size == 0:
        return [], pre_bins, post_bins, np.arange(-pre_bins, post_bins) * dt
    spd_thr = np.nanpercentile(flat_speed, speed_percentile)

    events = []  # list of (trial_idx, t_idx)
    for i in range(NT):
        sp = speed[i]
        if np.all(np.isnan(sp)):
            continue
        # candidate indices: above threshold and local maxima
        cand = np.where((sp > spd_thr) & (sp > np.roll(sp, 1)) & (sp >= np.roll(sp, -1)))[0]
        if cand.size == 0:
            continue

        # prune near edges (need full window)
        cand = cand[(cand >= pre_bins + amp_win_bins) & (cand < (T - 1) - post_bins - amp_win_bins)]
        if cand.size == 0:
            continue

        # amplitude check across ±10 ms around the candidate
        ex_i, ey_i = ex[i], ey[i]
        good = []
        for t0 in cand:
            if np.isnan(ex_i[t0-amp_win_bins]) or np.isnan(ex_i[t0+amp_win_bins]) \
               or np.isnan(ey_i[t0-amp_win_bins]) or np.isnan(ey_i[t0+amp_win_bins]):
                continue
            disp = np.hypot(ex_i[t0+amp_win_bins] - ex_i[t0-amp_win_bins],
                            ey_i[t0+amp_win_bins] - ey_i[t0-amp_win_bins])
            if disp >= amp_thresh:
                good.append(t0)
        if len(good) == 0:
            continue

        # enforce refractory period
        good = np.array(sorted(good), dtype=int)
        keep = [good[0]]
        for t0 in good[1:]:
            if t0 - keep[-1] >= refrac_bins:
                keep.append(t0)
        for t0 in keep:
            events.append((i, t0))

    t_rel = np.arange(-pre_bins, post_bins) * dt
    return events, pre_bins, post_bins, t_rel

# Build saccade-triggered matrices and plot

def build_saccade_triggered_matrix(binned_spikes, events, cids, pre_bins, post_bins, smooth_sigma_bins=0):
    if len(events) == 0 or len(cids) == 0:
        return np.zeros((0, pre_bins + post_bins)), None
    cids = np.asarray(cids, dtype=int)
    cids = cids[(cids >= 0) & (cids < binned_spikes.shape[2])]
    if cids.size == 0:
        return np.zeros((0, pre_bins + post_bins)), None

    window_len = pre_bins + post_bins
    rows = []
    for (tri, t0) in events:
        seg = binned_spikes[tri, t0 - pre_bins: t0 + post_bins, :][:, cids]  # (W, Nc)
        if seg.shape[0] != window_len:
            continue
        rows.append(np.nanmean(seg, axis=1))  # (W,)
    if len(rows) == 0:
        return np.zeros((0, window_len)), None

    mat = np.stack(rows, axis=0)  # (Nsacc, W)
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        mat = gaussian_filter1d(mat, smooth_sigma_bins, axis=1, mode='nearest')
    return mat, cids

def row_z(mat, eps=1e-9):
    m = np.nanmean(mat, axis=1, keepdims=True)
    s = np.nanstd(mat, axis=1, keepdims=True)
    return (mat - m) / (s + eps)

def plot_saccade_heatmap(mat, t_rel, title, vmin=None, vmax=None, sort_by='peak'):
    if mat.shape[0] == 0:
        print(f"{title}: no saccades to plot.")
        return
    if sort_by == 'peak':
        peak_idx = np.nanargmax(mat, axis=1)
        order = np.argsort(peak_idx)
    elif sort_by == 'max':
        order = np.argsort(np.nanmax(mat, axis=1))[::-1]
    else:
        order = np.arange(mat.shape[0])
    mat_sorted = mat[order]

    plt.figure(figsize=(10, max(4, mat_sorted.shape[0] / 40)))
    im = plt.imshow(
        mat_sorted,
        aspect='auto',
        origin='lower',
        extent=[t_rel[0], t_rel[-1], 0, mat_sorted.shape[0]],
        cmap='magma',
        vmin=vmin, vmax=vmax
    )
    plt.axvline(0, color='w', lw=1, ls='--', alpha=0.7)
    plt.colorbar(im, label='Mean firing rate (a.u.)')
    plt.xlabel('Time relative to saccade (s)')
    plt.ylabel('Saccades (sorted)')
    plt.title(title)
    plt.tight_layout()

# 1) Detect saccades (eyepos_xy is eyepos[:, :, :2])
# Ensure eyepos exists: build from EXP eyeData if missing
if 'eyepos' not in globals():
    try:
        NT = len(fixrsvp_trials)
    except Exception:
        NT = 0
    if NT > 0:
        eyepos = np.nan * np.zeros((NT, len(time_bins) - 1, 3))
        for i in range(NT):
            itrial = fixrsvp_trials[i]
            NH = exp['D'][itrial]['PR'].get('NoiseHistory', None)
            if NH is None or NH.shape[0] < 10:
                continue
            eye_data = exp['D'][itrial].get('eyeData', None)
            if eye_data is None or len(eye_data) == 0:
                continue
            eyecal_c = exp['D'][itrial]['C']['c']
            eyecal_dx = exp['D'][itrial]['C']['dx']
            eyecal_dy = exp['D'][itrial]['C']['dy']
            eye_data = eye_data.copy()
            eye_data[:, 1] = (eye_data[:, 1] - eyecal_c[0]) * eyecal_dx
            eye_data[:, 2] = (eye_data[:, 2] - eyecal_c[1]) * eyecal_dy
            eye_time = ptb2ephys(eye_data[:, 0])
            frame_times = ptb2ephys(NH[:, 0])
            eye_time = eye_time - frame_times[0]
            eye_digi = np.digitize(eye_time, time_bins) - 1
            eye_iix = (eye_digi >= 0) & (eye_digi < len(time_bins) - 1)
            eyepos[i, eye_digi[eye_iix], :] = eye_data[eye_iix, 1:4]

eyepos_xy = eyepos[:, :, :2]
events, pre_bins, post_bins, t_rel = detect_saccades_from_eyepos(
    eyepos_xy,
    time_bins,
    smooth_sigma_bins=2,
    speed_percentile=99.5,
    amp_thresh=0.3,
    refrac_s=0.02,
    pre_s=0.2,
    post_s=0.3,
)
print(f"Detected {len(events)} saccades across all trials.")

# 2) Build matrices for V1 and V2
W_sigma = 2  # temporal smoothing of STA (bins)
mat_v1, _ = build_saccade_triggered_matrix(binned_spikes, events, V1_cids, pre_bins, post_bins, smooth_sigma_bins=W_sigma)
mat_v2, _ = build_saccade_triggered_matrix(binned_spikes, events, V2_cids, pre_bins, post_bins, smooth_sigma_bins=W_sigma)

# 3) Plot heatmaps (raw and z-scored)
plot_saccade_heatmap(mat_v1, t_rel, f"Saccade-triggered response (V1), Nsacc={mat_v1.shape[0]}", sort_by='none')
plot_saccade_heatmap(mat_v2, t_rel, f"Saccade-triggered response (V2), Nsacc={mat_v2.shape[0]}", sort_by='none')

plot_saccade_heatmap(row_z(mat_v1), t_rel, f"Saccade-triggered response Z (V1), Nsacc={mat_v1.shape[0]}", vmin=-1.5, vmax=3.0, sort_by='none')
plot_saccade_heatmap(row_z(mat_v2), t_rel, f"Saccade-triggered response Z (V2), Nsacc={mat_v2.shape[0]}", vmin=-1.5, vmax=3.0, sort_by='none')
# 4) Overlaid average traces for V1 vs V2
plt.figure(figsize=(8, 4))
if mat_v1.shape[0] > 0:
    plt.plot(t_rel, np.nanmean(mat_v1, axis=0), label='V1', color='C0')
if mat_v2.shape[0] > 0:
    plt.plot(t_rel, np.nanmean(mat_v2, axis=0), label='V2', color='C3')
plt.axvline(0, color='k', lw=1, ls='--', alpha=0.7)
plt.xlabel('Time relative to saccade (s)')
plt.ylabel('Mean firing rate (a.u.)')
plt.title('Saccade-triggered average (area means)')
plt.legend(frameon=False)
plt.tight_layout()
# %%






#%%
from scipy.signal import savgol_filter
cids = np.arange(850, nclu)
for cc in cids:
    plt.plot(savgol_filter(np.nanmean(binned_spikes, 0)[:,cc], 10, 1))
    plt.title(f'Neuron {cc}')
    plt.show()

#%%
cids=[363,364,366,367,368,372,373,378,388,389,397,399,411,412,413,420,427,431,433,446,457,468,471,472,476,494,507,509,513,518,525,535,550,554]
for cc in cids:
    plt.plot(savgol_filter(np.nanmean(binned_spikes, 0)[:,cc], 10, 1))
    plt.title(f'Neuron {cc}')
    plt.show()

#%% V2 neurons
# cids = [144, 143, 140, 139, 137, 127, 105, 103, 97, 84, 83, 79, 69, 68, 65, 63, 61, 56, 51, 46, 44, 42, 41, 39, 36, 34, 33, 31, 30, 29, 18, 15, 12]
cids=[363,364,366,367,368,372,373,378,388,389,397,399,411,412,413,420,427,431,433,446,457,468,471,472,476,494,507,509,513,518,525,535,550,554]
for cc in cids:
    good_trials = np.isnan(binned_spikes[:,:,cc]).sum(1)==0
    jj, ii = np.where(binned_spikes[:,:,cc][good_trials])
    plot_raster(ii, jj, height=1)
    plt.title(f'Neuron {cc}')
    plt.show()
# plt.xlim(1000,1500)

#%% V1 neurons
cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
for cc in cids:
    good_trials = np.isnan(binned_spikes[:,:,cc]).sum(1)==0
    jj, ii = np.where(binned_spikes[:,:,cc][good_trials])
    plot_raster(ii, jj, height=1)
    plt.title(f'Neuron {cc}')
    plt.show()
# plt.xlim(1000,1500)


#%%

binned_spikes = np.nan*np.zeros((NT, len(time_bins)-1, nclu))
state = np.nan*np.zeros((NT, len(time_bins)-1))
eyepos = np.nan*np.zeros((NT, len(time_bins)-1, 3))
image_id = np.nan*np.zeros((NT, len(time_bins)-1))

for i in range(NT):
    itrial = fixrsvp_trials[i]
    NH = exp['D'][itrial]['PR']['NoiseHistory']
    if NH is None or NH.shape[0] < 10:
        continue

    eye_data = exp['D'][itrial]['eyeData']

    eyecal_c=exp['D'][itrial]['C']['c']
    eyecal_dx=exp['D'][itrial]['C']['dx']
    eyecal_dy=exp['D'][itrial]['C']['dy']

    eye_data[:,1] = (eye_data[:,1]-eyecal_c[0])*eyecal_dx
    eye_data[:,2] = (eye_data[:,2]-eyecal_c[1])*eyecal_dy

    t2 = np.where(NH[:,3]==2)[0][0]
    eye_time = ptb2ephys(eye_data[:,0])
    frame_times = ptb2ephys(NH[:,0])

    eye_time = eye_time - frame_times[0]
    # digitize to time_bins
    eye_digi = np.digitize(eye_time, time_bins) - 1
    eye_iix = (eye_digi >= 0) & (eye_digi < len(time_bins)-1)
    nh_digi = np.digitize(frame_times-frame_times[0], time_bins) - 1
    nh_good = (nh_digi >= 0) & (nh_digi < len(time_bins)-1)
    
    image_id[i, nh_digi[nh_good]] = NH[nh_good,3]
    eyepos[i, eye_digi[eye_iix], :] = eye_data[eye_iix, 1:4]
    state[i, eye_digi[eye_iix]] = eye_data[eye_iix, 4]
    
    st_ix = (st > frame_times[0]-1) & (st < frame_times[0]+1)
    

    trial_st = st[st_ix] - frame_times[0]
    trial_clu = clu[st_ix]
    st_binned = bin_spikes(trial_st, trial_clu, time_bins, n_units=nclu)
    binned_spikes[i] = st_binned


#%%
itrial += 1
if itrial >= binned_spikes.shape[0]:
    itrial = 0
plt.plot(time_bins[:-1], binned_spikes[good_trials][:,:,800:][itrial].mean(-1))
plt.gca().twinx()
plt.plot(time_bins[:-1], eyepos[good_trials][itrial][:,0], '.r')
plt.plot(time_bins[:-1], eyepos[good_trials][itrial][:,1], '.g')
plt.gca().twinx()
plt.plot(time_bins[:-1], image_id[good_trials][itrial], '.b')
plt.title(f"Trial {itrial}")

#%%plot eyepos timeline

plt.plot(eyepos[:,:,0].flatten(), '.r', alpha=0.1)
plt.plot(eyepos[:,:,1].flatten(), '.g', alpha=0.1)
plt.xlabel('Time bins')
plt.ylabel('Eye position (normed)')
plt.title('Eye Position over Time')

#%% Eye position histogram
eyex=eyepos[:,:,0].flatten()
eyey=eyepos[:,:,1].flatten()
bad_ix = np.isnan(eyex) | np.isnan(eyey)
eyex = eyex[~bad_ix]
eyey = eyey[~bad_ix]
edges = np.linspace(-2, 2, 101)
hist2d, xedges, yedges = np.histogram2d(eyex, eyey, bins=edges)
plt.figure()
plt.imshow(hist2d.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto') 
# Log scale color axis
plt.xlabel('Eye X position')
plt.ylabel('Eye Y position')
plt.title('Eye Position Histogram')
plt.colorbar(label='Counts')
plt.clim(0, np.max(hist2d)/10)


#%% Sort trials by trial length, by length of NoiseHistory
neuron_id = 932
good_trials = np.isnan(binned_spikes[:,:,neuron_id]).sum(1)==0
trial_lengths = []
for i in range(NT):
    itrial = fixrsvp_trials[i]
    NH = exp['D'][itrial]['PR']['NoiseHistory']
    if NH is None or NH.shape[0] < 10:
        trial_lengths.append(0)
    else:
        trial_lengths.append(NH.shape[0])
trial_lengths = np.array(trial_lengths)
sorted_trials = np.argsort(trial_lengths[good_trials])
plt.imshow(trial_lengths[good_trials][sorted_trials][:,np.newaxis], aspect='auto', cmap='viridis')
plt.colorbar(label='Trial Length (Number of NoiseHistory Frames)')
plt.xlabel('Trial Length')
plt.ylabel('Trials (sorted)')
plt.title('Trials Sorted by Trial Length')

#%% Plot spike rasters for a neuron, unsorted then sorted by trial length
for cc in cids:
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)
    ii, jj = np.where(binned_spikes[:,:,cc][good_trials][sorted_trials])
    plt.title(f'Neuron {cc} Spike Raster Sorted by Trial Length')
    plt.xlabel('Time Bins')
    plt.ylabel('Trials (sorted)')
    plot_raster(jj, ii, height=1)
    plt.show()

#%% Sort trials by mean eye position on x axis, enforce minimum trial length, spikes in trial
good_trials = np.isnan(binned_spikes[:,:,neuron_id]).sum(1)==0
min_trial_length = 10
good_trials = good_trials & (trial_lengths >= min_trial_length)

mean_eyex = np.nanmean(eyepos[:,:,0][good_trials], axis=1)
sorted_trials = np.argsort(mean_eyex)
plt.imshow(mean_eyex[sorted_trials][:,np.newaxis], aspect='auto', cmap='bwr', vmin=-2, vmax=2)
plt.colorbar(label='Mean Eye X Position')
plt.xlabel('Mean Eye X Position')
plt.ylabel('Trials (sorted)')
plt.title('Trials Sorted by Mean Eye X Position')

#%% Plot spike rasters for a neuron, unsorted then sorted by eye X position
neuron_id = 932
plt.figure(figsize=(10, 6))
plt.subplots_adjust(hspace=0.4)
plt.subplot(2, 1, 1)

jj, ii = np.where(binned_spikes[:,:,neuron_id][good_trials])
minimum_bins = 5 # count jj spike bins per trial per ii trial
unique_jj, counts = np.unique(jj, return_counts=True)
valid_jj = unique_jj[counts >= minimum_bins]
# only keep ii, jj where jj is in valid_jj, but that will leave blank y-axis rows
# so we need to filter ii, jj, then re-index as jj_new = np.arange(len(valid_jj))
valid_mask = np.isin(jj, valid_jj)
ii = ii[valid_mask]
jj = jj[valid_mask]
jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

plot_raster(ii, jj_new, height=1)
plt.title(f'Neuron {neuron_id} Spike Raster Unsorted')
plt.xlabel('Time Bins')
plt.ylabel('Trials')
plt.subplot(2, 1, 2)

jj, ii = np.where(binned_spikes[:,:,neuron_id][good_trials][sorted_trials])
#minimum_bins = 10 # count jj spike bins per trial per ii trial
unique_jj, counts = np.unique(jj, return_counts=True)
valid_jj = unique_jj[counts >= minimum_bins]
valid_mask = np.isin(jj, valid_jj)
ii = ii[valid_mask]
jj = jj[valid_mask]
jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

plot_raster(ii, jj_new, height=1)
plt.title(f'Neuron {neuron_id} Spike Raster Sorted by Eye X Position')
plt.xlabel('Time Bins')
plt.ylabel('Trials (sorted)')

#%%
minimum_bins = 5
# cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
cids=[363,364,366,367,368,372,373,378,388,389,397,399,411,412,413,420,427,431,433,446,457,468,471,472,476,494,507,509,513,518,525,535,550,554]
for cc in cids:
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)

    jj, ii = np.where(binned_spikes[:,:,cc][good_trials][sorted_trials])
    #minimum_bins = 10 # count jj spike bins per trial per ii trial
    unique_jj, counts = np.unique(jj, return_counts=True)
    valid_jj = unique_jj[counts >= minimum_bins]
    valid_mask = np.isin(jj, valid_jj)
    ii = ii[valid_mask]
    jj = jj[valid_mask]
    jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

    plt.title(f'Neuron {cc} Spike Raster Sorted by Eye X Position')
    plt.xlabel('Time Bins')
    plt.ylabel('Trials (sorted)')
    plot_raster(ii, jj_new, height=1)
    plt.show()


#%% Sort trials by mean eye position on y axis
good_trials = np.isnan(binned_spikes[:,:,neuron_id]).sum(1)==0
mean_eyey = np.nanmean(eyepos[:,:,1][good_trials], axis=1)
sorted_trials = np.argsort(mean_eyey)
plt.imshow(mean_eyey[sorted_trials][:,np.newaxis], aspect='auto', cmap='bwr', vmin=-2, vmax=2)
plt.colorbar(label='Mean Eye Y Position')
plt.xlabel('Mean Eye Y Position')
plt.ylabel('Trials (sorted)')
plt.title('Trials Sorted by Mean Eye Y Position')

#%% Plot spike rasters for a neuron, unsorted then sorted by eye Y position
neuron_id = 932
plt.figure(figsize=(10, 6))
plt.subplots_adjust(hspace=0.4)
plt.subplot(2, 1, 1)

jj, ii = np.where(binned_spikes[:,:,neuron_id][good_trials])
minimum_bins = 5 # count jj spike bins per trial per ii trial
unique_jj, counts = np.unique(jj, return_counts=True)
valid_jj = unique_jj[counts >= minimum_bins]
# only keep ii, jj where jj is in valid_jj, but that will leave blank y-axis rows
# so we need to filter ii, jj, then re-index as jj_new = np.arange(len(valid_jj))
valid_mask = np.isin(jj, valid_jj)
ii = ii[valid_mask]
jj = jj[valid_mask]
jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

plot_raster(ii, jj_new, height=1)
plt.title(f'Neuron {neuron_id} Spike Raster Unsorted')
plt.xlabel('Time Bins')
plt.ylabel('Trials')
plt.subplot(2, 1, 2)

jj, ii = np.where(binned_spikes[:,:,neuron_id][good_trials][sorted_trials])
#minimum_bins = 10 # count jj spike bins per trial per ii trial
unique_jj, counts = np.unique(jj, return_counts=True)
valid_jj = unique_jj[counts >= minimum_bins]
valid_mask = np.isin(jj, valid_jj)
ii = ii[valid_mask]
jj = jj[valid_mask]
jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

plot_raster(ii, jj_new, height=1)
plt.title(f'Neuron {neuron_id} Spike Raster Sorted by Eye Y Position')
plt.xlabel('Time Bins')
plt.ylabel('Trials (sorted)')

#%%
cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
for cc in cids:
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)

    jj, ii = np.where(binned_spikes[:,:,cc][good_trials][sorted_trials])
    #minimum_bins = 10 # count jj spike bins per trial per ii trial
    unique_jj, counts = np.unique(jj, return_counts=True)
    valid_jj = unique_jj[counts >= minimum_bins]
    valid_mask = np.isin(jj, valid_jj)
    ii = ii[valid_mask]
    jj = jj[valid_mask]
    jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

    plt.title(f'Neuron {cc} Spike Raster Sorted by Eye Y Position')
    plt.xlabel('Time Bins')
    plt.ylabel('Trials (sorted)')
    plot_raster(ii, jj_new, height=1)
    plt.show()


#%% Sort trials by mean eye position along a diagonal axis
good_trials = np.isnan(binned_spikes[:,:,neuron_id]).sum(1)==0
mean_eyex = np.nanmean(eyepos[:,:,0][good_trials], axis=1)
mean_eyey = np.nanmean(eyepos[:,:,1][good_trials], axis=1)
mean_eye_diag = (mean_eyex + mean_eyey) / np.sqrt(2)
sorted_trials = np.argsort(mean_eye_diag)
plt.imshow(mean_eye_diag[sorted_trials][:,np.newaxis], aspect='auto', cmap='bwr', vmin=-2, vmax=2)
plt.colorbar(label='Mean Eye Diagonal Position')
plt.xlabel('Mean Eye Diagonal Position')
plt.ylabel('Trials (sorted)')
plt.title('Trials Sorted by Mean Eye Diagonal Position')

#%%
cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
for cc in cids:
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)

    jj, ii = np.where(binned_spikes[:,:,cc][good_trials][sorted_trials])
    #minimum_bins = 10 # count jj spike bins per trial per ii trial
    unique_jj, counts = np.unique(jj, return_counts=True)
    valid_jj = unique_jj[counts >= minimum_bins]
    valid_mask = np.isin(jj, valid_jj)
    ii = ii[valid_mask]
    jj = jj[valid_mask]
    jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

    plt.title(f'Neuron {cc} Spike Raster Sorted by Diagonal Position')
    plt.xlabel('Time Bins')
    plt.ylabel('Trials (sorted)')
    plot_raster(ii, jj_new, height=1)
    plt.show()


#%% Sort trials by mean eye position along a negative diagonal axis
good_trials = np.isnan(binned_spikes[:,:,neuron_id]).sum(1)==0
mean_eyex = np.nanmean(eyepos[:,:,0][good_trials], axis=1)
mean_eyey = np.nanmean(eyepos[:,:,1][good_trials], axis=1)
mean_eye_diag = (mean_eyex - mean_eyey) / np.sqrt(2)
sorted_trials = np.argsort(mean_eye_diag)
plt.imshow(mean_eye_diag[sorted_trials][:,np.newaxis], aspect='auto', cmap='bwr', vmin=-2, vmax=2)
plt.colorbar(label='Mean Eye Diagonal Position')
plt.xlabel('Mean Eye Diagonal Position')
plt.ylabel('Trials (sorted)')
plt.title('Trials Sorted by Mean Eye Diagonal Position')
#%%
#%%
cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
for cc in cids:
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)

    jj, ii = np.where(binned_spikes[:,:,cc][good_trials][sorted_trials])
    #minimum_bins = 10 # count jj spike bins per trial per ii trial
    unique_jj, counts = np.unique(jj, return_counts=True)
    valid_jj = unique_jj[counts >= minimum_bins]
    valid_mask = np.isin(jj, valid_jj)
    ii = ii[valid_mask]
    jj = jj[valid_mask]
    jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

    plt.title(f'Neuron {cc} Spike Raster Sorted by Eye -ve diagonal Position')
    plt.xlabel('Time Bins')
    plt.ylabel('Trials (sorted)')
    plot_raster(ii, jj_new, height=1)
    plt.show()




#%% Check V2 neurons
#these were likely deep V1 neurons:cids = [144, 143, 140, 139, 137, 127, 105, 103, 97, 84, 83, 79, 69, 68, 65, 63, 61, 56, 51, 46, 44, 42, 41, 39, 36, 34, 33, 31, 30, 29, 18, 15, 12]
# full set, cids=V2_neurons
V2_cids=[363,364,366,367,368,372,373,378,388,389,397,399,411,412,413,420,427,431,433,446,457,468,471,472,476,494,507,509,513,518,525,535,550,554]
V1_cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
cids=V1_cids
for cc in cids:
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)

    jj, ii = np.where(binned_spikes[:,:,cc][good_trials][sorted_trials])
    #minimum_bins = 10 # count jj spike bins per trial per ii trial
    unique_jj, counts = np.unique(jj, return_counts=True)
    valid_jj = unique_jj[counts >= minimum_bins]
    valid_mask = np.isin(jj, valid_jj)
    ii = ii[valid_mask]
    jj = jj[valid_mask]
    jj_new = np.arange(len(valid_jj))[np.searchsorted(valid_jj, jj)]

    plt.title(f'Neuron {cc} Spike Raster Sorted by Eye -ve diagonal Position')
    plt.xlabel('Time Bins')
    plt.ylabel('Trials (sorted)')
    plot_raster(ii, jj_new, height=1)
    plt.show()


#%%


# %% V2 analysis: find two groups of 4 long trials with maximally different average PSTHs
# Goal: among "longish" FixRsvpStim trials, find disjoint groups A,B (size=4) maximizing
# the distance between their mean V2 population PSTHs.

def _compute_fixrsvp_trial_lengths(exp, fixrsvp_trials):
    lens = np.zeros(len(fixrsvp_trials), dtype=int)
    for i in range(len(fixrsvp_trials)):
        itrial = fixrsvp_trials[i]
        NH = exp['D'][itrial]['PR'].get('NoiseHistory', None)
        if NH is None:
            lens[i] = 0
            continue
        arr = np.asarray(NH)
        if arr.ndim == 1:
            lens[i] = int(arr.size)
        else:
            lens[i] = int(arr.shape[0])
    return lens

def _trial_population_psth_hz(binned_spikes, cids, dt_s):
    cids = np.asarray(cids, dtype=int).ravel()
    cids = cids[(cids >= 0) & (cids < binned_spikes.shape[2])]
    if cids.size == 0:
        return np.zeros((binned_spikes.shape[0], binned_spikes.shape[1]), dtype=float)
    # mean across neurons, convert counts/bin to Hz
    pop = np.nanmean(binned_spikes[:, :, cids], axis=2)
    return pop / float(dt_s)

def _pick_two_groups_maxdiff(pop_by_trial, candidate_idx, group_size=4, n_iter=50000, seed=0,
                             score='l2', smooth_bins=0.0):
    rng = np.random.default_rng(seed)
    cand = np.asarray(candidate_idx, dtype=int)
    if cand.size < 2 * group_size:
        return None

    best = None
    best_score = -np.inf
    X = pop_by_trial  # (NT, T)

    def _gauss_smooth_1d(y, sigma_bins):
        y = np.asarray(y, dtype=float)
        if sigma_bins is None or float(sigma_bins) <= 0:
            return y
        sigma = float(sigma_bins)
        radius = int(max(1, round(4.0 * sigma)))
        x = np.arange(-radius, radius + 1)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k = k / np.sum(k)
        # NaN-aware smoothing via weighted conv
        y0 = np.where(np.isfinite(y), y, 0.0)
        w = np.isfinite(y).astype(float)
        ys = np.convolve(y0, k, mode='same')
        ws = np.convolve(w, k, mode='same')
        out = np.full_like(y, np.nan, dtype=float)
        m = ws > 1e-12
        out[m] = ys[m] / ws[m]
        return out

    def _nan_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 3:
            return np.nan
        aa = a[m] - np.nanmean(a[m])
        bb = b[m] - np.nanmean(b[m])
        da = np.sqrt(np.nansum(aa * aa))
        db = np.sqrt(np.nansum(bb * bb))
        if da <= 0 or db <= 0:
            return np.nan
        return float(np.nansum(aa * bb) / (da * db))

    for _ in range(int(n_iter)):
        pick = rng.choice(cand, size=2 * group_size, replace=False)
        a = pick[:group_size]
        b = pick[group_size:]
        ma = np.nanmean(X[a], axis=0)
        mb = np.nanmean(X[b], axis=0)
        if score in ('anti_corr', 'anti_corr_x_l2'):
            # Prefer groups whose mean PSTHs are anti-correlated on slow timescales.
            # This is scale-insensitive (unlike L2) and emphasizes timing/shape differences.
            ma_s = _gauss_smooth_1d(ma, smooth_bins)
            mb_s = _gauss_smooth_1d(mb, smooth_bins)
            corr = _nan_corr(ma_s, mb_s)
            if not np.isfinite(corr):
                continue
            # Enforce anticorrelation (avoid picking purely correlated-but-large splits)
            if corr >= 0:
                continue
            if score == 'anti_corr_x_l2':
                d = ma_s - mb_s
                effect = float(np.sqrt(np.nansum(d * d)))
                s = (-corr) * effect
            else:
                s = -corr  # maximize anti-correlation (corr -> -1 => s -> +1)
        else:
            d = ma - mb
            if score == 'l1':
                s = float(np.nansum(np.abs(d)))
            else:
                s = float(np.sqrt(np.nansum(d * d)))
        if s > best_score:
            best_score = s
            best = (a.copy(), b.copy(), ma, mb)

    if best is None:
        return None
    return best_score, best


try:
    trial_lengths_fix = _compute_fixrsvp_trial_lengths(exp, fixrsvp_trials)
except Exception as e:
    print('V2 4v4 analysis: could not compute trial lengths:', repr(e))
    trial_lengths_fix = None

dt_s = float(time_bins[1] - time_bins[0])
t_centers = 0.5 * (time_bins[:-1] + time_bins[1:])
t_mask = (t_centers >= -0.25) & (t_centers <= 1.25)

# Use coarser bins for grouping + PSTH plots
group_bin_ms = 10.0
group_bin_size = int(max(1, round((group_bin_ms / 1000.0) / dt_s)))

pop_v2_hz = _trial_population_psth_hz(binned_spikes, V2_cids, dt_s)
pop_v2_hz = pop_v2_hz[:, t_mask]
t_show = t_centers[t_mask]

# Re-bin population PSTHs in time (for selection + plotting)
Tw = pop_v2_hz.shape[1]
Tw2 = (Tw // group_bin_size) * group_bin_size
if Tw2 < group_bin_size:
    pop_v2_hz_bin = pop_v2_hz
    t_show_bin = t_show
else:
    pop2 = pop_v2_hz[:, :Tw2].reshape(pop_v2_hz.shape[0], Tw2 // group_bin_size, group_bin_size)
    pop_v2_hz_bin = np.nanmean(pop2, axis=2)
    t2 = t_show[:Tw2].reshape(Tw2 // group_bin_size, group_bin_size)
    t_show_bin = np.nanmean(t2, axis=1)

if trial_lengths_fix is None:
    print('V2 4v4 analysis: skipping (no trial lengths).')
else:
    # "Longish" = above this quantile among nonzero lengths
    long_quantile = 0.70
    group_size = 4
    n_iter = 50000
    seed = 0
    score_kind = 'anti_corr_x_l2'  # 'anti_corr_x_l2'|'anti_corr'|'l2'|'l1'
    anti_corr_smooth_ms = 100.0  # slow-timescale smoothing for the selection criterion

    valid_len = trial_lengths_fix[trial_lengths_fix > 0]
    if valid_len.size == 0:
        print('V2 4v4 analysis: no trials with NoiseHistory length > 0')
    else:
        thr = int(np.quantile(valid_len, long_quantile))
        cand = np.where(trial_lengths_fix >= thr)[0]
        if cand.size < 2 * group_size:
            # fallback: just take top-K
            cand = np.argsort(trial_lengths_fix)[::-1]
            cand = cand[trial_lengths_fix[cand] > 0]

        # Keep only trials where PSTH has finite values
        finite_trial = np.any(np.isfinite(pop_v2_hz_bin), axis=1)
        cand = cand[finite_trial[cand]]

        print(f'V2 4v4 analysis: candidates={cand.size}, length_thr={thr} (quantile={long_quantile})')
        smooth_bins = 0
        if score_kind == 'anti_corr':
            smooth_bins = float(anti_corr_smooth_ms) / float(group_bin_ms)
        res_out = _pick_two_groups_maxdiff(
            pop_v2_hz_bin,
            cand,
            group_size=group_size,
            n_iter=n_iter,
            seed=seed,
            score=score_kind,
            smooth_bins=smooth_bins,
        )
        if res_out is None:
            print('V2 4v4 analysis: no valid split found (not enough candidates, or score rejected all candidates).')
        else:
            best_score, best = res_out
            A, B, meanA, meanB = best
            A = np.asarray(A, dtype=int)
            B = np.asarray(B, dtype=int)

            # Map indices back to original exp trial numbers
            A_trials = [fixrsvp_trials[i] for i in A]
            B_trials = [fixrsvp_trials[i] for i in B]

            if score_kind in ('anti_corr', 'anti_corr_x_l2'):
                # Recompute corr/effect for reporting
                # (use the same smoothing/time binning as the selection)
                def _gauss_smooth_1d(y, sigma_bins):
                    y = np.asarray(y, dtype=float)
                    if sigma_bins is None or float(sigma_bins) <= 0:
                        return y
                    sigma = float(sigma_bins)
                    radius = int(max(1, round(4.0 * sigma)))
                    x = np.arange(-radius, radius + 1)
                    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
                    k = k / np.sum(k)
                    y0 = np.where(np.isfinite(y), y, 0.0)
                    w = np.isfinite(y).astype(float)
                    ys = np.convolve(y0, k, mode='same')
                    ws = np.convolve(w, k, mode='same')
                    out = np.full_like(y, np.nan, dtype=float)
                    m = ws > 1e-12
                    out[m] = ys[m] / ws[m]
                    return out

                def _nan_corr(a, b):
                    a = np.asarray(a, dtype=float)
                    b = np.asarray(b, dtype=float)
                    m = np.isfinite(a) & np.isfinite(b)
                    if np.sum(m) < 3:
                        return np.nan
                    aa = a[m] - np.nanmean(a[m])
                    bb = b[m] - np.nanmean(b[m])
                    da = np.sqrt(np.nansum(aa * aa))
                    db = np.sqrt(np.nansum(bb * bb))
                    if da <= 0 or db <= 0:
                        return np.nan
                    return float(np.nansum(aa * bb) / (da * db))

                ma_s = _gauss_smooth_1d(meanA, smooth_bins)
                mb_s = _gauss_smooth_1d(meanB, smooth_bins)
                corr = _nan_corr(ma_s, mb_s)
                d = ma_s - mb_s
                effect_l2 = float(np.sqrt(np.nansum(d * d)))
                if score_kind == 'anti_corr_x_l2':
                    print(
                        f'Best: corr={corr:.4g}, effect_l2={effect_l2:.4g}, score={best_score:.4g} '
                        f'(bin={int(group_bin_ms)}ms, smooth={int(anti_corr_smooth_ms)}ms)'
                    )
                else:
                    print(
                        f'Best: corr={corr:.4g}, score={best_score:.4g} '
                        f'(bin={int(group_bin_ms)}ms, smooth={int(anti_corr_smooth_ms)}ms)'
                    )
            else:
                print(f'Best score ({score_kind}) = {best_score:.4g}')
            print('Group A (fixrsvp idx → exp trial id, length):')
            for i in A:
                print(f'  {i:4d} → {fixrsvp_trials[i]:4d}, len={trial_lengths_fix[i]}')
            print('Group B (fixrsvp idx → exp trial id, length):')
            for i in B:
                print(f'  {i:4d} → {fixrsvp_trials[i]:4d}, len={trial_lengths_fix[i]}')

            # Plot: mean PSTHs + difference
            plt.figure(figsize=(10, 4))
            plt.plot(t_show_bin, meanA, label='Group A (mean)', color='C0', lw=2)
            plt.plot(t_show_bin, meanB, label='Group B (mean)', color='C3', lw=2)
            plt.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
            plt.xlim(-0.25, 1.25)
            plt.xlabel('Time from RSVP start (s)')
            plt.ylabel('V2 population rate (Hz)')
            if score_kind in ('anti_corr', 'anti_corr_x_l2'):
                # Note: best_score is either -corr or (-corr)*effect depending on score_kind
                plt.title(
                    f'Best 4v4 split (V2), score={best_score:.3g}, thr_len={thr}, cand={cand.size}, '
                    f'bin={int(group_bin_ms)}ms, smooth={int(anti_corr_smooth_ms)}ms'
                )
            else:
                plt.title(f'Best 4v4 split (V2), score={best_score:.3g}, thr_len={thr}, cand={cand.size}, bin={int(group_bin_ms)}ms')
            plt.legend(frameon=False)
            plt.tight_layout()

            plt.figure(figsize=(10, 3))
            plt.plot(t_show_bin, meanA - meanB, color='k', lw=2)
            plt.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
            plt.xlim(-0.25, 1.25)
            plt.xlabel('Time from RSVP start (s)')
            plt.ylabel('A − B (Hz)')
            plt.title('Difference of group means')
            plt.tight_layout()

            # Plot: per-trial overlays
            plt.figure(figsize=(10, 4))
            for i in A:
                plt.plot(t_show_bin, pop_v2_hz_bin[i], color='C0', alpha=0.25)
            for i in B:
                plt.plot(t_show_bin, pop_v2_hz_bin[i], color='C3', alpha=0.25)
            plt.plot(t_show_bin, meanA, color='C0', lw=2)
            plt.plot(t_show_bin, meanB, color='C3', lw=2)
            plt.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
            plt.xlim(-0.25, 1.25)
            plt.xlabel('Time from RSVP start (s)')
            plt.ylabel('V2 population rate (Hz)')
            plt.title('Per-trial PSTHs (thin) + group means (thick)')
            plt.tight_layout()

            # Persist best groups for follow-on analysis
            best_v2_A_fixidx = A
            best_v2_B_fixidx = B
            best_v2_A_exp_trials = A_trials
            best_v2_B_exp_trials = B_trials
            best_v2_score = best_score
            best_v2_len_thr = thr

            # Plot: rasters for the two winning groups (V2 units)
            t_mask_r = (t_centers >= -0.05) & (t_centers <= 0.80)
            t_show_r = t_centers[t_mask_r]

            def _style_pretty_raster_axes(ax, show_xticklabels=False):
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_yticks([])
                ax.tick_params(axis='y', left=False, labelleft=False)
                # Remove time axes entirely (ticks/labels)
                ax.set_xticks([])
                ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('')

            def _plot_trial_v2_raster(ax, fixidx, cids, binned_spikes, t_show, t_mask, trial_lengths_fix, fixrsvp_trials):
                cids = np.asarray(cids, dtype=int).ravel()
                cids = cids[(cids >= 0) & (cids < binned_spikes.shape[2])]
                if cids.size == 0:
                    ax.text(0.5, 0.5, 'No valid V2 units', ha='center', va='center', color='0.3', transform=ax.transAxes)
                    return

                sp = np.asarray(binned_spikes[fixidx, :, :], dtype=float)[:, cids]  # (T, Nc)
                sp_win = sp[t_mask, :]
                ti, ui = np.nonzero(sp_win > 0)
                if ti.size == 0:
                    ax.text(0.5, 0.5, 'No spikes', ha='center', va='center', color='0.3', transform=ax.transAxes)
                else:
                    cnt = sp_win[ti, ui].astype(int, copy=False)
                    if np.max(cnt) > 1:
                        x = np.repeat(t_show[ti], cnt)
                        y = np.repeat(ui, cnt)
                    else:
                        x = t_show[ti]
                        y = ui
                    plot_raster(x, y, height=0.8, ax=ax, linewidth=0.4, color='k')
                ax.axvline(0, color='0.2', lw=1, ls='--', alpha=0.8)
                ax.set_xlim(-0.05, 0.80)
                ax.set_ylim(-1, cids.size)

            def _plot_two_group_rasters(groupA_fixidx, groupB_fixidx, labelA='Group A', labelB='Group B'):
                from matplotlib.gridspec import GridSpec
                from matplotlib.lines import Line2D
                from pathlib import Path

                psth_bin_ms = 10.0  # coarser binning for the average PSTHs

                groupA_fixidx = np.asarray(groupA_fixidx, dtype=int).ravel()
                groupB_fixidx = np.asarray(groupB_fixidx, dtype=int).ravel()

                def _plot_group_mean_psth(ax, group_fixidx, color):
                    group_fixidx = np.asarray(group_fixidx, dtype=int).ravel()
                    cids = np.asarray(V2_cids, dtype=int).ravel()
                    cids = cids[(cids >= 0) & (cids < binned_spikes.shape[2])]
                    if group_fixidx.size == 0 or cids.size == 0:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='0.3', transform=ax.transAxes)
                        return
                    dt_s = float(np.median(np.diff(t_show_r))) if t_show_r.size > 1 else 0.001
                    sp = np.asarray(binned_spikes[group_fixidx, :, :], dtype=float)  # (ntr, T, n_units)
                    sp = sp[:, t_mask_r, :][:, :, cids]  # (ntr, Tw, Nc)
                    pop_per_trial = np.nanmean(sp, axis=2) / dt_s  # (ntr, Tw) Hz (mean across neurons)

                    # Re-bin to coarser time bins for display
                    bin_size = int(max(1, round((psth_bin_ms / 1000.0) / dt_s)))
                    Tw = pop_per_trial.shape[1]
                    Tw2 = (Tw // bin_size) * bin_size
                    if Tw2 < bin_size:
                        mean = np.nanmean(pop_per_trial, axis=0)
                        sem = np.nanstd(pop_per_trial, axis=0) / np.sqrt(max(1, pop_per_trial.shape[0]))
                        t_plot = t_show_r
                    else:
                        pop2 = pop_per_trial[:, :Tw2]
                        pop2 = pop2.reshape(pop2.shape[0], Tw2 // bin_size, bin_size)
                        pop_bin = np.nanmean(pop2, axis=2)  # (ntr, nbin)
                        mean = np.nanmean(pop_bin, axis=0)
                        sem = np.nanstd(pop_bin, axis=0) / np.sqrt(max(1, pop_bin.shape[0]))
                        t2 = t_show_r[:Tw2]
                        t2 = t2.reshape(Tw2 // bin_size, bin_size)
                        t_plot = np.nanmean(t2, axis=1)

                    ax.plot(t_plot, mean, color=color, lw=1.5)
                    # Fill under mean curve (no time axes; use scale bar)
                    ax.fill_between(t_plot, 0.0, mean, color=color, alpha=0.25, linewidth=0)
                    ax.axvline(0, color='0.2', lw=1, ls='--', alpha=0.8)
                    ax.set_xlim(-0.05, 0.80)
                    ax.set_ylim(bottom=0.0)

                fig = plt.figure(figsize=(10.5, 7.5))
                gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 0.9], hspace=0.10, wspace=0.10)

                # 4 rasters per group (two columns)
                axesA = [fig.add_subplot(gs[r, 0]) for r in range(4)]
                axesB = [fig.add_subplot(gs[r, 1]) for r in range(4)]
                axA_psth = fig.add_subplot(gs[4, 0])
                axB_psth = fig.add_subplot(gs[4, 1])

                for k, ax in enumerate(axesA):
                    if k >= groupA_fixidx.size:
                        ax.set_visible(False)
                        continue
                    fi = int(groupA_fixidx[k])
                    _plot_trial_v2_raster(ax, fi, V2_cids, binned_spikes, t_show_r, t_mask_r, trial_lengths_fix, fixrsvp_trials)
                    _style_pretty_raster_axes(ax, show_xticklabels=False)

                for k, ax in enumerate(axesB):
                    if k >= groupB_fixidx.size:
                        ax.set_visible(False)
                        continue
                    fi = int(groupB_fixidx[k])
                    _plot_trial_v2_raster(ax, fi, V2_cids, binned_spikes, t_show_r, t_mask_r, trial_lengths_fix, fixrsvp_trials)
                    _style_pretty_raster_axes(ax, show_xticklabels=False)

                # Bottom row: mean PSTH per group
                _plot_group_mean_psth(axA_psth, groupA_fixidx, color='C0')
                _plot_group_mean_psth(axB_psth, groupB_fixidx, color='C3')

                # PSTH styling (remove time axes; keep only traces + scale bar)
                for ax in (axA_psth, axB_psth):
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(axis='y', left=False, labelleft=False)
                    ax.set_xticks([])
                    ax.tick_params(axis='x', bottom=False, labelbottom=False)
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                # Column labels (subtle)
                fig.text(0.25, 0.98, str(labelA), ha='center', va='top', color='0.25', fontsize=10)
                fig.text(0.75, 0.98, str(labelB), ha='center', va='top', color='0.25', fontsize=10)

                # Dotted separators between raster trials (across both columns)
                visible_axes = [ax for ax in (axesA + axesB) if ax.get_visible()]
                if len(visible_axes) > 0:
                    x_left = min(ax.get_position().x0 for ax in visible_axes)
                    x_right = max(ax.get_position().x1 for ax in visible_axes)
                    for r in range(3):
                        y_upper_bottom = max(axesA[r].get_position().y0, axesB[r].get_position().y0)
                        y_lower_top = min(axesA[r + 1].get_position().y1, axesB[r + 1].get_position().y1)
                        y_sep = 0.5 * (y_upper_bottom + y_lower_top)
                        fig.add_artist(
                            Line2D(
                                [x_left, x_right], [y_sep, y_sep],
                                transform=fig.transFigure,
                                linestyle=':',
                                linewidth=1.0,
                                color='0.35',
                                alpha=0.9,
                            )
                        )

                # 500 ms time bar (draw once; in Group A PSTH panel)
                # Window is [-0.05, 0.80] => width = 0.85 s
                bar_s = 0.5
                win_s = 0.80 - (-0.05)
                bar_frac = bar_s / win_s
                x0 = 0.08
                x1 = x0 + bar_frac
                y0 = 0.20
                axA_psth.plot([x0, x1], [y0, y0], transform=axA_psth.transAxes, clip_on=False, color='0.2', lw=2.0)
                axA_psth.plot([x0, x0], [y0 - 0.04, y0 + 0.04], transform=axA_psth.transAxes, clip_on=False, color='0.2', lw=2.0)
                axA_psth.plot([x1, x1], [y0 - 0.04, y0 + 0.04], transform=axA_psth.transAxes, clip_on=False, color='0.2', lw=2.0)
                axA_psth.text((x0 + x1) / 2.0, y0 + 0.06, '500 ms', transform=axA_psth.transAxes,
                              ha='center', va='bottom', color='0.2', fontsize=8)

                # Tighten margins
                fig.subplots_adjust(left=0.03, right=0.995, bottom=0.06, top=0.94)

                return fig

            fig_groups = _plot_two_group_rasters(best_v2_A_fixidx, best_v2_B_fixidx, labelA='Group A', labelB='Group B')
            try:
                from pathlib import Path
                out_path = Path('figures') / 'Luke0804_V2_4v4_groups_rasters_psth.svg'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig_groups.savefig(out_path, format='svg', bbox_inches='tight')
                print(f'Saved SVG: {out_path}')
            except Exception as e:
                print('Warning: could not save SVG:', repr(e))

#%%

#%% Eye traces for the selected 4v4 trials (color-coded by group)
if 'best_v2_A_exp_trials' not in globals() or 'best_v2_B_exp_trials' not in globals():
    print('Eye traces: best_v2_A_exp_trials / best_v2_B_exp_trials not defined; run V2 4v4 selection first.')
else:
    if 'exp' not in globals() or exp is None:
        session = globals().get('session', None)
        if session is None:
            raise RuntimeError('Eye traces: session not defined; run the loader cells first.')
        exp = session.load_exp()

    t0, t1 = -0.05, 0.80
    A_trials = np.asarray(best_v2_A_exp_trials, dtype=int).ravel()
    B_trials = np.asarray(best_v2_B_exp_trials, dtype=int).ravel()

    def _get_eye_data_trial(D):
        for k in ('eyeData', 'eyedata', 'EyeData'):
            if k in D:
                ed = D[k]
                if ed is not None:
                    return np.asarray(ed)
        return None

    def _trial_rsvp_start_ptb(D):
        PR = D.get('PR', {})
        NH = PR.get('NoiseHistory', None)
        if NH is None:
            st = PR.get('startTime', np.nan)
            return float(st) if np.isfinite(st) else np.nan
        arr = np.asarray(NH, dtype=float)
        if arr.size == 0:
            return np.nan
        if arr.ndim == 1:
            return float(arr[0])
        return float(arr[0, 0])

    def _get_trial_eye_trace_rel(D):
        eye_data = _get_eye_data_trial(D)
        if eye_data is None or eye_data.ndim != 2 or eye_data.shape[1] < 3:
            return None

        t_ptb = np.asarray(eye_data[:, 0], dtype=float)
        x = np.asarray(eye_data[:, 1], dtype=float)
        y = np.asarray(eye_data[:, 2], dtype=float)

        # Apply calibration if present (matches earlier eyepos construction)
        try:
            C = D.get('C', None)
            if C is not None and 'c' in C and 'dx' in C and 'dy' in C:
                c = np.asarray(C['c'], dtype=float).ravel()
                dx = float(C['dx'])
                dy = float(C['dy'])
                if c.size >= 2 and np.isfinite(dx) and np.isfinite(dy):
                    x = (x - c[0]) * dx
                    y = (y - c[1]) * dy
        except Exception:
            pass

        start_ptb = _trial_rsvp_start_ptb(D)
        if not np.isfinite(start_ptb):
            return None

        # Align to RSVP start in ephys timebase for consistency with spikes
        try:
            t_e = ptb2ephys(t_ptb)
            t0_e = ptb2ephys(start_ptb)
        except Exception:
            # Fallback: stay in PTB timebase if mapping unavailable
            t_e = t_ptb
            t0_e = start_ptb

        t_rel = np.asarray(t_e, dtype=float) - float(t0_e)
        m = np.isfinite(t_rel) & np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return None
        return t_rel[m], x[m], y[m]

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)

    def _plot_group(trials, color, label):
        n_ok = 0
        for tr in trials:
            if tr < 0 or tr >= len(exp['D']):
                continue
            out = _get_trial_eye_trace_rel(exp['D'][int(tr)])
            if out is None:
                continue
            t_rel, x, y = out
            w = (t_rel >= t0) & (t_rel <= t1)
            if not np.any(w):
                continue
            ax_x.plot(t_rel[w], x[w], color=color, alpha=0.35, lw=1.0)
            ax_y.plot(t_rel[w], y[w], color=color, alpha=0.35, lw=1.0)
            n_ok += 1
        ax_x.plot([], [], color=color, lw=2, label=f'{label} (n={n_ok})')
        return n_ok

    _plot_group(A_trials, 'C0', 'Group A')
    _plot_group(B_trials, 'C3', 'Group B')

    for ax in (ax_x, ax_y):
        ax.axvline(0, color='0.2', lw=1, ls='--', alpha=0.8)
        ax.set_xlim(t0, t1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_x.set_ylabel('Eye X (cal units)')
    ax_y.set_ylabel('Eye Y (cal units)')
    ax_y.set_xlabel('Time from RSVP start (s)')
    ax_x.legend(frameon=False, loc='upper right')
    fig.tight_layout()

    # Save SVG
    try:
        from pathlib import Path
        out_path = Path('figures') / 'Luke0804_V2_4v4_groups_eye_traces.svg'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format='svg', bbox_inches='tight')
        print(f'Saved SVG: {out_path}')
    except Exception as e:
        print('Warning: could not save eye-trace SVG:', repr(e))
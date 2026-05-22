#%%
"""
Saccade-triggered average (STA) responses early vs late in session.

Loads spikes and eye tracker data, detects saccades from pupil positions,
aligns saccade times to ephys clock, and computes/plots population PSTHs
around saccades for early and late portions of the session.
"""

import numpy as np
import matplotlib.pyplot as plt
from DataRowleyV1V2.data.registry import get_session

#%%
# --- Rowley loader parity with testingRowley0.py ---
# Probe metadata to derive V1/V2 CIDs from mean depth, matching testingRowley0.py
datadir = '/mnt/ssd/RowleyMarmoV1V2/raw/Luke_2025-08-04/'
imec1_path = datadir + '/pipeline_results_Luke0804_V2V1_g0_imec1/cur/cur_sorter_output'
imec0_path = datadir + '/pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_sorter_output'

try:
	spike_positions_imec0 = np.load(imec0_path + '/spike_positions.npy')
	spike_positions_imec1 = np.load(imec1_path + '/spike_positions.npy')

	spike_clusters_imec0 = np.load(imec0_path + '/spike_clusters.npy')
	spike_clusters_imec1 = np.load(imec1_path + '/spike_clusters.npy')

	spike_cids_imec0 = np.unique(spike_clusters_imec0)
	spike_cids_imec1 = np.unique(spike_clusters_imec1)

	mean_pos_imec0 = np.nan * np.zeros((int(spike_clusters_imec0.max()) + 1, 1))
	mean_pos_imec1 = np.nan * np.zeros((int(spike_clusters_imec1.max()) + 1, 1))

	for cc in range(int(spike_clusters_imec0.max()) + 1):
		cc_pos = spike_positions_imec0[spike_clusters_imec0 == cc]
		if cc_pos.size == 0:
			continue
		mean_pos_imec0[cc] = np.mean(np.mean(cc_pos, axis=0))

	for cc in range(int(spike_clusters_imec1.max()) + 1):
		cc_pos = spike_positions_imec1[spike_clusters_imec1 == cc]
		if cc_pos.size == 0:
			continue
		mean_pos_imec1[cc] = np.mean(np.mean(cc_pos, axis=0))

	# Threshold to separate shallow/deep (1250 um), consistent with testingRowley0.py
	V1_cids = (np.max(spike_cids_imec0) + 1 + spike_cids_imec1[mean_pos_imec1[:, 0] < 1250]).astype(int)
	V2_cids = (spike_cids_imec0[mean_pos_imec0[:, 0] >= 1250]).astype(int)
except Exception as e:
	# If paths are unavailable in this environment, fall back to empty lists
	print(f"Warning: imec metadata not loaded ({e}). V1/V2 CID sets empty.")
	V1_cids, V2_cids = np.array([], dtype=int), np.array([], dtype=int)


def _pick_eye_columns(dpi_df, eye='right'):
	"""Select time and pupil position columns from DPI DataFrame for a given eye.

	Tries common OpenIris column names. Returns (t, x, y) as numpy arrays.
	"""
	assert eye in ('left', 'right'), "eye must be 'left' or 'right'"

	# Time columns typically exist per-eye
	t_col = f'{eye.capitalize()}Seconds'
	if t_col not in dpi_df.columns:
		# Fallback: sometimes only a unified seconds column exists
		t_col = 'Seconds' if 'Seconds' in dpi_df.columns else None

	# Pupil position (raw) columns
	x_col = f'{eye.capitalize()}PupilX'
	y_col = f'{eye.capitalize()}PupilY'

	# If raw pupil isn't present, try CR-Pupil relative (proxy for gaze)
	if x_col not in dpi_df.columns or y_col not in dpi_df.columns:
		x_col = f'{eye.capitalize()}CR1X'
		y_col = f'{eye.capitalize()}CR1Y'

	# Final check
	missing = [c for c in (t_col, x_col, y_col) if c is None or c not in dpi_df.columns]
	if len(missing) > 0:
		raise KeyError(f"Missing required DPI columns: {missing}")

	t = dpi_df[t_col].to_numpy()
	x = dpi_df[x_col].to_numpy()
	y = dpi_df[y_col].to_numpy()
	return t, x, y


def detect_saccades(t, x, y,
					vel_threshold=250.0,
					min_interval=0.120):
	"""Detect saccade onset times from pupil position using speed threshold.

	Parameters
	- t: time (s)
	- x, y: positions (px or arbitrary units)
	- vel_threshold: speed threshold (units/s)
	- min_interval: minimum interval between saccades (s)

	Returns: saccade onset times (same timebase as t)
	"""
	# Denoise with simple moving average (3 samples) to reduce noise
	def smooth(arr, k=3):
		if k <= 1:
			return arr
		k = int(k)
		pad = k // 2
		pad_left = arr[0:1].repeat(pad)
		pad_right = arr[-1:].repeat(pad)
		arr_pad = np.concatenate([pad_left, arr, pad_right])
		kernel = np.ones(k) / k
		return np.convolve(arr_pad, kernel, mode='valid')

	x_s = smooth(x, 5)
	y_s = smooth(y, 5)

	# Compute velocity magnitude using central differences
	dt = np.gradient(t)
	# Guard against zeros in dt
	dt[dt == 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1.0
	vx = np.gradient(x_s) / dt
	vy = np.gradient(y_s) / dt
	speed = np.sqrt(vx**2 + vy**2)

	# Threshold crossings: rising edges
	above = speed > vel_threshold
	edges = np.where(np.diff(above.astype(np.int8)) == 1)[0] + 1
	cand_times = t[edges]

	# Enforce refractory period
	sacc_times = []
	last_t = -np.inf
	for tt in cand_times:
		if tt - last_t >= min_interval:
			sacc_times.append(tt)
			last_t = tt

	return np.array(sacc_times)


def compute_sta(st, clu, cids, event_times,
				 t_pre=0.200, t_post=0.500, bin_ms=10.0):
	"""Fast population PSTH via searchsorted + bincount."""
	bin_sec = bin_ms / 1000.0
	rel_edges = np.arange(-t_pre, t_post + 1e-9, bin_sec).astype(np.float64)
	n_bins = len(rel_edges) - 1

	st_min = float(np.min(st))
	st_max = float(np.max(st))
	# Precompute per-unit, sorted spike times
	st_by_unit = {cid: np.sort(np.asarray(st[clu == cid], dtype=np.float64)) for cid in cids}

	counts_sum = np.zeros(n_bins, dtype=np.float64)
	n_events_used = 0

	for et in np.asarray(event_times, dtype=np.float64):
		win_start = et - t_pre
		win_end = et + t_post
		if win_end < st_min or win_start > st_max:
			continue

		event_counts = np.zeros(n_bins, dtype=np.float64)
		for cid in cids:
			s = st_by_unit[cid]
			i0 = np.searchsorted(s, win_start, side='left')
			i1 = np.searchsorted(s, win_end, side='right')
			s_rel = s[i0:i1] - et
			if s_rel.size == 0:
				continue
			idx = np.searchsorted(rel_edges, s_rel, side='right') - 1
			valid = (idx >= 0) & (idx < n_bins)
			if np.any(valid):
				event_counts += np.bincount(idx[valid], minlength=n_bins)

		counts_sum += event_counts / max(1, len(cids))
		n_events_used += 1

	if n_events_used == 0:
		return rel_edges, np.zeros(n_bins, dtype=np.float64)

	psth = counts_sum / n_events_used / bin_sec
	return rel_edges, psth


def compute_sta_matrix(st, clu, cids, event_times,
						 t_pre=0.200, t_post=0.500, bin_ms=10.0):
	"""Fast per-event mean firing rate over time bins."""
	bin_sec = bin_ms / 1000.0
	rel_edges = np.arange(-t_pre, t_post + 1e-9, bin_sec).astype(np.float64)
	n_bins = len(rel_edges) - 1

	st_min = float(np.min(st))
	st_max = float(np.max(st))
	st_by_unit = {cid: np.sort(np.asarray(st[clu == cid], dtype=np.float64)) for cid in cids}

	rows = []
	for et in np.asarray(event_times, dtype=np.float64):
		win_start = et - t_pre
		win_end = et + t_post
		if win_end < st_min or win_start > st_max:
			continue

		counts = np.zeros(n_bins, dtype=np.float64)
		for cid in cids:
			s = st_by_unit[cid]
			i0 = np.searchsorted(s, win_start, side='left')
			i1 = np.searchsorted(s, win_end, side='right')
			s_rel = s[i0:i1] - et
			if s_rel.size == 0:
				continue
			idx = np.searchsorted(rel_edges, s_rel, side='right') - 1
			valid = (idx >= 0) & (idx < n_bins)
			if np.any(valid):
				counts += np.bincount(idx[valid], minlength=n_bins)

		rows.append((counts / max(1, len(cids))) / bin_sec)

	if len(rows) == 0:
		return rel_edges, np.zeros((0, n_bins), dtype=np.float64)

	return rel_edges, np.stack(rows, axis=0)

def subsample_events_uniform(times, max_events=1000):
	"""Uniformly subsample events across the session to a maximum count."""
	times = np.asarray(times)
	if times.size <= max_events:
		return times
	order = np.argsort(times)
	idx = np.linspace(0, times.size - 1, max_events).astype(int)
	return times[order][idx]

from DataRowleyV1V2.exp.general import get_trial_protocols

# Helper: get BackImage trial start times and align to ephys (using EXP online eyetrace)
def _get_backimage_starts_ephys(session, ptb2ephys, draw_latency=0):
    """
    Returns ephys-aligned trial start times for BackImage trials by
    parsing the marmoview exp structure, consistent with BackImageTrial.
    """
    exp = session.load_exp()
    protocols = get_trial_protocols(exp)

    # Collect PTB start times for BackImage trials
    ptb_starts = []
    for iT in range(len(exp['D'])):
        if protocols[iT] == 'BackImage':
            # BackImageTrial uses PR.startTime + draw_latency
            start_ptb = exp['D'][iT]['PR']['startTime'] + draw_latency
            ptb_starts.append(start_ptb)

    if len(ptb_starts) == 0:
        raise RuntimeError("No BackImage trials found in exp structure.")

    ptb_starts = np.asarray(ptb_starts, dtype=float)
    ptb_starts = ptb_starts[np.isfinite(ptb_starts)]

	# Map PTB start times to ephys
    t_ephys = ptb2ephys(ptb_starts)
    return np.sort(t_ephys)

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


# Helper: get FixRSVP trial start times in ephys (robust to 1D/2D NoiseHistory)
def _get_fixRSVP_starts_ephys(session, ptb2ephys, draw_latency=0,
                              include=('FixRsvpStim',)):
    exp = session.load_exp()
    protocols = get_trial_protocols(exp)

    starts_ptb = []
    for iT in range(len(exp['D'])):
        proto = protocols[iT]
        if proto not in include:
            continue
        PR = exp['D'][iT]['PR']
        NH = PR.get('NoiseHistory', None)
        if NH is None:
            continue
        arr = np.asarray(NH, dtype=float)
        if arr.size == 0:
            continue
        first_flip = float(arr[0]) if arr.ndim == 1 else float(arr[0, 0])
        starts_ptb.append(first_flip + draw_latency)

    if len(starts_ptb) == 0:
        return np.array([], dtype=float)

    starts_ptb = np.asarray(starts_ptb, dtype=float)
    starts_ptb = starts_ptb[np.isfinite(starts_ptb)]
    starts_ephys = ptb2ephys(starts_ptb)
    return np.sort(starts_ephys)


#%%
# Load session (match testingRowley0.py date and loaders)
session = get_session('Luke', '2025-08-04')
assert session is not None, 'Failed to load session.'

# Load spikes (testingRowley0.py uses tuple indexing; handle both signatures)
spikes = session.load_spikes()
if isinstance(spikes, (list, tuple)) and len(spikes) >= 2:
	st, clu = spikes[0], spikes[1]
else:
	st, clu = spikes

# Ensure numpy arrays (handles torch/array-like inputs)
def _to_numpy(a):
    try:
        # torch tensors
        return a.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(a)

st = _to_numpy(st)
clu = _to_numpy(clu)
# Derive unique cluster IDs from clu (parity with testingRowley0.py usage)
cids = np.unique(clu)

# PTB → ephys mapper (parity with testingRowley0.py)
ptb2ephys = session.load_ptb2ephys()[0]

# Get BackImage trial starts in ephys timebase
trial_starts = _get_backimage_starts_ephys(session, ptb2ephys)

# Keep events within spike time range with margins
t_min = st.min() + 2.50
t_max = st.max() - 2.50
trial_starts = trial_starts[(trial_starts >= t_min) & (trial_starts <= t_max)]

if trial_starts.size < 20:
    print(f"Warning: few BackImage trials in spike range ({trial_starts.size}). Results may be noisy.")

# Split early vs late using percentiles
early_cut = np.percentile(trial_starts, 20)
late_cut = np.percentile(trial_starts, 80)
starts_early = trial_starts[trial_starts <= early_cut]
starts_late  = trial_starts[trial_starts >= late_cut]

# Compute STAs, on every 5th trial to speed up computation
edges, psth_early = compute_sta(st, clu, cids, starts_early[0::5],
                                t_pre=1.200, t_post=2.500, bin_ms=100.0)
_, psth_late = compute_sta(st, clu, cids, starts_late[0::5],
                           t_pre=1.200, t_post=2.500, bin_ms=100.0)

centers = 0.5 * (edges[:-1] + edges[1:])

# Plot BackImage trial-start-triggered averages
plt.figure(figsize=(10, 6))
plt.plot(centers, psth_early, label=f'Early (n={len(starts_early)})', color='C0')
plt.plot(centers, psth_late,  label=f'Late (n={len(starts_late)})',  color='C1')
plt.axvline(0, color='k', lw=1, alpha=0.6)
plt.xlabel('Time from BackImage trial start (s)')
plt.ylabel('Population firing rate (Hz)')
plt.title('BackImage trial-start-triggered average responses (early vs late)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# (Removed DPI-based saccade section; we rely on EXP online eyetrace only.)

#%%
# Forage (Dots/Gaborium pregen) trial-start-triggered responses
forage_starts = _get_forage_starts_ephys(
	session, ptb2ephys, draw_latency=0,
	include=('ForageDots', 'ForagePregenGabor')
)

# Keep events within spike time range with margins
forage_starts = forage_starts[(forage_starts >= t_min) & (forage_starts <= t_max)]

if forage_starts.size < 20:
	print(f"Warning: few Forage trials in spike range ({forage_starts.size}).")

# Split early vs late using percentiles
f_early_cut = np.percentile(forage_starts, 20)
f_late_cut = np.percentile(forage_starts, 80)
forage_early = forage_starts[forage_starts <= f_early_cut]
forage_late  = forage_starts[forage_starts >= f_late_cut]

# Compute STAs
edges_f, psth_f_early = compute_sta(
	st, clu, cids, forage_early[0::5],
	t_pre=.0500, t_post=0.050, bin_ms=2.0
)
_, psth_f_late = compute_sta(
	st, clu, cids, forage_late[0::5],
	t_pre=.0500, t_post=0.050, bin_ms=2.0
)
_, psth_f_late = compute_sta(
	st, clu, cids, forage_late[0::5],
	t_pre=.0500, t_post=0.050, bin_ms=2.0
)

centers_f = 0.5 * (edges_f[:-1] + edges_f[1:])

plt.figure(figsize=(10, 6))
plt.plot(centers_f, psth_f_early, label=f'Early (n={len(forage_early)})', color='C2')
plt.plot(centers_f, psth_f_late,  label=f'Late (n={len(forage_late)})',  color='C3')
plt.axvline(0, color='k', lw=1, alpha=0.6)
plt.xlabel('Time from Forage trial start (s)')
plt.ylabel('Population firing rate (Hz)')
plt.title('Forage trial-start-triggered average responses (Dots/PregenGabor)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Raw per-trial heatmap for Forage
sample_every_f = 1
starts_sorted_f  = np.sort(forage_starts)
starts_sampled_f = starts_sorted_f[0::sample_every_f]

edges2_f, mat_f = compute_sta_matrix(
	st, clu, cids, starts_sampled_f,
	t_pre=0.200, t_post=0.500, bin_ms=10.0
)

centers2_f = 0.5 * (edges2_f[:-1] + edges2_f[1:])

plt.figure(figsize=(12, 8))
plt.imshow(
	mat_f,
	aspect='auto',
	cmap='viridis',
	extent=(centers2_f[0], centers2_f[-1], float(len(mat_f)), 0.0)
)
plt.colorbar(label='Mean rate (Hz)')
plt.xlabel('Time from Forage trial start (s)')
plt.ylabel('Trial index (start → end)')
plt.title(f'Per-trial mean rates (Forage: Dots/PregenGabor; every {sample_every_f}th)')
plt.axvline(0, color='w', lw=1, alpha=0.6)
plt.tight_layout()
plt.show()

# Raw per-trial heatmap across the full session
sample_every = 1  # downsample rows for readability
starts_sorted  = np.sort(trial_starts)
starts_sampled = starts_sorted[0::sample_every]

edges2, mat = compute_sta_matrix(
    st, clu, cids, starts_sampled,
    t_pre=0.200, t_post=0.500, bin_ms=10.0
)

centers2 = 0.5 * (edges2[:-1] + edges2[1:])

plt.figure(figsize=(12, 8))
plt.imshow(
    mat,
    aspect='auto',
    cmap='viridis',
	extent=(centers2[0], centers2[-1], float(len(mat)), 0.0)
)
plt.colorbar(label='Mean rate (Hz)')
plt.xlabel('Time from BackImage trial start (s)')
plt.ylabel('Trial index (start → end)')
plt.title(f'Per-trial mean rates (BackImage; every {sample_every}th trial)')
plt.axvline(0, color='w', lw=1, alpha=0.6)
plt.tight_layout()
plt.show()


#%%
def _get_exp_eyetraces_ptb(session):
    """
    Concatenate Exp(:).eyedata(:,0:2) across trials and return PTB (t), x, y.
    Handles common key variants: 'eyedata', 'eyeData', 'EyeData'.
    """
    exp = session.load_exp()
    t_list, x_list, y_list = [], [], []

    for iT in range(len(exp['D'])):
        D = exp['D'][iT]
        ed_key = next((k for k in ('eyedata', 'eyeData', 'EyeData') if k in D), None)
        if ed_key is None:
            continue
        ed = np.asarray(D[ed_key])
        if ed.ndim != 2 or ed.shape[1] < 3:
            continue

        t = ed[:, 0]
        x = ed[:, 1]
        y = ed[:, 2]
        mask = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue

        t_list.append(t[mask])
        x_list.append(x[mask])
        y_list.append(y[mask])

    if len(t_list) == 0:
        raise RuntimeError("No valid eyedata found in exp.")

    t_ptb = np.concatenate(t_list)
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    # Ensure monotonic order by time
    order = np.argsort(t_ptb)
    return t_ptb[order], x[order], y[order]

#%%
# Saccade-triggered raw heatmap using ALL detected saccades (EXP eyedata)
t_ptb, x_ptb, y_ptb = _get_exp_eyetraces_ptb(session)

# Detect saccades in PTB timebase, then map to ephys
sacc_ptb = detect_saccades(t_ptb, x_ptb, y_ptb)
sacc_ephys_exp = ptb2ephys(sacc_ptb)

# Keep saccades within spike time range for this window
s_t_pre = 0.200
s_t_post = 0.500
s_margin = max(s_t_pre, s_t_post)
s_t_min = st.min() + s_margin
s_t_max = st.max() - s_margin
sacc_ephys_exp = sacc_ephys_exp[(sacc_ephys_exp >= s_t_min) & (sacc_ephys_exp <= s_t_max)]

edges_se, mat_se = compute_sta_matrix(
	st, clu, cids, subsample_events_uniform(sacc_ephys_exp, max_events=1000),
    t_pre=s_t_pre, t_post=s_t_post, bin_ms=10.0
)

centers_se = 0.5 * (edges_se[:-1] + edges_se[1:])

plt.figure(figsize=(12, 8))
plt.imshow(
    mat_se,
    aspect='auto',
    cmap='viridis',
	extent=(centers_se[0], centers_se[-1], float(len(mat_se)), 0.0)
)
plt.colorbar(label='Mean rate (Hz)')
plt.xlabel('Time from saccade (s)')
plt.ylabel('Saccade index (start → end)')
plt.title('Per-saccade mean rates (EXP eyedata; all saccades)')
plt.axvline(0, color='w', lw=1, alpha=0.6)
plt.tight_layout()
plt.show()


#%% Find and allign RSVP trials


def compute_sta_by_neuron(st, clu, cids, event_times, t_pre=0.200, t_post=0.500, bin_ms=10.0):
    """Mean PSTH per neuron across events. Returns (edges, Nc×Nbins matrix)."""
    bin_sec = bin_ms / 1000.0
    rel_edges = np.arange(-t_pre, t_post + 1e-9, bin_sec).astype(np.float64)
    n_bins = len(rel_edges) - 1
    cids = np.asarray(cids, dtype=int)
    st_min, st_max = float(np.min(st)), float(np.max(st))
    st_by_unit = {cid: np.sort(np.asarray(st[clu == cid], dtype=np.float64)) for cid in cids}
    psth_mat = np.zeros((len(cids), n_bins), dtype=np.float64)
    n_used = 0
    ev = np.asarray(event_times, dtype=np.float64)
    for et in ev:
        w0, w1 = et - t_pre, et + t_post
        if w1 < st_min or w0 > st_max:
            continue
        for ui, cid in enumerate(cids):
            s = st_by_unit[cid]
            i0 = np.searchsorted(s, w0, side='left')
            i1 = np.searchsorted(s, w1, side='right')
            s_rel = s[i0:i1] - et
            if s_rel.size == 0:
                continue
            idx = np.searchsorted(rel_edges, s_rel, side='right') - 1
            valid = (idx >= 0) & (idx < n_bins)
            if np.any(valid):
                psth_mat[ui] += np.bincount(idx[valid], minlength=n_bins)
        n_used += 1
    if n_used == 0:
        return rel_edges, np.zeros((len(cids), n_bins), dtype=np.float64)
    psth_mat = (psth_mat / n_used) / bin_sec
    return rel_edges, psth_mat

def zscore_rows(mat, eps=1e-9):
    m = np.nanmean(mat, axis=1, keepdims=True)
    s = np.nanstd(mat, axis=1, keepdims=True)
    return (mat - m) / (s + eps)

#%% RSVP (FixRsvpStim) — neuron×time PSTH colormap and per-trial heatmap
rsvp_starts = _get_fixRSVP_starts_ephys(session, ptb2ephys, draw_latency=0)

# Keep events within spike time range
margin_pre, margin_post = 0.200, 0.500
t_min = st.min() + margin_pre
t_max = st.max() - margin_post
rsvp_starts = rsvp_starts[(rsvp_starts >= t_min) & (rsvp_starts <= t_max)]


if rsvp_starts.size == 0:
    print("No FixRsvpStim starts found in spike range.")
else:
    # Neuron×time PSTH colormap (use V2_cids if available, else all)
    sel_cids = V2_cids if 'V2_cids' in globals() and V2_cids.size > 0 else cids
    edges_n, psth_neu = compute_sta_by_neuron(
        st, clu, sel_cids, rsvp_starts, t_pre=0.200, t_post=1.50, bin_ms=25.0
    )
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    valid_rows = ~np.all(np.isnan(psth_neu), axis=1)
    if np.any(valid_rows):
        psth_neu = psth_neu[valid_rows]
        peak_idx = np.nanargmax(psth_neu, axis=1)
        order = np.argsort(peak_idx)
        psth_sorted_z = zscore_rows(psth_neu[order])
        plt.figure(figsize=(10, max(4, psth_sorted_z.shape[0] / 40)))
        im = plt.imshow(
            psth_sorted_z,
            aspect='auto',
            origin='lower',
            extent=[centers_n[0], centers_n[-1], 0, psth_sorted_z.shape[0]],
            cmap='magma', vmin=-1.5, vmax=3.0
        )
        plt.colorbar(im, label='Z-scored firing rate')
        plt.axvline(0, color='w', lw=1, ls='--', alpha=0.6)
        plt.xlabel('Time from RSVP start (s)')
        plt.ylabel('Neurons (sorted by peak)')
        plt.title(f'RSVP PSTH colormap (mean across trials), N={psth_sorted_z.shape[0]}')
        plt.tight_layout()
    else:
        print("RSVP PSTH: all selected neurons are NaN; skipping colormap.")

    # Per-trial population heatmap sorted by trial length
    exp = session.load_exp()
    protocols = get_trial_protocols(exp)
    starts_ephys_list, trial_len_list = [], []
    for iT in range(len(exp['D'])):
        if protocols[iT] != 'FixRsvpStim':
            continue
        PR = exp['D'][iT]['PR']
        NH = PR.get('NoiseHistory', None)
        if NH is None:
            continue
        arr = np.asarray(NH, dtype=float)
        if arr.size == 0:
            continue
        # First flip time (PTB) and trial length (rows)
        first_ptb = float(arr[0]) if arr.ndim == 1 else float(arr[0, 0])
        start_e = ptb2ephys(first_ptb)
        if start_e < t_min or start_e > t_max:
            continue
        starts_ephys_list.append(start_e)
        trial_len_list.append(arr.size if arr.ndim == 1 else arr.shape[0])

    starts_ephys_arr = np.asarray(starts_ephys_list, dtype=float)
    trial_len_arr = np.asarray(trial_len_list, dtype=int)

    edges_evt, mat_evt = compute_sta_matrix(
        st, clu, sel_cids, starts_ephys_arr, t_pre=0.200, t_post=1.50, bin_ms=25.0
    )
    centers_evt = 0.5 * (edges_evt[:-1] + edges_evt[1:])
    if mat_evt.shape[0] > 0 and trial_len_arr.size == mat_evt.shape[0]:
        order_trials = np.argsort(trial_len_arr)[::-1]
        mat_sorted = mat_evt[order_trials]
        plt.figure(figsize=(12, max(6, mat_sorted.shape[0] / 25)))
        im = plt.imshow(
            mat_sorted,
            aspect='auto',
            origin='lower',
            extent=[centers_evt[0], centers_evt[-1], 0, mat_sorted.shape[0]],
            cmap='viridis'
        )
        plt.colorbar(im, label='Mean rate (Hz)')
        plt.axvline(0, color='w', lw=1, alpha=0.6)
        plt.xlabel('Time from RSVP start (s)')
        plt.ylabel('Trials (sorted by length)')
        plt.title(f'RSVP per-trial population (sorted), Ntrials={mat_sorted.shape[0]}')
        plt.tight_layout()
    else:
        print("RSVP per-trial heatmap: no valid events or mismatch with lengths.")
		

#%% RSVP (FixRsvpStim) — V1, neuron×time PSTH colormap and per-trial heatmap
rsvp_starts = _get_fixRSVP_starts_ephys(session, ptb2ephys, draw_latency=0)

# Keep events within spike time range
margin_pre, margin_post = 0.200, 0.500
t_min = st.min() + margin_pre
t_max = st.max() - margin_post
rsvp_starts = rsvp_starts[(rsvp_starts >= t_min) & (rsvp_starts <= t_max)]


if rsvp_starts.size == 0:
    print("No FixRsvpStim starts found in spike range.")
else:
    # Neuron×time PSTH colormap (use V1_cids if available, else all)
    sel_cids = V1_cids if 'V1_cids' in globals() and V1_cids.size > 0 else cids
    edges_n, psth_neu = compute_sta_by_neuron(
        st, clu, sel_cids, rsvp_starts, t_pre=0.200, t_post=1.50, bin_ms=25.0
    )
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    valid_rows = ~np.all(np.isnan(psth_neu), axis=1)
    if np.any(valid_rows):
        psth_neu = psth_neu[valid_rows]
        peak_idx = np.nanargmax(psth_neu, axis=1)
        order = np.argsort(peak_idx)
        psth_sorted_z = zscore_rows(psth_neu[order])
        plt.figure(figsize=(10, max(4, psth_sorted_z.shape[0] / 40)))
        im = plt.imshow(
            psth_sorted_z,
            aspect='auto',
            origin='lower',
            extent=[centers_n[0], centers_n[-1], 0, psth_sorted_z.shape[0]],
            cmap='magma', vmin=-1.5, vmax=3.0
        )
        plt.colorbar(im, label='Z-scored firing rate')
        plt.axvline(0, color='w', lw=1, ls='--', alpha=0.6)
        plt.xlabel('Time from RSVP start (s)')
        plt.ylabel('Neurons (sorted by peak)')
        plt.title(f'RSVP PSTH colormap (mean across trials), N={psth_sorted_z.shape[0]}')
        plt.tight_layout()
    else:
        print("RSVP PSTH: all selected neurons are NaN; skipping colormap.")

    # Per-trial population heatmap sorted by trial length
    exp = session.load_exp()
    protocols = get_trial_protocols(exp)
    starts_ephys_list, trial_len_list = [], []
    for iT in range(len(exp['D'])):
        if protocols[iT] != 'FixRsvpStim':
            continue
        PR = exp['D'][iT]['PR']
        NH = PR.get('NoiseHistory', None)
        if NH is None:
            continue
        arr = np.asarray(NH, dtype=float)
        if arr.size == 0:
            continue
        # First flip time (PTB) and trial length (rows)
        first_ptb = float(arr[0]) if arr.ndim == 1 else float(arr[0, 0])
        start_e = ptb2ephys(first_ptb)
        if start_e < t_min or start_e > t_max:
            continue
        starts_ephys_list.append(start_e)
        trial_len_list.append(arr.size if arr.ndim == 1 else arr.shape[0])

    starts_ephys_arr = np.asarray(starts_ephys_list, dtype=float)
    trial_len_arr = np.asarray(trial_len_list, dtype=int)

    edges_evt, mat_evt = compute_sta_matrix(
        st, clu, sel_cids, starts_ephys_arr, t_pre=0.200, t_post=1.50, bin_ms=25.0
    )
    centers_evt = 0.5 * (edges_evt[:-1] + edges_evt[1:])
    if mat_evt.shape[0] > 0 and trial_len_arr.size == mat_evt.shape[0]:
        order_trials = np.argsort(trial_len_arr)[::-1]
        mat_sorted = mat_evt[order_trials]
        plt.figure(figsize=(12, max(6, mat_sorted.shape[0] / 25)))
        im = plt.imshow(
            mat_sorted,
            aspect='auto',
            origin='lower',
            extent=[centers_evt[0], centers_evt[-1], 0, mat_sorted.shape[0]],
            cmap='viridis'
        )
        plt.colorbar(im, label='Mean rate (Hz)')
        plt.axvline(0, color='w', lw=1, alpha=0.6)
        plt.xlabel('Time from RSVP start (s)')
        plt.ylabel('Trials (sorted by length)')
        plt.title(f'RSVP per-trial population (sorted), Ntrials={mat_sorted.shape[0]}')
        plt.tight_layout()
    else:
        print("RSVP per-trial heatmap: no valid events or mismatch with lengths.")
# %%

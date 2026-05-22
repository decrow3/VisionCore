# Editing to test out plotting for Luke-08-04 covariance decomposition
#%% Setup and Imports
#420 V2 unit
V1_cids = [962, 950, 949,945,942,941,939,938,937,935,934,932,931,930,928,924,923,922,916,914,910,909,907,903,901,894,893,887,886,883,882,874,873,866,862,861,852,857,856,855,853,852,851]
V2_cids=[363,364,366,367,368,372,373,378,388,389,397,399,411,412,413,420,427,431,433,446,457,468,471,472,476,494,507,509,513,518,525,535,550,554]
SELECTED_CIDS = V1_cids
# ----------------------------
# Rowley Luke0804 no-model diagnostics
# ----------------------------
# This script is a sandbox for understanding how dual-window binning plays out for
# Rowley Luke_2025-08-04 ("Luke0804") and for validating eye-distance distributions.
#
# Key changes vs older usage:
# - We explicitly restrict the analysis to the V1 unit set (V1_cids).
# - We add per-window histograms of pairwise eye-trajectory distances.
# - We intentionally SKIP ccmax filtering for now, but leave a clear placeholder
#   where ccmax/high-ccmax unit filtering can be reintroduced.

# Toggle: recompute from dataset (recommended for distance hist diagnostics)
RUN_ANALYSIS = True

# Dataset config for Luke0804 (Rowley). This parent YAML now resolves to the
# refreshed per-eye session YAML in experiments/dataset_configs/sessions/Luke_2025-08-04_left_V1.yaml.
from pathlib import Path
DATASET_CONFIGS_PATH = str(
    Path(__file__).resolve().parent.parent / "experiments/dataset_configs/single_Luke0804_120_noshift_rowley.yaml"
)

# DualWindowAnalysis sweep parameters
# Note: effective minimum bin size is dt*1000 ms.
# When sampling is forced to 240 Hz, dt = 1/240 s => 4.1667 ms/bin.
# To avoid confusion from flooring/quantization, we define windows in integer bins and
# compute the corresponding ms after dt is known.
USE_WINDOW_BINS = True
WINDOW_BINS = [1, 2, 4, 8, 16, 32]

# Fallback if USE_WINDOW_BINS=False
WINDOWS_MS = [5, 10, 20, 40, 80]
T_HIST_MS = 100
VALID_TIME_BINS = 240
TOTAL_SPIKES_THRESHOLD = 50 

# Override the dataset-config resampling. The default Luke0804 config downsamples 240->120.
# For binning diagnostics we often want to stay at 240 Hz.
FORCE_TARGET_RATE_HZ = 240

# Eye-distance histogram parameters
# IMPORTANT: To match scripts/mcfarland_sim.py, use percentile-based bin edges.
# This creates equal-count bins over the (subsampled) upper-triangular RMS distances
# and intentionally leaves the extreme tail (>= last edge) unbinned.
EYE_DIST_N_BINS = 30
EYE_BINNING_MODE = 'uniform'   # 'mcfarland' | 'uniform' (linear) | 'quantile' (equal-count)
EYE_BIN_Q = 0.95  # only used by some modes; ignored by 'mcfarland'

# Intercept estimation mode
# - 'linear': match mcfarland_sim.py default (weighted local linear regression on Ceye)
# - 'raw_first_bin': use Ceye at the first valid bin (no fitting)
INTERCEPT_MODE = 'linear'
INTERCEPT_D_MAX = 0.6
INTERCEPT_EVAL_AT_FIRST_BIN = True
INTERCEPT_FORCE_NONPOS_SLOPE = False  # keep False for fidelity to current mcfarland_sim.py

# Fit/diagnostics will only use distance bins with at least this many pairs.
# With many small bins, reduce this so near-zero bins still participate.
MIN_BIN_COUNT_FOR_FIT = 5

# If True, condition eye-distance bins only within the SAME aligned time index (T_idx).
# This matches the McFarland-style "time-matched" conditioning and does NOT require a model.
TIME_MATCHED_CONDITIONING = True

# Cap per-time-bin samples when doing time-matched conditioning to keep O(M^2) tractable.
MAX_SAMPLES_PER_TIMEBIN = 960

# Cap samples per window to keep distance histograms tractable.
# WARNING: distance calculation is O(N^2) in the number of extracted windows.
MAX_SAMPLES_PER_WINDOW = 15000

# Save outputs/figures for reproducibility and fast iteration
SAVE_OUTPUTS = True
SAVE_FIGURES = True
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "mcfarland"

# Optional: inspect diagonal (i,i) covariance-vs-distance curves on a grid
PLOT_INSPECTION_GRID = True

# Use smallest window for inspection (more same-time samples; more small-distance pairs).
# If USE_WINDOW_BINS is enabled, this will be set from WINDOW_BINS after dt is known.
INSPECT_WIN_MS = None

# Plot all neurons (chunked into multiple PDFs).
INSPECT_CHUNK_SIZE = 36
INSPECT_NCOLS = 6
SHOW_INSPECTION_GRIDS = True

# Save vector graphics for paper-quality inspection
SAVE_SVG = True

# ----------------------------
# Intercept-fit filtering
# ----------------------------
# We can filter out cells with unreliable diagonal intercept fits (Crate) before
# running the full sweep. This uses a single "probe" window (default: 2 bins).
ENABLE_CELL_FIT_FILTER = True

# Probe window selection: prefer bins when USE_WINDOW_BINS=True.
FIT_FILTER_WINDOW_BINS = 2
FIT_FILTER_WINDOW_MS = 8.33333333333334

# Minimum number of cells to keep; abort analysis if we drop below this.
MIN_CELLS_AFTER_FILTER = 10

# Fit-quality thresholds (diagonal only)
FIT_MIN_R2 = 0.60
FIT_MIN_LOG_POINTS = 4
FIT_MAX_FALLBACK_FRAC = 1.0  # (unused currently; placeholder)
FIT_DISALLOW_CLAMPED = False
FIT_REQUIRE_NONFALLBACK = False
FIT_ALLOW_FALLBACK_IF_FINITE = True
FIT_MIN_VALID_BINS = 4

# ----------------------------
# Intercept fit: weighting + slope expectation
# ----------------------------
# We expect eye-conditioned second moments to *decrease* with distance, i.e.
# log(y-plateau) should have a negative slope vs distance.
FIT_EXPECT_NEG_SLOPE = True

# Weight earlier (small-distance) bins more heavily in the log-linear fit.
# This helps focus the fit on the near-zero regime that determines the intercept.
FIT_WEIGHT_POWER = 2.0          # larger => more emphasis on small distances
FIT_WEIGHT_EPS = 1e-6           # avoids division by zero when x≈0
FIT_WEIGHT_USE_COUNTS = True    # additionally weight by sqrt(bin_counts)

# Fallback intercept uses earliest bins (small distance) instead of the full mean.
FIT_FALLBACK_EARLY_K = 3

# If True, try progressively looser masks to keep >= MIN_CELLS_AFTER_FILTER
# before aborting. This avoids brittle failures when the curve doesn't show
# a clean exponential decay (common at small windows / low spike counts).
FIT_FILTER_AUTO_RELAX = True

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np
#%%
from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()
device = get_free_device()

# Use the canonical McFarland implementation to avoid method drift.
# We alias it to avoid clashing with the sandbox class defined below.
from mcfarland_sim import DualWindowAnalysis as McFarlandDualWindowAnalysis

# If True, run analysis via scripts/mcfarland_sim.py exactly.
USE_CANONICAL_MCFARLAND = False

#%% Utility function for smoothing eye position
import numpy as np
import torch

from scipy.signal import savgol_filter

def _savgol_1d_nan(y, window_length=15, polyorder=3):
    """
    Apply Savitzky–Golay to a 1D array with NaNs.
    NaNs are interpolated for filtering and then restored.
    """
    y = np.asarray(y, float)
    mask = np.isfinite(y)

    # If too few valid points, just return original
    if mask.sum() < polyorder + 2:
        return y

    yy = y.copy()
    idx_valid = np.where(mask)[0]
    idx_nan   = np.where(~mask)[0]

    # Linear interp over NaNs so savgol_filter has no gaps
    yy[idx_nan] = np.interp(idx_nan, idx_valid, yy[idx_valid])

    # Apply SG filter
    ys = savgol_filter(
        yy,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )

    # Restore original NaNs
    ys[~mask] = np.nan
    return ys


def savgol_nan_numpy(x, axis=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along a given axis for a NumPy array.
    """
    return np.apply_along_axis(
        _savgol_1d_nan,
        axis=axis,
        arr=x,
        window_length=window_length,
        polyorder=polyorder,
    )


def savgol_nan_torch(x, dim=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along dim for a torch.Tensor.
    - x: (..., T, ...) tensor
    - dim: time dimension (default 1)
    """
    # Move target dim to last for easier NumPy apply
    x_np = x.detach().cpu().numpy()
    x_np = np.moveaxis(x_np, dim, -1)

    y_np = savgol_nan_numpy(
        x_np,
        axis=-1,
        window_length=window_length,
        polyorder=polyorder,
    )

    # Move axis back and convert to torch
    y_np = np.moveaxis(y_np, -1, dim)
    y = torch.from_numpy(y_np).to(x.device).type_as(x)
    return y

#%% Law of total covariance decomposition
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
from tqdm import tqdm
import time


def _fmt_ms(ms, decimals=2):
    """Consistent window label for plots."""
    try:
        return f"{float(ms):.{int(decimals)}f}ms"
    except Exception:
        return f"{ms}ms"


def _fmt_ms_tag(ms, decimals=2):
    """Consistent window tag for filenames."""
    # Keep '.' in filenames; it's fine on Linux.
    return _fmt_ms(ms, decimals=decimals)

class DualWindowAnalysis:
    def __init__(self, robs, eyepos, valid_mask, dt=1/240, device='cuda'):
        """
        Turbo-Charged Covariance Decomposition.
        Uses GPU for stats + Vectorized Linear Regression for fitting.
        """
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Cache per-window summaries for inspection/plotting (cribbed from scripts/mcfarland_sim.py)
        self.window_summaries = {}
        
        print(f"Initializing on {self.device}...")
        t0 = time.time()
        
        # 1. Load & Sanitize
        if np.isnan(robs).any():
            robs = np.nan_to_num(robs, nan=0.0)
        eyepos = np.nan_to_num(eyepos, nan=0.0)

        self.robs = torch.tensor(robs, dtype=torch.float32, device=self.device)
        self.eyepos = torch.tensor(eyepos, dtype=torch.float32, device=self.device)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        
        self.n_trials, self.n_time, self.n_cells = robs.shape
        
        # 2. Pre-compute PSTH
        valid_float = self.valid_mask.float().unsqueeze(-1)
        sum_spikes = torch.sum(self.robs * valid_float, dim=0)
        count_trials = torch.sum(valid_float, dim=0)
        count_trials[count_trials == 0] = 1.0 
        self.psth = sum_spikes / count_trials
        
        # 3. Valid Segments
        self.segments = self._get_valid_segments(min_len_bins=36)
        print(f"Loaded {len(self.segments)} valid segments. Init took {time.time()-t0:.2f}s")

    def _get_valid_segments(self, min_len_bins):
        segments = []
        mask_cpu = self.valid_mask.cpu().numpy()
        for tr in range(self.n_trials):
            padded = np.concatenate(([False], mask_cpu[tr], [False]))
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            stops = np.where(diffs == -1)[0]
            for start, stop in zip(starts, stops):
                if (stop - start) >= min_len_bins:
                    segments.append((tr, start, stop))
        return segments

    def _extract_windows_gpu(self, t_count, t_hist, max_samples=10000):
        total_len = t_count + t_hist
        trial_indices, time_indices = [], []
        
        for (tr, start, stop) in self.segments:
            if (stop - start) < total_len: continue
            t_starts = np.arange(start, stop - total_len + 1, t_count)
            trial_indices.extend([tr] * len(t_starts))
            time_indices.extend(t_starts)
            
        if not trial_indices:
            return None, None, None, None
            
        # Subsample to cap VRAM/Compute usage
        n_total = len(trial_indices)
        if n_total > max_samples:
            np.random.seed(42) 
            keep_idx = np.random.choice(n_total, max_samples, replace=False)
            trial_indices = np.array(trial_indices)[keep_idx]
            time_indices = np.array(time_indices)[keep_idx]
            
        idx_tr = torch.tensor(trial_indices, device=self.device, dtype=torch.long)
        idx_t0 = torch.tensor(time_indices, device=self.device, dtype=torch.long)
        
        # GPU Gather
        offsets = torch.arange(total_len, device=self.device).unsqueeze(0)
        gather_t = idx_t0.unsqueeze(1) + offsets
        gather_tr = idx_tr.unsqueeze(1).expand(-1, total_len)
        E = self.eyepos[gather_tr, gather_t, :]
        
        spike_offsets = torch.arange(t_hist, total_len, device=self.device).unsqueeze(0)
        gather_t_spike = idx_t0.unsqueeze(1) + spike_offsets
        gather_tr_spike = idx_tr.unsqueeze(1).expand(-1, t_count)
        
        S_raw = self.robs[gather_tr_spike, gather_t_spike, :]
        S = torch.sum(S_raw, dim=1)
        T_idx = idx_t0 + t_hist

        return S, E, T_idx, idx_tr

    def _choose_eye_bins(self, E, n_bins=15, q=0.95, sample_n=2000, mode='quantile'):
        """Choose global distance bins from a subsample of eye trajectories.

        Uses RMS distance (cdist / sqrt(T)) similar to scripts/mcfarland_sim.py.
        """
        mode = str(mode).lower()
        N = E.shape[0]
        if N < 2:
            bins = torch.linspace(0, 1.0, n_bins + 1, device=self.device)
            return bins

        sample_n = int(min(sample_n, N))
        perm = torch.randperm(N, device=self.device)[:sample_n]
        X = E[perm]
        T = X.shape[1]
        Xflat = X.reshape(sample_n, -1)
        D = torch.cdist(Xflat, Xflat) / torch.sqrt(torch.tensor(float(T), device=self.device))
        ii, jj = torch.triu_indices(sample_n, sample_n, offset=1, device=self.device)
        d = D[ii, jj]
        # Match scripts/mcfarland_sim.py binning exactly: percentile edges with
        # step=100/(n_bins+1), intentionally excluding 100%.
        if mode == 'mcfarland':
            if d.numel() == 0:
                return torch.linspace(0.0, 1.0, n_bins + 1, device=self.device)
            dist_np = d.detach().cpu().numpy()
            pct = np.arange(0, 100, 100 / float(n_bins + 1))
            edges = np.percentile(dist_np, pct)
            edges = np.asarray(edges, dtype=np.float32)
            # Ensure non-decreasing and include 0 explicitly
            edges[0] = 0.0
            edges = np.unique(edges)
            if edges.size < 2:
                edges = np.linspace(0.0, float(np.max(dist_np) if dist_np.size else 1.0), n_bins + 1, dtype=np.float32)
            return torch.as_tensor(edges, device=self.device, dtype=torch.float32)

        # For other modes we cap the tail using q-quantile.
        if d.numel() == 0:
            max_dist = torch.tensor(1.0, device=self.device)
        else:
            max_dist = torch.quantile(d, q)
            if not torch.isfinite(max_dist) or max_dist <= 0:
                max_dist = torch.tensor(1.0, device=self.device)

        # Always include 0.0 as the first edge
        if mode == 'uniform':
            bins = torch.linspace(0.0, max_dist, n_bins + 1, device=self.device)
            return bins

        # Quantile bins: equal-count bins for robust fitting (more pairs per bin)
        d_clip = d[d <= max_dist]
        if d_clip.numel() < 10:
            bins = torch.linspace(0.0, max_dist, n_bins + 1, device=self.device)
            return bins

        qs = torch.linspace(0.0, 1.0, n_bins + 1, device=self.device)
        bins = torch.quantile(d_clip, qs)
        bins[0] = 0.0
        bins[-1] = max_dist

        # Guard against repeated edges (can happen if many distances are identical)
        bins_u = torch.unique_consecutive(bins)
        if bins_u.numel() < (n_bins + 1):
            bins = torch.linspace(0.0, max_dist, n_bins + 1, device=self.device)
        return bins

    def _fit_intercepts_linear(self, Ceye, bin_centers, count_e, d_max=0.4, min_bins=3, eps=1e-8, eval_at_first_bin=True):
        """Match scripts/mcfarland_sim.py::_fit_intercepts_linear (local weighted linear fit)."""
        n_bins, n_cells, _ = Ceye.shape
        C_intercept = np.full((n_cells, n_cells), np.nan, dtype=Ceye.dtype)

        x = np.asarray(bin_centers, dtype=np.float64)
        w_all = np.asarray(count_e, dtype=np.float64)

        use_mask = np.isfinite(x) & (x > 0) & (x <= float(d_max)) & np.isfinite(w_all) & (w_all > 0)
        idx = np.where(use_mask)[0]
        if idx.size < int(min_bins):
            k0 = np.where(np.isfinite(x) & np.isfinite(w_all) & (w_all > 0))[0]
            if k0.size > 0:
                return Ceye[k0[0]].copy()
            return C_intercept

        x_loc = x[idx]
        w_loc = w_all[idx]
        x_eval = x_loc[0] if bool(eval_at_first_bin) else 0.0

        S0 = np.sum(w_loc)
        Sx = np.sum(w_loc * x_loc)
        Sxx = np.sum(w_loc * x_loc**2)
        det = S0 * Sxx - Sx**2

        if S0 == 0 or (det / (S0 * S0)) < float(eps):
            return Ceye[idx[0]].copy()

        for i in range(n_cells):
            self._fit_single_pair_linear(Ceye, C_intercept, idx, w_loc, x_loc, x_eval, S0, Sx, Sxx, det, i, i)
            for j in range(i + 1, n_cells):
                self._fit_single_pair_linear(Ceye, C_intercept, idx, w_loc, x_loc, x_eval, S0, Sx, Sxx, det, i, j)
                C_intercept[j, i] = C_intercept[i, j]

        return C_intercept

    def _fit_single_pair_linear(self, Ceye, C_intercept, idx, w_loc, x_loc, x_eval, S0, Sx, Sxx, det, i, j):
        """Helper for the weighted linear fit."""
        y = Ceye[idx, i, j]

        if not np.isfinite(y).all():
            v = np.isfinite(y)
            if np.sum(v) < 3:
                return
            wv, xv, yv = w_loc[v], x_loc[v], y[v]
            s0 = np.sum(wv); sx = np.sum(wv * xv); sxx = np.sum(wv * xv**2)
            d = s0 * sxx - sx**2
            if d <= 0:
                return
            sy = np.sum(wv * yv)
            sxy = np.sum(wv * xv * yv)
            beta1 = (s0 * sxy - sx * sy) / d
            beta0 = (sxx * sy - sx * sxy) / d
            Sy = sy
        else:
            Sy = np.sum(w_loc * y)
            Sxy = np.sum(w_loc * x_loc * y)
            beta1 = (S0 * Sxy - Sx * Sy) / det
            beta0 = (Sxx * Sy - Sx * Sxy) / det

        if INTERCEPT_FORCE_NONPOS_SLOPE and (beta1 > 0):
            beta1 = 0.0
            # Recompute beta0 as weighted mean
            if not np.isfinite(y).all():
                v = np.isfinite(y)
                beta0 = np.average(y[v], weights=w_loc[v])
            else:
                beta0 = Sy / S0

        C_intercept[i, j] = beta0 + beta1 * x_eval

    def _fit_diag_intercepts_linear_with_quality(self, Ceye, bin_centers, count_e, d_max=0.4, min_bins=3, eps=1e-8, eval_at_first_bin=True):
        """Diagonal-only version of the McFarland linear fit, with lightweight diagnostics."""
        n_bins, n_cells, _ = Ceye.shape
        diag = np.full((n_cells,), np.nan, dtype=float)
        slope = np.full((n_cells,), np.nan, dtype=float)
        r2 = np.full((n_cells,), np.nan, dtype=float)

        x = np.asarray(bin_centers, dtype=np.float64)
        w_all = np.asarray(count_e, dtype=np.float64)
        use_mask = np.isfinite(x) & (x > 0) & (x <= float(d_max)) & np.isfinite(w_all) & (w_all > 0)
        idx = np.where(use_mask)[0]
        if idx.size < int(min_bins):
            return diag, {'slope': slope, 'r2': r2, 'n_valid_bins': np.zeros((n_cells,), dtype=int)}

        x_loc = x[idx]
        w_loc = w_all[idx]
        x_eval = x_loc[0] if bool(eval_at_first_bin) else 0.0

        S0 = np.sum(w_loc)
        Sx = np.sum(w_loc * x_loc)
        Sxx = np.sum(w_loc * x_loc**2)
        det = S0 * Sxx - Sx**2
        if S0 == 0 or (det / (S0 * S0)) < float(eps):
            return diag, {'slope': slope, 'r2': r2, 'n_valid_bins': np.zeros((n_cells,), dtype=int)}

        n_valid_bins = np.full((n_cells,), int(idx.size), dtype=int)

        for i in range(n_cells):
            y = np.asarray(Ceye)[idx, i, i]
            v = np.isfinite(y)
            if np.sum(v) < int(min_bins):
                continue
            ww = w_loc[v]; xx = x_loc[v]; yy = y[v]
            s0 = np.sum(ww); sx = np.sum(ww * xx); sxx = np.sum(ww * xx * xx)
            d = s0 * sxx - sx * sx
            if d <= 0:
                continue
            sy = np.sum(ww * yy)
            sxy = np.sum(ww * xx * yy)
            b1 = (s0 * sxy - sx * sy) / d
            b0 = (sxx * sy - sx * sxy) / d
            if INTERCEPT_FORCE_NONPOS_SLOPE and (b1 > 0):
                b1 = 0.0
                b0 = sy / s0
            slope[i] = float(b1)
            diag[i] = float(b0 + b1 * x_eval)

            # R^2 in y-space (weighted)
            yhat = b0 + b1 * xx
            ybar = np.average(yy, weights=ww)
            ss_res = float(np.sum(ww * (yy - yhat) ** 2))
            ss_tot = float(np.sum(ww * (yy - ybar) ** 2))
            if ss_tot > 0:
                r2[i] = 1.0 - ss_res / ss_tot

        return diag, {'slope': slope, 'r2': r2, 'n_valid_bins': n_valid_bins}

    def _compute_binned_stats_gpu(self, S, E, n_bins=15, bins=None):
        N = S.shape[0]
        dists = torch.zeros((N, N), device=self.device, dtype=torch.float32)
        block_size = 2000 
        
        # Blocked Distance Calc
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            E_i = E[i:i_end]
            for j in range(0, N, block_size):
                j_end = min(j + block_size, N)
                E_j = E[j:j_end]
                diff = E_i.unsqueeze(1) - E_j.unsqueeze(0)
                dists[i:i_end, j:j_end] = torch.mean(torch.norm(diff, dim=-1), dim=-1)
        
        mask_triu = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        valid_dists = dists[mask_triu]
        
        if bins is None:
            # Estimate bins robustly from a subsample (avoids building global percentiles on huge arrays)
            bins = self._choose_eye_bins(E, n_bins=n_bins)
        else:
            bins = torch.as_tensor(bins, device=self.device, dtype=torch.float32)
            n_bins = int(bins.numel() - 1)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        n_cells = S.shape[1]
        binned_covs = torch.zeros((n_bins, n_cells, n_cells), device=self.device)
        bin_counts  = torch.zeros(n_bins, device=self.device)
        ST = S.T
        
        # Binned Covariance
        for k in range(n_bins):
            mask_bin = (dists >= bins[k]) & (dists < bins[k+1]) & mask_triu
            count = mask_bin.sum()
            if count < 5: continue
            cov_sum = torch.linalg.multi_dot([ST, mask_bin.float(), S])
            binned_covs[k] = cov_sum / count
            bin_counts[k] = count
            
        # Return extra info so we can plot eye-distance histograms.
        return (
            binned_covs.cpu().numpy(),
            bin_centers.cpu().numpy(),
            bin_counts.cpu().numpy(),
            bins.cpu().numpy(),
        )

    def _compute_binned_stats_time_matched_gpu(self, S, E, T_idx, idx_tr, n_bins=15, bins=None, max_per_time=96, min_trials_per_time=10):
        """Time-matched second moment estimation.

        Only forms pairs of windows that share the same aligned time index (T_idx).
        This matches the logic in scripts/mcfarland_sim.py::_calculate_second_moment.
        """
        if bins is None:
            bins_t = self._choose_eye_bins(E, n_bins=n_bins)
        else:
            bins_t = torch.as_tensor(bins, device=self.device, dtype=torch.float32)
            n_bins = int(bins_t.numel() - 1)

        bin_centers_t = 0.5 * (bins_t[:-1] + bins_t[1:])
        n_cells = S.shape[1]

        # Accumulate on CPU to avoid GPU memory growth
        SS_sum = torch.zeros((n_bins, n_cells, n_cells), device='cpu', dtype=torch.float64)
        count_e = torch.zeros((n_bins,), device='cpu', dtype=torch.long)

        unique_times = torch.unique(T_idx).detach().cpu().numpy()
        inv_sqrt_T = 1.0 / torch.sqrt(torch.tensor(float(E.shape[1]), device=self.device, dtype=torch.float32))

        for t in unique_times:
            ix = torch.where(T_idx == int(t))[0]
            n_t = int(ix.numel())
            if n_t < min_trials_per_time:
                continue

            # Subsample per time to keep O(M^2) reasonable
            if n_t > max_per_time:
                perm = torch.randperm(n_t, device=self.device)[:max_per_time]
                ix = ix[perm]
                n_t = int(ix.numel())
                if n_t < 2:
                    continue

            X = E[ix]         # (M, T, 2)
            Sc = S[ix]        # (M, C)

            # cross-trial pairs (upper triangle)
            ii, jj = torch.triu_indices(n_t, n_t, offset=1, device=self.device)
            if ii.numel() == 0:
                continue

            Xflat = X.reshape(n_t, -1)
            D = torch.cdist(Xflat, Xflat) * inv_sqrt_T
            d = D[ii, jj]

            bid = torch.bucketize(d, bins_t, right=False)
            ok = (bid >= 1) & (bid <= n_bins)
            if not ok.any():
                continue

            ii = ii[ok]; jj = jj[ok]; bid = bid[ok]

            # accumulate sum of outer products per bin
            ST = Sc.T
            for k in range(1, n_bins + 1):
                mk = (bid == k)
                if not mk.any():
                    continue
                Si = Sc[ii[mk]]
                Sj = Sc[jj[mk]]
                M = Si.transpose(0, 1).matmul(Sj)  # (C, C)
                SS_sum[k-1] += M.detach().cpu().to(torch.float64)
                count_e[k-1] += mk.sum().detach().cpu()

        # Convert sums to second moments
        SS = SS_sum.numpy()
        cnt = count_e.numpy().astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            MM = SS / cnt[:, None, None]
        MM = 0.5 * (MM + np.swapaxes(MM, -1, -2))
        return MM, bin_centers_t.detach().cpu().numpy(), count_e.numpy(), bins_t.detach().cpu().numpy()
    
    def _compute_cv_psth_sigma(self, t_count, T_idx, valid_mask_float):
        """
        Computes UNBIASED PSTH Covariance using Split-Half Cross-Validation.
        Removes the '1/N' noise floor bias that causes negative Fano Factors.
        """
        # 1. Generate Split Masks (A/B)
        # We want to split the VALID TRIALS for each timepoint.
        # This is tricky because validity varies by time.
        # Heuristic: Randomly assign every trial ID to group A or B.
        
        n_trials = self.n_trials
        perm = torch.randperm(n_trials, device=self.device)
        idx_A = perm[:n_trials//2]
        idx_B = perm[n_trials//2:]
        
        # Create masks
        mask_A = torch.zeros((n_trials, 1, 1), device=self.device)
        mask_B = torch.zeros((n_trials, 1, 1), device=self.device)
        mask_A[idx_A] = 1.0
        mask_B[idx_B] = 1.0
        
        # 2. Compute PSTH_A and PSTH_B
        # valid_mask_float is [Trials, Time, 1]
        
        # Weighted Sums
        sum_A = torch.sum(self.robs * valid_mask_float * mask_A, dim=0)
        cnt_A = torch.sum(valid_mask_float * mask_A, dim=0)
        cnt_A[cnt_A==0] = 1.0
        psth_A = sum_A / cnt_A # [Time, Cells]
        
        sum_B = torch.sum(self.robs * valid_mask_float * mask_B, dim=0)
        cnt_B = torch.sum(valid_mask_float * mask_B, dim=0)
        cnt_B[cnt_B==0] = 1.0
        psth_B = sum_B / cnt_B # [Time, Cells]
        
        # 3. Gather Windows for Covariance
        # We need the sums of PSTH over the specific windows used in this sweep
        offsets = torch.arange(t_count, device=self.device).unsqueeze(0)
        gather_t = T_idx.unsqueeze(1) + offsets
        
        # Extract windows from both independent PSTHs
        # [N_samples, t_count, C] -> Sum -> [N_samples, C]
        win_A = torch.sum(psth_A[gather_t, :], dim=1)
        win_B = torch.sum(psth_B[gather_t, :], dim=1)
        
        # 4. Compute Cross-Covariance Matrix
        # Cov(A, B) = E[(A - muA)(B - muB).T]
        mu_A = torch.mean(win_A, dim=0, keepdim=True)
        mu_B = torch.mean(win_B, dim=0, keepdim=True)
        
        centered_A = win_A - mu_A
        centered_B = win_B - mu_B
        
        n_samples = win_A.shape[0]
        # Cross-Covariance
        Sigma_CV = (centered_A.T @ centered_B) / (n_samples - 1)
        
        # Symmetrize (Optional but good for numerical stability)
        Sigma_CV = 0.5 * (Sigma_CV + Sigma_CV.T)
        
        return Sigma_CV
    
    def _fit_intercepts_vectorized(self, binned_covs, bin_centers, bin_counts, diag_limit=None):
        """
        Vectorized Linear Fitting.
        Replaces slow iterative curve_fit with instant O(1) algebra.
        Model: y = A * exp(-x/tau) + Plateau
        Linearized: log(y - Plateau) = log(A) - (1/tau)*x
        """
        n_bins, n_cells, _ = binned_covs.shape
        Sigma_intercept = np.full((n_cells, n_cells), np.nan, dtype=float)
        
        # Identify valid bins (Global, same for all pairs)
        valid_bins = bin_counts > MIN_BIN_COUNT_FOR_FIT
        if np.sum(valid_bins) < 4:
            return Sigma_intercept
            
        x = np.asarray(bin_centers)[valid_bins]
        c = np.asarray(bin_counts)[valid_bins]
        
        # Loop over pairs (Python loop over simple numpy ops is fast enough for 7000 pairs)
        # ~0.1 seconds total
        for i in range(n_cells):
            for j in range(i, n_cells):
                y = binned_covs[valid_bins, i, j]
                
                # 1. Estimate Plateau (Noise Floor) from tail
                plateau = np.mean(y[-3:]) 
                
                # 2. Linearize
                y_sub = y - plateau
                
                # Filter positive values only for Log
                mask_log = y_sub > (1e-6 * np.max(np.abs(y)))
                
                if np.sum(mask_log) < 3:
                    # Fallback if no decay visible
                    k = int(min(FIT_FALLBACK_EARLY_K, y.size))
                    intercept = float(np.nanmean(y[:k])) if k > 0 else float(np.nanmean(y))
                else:
                    # Weighted log-linear fit (emphasize small distances)
                    try:
                        x_fit = x[mask_log]
                        y_fit = np.log(y_sub[mask_log])
                        c_fit = c[mask_log]

                        w = 1.0 / np.power(x_fit + float(FIT_WEIGHT_EPS), float(FIT_WEIGHT_POWER))
                        if FIT_WEIGHT_USE_COUNTS:
                            w = w * np.sqrt(np.maximum(c_fit, 1.0))
                        w = w / np.nanmax(w)

                        m, b = np.polyfit(x_fit, y_fit, 1, w=w)

                        # Enforce expected negative slope; otherwise fallback to early bins.
                        if FIT_EXPECT_NEG_SLOPE and (m >= 0):
                            k = int(min(FIT_FALLBACK_EARLY_K, y.size))
                            intercept = float(np.nanmean(y[:k])) if k > 0 else float(np.nanmean(y))
                        else:
                            A = np.exp(b)
                            intercept = float(A + plateau)
                    except:
                        intercept = np.nan
                
                # 3. Clamp (signal covariance should not exceed total covariance)
                if (diag_limit is not None) and (i == j):
                    limit = float(diag_limit[i])
                    if np.isfinite(limit) and np.isfinite(intercept) and intercept > 0.99 * limit:
                        intercept = 0.99 * limit
                        
                Sigma_intercept[i, j] = intercept
                Sigma_intercept[j, i] = intercept
                
        return Sigma_intercept

    def _fit_diag_intercepts_with_quality(self, binned_covs, bin_centers, bin_counts, diag_limit=None):
        """Fit ONLY diagonal intercepts with basic reliability diagnostics.

        Diagnostics are computed in the same spirit as `_fit_intercepts_vectorized`:
        - plateau from the tail of the valid-bin curve
        - log-linear fit on (y - plateau) where positive
        R² is computed in log-space over the points used for the fit.

        Returns
        -------
        diag_intercepts : (C,) float array
        diag_fit : dict of arrays
        """
        n_bins, n_cells, _ = binned_covs.shape

        diag_intercepts = np.full((n_cells,), np.nan, dtype=float)
        fit_r2 = np.full((n_cells,), np.nan, dtype=float)
        fit_nbins = np.zeros((n_cells,), dtype=int)
        fit_nlog = np.zeros((n_cells,), dtype=int)
        fit_fallback = np.zeros((n_cells,), dtype=bool)
        fit_clamped = np.zeros((n_cells,), dtype=bool)
        fit_slope = np.full((n_cells,), np.nan, dtype=float)

        valid_bins = np.asarray(bin_counts) > MIN_BIN_COUNT_FOR_FIT
        if np.sum(valid_bins) < 4:
            return diag_intercepts, {
                'r2': fit_r2,
                'n_valid_bins': fit_nbins,
                'n_log_points': fit_nlog,
                'fallback': fit_fallback,
                'clamped': fit_clamped,
                'slope': fit_slope,
            }

        x = np.asarray(bin_centers)[valid_bins]
        c = np.asarray(bin_counts)[valid_bins]

        for i in range(n_cells):
            y = np.asarray(binned_covs)[valid_bins, i, i]
            fit_nbins[i] = int(y.size)

            # plateau from tail
            plateau = float(np.nanmean(y[-3:])) if y.size >= 3 else float(np.nanmean(y))
            y_sub = y - plateau

            # keep positive values for log-linear fit
            scale = np.nanmax(np.abs(y))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            mask_log = y_sub > (1e-6 * scale)
            fit_nlog[i] = int(np.sum(mask_log))

            if np.sum(mask_log) < 3:
                k = int(min(FIT_FALLBACK_EARLY_K, y.size))
                intercept = float(np.nanmean(y[:k])) if k > 0 else float(np.nanmean(y))
                fit_fallback[i] = True
            else:
                x_fit = x[mask_log]
                y_fit = np.log(y_sub[mask_log])
                try:
                    c_fit = c[mask_log]
                    w = 1.0 / np.power(x_fit + float(FIT_WEIGHT_EPS), float(FIT_WEIGHT_POWER))
                    if FIT_WEIGHT_USE_COUNTS:
                        w = w * np.sqrt(np.maximum(c_fit, 1.0))
                    w = w / np.nanmax(w)

                    m, b = np.polyfit(x_fit, y_fit, 1, w=w)
                    fit_slope[i] = float(m)

                    y_hat = m * x_fit + b
                    ss_res = float(np.sum((y_fit - y_hat) ** 2))
                    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                    if ss_tot > 0:
                        fit_r2[i] = 1.0 - (ss_res / ss_tot)

                    if FIT_EXPECT_NEG_SLOPE and (m >= 0):
                        k = int(min(FIT_FALLBACK_EARLY_K, y.size))
                        intercept = float(np.nanmean(y[:k])) if k > 0 else float(np.nanmean(y))
                        fit_fallback[i] = True
                    else:
                        intercept = float(np.exp(b) + plateau)
                except Exception:
                    intercept = np.nan
                    fit_fallback[i] = True

            # clamp diag intercept to below total covariance diag
            if diag_limit is not None:
                limit = float(diag_limit[i])
                if np.isfinite(limit) and np.isfinite(intercept) and intercept > 0.99 * limit:
                    intercept = 0.99 * limit
                    fit_clamped[i] = True

            diag_intercepts[i] = intercept

        return diag_intercepts, {
            'r2': fit_r2,
            'n_valid_bins': fit_nbins,
            'n_log_points': fit_nlog,
            'fallback': fit_fallback,
            'clamped': fit_clamped,
            'slope': fit_slope,
        }

    def run_sweep(self, window_sizes_ms, t_hist_ms=100):
        t_hist_bins = int(t_hist_ms / (self.dt * 1000))
        results = []
        mats_save = []
        
        print(f"Starting Sweep (Hist={t_hist_ms}ms)...")
        
        for win_ms in tqdm(window_sizes_ms):
            t0 = time.time()
            t_count_bins = int(win_ms / (self.dt * 1000))
            if t_count_bins < 1: t_count_bins = 1
            
            # 1. GPU Extract
            S, E, T_idx, idx_tr = self._extract_windows_gpu(t_count_bins, t_hist_bins, max_samples=MAX_SAMPLES_PER_WINDOW)
            if (S is None) or (E is None) or (T_idx is None) or (idx_tr is None):
                continue
            if S.shape[0] < 100:
                continue
            t1 = time.time()
            
            # 2. GPU Stats
            n_samples = S.shape[0]
            Sigma_Total_Raw = (S.T @ S) / n_samples
            mean_N = torch.mean(S, dim=0)
            Sigma_Total_Cov = Sigma_Total_Raw - torch.outer(mean_N, mean_N)
            
            offsets = torch.arange(t_count_bins, device=self.device).unsqueeze(0)
            gather_t = T_idx.unsqueeze(1) + offsets
            psth_vals = self.psth[gather_t, :]
            psth_sums = torch.sum(psth_vals, dim=1)
            # Sigma_PSTH = torch.cov(psth_sums.T)
            valid_float = self.valid_mask.float().unsqueeze(-1)
            Sigma_PSTH = self._compute_cv_psth_sigma(t_count_bins, T_idx, valid_float)
            
            # 3. McFarland Stats: binned SECOND MOMENTS E[S_i S_j | d]
            bins = self._choose_eye_bins(E, n_bins=EYE_DIST_N_BINS, q=EYE_BIN_Q, mode=EYE_BINNING_MODE)
            if TIME_MATCHED_CONDITIONING:
                binned_2nd, bin_centers, bin_counts, bin_edges = self._compute_binned_stats_time_matched_gpu(
                    S,
                    E,
                    T_idx,
                    idx_tr,
                    n_bins=EYE_DIST_N_BINS,
                    bins=bins,
                    max_per_time=MAX_SAMPLES_PER_TIMEBIN,
                    min_trials_per_time=10,
                )
            else:
                binned_2nd, bin_centers, bin_counts, bin_edges = self._compute_binned_stats_gpu(
                    S, E, n_bins=EYE_DIST_N_BINS, bins=bins
                )
            t2 = time.time()
            
            # 4. McFarland Crate estimation:
            # Ceye(d) = E[S_i S_j | d] - mu_i mu_j, with mu_i = global mean SpikeCounts.
            # Then fit intercept using the McFarland linear method (default).
            mean_N_np = mean_N.detach().cpu().numpy()
            Ceye = binned_2nd - np.outer(mean_N_np, mean_N_np)

            if str(INTERCEPT_MODE).lower() == 'raw_first_bin':
                # First valid bin only (no fit)
                k0 = np.where(np.asarray(bin_counts) > 0)[0]
                Crate = Ceye[int(k0[0])].copy() if k0.size else np.full_like(Ceye[0], np.nan)
            else:
                # Match scripts/mcfarland_sim.py (evaluate at first bin center for conservative estimate)
                Crate = self._fit_intercepts_linear(
                    Ceye,
                    bin_centers,
                    bin_counts,
                    d_max=INTERCEPT_D_MAX,
                    eval_at_first_bin=INTERCEPT_EVAL_AT_FIRST_BIN,
                )

            # Optional physical-limit masking (as in mcfarland_sim.py::_calculate_Crate)
            Sigma_Total_Cov_np = Sigma_Total_Cov.detach().cpu().numpy()
            bad_mask = np.isfinite(np.diag(Crate)) & np.isfinite(np.diag(Sigma_Total_Cov_np)) & (np.diag(Crate) > 0.99 * np.diag(Sigma_Total_Cov_np))
            if np.any(bad_mask):
                Crate[bad_mask, :] = np.nan
                Crate[:, bad_mask] = np.nan
                Ceye[:, bad_mask, :] = np.nan
                Ceye[:, :, bad_mask] = np.nan

            # Diagonal diagnostics (linear-fit space)
            diag_intercepts, diag_fit = self._fit_diag_intercepts_linear_with_quality(
                Ceye,
                bin_centers,
                bin_counts,
                d_max=INTERCEPT_D_MAX,
                eval_at_first_bin=INTERCEPT_EVAL_AT_FIRST_BIN,
            )

            # For convenience/debug: implied second-moment intercept
            Intercept_2nd = Crate + np.outer(mean_N_np, mean_N_np)
            t3 = time.time()
            
            # 5. Algebra
            Sigma_Total_Cov = Sigma_Total_Cov_np
            Sigma_PSTH = Sigma_PSTH.cpu().numpy()

            Sigma_Rate = Crate
            Sigma_FEM = Sigma_Rate - Sigma_PSTH
            Sigma_Noise_Uncorr = Sigma_Total_Cov - Sigma_PSTH
            Sigma_Noise_Corr = Sigma_Total_Cov - Sigma_Rate
            
            # Metrics
            mu = mean_N_np.copy()
            mu[mu==0] = 1e-9
            ff_uncorr = np.diag(Sigma_Noise_Uncorr) / mu
            ff_corr = np.diag(Sigma_Noise_Corr) / mu

            # alpha and (1-alpha) like scripts/mcfarland_sim.py
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha = np.diag(Sigma_PSTH) / np.diag(Sigma_Rate)
            one_minus_alpha = 1.0 - alpha
            bad = ~np.isfinite(alpha)
            alpha[bad] = np.nan
            one_minus_alpha[bad] = np.nan
            
            if np.isnan(Sigma_FEM).any():
                rank = np.nan
            else:
                evals = np.linalg.eigvalsh(Sigma_FEM)[::-1]
                pos = evals[evals > 0]
                rank = (np.sum(pos[:2])/np.sum(pos)) if len(pos)>2 else 1.0
            
            results.append({
                'window_ms': win_ms,
                'ff_uncorr': ff_uncorr,
                'ff_corr': ff_corr,
                'ff_uncorr_mean': np.nanmean(ff_uncorr),
                'ff_corr_mean': np.nanmean(ff_corr),
                'alpha': alpha,
                'one_minus_alpha': one_minus_alpha,
                'fem_rank_ratio': rank,
                'n_samples': n_samples,
                'mean_counts': mean_N_np,

                # Intercept-fit diagonal diagnostics (linear fit on Ceye like mcfarland_sim.py)
                'diag_intercept': diag_intercepts,
                'diag_fit_r2': diag_fit.get('r2', np.full_like(diag_intercepts, np.nan, dtype=float)),
                'diag_fit_n_valid_bins': diag_fit.get('n_valid_bins', np.zeros_like(diag_intercepts, dtype=int)),
                'diag_fit_slope': diag_fit.get('slope', np.full_like(diag_intercepts, np.nan, dtype=float)),
                'diag_fit_mode': str(INTERCEPT_MODE),

                # Eye-distance histogram (upper-triangular pairs) used by the McFarland binning.
                # Counts are the number of (i,j) pairs of extracted windows whose eye-trajectory
                # distance falls in each bin.
                'eye_dist_bin_centers': bin_centers,
                'eye_dist_bin_counts': bin_counts,
                'eye_dist_bin_edges': bin_edges,
            })

            mats_save.append({
                'Total': Sigma_Total_Cov,
                'PSTH': Sigma_PSTH,
                'FEM': Sigma_FEM,
                'Noise_Corr': Sigma_Noise_Corr,
                # Store covariance intercept like mcfarland_sim.py (Crate)
                'Intercept': Sigma_Rate,
                # Also store implied second-moment intercept and binned second moments (for inspection)
                'Intercept_2nd': Intercept_2nd,
                # Also store binned curves for inspection/debug
                'Ceye_binned': Ceye,
                'SecondMoment_binned': binned_2nd,
                'bin_centers': bin_centers,
                'bin_counts': bin_counts,
            })

            # Cache for inspect_neuron_pair
            self.window_summaries[float(win_ms)] = {
                'bin_centers': np.asarray(bin_centers),
                'bin_counts': np.asarray(bin_counts),
                'Sigma_Intercept': np.asarray(Sigma_Rate),
                'Sigma_Intercept_2nd': np.asarray(Intercept_2nd),
                'Sigma_PSTH': np.asarray(Sigma_PSTH),
                'Sigma_Total': np.asarray(Sigma_Total_Cov),
                'Sigma_FEM': np.asarray(Sigma_FEM),
                'mean_counts': np.asarray(mean_N_np),
                'SecondMoment_binned': np.asarray(binned_2nd),
            }

            # Debug Timing
            # tqdm.write(f"  {win_ms}ms: Extract={t1-t0:.2f}s, Stats={t2-t1:.2f}s, Fit={t3-t2:.2f}s")
            
        return results, mats_save

    def inspect_neuron_pair(self, i, j, win_ms, ax=None, show=True):
        """Plot covariance vs eye-distance bin for a neuron pair.

        Cribbed from scripts/mcfarland_sim.py. Uses cached per-window summaries.
        """
        import matplotlib.pyplot as plt

        if not self.window_summaries:
            raise RuntimeError("run_sweep must be called before inspecting neuron pairs.")

        win_key = float(win_ms)
        if win_key not in self.window_summaries:
            avail = ", ".join(str(k) for k in sorted(self.window_summaries.keys()))
            raise KeyError(f"Window {win_ms}ms not cached. Available: {avail}")

        summary = self.window_summaries[win_key]
        bin_centers = summary['bin_centers']
        counts = summary['bin_counts']
        valid = counts > 0
        if not np.any(valid):
            raise RuntimeError("No histogram bins with data for this neuron pair.")

        mu = np.asarray(summary.get('mean_counts', None), dtype=float)
        if mu.ndim != 1:
            raise RuntimeError("Cached mean_counts missing or invalid; rerun run_sweep.")
        muij = float(mu[i] * mu[j])

        m2 = np.asarray(summary['SecondMoment_binned'][:, i, j], dtype=float)
        covs = m2 - muij

        intercept_cov = summary['Sigma_Intercept'][i, j]
        psth_cov = summary['Sigma_PSTH'][i, j]

        created = False
        if ax is None:
            created = True
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
        else:
            fig = ax.figure

        x_all = np.asarray(bin_centers[valid], dtype=float)
        y_cov = np.asarray(covs[valid], dtype=float)
        y_m2 = np.asarray(m2[valid], dtype=float)
        c_all = np.asarray(counts[valid], dtype=float)

        ax.plot(x_all, y_cov, 'o', alpha=0.6, label='Measured Covariance')
        ax.axhline(psth_cov, linestyle='--', linewidth=2, label='PSTH Covariance')
        ax.axhline(intercept_cov, linestyle=':', linewidth=2, label=f"Intercept (Crate)={intercept_cov:.3g}")

        # Overlay McFarland-style weighted local linear fit on Ceye(d)
        fit_r2 = np.nan
        fit_ok = False
        x_eval = np.nan
        if x_all.size >= 3:
            x = np.asarray(x_all, dtype=float)
            w_all = np.asarray(c_all, dtype=float)
            use = np.isfinite(x) & (x > 0) & (x <= float(INTERCEPT_D_MAX)) & np.isfinite(w_all) & (w_all > 0)
            idx = np.where(use)[0]
            if idx.size >= 3:
                xx = x[idx]
                ww = w_all[idx]
                yy = y_cov[idx]
                x_eval = float(xx[0]) if bool(INTERCEPT_EVAL_AT_FIRST_BIN) else 0.0

                S0 = np.sum(ww)
                Sx = np.sum(ww * xx)
                Sxx = np.sum(ww * xx * xx)
                det = S0 * Sxx - Sx * Sx
                if (S0 > 0) and (det > 0):
                    Sy = np.sum(ww * yy)
                    Sxy = np.sum(ww * xx * yy)
                    b1 = (S0 * Sxy - Sx * Sy) / det
                    b0 = (Sxx * Sy - Sx * Sxy) / det
                    if INTERCEPT_FORCE_NONPOS_SLOPE and (b1 > 0):
                        b1 = 0.0
                        b0 = Sy / S0

                    yhat = b0 + b1 * xx
                    ybar = np.average(yy, weights=ww)
                    ss_res = float(np.sum(ww * (yy - yhat) ** 2))
                    ss_tot = float(np.sum(ww * (yy - ybar) ** 2))
                    if ss_tot > 0:
                        fit_r2 = 1.0 - ss_res / ss_tot

                    x_draw = np.linspace(float(np.nanmin(x_all)), float(np.nanmax(x_all)), 200)
                    y_draw = b0 + b1 * x_draw
                    ax.plot(x_draw, y_draw, '-', linewidth=1.8, alpha=0.9, label=f"Linear fit (R²={fit_r2:.2f})")
                    fit_ok = True

        # Explicit intercept marker at evaluation point
        if np.isfinite(x_eval):
            ax.plot([x_eval], [intercept_cov], marker='x', markersize=7, color='k', alpha=0.85)

        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Δ Eye Trajectory (a.u.)')
        ax.set_ylabel('Covariance')
        title = f"Pair ({i},{j}) | win={_fmt_ms(win_ms)}"
        if fit_ok and np.isfinite(fit_r2):
            title += f" | R²={fit_r2:.2f}"
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        #ax.legend(frameon=False, loc='best')

        if show and created:
            plt.show()

        return fig, ax


def plot_inspect_neuron_grid(analyzer, win_ms, pairs, ncols=5, figsize_per=(3.8, 3.0)):
    """Plot a grid of inspect_neuron_pair subplots.

    Example: pairs=[(20,20),(21,21),...]
    """
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        raise ValueError('pairs must be non-empty')
    ncols = min(int(ncols), n)
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        constrained_layout=True,
    )
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    for k, (i, j) in enumerate(pairs):
        ax = axs[k // ncols, k % ncols]
        analyzer.inspect_neuron_pair(i, j, win_ms, ax=ax, show=False)

    for k in range(n, nrows * ncols):
        axs[k // ncols, k % ncols].axis('off')

    fig.suptitle(f"inspect_neuron_pair grid | win={_fmt_ms(win_ms)}")
    return fig, axs


def _print_window_bin_diagnostics(dt, window_sizes_ms, t_hist_ms):
    print(f"dt = {dt:.8f} s  ({dt*1000:.3f} ms/bin)")
    t_hist_bins = int(t_hist_ms / (dt * 1000))
    print(f"t_hist: requested={t_hist_ms} ms -> bins={t_hist_bins} -> eff={t_hist_bins*dt*1000:.3f} ms")
    for win_ms in window_sizes_ms:
        t_count_bins = int(win_ms / (dt * 1000))
        t_count_bins = max(t_count_bins, 1)
        eff_ms = t_count_bins * dt * 1000
        print(f"window: requested={win_ms:>4} ms -> bins={t_count_bins:>3} -> eff={eff_ms:>7.3f} ms")


def plot_eye_distance_histograms(results, title=None, logy=False, normalize=False):
    """Plot per-window eye-distance pair histograms.

    Parameters
    ----------
    results : list of dict
        Output from DualWindowAnalysis.run_sweep. If using the canonical McFarland
        analyzer, you can attach per-window histogram fields from
        `analyzer.window_summaries[win]['bin_centers'/'bin_counts']`.
    logy : bool
        If True, use log scale on counts.
    normalize : bool
        If True, plot fraction of pairs per bin rather than raw counts.
    """
    # Keep only windows where histogram exists
    rr = [r for r in results if 'eye_dist_bin_counts' in r]
    if len(rr) == 0:
        raise ValueError("No eye-distance histogram fields found in results.")

    rr = sorted(rr, key=lambda x: float(x.get('window_ms', np.nan)))
    n = len(rr)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows), constrained_layout=True)
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    for i, r in enumerate(rr):
        ax = axs[i // ncols, i % ncols]
        centers = np.asarray(r['eye_dist_bin_centers'], dtype=float)
        counts = np.asarray(r['eye_dist_bin_counts'], dtype=float)
        edges = r.get('eye_dist_bin_edges', None)
        if edges is not None:
            edges = np.asarray(edges, dtype=float)
            widths = np.diff(edges)
            if widths.size == centers.size:
                w = widths
            else:
                w = None
        else:
            w = None

        if w is None:
            # fallback: estimate bar widths from centers
            if centers.size >= 2:
                dc = np.diff(centers)
                step = float(np.nanmedian(dc[np.isfinite(dc)])) if np.isfinite(dc).any() else 1.0
            else:
                step = 1.0
            w = np.full_like(centers, fill_value=step, dtype=float)

        y = counts
        if normalize and np.isfinite(y).any():
            denom = np.nansum(y)
            if denom > 0:
                y = y / denom

        ax.bar(centers, y, width=w, align='center', alpha=0.85, edgecolor='none')
        ax.set_title(f"{_fmt_ms(float(r['window_ms']))} (n_samples={int(r.get('n_samples', -1))})")
        ax.set_xlabel("Eye-trajectory distance (binned)")
        ax.set_ylabel("# pairs" if not normalize else "fraction of pairs")
        if logy and not normalize:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(n, nrows*ncols):
        axs[j // ncols, j % ncols].axis('off')

    if title:
        fig.suptitle(title)
    return fig, axs

#%% Run Luke0804 (Rowley) without a model (recompute)
if RUN_ANALYSIS:
    dataset_configs = load_dataset_configs(DATASET_CONFIGS_PATH)
    dataset_config = dataset_configs[0].copy()
    dataset_config['types'] = ['fixrsvp']

    # Force sampling to 240 Hz (avoid prepare_data downsampling to 120 Hz)
    if FORCE_TARGET_RATE_HZ is not None:
        dataset_config.setdefault('sampling', {})
        dataset_config['sampling']['source_rate'] = int(FORCE_TARGET_RATE_HZ)
        dataset_config['sampling']['target_rate'] = int(FORCE_TARGET_RATE_HZ)
        print(f"Forcing sampling target_rate_hz={FORCE_TARGET_RATE_HZ}")

    print("Requested dataset_config sampling:", dataset_config.get('sampling', None))

    train_data, val_data, dataset_config = prepare_data(dataset_config, strict=False)
    print("Prepared dataset_config sampling:", dataset_config.get('sampling', None))
    sess = train_data.dsets[0].metadata.get('sess', None)
    sess_name = getattr(sess, 'name', 'Luke_2025-08-04')
    cids = np.asarray(dataset_config.get('cids', []))
    print(f"Running Rowley no-model DWA on session: {sess_name}")
    print(f"Dataset has {cids.size} cids")

    # Select fixrsvp inds and build a shallow combined dataset
    inds = torch.concatenate([
        train_data.get_dataset_inds('fixrsvp'),
        val_data.get_dataset_inds('fixrsvp')
    ], dim=0)
    dataset = train_data.shallow_copy()
    dataset.inds = inds

    dset_idx = inds[:, 0].unique().item()
    dset_fix = dataset.dsets[dset_idx]
    trial_inds = dset_fix.covariates['trial_inds'].numpy()
    trials = np.unique(trial_inds)

    NC = dset_fix['robs'].shape[1]
    T = np.max(dset_fix.covariates['psth_inds'][:].numpy()).item() + 1
    NT = len(trials)

    fixation = np.hypot(dset_fix['eyepos'][:, 0].numpy(), dset_fix['eyepos'][:, 1].numpy()) < 1

    # Trial-align arrays
    robs = np.nan * np.zeros((NT, T, NC), dtype=np.float32)
    dfs = np.nan * np.zeros((NT, T, NC), dtype=np.float32)
    eyepos = np.nan * np.zeros((NT, T, 2), dtype=np.float32)
    fix_dur = np.nan * np.zeros((NT,), dtype=np.float32)

    for itrial in tqdm(range(NT)):
        ix = (trials[itrial] == trial_inds) & fixation
        if np.sum(ix) == 0:
            continue

        psth_inds = dset_fix.covariates['psth_inds'][ix].numpy()
        fix_dur[itrial] = len(psth_inds)
        robs[itrial][psth_inds] = dset_fix['robs'][ix].numpy()
        dfs[itrial][psth_inds] = dset_fix['dfs'][ix].numpy()
        eyepos[itrial][psth_inds] = dset_fix['eyepos'][ix].numpy()

    good_trials = fix_dur > 100
    robs = robs[good_trials]
    dfs = dfs[good_trials]
    eyepos = eyepos[good_trials]

    # dt: prefer dataset_config sampling target rate (forced above if requested)
    target_rate = float(dataset_config.get('sampling', {}).get('target_rate', 120))
    dt = 1.0 / target_rate

    # Decide windows after dt is known
    if USE_WINDOW_BINS:
        windows_ms = [float(b) * dt * 1000.0 for b in WINDOW_BINS]
        if INSPECT_WIN_MS is None:
            INSPECT_WIN_MS = windows_ms[0]
    else:
        windows_ms = list(WINDOWS_MS)
        if INSPECT_WIN_MS is None:
            INSPECT_WIN_MS = windows_ms[0]

    valid_time_bins = int(min(VALID_TIME_BINS, robs.shape[1]))
    _print_window_bin_diagnostics(dt, windows_ms, T_HIST_MS)

    # ----------------------------
    # Neuron selection: SELECTED_CIDS only (no ccmax filtering here)
    # ----------------------------
    sel_mask = np.isin(cids, np.asarray(SELECTED_CIDS)) if cids.size == NC else np.ones((NC,), dtype=bool)
    spike_ok = np.nansum(robs, axis=(0, 1)) > TOTAL_SPIKES_THRESHOLD

    # Placeholder for ccmax-based unit filtering (reintroduce later)
    # -----------------------------------------------------------
    # Example idea:
    #   high_cc_cids = np.load(f"../figures/{sess_name}_high_ccmax_cids.npy")
    #   ccmax_mask = np.isin(cids, high_cc_cids)
    #   combined_mask = spike_ok & sel_mask & ccmax_mask
    # For now, skip ccmax and only use spikes + SELECTED_CIDS restriction.
    # -----------------------------------------------------------

    combined_mask = spike_ok & sel_mask
    neuron_mask = np.where(combined_mask)[0]
    print(f"Using {len(neuron_mask)} neurons / {NC} total (spikes>{TOTAL_SPIKES_THRESHOLD} & SELECTED_CIDS)")

    dfs_valid = np.nanmean(dfs[:, :, neuron_mask], axis=2) > 0.5
    valid_mask = (
        dfs_valid &
        np.isfinite(np.sum(robs[:, :, neuron_mask], axis=2)) &
        np.isfinite(np.sum(eyepos, axis=2))
    )

    iix = np.arange(valid_time_bins)
    robs_used = robs[:, iix][:, :, neuron_mask]
    eyepos_used = eyepos[:, iix]
    valid_used = valid_mask[:, iix]

    # ------------------------------------------------------------
    # Optional: filter cells by diagonal intercept-fit quality
    # ------------------------------------------------------------
    # NOTE: The canonical McFarland analyzer does not expose the exp-fit diagnostics
    # this sandbox filter uses (R² / n_log_points / fallback flags). To keep method
    # fidelity, we skip this filter when USE_CANONICAL_MCFARLAND=True.
    if ENABLE_CELL_FIT_FILTER and (not USE_CANONICAL_MCFARLAND):
        if USE_WINDOW_BINS:
            probe_ms = float(FIT_FILTER_WINDOW_BINS) * dt * 1000.0
        else:
            probe_ms = float(FIT_FILTER_WINDOW_MS)

        print(f"\n[fit-filter] probing intercept fits at win={_fmt_ms(probe_ms)}")
        analyzer_probe = DualWindowAnalysis(robs_used, eyepos_used, valid_used, dt=dt, device='cuda')
        probe_results, _ = analyzer_probe.run_sweep([probe_ms], t_hist_ms=T_HIST_MS)
        if len(probe_results) == 0:
            raise RuntimeError("[fit-filter] probe sweep returned no results; cannot filter cells")

        pr = probe_results[0]
        r2 = np.asarray(pr.get('diag_fit_r2', []), dtype=float)
        nvb = np.asarray(pr.get('diag_fit_n_valid_bins', []), dtype=float)
        # exp-fit-only diagnostics may be missing when using McFarland-linear intercepts
        nlog = np.asarray(pr.get('diag_fit_n_log_points', nvb), dtype=float)
        fallback = np.asarray(pr.get('diag_fit_fallback', np.zeros_like(r2, dtype=bool)), dtype=bool)
        clamped = np.asarray(pr.get('diag_fit_clamped', np.zeros_like(r2, dtype=bool)), dtype=bool)
        slope = np.asarray(pr.get('diag_fit_slope', []), dtype=float)
        diag_int = np.asarray(pr.get('diag_intercept', []), dtype=float)

        if r2.size == 0:
            raise RuntimeError("[fit-filter] missing diag fit metrics; run_sweep did not populate them")

        # Base availability: enough bins + finite intercept estimate
        has_bins = np.isfinite(nvb) & (nvb >= float(FIT_MIN_VALID_BINS))
        has_intercept = np.isfinite(diag_int)

        # Model-fit quality (log-linearized exp+plateau fit)
        slope_ok = np.isfinite(slope) & (slope < 0) if FIT_EXPECT_NEG_SLOPE else np.ones_like(r2, dtype=bool)
        good_fit = (
            np.isfinite(r2)
            & (r2 >= float(FIT_MIN_R2))
            & np.isfinite(nlog)
            & (nlog >= float(FIT_MIN_LOG_POINTS))
            & has_bins
            & slope_ok
        )
        strict_good = good_fit & (~fallback)

        # Fallback acceptability: keep if intercept is finite and there are enough bins
        good_fallback = fallback & bool(FIT_ALLOW_FALLBACK_IF_FINITE) & has_bins & has_intercept

        # Candidate masks (strict -> relaxed)
        masks = []
        masks.append(("strict_fit", strict_good))
        masks.append(("good_fit_or_fallback", good_fit | good_fallback))
        masks.append(("finite_intercept", has_bins & has_intercept))

        # Apply clamp veto late
        if FIT_DISALLOW_CLAMPED:
            masks = [(name, (m & (~clamped))) for (name, m) in masks]

        # If explicitly requiring non-fallback, constrain all masks
        if FIT_REQUIRE_NONFALLBACK:
            masks = [(name + "+nonfallback", (m & (~fallback))) for (name, m) in masks]

        # Diagnostics
        def _nanmedian(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            return float(np.median(x)) if x.size else float('nan')

        print(
            "[fit-filter] probe stats: "
            f"fallback={int(np.sum(fallback))}/{int(fallback.size)}, "
            f"median(n_valid_bins)={_nanmedian(nvb):.1f}, "
            f"median(n_log_points)={_nanmedian(nlog):.1f}, "
            f"median(R²)={_nanmedian(r2):.2f}"
        )

        # Choose the strictest mask that keeps enough cells
        chosen_name = None
        good = None
        for name, m in masks:
            n_keep = int(np.sum(m))
            if (not FIT_FILTER_AUTO_RELAX) and (name != masks[0][0]):
                continue
            if n_keep >= int(MIN_CELLS_AFTER_FILTER):
                chosen_name = name
                good = m
                break

        # If none meet the minimum, fall back to the last (loosest) only if auto-relax is off
        if good is None:
            chosen_name, good = masks[0]

        n_before = int(neuron_mask.size)
        n_strict = int(np.sum(masks[0][1]))
        n_after = int(np.sum(good))

        print(f"[fit-filter] strict keep {n_strict}/{n_before} (R²>={FIT_MIN_R2}, nlog>={FIT_MIN_LOG_POINTS}, !fallback)")
        print(f"[fit-filter] chosen mask: {chosen_name} -> keep {n_after}/{n_before}")

        if n_after < int(MIN_CELLS_AFTER_FILTER):
            raise RuntimeError(
                f"[fit-filter] aborting: only {n_after} cells remain (<{MIN_CELLS_AFTER_FILTER}). "
                "Consider lowering FIT_MIN_R2/FIT_MIN_LOG_POINTS or increasing TOTAL_SPIKES_THRESHOLD."
            )

        # Apply filter and rebuild arrays
        neuron_mask = neuron_mask[good]
        robs_used = robs[:, iix][:, :, neuron_mask]

        # recompute validity with the filtered cell set
        dfs_valid = np.nanmean(dfs[:, :, neuron_mask], axis=2) > 0.5
        valid_mask = (
            dfs_valid &
            np.isfinite(np.sum(robs[:, :, neuron_mask], axis=2)) &
            np.isfinite(np.sum(eyepos, axis=2))
        )
        valid_used = valid_mask[:, iix]
        print(f"[fit-filter] updated neuron_mask size={int(neuron_mask.size)}")

    AnalyzerCls = McFarlandDualWindowAnalysis if USE_CANONICAL_MCFARLAND else DualWindowAnalysis
    analyzer = AnalyzerCls(robs_used, eyepos_used, valid_used, dt=dt, device='cuda')
    if USE_CANONICAL_MCFARLAND:
        results, last_mats = analyzer.run_sweep(
            windows_ms,
            t_hist_ms=T_HIST_MS,
            n_bins=EYE_DIST_N_BINS,
            n_shuffles=0,
            seed=42,
            intercept_mode='linear',
        )

        # Compatibility aliases (older sandbox code expects these names)
        for m in last_mats:
            if isinstance(m, dict):
                if 'Noise_Corr' not in m and 'NoiseCorrC' in m:
                    m['Noise_Corr'] = m['NoiseCorrC']
                if 'Noise_Uncorr' not in m and 'NoiseCorrU' in m:
                    m['Noise_Uncorr'] = m['NoiseCorrU']

        # Attach per-window eye-distance histograms from canonical window_summaries
        for r in results:
            win = float(r.get('window_ms', np.nan))
            s = analyzer.window_summaries.get(win, None)
            if s is None:
                continue
            r['eye_dist_bin_centers'] = np.asarray(s.get('bin_centers', []), dtype=float)
            r['eye_dist_bin_counts'] = np.asarray(s.get('bin_counts', []), dtype=float)
            # bin edges are not stored in canonical summaries; plotting code infers widths.
    else:
        results, last_mats = analyzer.run_sweep(windows_ms, t_hist_ms=T_HIST_MS)
    windows = [r['window_ms'] for r in results]

    if SAVE_OUTPUTS:
        from pathlib import Path
        import pickle

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            'sess': sess_name,
            'cids': cids,
            'neuron_mask': neuron_mask,
            'windows': windows,
            'cids_used': cids[neuron_mask] if cids.size == NC else None,
            'results': results,
            'last_mats': last_mats,
            'meta': {
                'notes': 'Rowley Luke0804 (no-model) covariance decomposition; V2-only; includes eye-distance hist counts.',
                'dataset_configs_path': DATASET_CONFIGS_PATH,
                'windows_ms': windows_ms,
                'use_window_bins': USE_WINDOW_BINS,
                'window_bins': WINDOW_BINS if USE_WINDOW_BINS else None,
                't_hist_ms': T_HIST_MS,
                'valid_time_bins': valid_time_bins,
                'dt': dt,
                'target_rate_hz': target_rate,
                'force_target_rate_hz': FORCE_TARGET_RATE_HZ,
                'total_spikes_threshold': TOTAL_SPIKES_THRESHOLD,
                'use_canonical_mcfarland': USE_CANONICAL_MCFARLAND,
                'eye_dist_n_bins': EYE_DIST_N_BINS,
                'eye_binning_mode': EYE_BINNING_MODE,
                'eye_bin_q': EYE_BIN_Q,
                # Sandbox-only analysis knobs (not used when canonical McFarland is enabled)
                'min_bin_count_for_fit': (None if USE_CANONICAL_MCFARLAND else MIN_BIN_COUNT_FOR_FIT),
                'time_matched_conditioning': (None if USE_CANONICAL_MCFARLAND else TIME_MATCHED_CONDITIONING),
                'max_samples_per_timebin': (None if USE_CANONICAL_MCFARLAND else MAX_SAMPLES_PER_TIMEBIN),
                'max_samples_per_window': (None if USE_CANONICAL_MCFARLAND else MAX_SAMPLES_PER_WINDOW),
                'enable_cell_fit_filter': (False if USE_CANONICAL_MCFARLAND else ENABLE_CELL_FIT_FILTER),
                'canonical_intercept_mode': ('linear' if USE_CANONICAL_MCFARLAND else None),
                'fit_filter_window_bins': FIT_FILTER_WINDOW_BINS if USE_WINDOW_BINS else None,
                'fit_filter_window_ms': FIT_FILTER_WINDOW_MS if not USE_WINDOW_BINS else None,
                'min_cells_after_filter': MIN_CELLS_AFTER_FILTER,
                'fit_min_r2': FIT_MIN_R2,
                'fit_min_log_points': FIT_MIN_LOG_POINTS,
                'fit_min_valid_bins': FIT_MIN_VALID_BINS,
                'fit_disallow_clamped': FIT_DISALLOW_CLAMPED,
                'fit_require_nonfallback': FIT_REQUIRE_NONFALLBACK,
                'fit_allow_fallback_if_finite': FIT_ALLOW_FALLBACK_IF_FINITE,
                'fit_filter_auto_relax': FIT_FILTER_AUTO_RELAX,
            }
        }

        pkl_path = FIGURES_DIR / f"mcfarland_fixrsvp_{sess_name}_V2_no_model.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(out, f)
        print(f"Saved outputs: {pkl_path}")

    # Plot: eye-distance pair histograms per window
    fig, axs = plot_eye_distance_histograms(
        results,
        title=f"Eye-distance pair histograms (Rowley {sess_name}, V2 only)",
        logy=True,
        normalize=False,
    )
    if SAVE_FIGURES:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = FIGURES_DIR / f"eye_distance_hist_{sess_name}_V2.pdf"
        fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure: {pdf_path}")
    plt.show()

    # Plot: inspect diagonal (variance) vs distance curves on a grid (all neurons, chunked)
    if PLOT_INSPECTION_GRID and (len(neuron_mask) > 0):
        n_total = int(len(neuron_mask))
        pairs_all = [(i, i) for i in range(n_total)]

        for start in range(0, n_total, int(INSPECT_CHUNK_SIZE)):
            end = min(n_total, start + int(INSPECT_CHUNK_SIZE))
            pairs = pairs_all[start:end]
            fig2, _ = plot_inspect_neuron_grid(
                analyzer,
                win_ms=INSPECT_WIN_MS,
                pairs=pairs,
                ncols=int(INSPECT_NCOLS),
            )
            if SAVE_FIGURES:
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                part = (start // int(INSPECT_CHUNK_SIZE)) + 1
                pdf_path2 = FIGURES_DIR / f"inspect_diag_grid_{sess_name}_V2_win{_fmt_ms_tag(INSPECT_WIN_MS)}_part{part:02d}.pdf"
                fig2.savefig(pdf_path2, bbox_inches='tight', dpi=300)
                print(f"Saved figure: {pdf_path2}")

            if SHOW_INSPECTION_GRIDS:
                plt.show()
            else:
                plt.close(fig2)

#%% Load cached results (optional)
# Note: use a separate `if not RUN_ANALYSIS:` block (instead of `else:`) so you can
# run code cells independently in VS Code without ever hitting a dangling `else:`.
if not RUN_ANALYSIS:
    import pickle

    results_file = Path(__file__).resolve().parent.parent / "figures" / "mcfarland_fixrsvp_Luke_2025-08-04.pkl"
    with open(results_file, 'rb') as f:
        output_Luke = pickle.load(f)

    results = output_Luke['results']
    last_mats = output_Luke['last_mats']
    windows = output_Luke.get('windows', [r['window_ms'] for r in results])
 #%% Load saved Allen McFarland results
# import pickle
# from pathlib import Path

# results_file = Path(__file__).resolve().parent.parent / "figures" / "mcfarland_fixrsvp_Allen_2022-03-04.pkl"
# with open(results_file, 'rb') as f:
#     output_Allen = pickle.load(f)

# results = output_Allen['results']
# last_mats = output_Allen['last_mats']
# windows = output_Allen.get('windows', [r['window_ms'] for r in results])

#%% 3. Plot Fano Factor Scaling
import pandas as pd
# Ensure mean Fano fields are present (compute if missing)
for r in results:
    if 'ff_uncorr_mean' not in r:
        r['ff_uncorr_mean'] = np.nanmean(r['ff_uncorr'])
    if 'ff_corr_mean' not in r:
        r['ff_corr_mean'] = np.nanmean(r['ff_corr'])

df = pd.DataFrame(results)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FIG_TAG = globals().get('sess_name', 'Luke_2025-08-04')

fig_ff, ax_ff = plt.subplots(figsize=(8, 6))
ax_ff.plot(df['window_ms'], df['ff_uncorr_mean'], 'o-', label='Standard (Uncorrected)')
ax_ff.plot(df['window_ms'], df['ff_corr_mean'], 'o-', label='FEM-Corrected')

ax_ff.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax_ff.set_xlabel('Count Window (ms)')
ax_ff.set_ylabel('Mean Fano Factor')
ax_ff.set_title('Integration of Noise: FEM Correction')
ax_ff.legend()
ax_ff.grid(True, alpha=0.3)

if SAVE_FIGURES:
    p = FIGURES_DIR / f"ff_scaling_{FIG_TAG}.pdf"
    fig_ff.savefig(p, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {p}")
if SAVE_SVG:
    p = FIGURES_DIR / f"ff_scaling_{FIG_TAG}.svg"
    fig_ff.savefig(p, bbox_inches='tight')
    print(f"Saved figure: {p}")
plt.show()

# Pick a window for detailed plots
# Use 5ms for the FF + eigenspectrum plots (requested). We choose the closest available.
PLOT_WIN_MS = 5

def _closest_window_idx(results_list, target_ms):
    w = np.asarray([float(r.get('window_ms', np.nan)) for r in results_list], dtype=float)
    return int(np.nanargmin(np.abs(w - float(target_ms))))

window_idx = _closest_window_idx(results, PLOT_WIN_MS)
win_ms = results[window_idx]['window_ms']
Sigma_FEM = last_mats[window_idx]['FEM']
u, s, vh = np.linalg.svd(Sigma_FEM)
fig_svd_fem, ax = plt.subplots()
ax.plot(s, 'o-')
ax.set_title(f"Singular Values of FEM Covariance ({_fmt_ms(win_ms)})")
ax.set_xlabel('component')
ax.set_ylabel('singular value')
ax.grid(True, alpha=0.3)
if SAVE_FIGURES:
    p = FIGURES_DIR / f"svd_fem_{FIG_TAG}_win{_fmt_ms_tag(win_ms)}.pdf"
    fig_svd_fem.savefig(p, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {p}")
if SAVE_SVG:
    p = FIGURES_DIR / f"svd_fem_{FIG_TAG}_win{_fmt_ms_tag(win_ms)}.svg"
    fig_svd_fem.savefig(p, bbox_inches='tight')
    print(f"Saved figure: {p}")
plt.show()

# same for total covariance
Sigma_Total = last_mats[window_idx]['Total']
u, s, vh = np.linalg.svd(Sigma_Total)
fig_svd_tot, ax = plt.subplots()
ax.plot(s, 'o-')
ax.set_title(f"Singular Values of Total Covariance ({_fmt_ms(win_ms)})")
ax.set_xlabel('component')
ax.set_ylabel('singular value')
ax.grid(True, alpha=0.3)
if SAVE_FIGURES:
    p = FIGURES_DIR / f"svd_total_{FIG_TAG}_win{_fmt_ms_tag(win_ms)}.pdf"
    fig_svd_tot.savefig(p, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {p}")
if SAVE_SVG:
    p = FIGURES_DIR / f"svd_total_{FIG_TAG}_win{_fmt_ms_tag(win_ms)}.svg"
    fig_svd_tot.savefig(p, bbox_inches='tight')
    print(f"Saved figure: {p}")
plt.show()

# now noise correlation (FEM-corrected)
Sigma_Noise = last_mats[window_idx]['Noise_Corr']
u, s, vh = np.linalg.svd(Sigma_Noise)
fig_svd_noise, ax = plt.subplots()
ax.plot(s, 'o-')
ax.set_title(f"Singular Values of Noise Correlation ({_fmt_ms(win_ms)})")
ax.set_xlabel('component')
ax.set_ylabel('singular value')
ax.grid(True, alpha=0.3)
if SAVE_FIGURES:
    p = FIGURES_DIR / f"svd_noise_{FIG_TAG}_win{_fmt_ms_tag(win_ms)}.pdf"
    fig_svd_noise.savefig(p, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {p}")
if SAVE_SVG:
    p = FIGURES_DIR / f"svd_noise_{FIG_TAG}_win{_fmt_ms_tag(win_ms)}.svg"
    fig_svd_noise.savefig(p, bbox_inches='tight')
    print(f"Saved figure: {p}")
plt.show()
# %%
i = window_idx
fig_scatter, ax = plt.subplots()
ax.plot(results[i]['ff_uncorr'], results[i]['ff_corr'], 'o', alpha=0.7)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'k')
ax.set_xlabel('Fano Factor (Uncorrected)')
ax.set_ylabel('Fano Factor (Corrected)')
ax.set_title(f"FF per-neuron scatter ({_fmt_ms(float(results[i]['window_ms']))})")
ax.grid(True, alpha=0.25)
if SAVE_FIGURES:
    p = FIGURES_DIR / f"ff_scatter_{FIG_TAG}_win{_fmt_ms_tag(float(results[i]['window_ms']))}.pdf"
    fig_scatter.savefig(p, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {p}")
if SAVE_SVG:
    p = FIGURES_DIR / f"ff_scatter_{FIG_TAG}_win{_fmt_ms_tag(float(results[i]['window_ms']))}.svg"
    fig_scatter.savefig(p, bbox_inches='tight')
    print(f"Saved figure: {p}")
plt.show()

#%%
results[0]
# %%
# show the total covariance matrix subtracting the diagonal
window_idx = _closest_window_idx(results, PLOT_WIN_MS)
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(last_mats[window_idx]['Total'] - np.diag(np.diag(last_mats[window_idx]['Total'])))
plt.title(f"Total Covariance ({_fmt_ms(windows[window_idx])})")

# show FEM
plt.subplot(1,3,2)
plt.imshow(last_mats[window_idx]['FEM'] - np.diag(np.diag(last_mats[window_idx]['FEM'])))
plt.title(f"FEM Covariance ({_fmt_ms(windows[window_idx])})")

# show Noise_Corr
plt.subplot(1,3,3)
plt.imshow(last_mats[window_idx]['PSTH'] - np.diag(np.diag(last_mats[window_idx]['PSTH'])))
plt.title(f"PSTH Covariance ({_fmt_ms(windows[window_idx])})")



# %%
plt.subplot(1,2,1)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH'])
plt.subplot(1,2,2)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'])

# %%

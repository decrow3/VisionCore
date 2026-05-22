"""
Compute spatial information from the model using reconstructed stimuli.
Allows counterfactual analysis with real vs fake eye traces.
"""
#%% Imports
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% Core imports

from DataYatesV1 import enable_autoreload, get_free_device
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye
from spatial_info import make_stimulus_stack, make_counterfactual_stim
from spatial_info import get_spatial_readout
from spatial_info import compute_rate_map, compute_rate_map_batched
from spatial_info import spatial_ssi_population, make_movie
from scripts.spatial_info_cache_declan import (
    load_backimage_fixation_results,
    load_fixrsvp_fixation_pool,
    load_backimage_image_cache,
)

enable_autoreload()
device = get_free_device()

from utils import get_model_and_dataset_configs

#%% Get model and data
model, dataset_configs = get_model_and_dataset_configs()
model = model.to(device)

import dill
SCRIPT_DIR = Path(__file__).resolve().parent
with open(SCRIPT_DIR / 'mcfarland_outputs_mono.pkl', 'rb') as f:
    outputs = dill.load(f)

readout = get_spatial_readout(model, outputs).to(device)

sessions = [outputs[i]['sess'] for i in range(len(outputs))]
#%% Helper functions

import pickle, hashlib, json

def cache_load_or_compute(cache_path: Path, compute_fn, *, meta: dict):
    """
    Loads cache if present and metadata matches; otherwise computes + saves.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

    meta_json = json.dumps(meta, sort_keys=True).encode("utf-8")
    meta_hash = hashlib.sha1(meta_json).hexdigest()

    if cache_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("meta_hash") == meta_hash:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    out = compute_fn()
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    meta_path.write_text(json.dumps({"meta_hash": meta_hash, "meta": meta}, indent=2))
    return out


def debug_eye_units_and_bounds(eyedeg_xy: np.ndarray, img_hw: tuple[int, int], ppd: float) -> None:
    """Sanity check: are eye positions (deg) consistent with an image centered at (0,0) deg?"""
    H, W = img_hw
    eyedeg_xy = np.asarray(eyedeg_xy)
    x_deg = eyedeg_xy[:, 0]
    y_deg = eyedeg_xy[:, 1]

    extent_w = (W / float(ppd)) / 2.0
    extent_h = (H / float(ppd)) / 2.0

    in_deg = (
        (x_deg >= -extent_w) & (x_deg <= extent_w) &
        (y_deg >= -extent_h) & (y_deg <= extent_h)
    )

    # Map degrees -> pixel indices (origin upper-left). +y in deg usually means up, hence the minus.
    x_pix = x_deg * float(ppd) + (W / 2.0)
    y_pix = -y_deg * float(ppd) + (H / 2.0)
    in_pix = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)

    print("\n=== EYE / IMAGE COORD CHECK ===")
    print(f"Image (H,W): {(H, W)}, ppd={ppd:.3f}")
    print(f"Deg extent: x±{extent_w:.2f}, y±{extent_h:.2f}")
    print(
        f"Eye deg range: x[{np.nanmin(x_deg):.2f},{np.nanmax(x_deg):.2f}] "
        f"y[{np.nanmin(y_deg):.2f},{np.nanmax(y_deg):.2f}]"
    )
    print(f"In deg extent: {np.nanmean(in_deg) * 100:.1f}%")
    print(f"In pixel bounds after deg->pix: {np.nanmean(in_pix) * 100:.1f}%")
    print("If these % are low, you likely have a unit/sign/center mismatch.")


def rescale_fixations_only(trace: np.ndarray, saccade_mask: np.ndarray, eye_scale: float) -> np.ndarray:
    """Rescale only fixational jitter (within fixation runs), leaving saccade frames untouched.

    trace: (T,2) in degrees
    saccade_mask: (T,) bool (True during saccade frames)
    eye_scale: scalar where 0 removes FEM, 1 keeps original
    """
    trace = np.asarray(trace, dtype=np.float32)
    saccade_mask = np.asarray(saccade_mask, dtype=bool)
    out = trace.copy()

    fix_idx = np.where(~saccade_mask)[0]
    if fix_idx.size == 0:
        return out

    split_pts = np.where(np.diff(fix_idx) != 1)[0] + 1
    runs = np.split(fix_idx, split_pts)
    scale = float(eye_scale)
    for r in runs:
        if r.size == 0:
            continue
        center = out[r].mean(axis=0, keepdims=True)
        out[r] = center + (out[r] - center) * scale
    return out



"""
Plotting code for making a nice figure with the spatial information over time on an image
"""
def plot_spatial_info_figure(full_stack, iframe, f, Pr, eyepos, itrial, I_t_null, I_t,
                             crop=(slice(250, 350), slice(250, 350)),
                             outpath=None, dpi=300):
    # -------------------------
    # Style (publication-ish)
    # -------------------------
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,   # nicer embedded fonts
        "ps.fonttype": 42,
    })

    # consistent colors
    c_fem  = "#1f77b4"  # matplotlib default blue
    c_null = "#ff7f0e"  # matplotlib default orange
    c_diag = "0.15"

    fig = plt.figure(figsize=(11.2, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.05, 1.0, 1.25, 1.25],
        hspace=0.45, wspace=0.38
    )

    # Helpers
    def prettify(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.15, linewidth=0.6)
        ax.set_axisbelow(True)

    def panel_label(ax, s):
        ax.text(0.0, 1.08, s, transform=ax.transAxes,
                ha="left", va="bottom", fontweight="bold")

    # -------------------------
    # A) Stimulus
    # -------------------------
    axA = fig.add_subplot(gs[0, 0])
    stim = full_stack[iframe][crop[0], crop[1]]
    axA.imshow(stim, cmap="gray", interpolation="nearest")
    axA.set_title("Stimulus")
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_visible(False)
    panel_label(axA, "A")

    # -------------------------
    # B) Power spectrum
    # -------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(f, Pr[iframe], color=c_fem, lw=1.8)
    axB.set_xlabel("Spatial Frequency (c/deg)")
    axB.set_ylabel("Power")
    axB.set_title("Power Spectrum")
    prettify(axB)
    # optional: focus x-range / log scales if you want
    # axB.set_xscale("log"); axB.set_yscale("log")
    panel_label(axB, "B")

    # -------------------------
    # C) Eye position (wide)
    # -------------------------
    axC = fig.add_subplot(gs[0, 2:4])
    ep = eyepos[itrial]  # expected shape: (T,2) or (T,) ; your screenshot shows 2 traces
    if ep.ndim == 1:
        axC.plot(ep, color=c_fem, lw=1.5)
    else:
        axC.plot(ep[:, 0], color=c_fem,  lw=1.5, label=None)
        axC.plot(ep[:, 1], color=c_null, lw=1.5, label=None)
    axC.set_xlabel("Time (frames)")
    axC.set_ylabel("Eye Position (deg)")
    axC.set_title("Eye Position")
    prettify(axC)
    panel_label(axC, "C")

    # -------------------------
    # D) Spatial info scatter (wide)
    # -------------------------
    axD = fig.add_subplot(gs[1, 0:2])
    x = I_t_null.mean(0)
    y = I_t.mean(0)

    axD.scatter(x, y, s=14, alpha=0.18, color=c_fem, edgecolors="none", rasterized=True)
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    pad = 0.03 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    axD.plot([lo, hi], [lo, hi], color=c_diag, lw=1.6)
    axD.set_xlim(lo, hi)
    axD.set_ylim(lo, hi)

    axD.set_xlabel("Bits (No FEM)")
    axD.set_ylabel("Bits (With FEM)")
    axD.set_title("Spatial Info (Units)")
    prettify(axD)
    panel_label(axD, "D")

    # -------------------------
    # E) Spatial info timecourse (wide)
    # -------------------------
    axE = fig.add_subplot(gs[1, 2:4])
    axE.plot(I_t.mean(1),      color=c_fem,  lw=1.6, label="FEM")
    axE.plot(I_t_null.mean(1), color=c_null, lw=1.6, label="Null")
    axE.set_xlabel("Time (frames)")
    axE.set_ylabel("Spatial Info (bits)")
    axE.legend(frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.85",
               loc="upper left")
    axE.set_title("")  # your original doesn’t title this panel; keep clean
    prettify(axE)
    panel_label(axE, "E")

    # Margins so labels breathe
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.12)

    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    return fig



def radial_power_spectra_np(imgs, ppd, nbins=None, window=True, return_2d=False, eps=0.0):
    """
    imgs: (N,H,W) float/uint, any range
    ppd: pixels per degree (float)
    nbins: number of radial bins (default ~ min(H,W)//2)
    window: apply 2D Hann window (recommended)
    return_2d: also return per-image 2D power spectra (fftshifted)
    Returns:
      f_centers: (B,) cycles/degree
      P_radial:  (N,B) mean power in annuli
      (optional) P2d: (N,H,W) 2D power spectra (fftshifted)
    """
    imgs = np.asarray(imgs, dtype=np.float32)
    N, H, W = imgs.shape
    if nbins is None:
        nbins = min(H, W) // 2

    # --- precompute frequency grid in cycles/degree ---
    fy = np.fft.fftfreq(H, d=1.0) * ppd  # cycles/deg
    fx = np.fft.fftfreq(W, d=1.0) * ppd
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R = np.sqrt(FX**2 + FY**2)
    R = np.fft.fftshift(R)

    # radial bins (0 .. max radius)
    r_max = R.max()
    edges = np.linspace(0.0, r_max + 1e-12, nbins + 1)
    bin_idx = np.digitize(R.ravel(), edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    counts = np.bincount(bin_idx, minlength=nbins).astype(np.float32)

    # optional window
    if window:
        wy = np.hanning(H).astype(np.float32)
        wx = np.hanning(W).astype(np.float32)
        win = wy[:, None] * wx[None, :]
    else:
        win = None

    P_radial = np.empty((N, nbins), dtype=np.float32)
    P2d_out = np.empty((N, H, W), dtype=np.float32) if return_2d else None

    for i in range(N):
        x = imgs[i]
        x = x - x.mean()
        if win is not None:
            x = x * win

        F = np.fft.fft2(x, norm="ortho")
        P = (F.real * F.real + F.imag * F.imag)  # |F|^2
        P = np.fft.fftshift(P)

        # radial mean via bincount
        sums = np.bincount(bin_idx, weights=P.ravel(), minlength=nbins).astype(np.float32)
        P_radial[i] = sums / (counts + eps)

        if return_2d:
            P2d_out[i] = P

    f_centers = 0.5 * (edges[:-1] + edges[1:])
    return (f_centers, P_radial, P2d_out) if return_2d else (f_centers, P_radial)

"""
This is the key simulation
Inputs:
    eyepos: (T,2) eye positions in degrees
    full_stack: (N,H,W) stimulus stack (N frames)
    out_size: (H_out, W_out) size of output stimulus
    n_lags: number of time lags to use
    scale: scale factor for stimulus
    plot: whether to plot the eyeposition and stimulus frame
"""
def get_trial_stim_and_rates(eyepos, full_stack,
                             out_size=(151, 151), n_lags=32, scale=1.0, plot=False):

    # `make_counterfactual_stim` expects `full_stack` shaped (T,H,W)
    if isinstance(full_stack, torch.Tensor):
        full_stack = full_stack.detach().cpu().numpy()
    full_stack = np.asarray(full_stack)
    if full_stack.ndim == 4 and full_stack.shape[1] == 1:
        full_stack = full_stack[:, 0]
    if full_stack.ndim != 3:
        raise ValueError(
            f"full_stack must be (T,H,W) or (T,1,H,W); got shape {full_stack.shape}"
        )

    nan_idx = np.where(np.isnan(eyepos[:, 0]))[0]
    T = int(nan_idx[0]) if nan_idx.size > 0 else int(eyepos.shape[0])

    # Ensure we have enough stimulus frames for lag embedding: full_stack[:T + n_lags]
    max_T = int(full_stack.shape[0] - n_lags)
    if T > max_T:
        T = max_T
    eyepos = eyepos[:T]
    eyepos = torch.from_numpy(eyepos).float()

    null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0)

    eye_stim = make_counterfactual_stim(full_stack, eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale)
    eye_stim_null = make_counterfactual_stim(full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale)
    # print(f"Reconstructed stim shape: {eye_stim.shape}")

    v = out_size[0]/ppd
    if plot:
        plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
        plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
        plt.show()

    # Compute rates on normalized stimulus
    # TODO: This assumes pixelnorm was called (which it almost certainly was, but we should do this better...)
    y = compute_rate_map_batched(model, readout, (eye_stim - 127.0)/255.0)
    y_null = compute_rate_map_batched(model, readout, (eye_stim_null - 127.0)/255.0)

    return y, y_null, eye_stim, eye_stim_null

"""
This is the key simulation now with multiple scales for the eyetrace
Inputs:
    eyepos: (T,2) eye positions in degrees, single trial of fixed max length (540) with NaNs to indicate end
    full_stack: (N,H,W) stimulus stack (N frames)
    out_size: (H_out, W_out) size of output stimulus
    n_lags: number of time lags to use
    stim_scale: scale factor for stimulus
    eye_scale: vecotr of scale factor for eye position (e.g. 1.0 for real, 0.0 for null), default to [1 0]
    plot: whether to plot the eyeposition and stimulus frame
    # now packages the outputs into (T, C, H_out, W_out, n_eyescales) 
    # for the stimulus and rates, where n_eyescales is the number of different eye scales we want to test
    # rather than two separate outputs for real and null, we can just have one output with the different scales
    returns: y: (T, C, H_out, W_out, n_eyescales) rate map for real eye trace
    eye_stim: (T, C, H_out, W_out, n_eyescales) stimulus for real eye trace
"""
def get_trial_stim_and_rates_eyescaled(eyepos, full_stack,
                             out_size=(151, 151), n_lags=32, stim_scale=1.0, eye_scale=[0, 0.5, 1.0, 2.0], plot=False):
    if isinstance(full_stack, torch.Tensor):
        full_stack = full_stack.detach().cpu().numpy()
    full_stack = np.asarray(full_stack)
    if full_stack.ndim == 4 and full_stack.shape[1] == 1:
        full_stack = full_stack[:, 0]
    if full_stack.ndim != 3:
        raise ValueError(
            f"full_stack must be (T,H,W) or (T,1,H,W); got shape {full_stack.shape}"
        )

    nan_idx = np.where(np.isnan(eyepos[:, 0]))[0]
    T = int(nan_idx[0]) if nan_idx.size > 0 else int(eyepos.shape[0])
    #For very fast frame rates, we may have fewer frames our eyetrack supports
    if T > (full_stack.shape[0]-n_lags-1):
        #print(f"Warning: eyepos length {T} greater than full_stack length {full_stack.shape[0]}, truncating eyepos")
        T = full_stack.shape[0]-n_lags-1 # ensure we have enough frames for lags full_stack[:eyepos.shape[0] + n_lags])
    eyepos = eyepos[:T]
    print(f"Using eyepos length {T}")
    eyepos = torch.from_numpy(eyepos).float()

    #scaled_eyepos= torch.zeros_like(eyepos) + eyepos.mean(0) * (1 - eye_scale) + eyepos * eye_scale
    #null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0) * (1 - eye_scale) + eyepos * eye_scale

    for i in range(eye_scale.shape[0]):
        scaled_eyepos = torch.zeros_like(eyepos) + eyepos * eye_scale[i] + (1 - eye_scale[i]) * eyepos.mean(0)
        
        eye_stim_loop = make_counterfactual_stim(full_stack, scaled_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=stim_scale)
        # append to the output tensor
        eye_stim = eye_stim_loop if i == 0 else torch.cat((eye_stim, eye_stim_loop), dim=4)
   
        #eye_stim_null = make_counterfactual_stim(full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=stim_scale)
        # print(f"Reconstructed stim shape: {eye_stim.shape}")

        v = out_size[0]/ppd
        if plot:
            plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
            plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
            plt.show()

        # Compute rates on normalized stimulus
        # TODO: This assumes pixelnorm was called (which it almost certainly was, but we should do this better...)
        #for i in range(eye_scale.shape[0]):
        y_loop = compute_rate_map_batched(model, readout, (eye_stim_loop - 127.0)/255.0)
        # y_loop should be (T, C, H_out, W_out), append to output tensor on new axis
        # y = y_loop if i == 0 else torch.cat((y, y_loop), dim=4)
        # but this errors "IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)"
        # so we need to unsqueeze y_loop first
        y_loop = y_loop.unsqueeze(4)
        y = y_loop if i == 0 else torch.cat((y, y_loop), dim=4)
        
    #y_null = compute_rate_map_batched(model, readout, (eye_stim_null - 127.0)/255.0)

    return y, eye_stim



#%% Now try for a single natural image to add in real saccades from that image, 
# interspersed with fixational eye movements of different length from our RSVP stim
# in order to manipulate saccade frequency. Here we just treat saccades as 
# instant jumps in eye position, but we use real positions of fixations from backimage
# stimuli viewed by the monkeys.


# Extract eye positions for a single natural image frame across all trials
from DataYatesV1.exp import BackImageTrial, get_trial_protocols
from DataYatesV1.utils.detect_saccades import detect_saccades
def get_fixations_for_backimage_across_sessions(
    model, sessions, image_file=None, n_images=27
):
    """
    Get fixation eye positions for multiple backimages across sessions.
    Aggregates fixation data (ignoring saccades for now).
    
    Parameters
    ----------
    model : VisionCore model
        Model with .names attribute
    sessions : list
        List of session names to process
    image_file : str, optional
        Specific image to analyze. If None, uses most common images.
    n_images : int
        Number of most-common images to return data for
    
    Returns
    -------
    results : dict
        Keys are image filenames, values are dicts with:
        - 'eyepos': (N_samples, 2) fixation eye positions
        - 'n_trials': number of trials
        - 'n_sessions': number of sessions with this image
    """
    
    from collections import Counter
    
    # First pass: count images across all sessions
    print("Scanning sessions for backimage data...")
    all_image_counts = Counter()
    
    for sess_idx, name in enumerate(sessions):
        try:
            dataset_idx = model.names.index(name)
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)
            
            inds = torch.concatenate([
                train_data.get_dataset_inds('backimage'),
                val_data.get_dataset_inds('backimage')
            ], dim=0)
            
            if len(inds) == 0:
                continue
            
            dataset = train_data.shallow_copy()
            dataset.inds = inds
            dset_idx = inds[:,0].unique().item()
            
            sess_obj = dataset.dsets[dset_idx].metadata['sess']
            exp = sess_obj.exp
            protocols = get_trial_protocols(exp)
            trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
            unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])
            
            backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
            backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
            
            if len(backimage_trial_inds) == 0:
                continue
            
            backimage_trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds]
            
            for t in backimage_trials:
                all_image_counts[t.image_file] += 1
                
        except Exception as e:
            print(f"  Warning: Failed to scan session {name}: {e}")
            continue
    
    if len(all_image_counts) == 0:
        raise ValueError("No backimage trials found across any session")
    
    # Get top N images
    top_images = [img for img, count in all_image_counts.most_common(n_images)]
    print(f"\nTop {n_images} images:")
    for img, count in all_image_counts.most_common(n_images):
        print(f"  {img}: {count} trials")
    
    # Second pass: collect fixation data for top images
    results = {}
    
    for image_file in top_images:
        print(f"\nCollecting fixations for '{image_file}'...")
        
        all_eyepos = []
        n_trials_total = 0
        n_sessions_with_image = 0
        
        for sess_idx, name in enumerate(sessions):
            try:
                dataset_idx = model.names.index(name)
                train_data, val_data, _ = load_single_dataset(model, dataset_idx)
                
                inds = torch.concatenate([
                    train_data.get_dataset_inds('backimage'),
                    val_data.get_dataset_inds('backimage')
                ], dim=0)
                
                if len(inds) == 0:
                    continue
                
                dataset = train_data.shallow_copy()
                dataset.inds = inds
                dset_idx = inds[:,0].unique().item()
                
                sess_obj = dataset.dsets[dset_idx].metadata['sess']
                exp = sess_obj.exp
                protocols = get_trial_protocols(exp)
                trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
                unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])
                
                backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
                backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
                
                if len(backimage_trial_inds) == 0:
                    continue
                
                backimage_trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds]
                
                # Find trials with this image
                trials_with_image = [(idx, t) for idx, t in zip(backimage_trial_inds, backimage_trials)
                                     if t.image_file == image_file]
                
                if len(trials_with_image) == 0:
                    continue
                
                # Get eye position data
                eyepos = dataset.dsets[dset_idx]['eyepos'][:].numpy()
                t_dpi = sess_obj.dpi['t_ephys'].values
                
                # Pre-compute saccade mask (from detect_saccades)
                saccades = detect_saccades(sess_obj)
                is_saccade_mask = np.zeros(len(t_dpi), dtype=bool)
                for sacc in saccades:
                    sacc_samples = (t_dpi >= sacc.start_time) & (t_dpi <= sacc.end_time)
                    is_saccade_mask |= sacc_samples
                
                # Collect fixation eye positions
                session_eyepos = []
                for global_trial_idx, trial_obj in trials_with_image:
                    trial_mask = (trial_inds == global_trial_idx)
                    trial_sample_inds = np.where(trial_mask)[0]
                    
                    if len(trial_sample_inds) == 0:
                        continue
                    
                    # Use saccade mask to exclude saccade periods
                    trial_is_saccade = is_saccade_mask[trial_sample_inds]
                    fixation_samples = trial_sample_inds[~trial_is_saccade]
                    
                    if len(fixation_samples) > 0:
                        session_eyepos.append(eyepos[fixation_samples])
                
                if session_eyepos:
                    all_eyepos.append(np.vstack(session_eyepos))
                    n_sessions_with_image += 1
                    n_trials_total += len(trials_with_image)
                    
                    print(f"  Session {sess_idx} ({name}): {len(trials_with_image)} trials, "
                          f"{sum(len(x) for x in session_eyepos)} fixation samples")
                
            except Exception as e:
                print(f"  Warning: Failed to process session {name}: {e}")
                continue
        
        if all_eyepos:
            eyepos_aggregated = np.vstack(all_eyepos)
            results[image_file] = {
                'eyepos': eyepos_aggregated,
                'n_trials': n_trials_total,
                'n_sessions': n_sessions_with_image,
            }
            print(f"  Total: {n_sessions_with_image} sessions, {n_trials_total} trials, "
                  f"{len(eyepos_aggregated)} fixation samples")
    
    return results


# %% Example usage - multiple images, multiple sessions (default to 3)
print("=" * 60)
print("BACKIMAGE FIXATION ANALYSIS")
print("=" * 60)
rerun = False  # True to recompute
# Use centralized cache utility
results = load_backimage_fixation_results(
    model=model,
    sessions=sessions,
    n_images=27,
    force_recompute=rerun,
)

# %% Visualize results
# Use centralized image cache builder
image_cache = load_backimage_image_cache(
    model=model,
    sessions=sessions,
    results=results,
)

# --- Plotting ---
n_images = len(results)
sx = int(np.sqrt(n_images))
sy = int(np.ceil(n_images / sx))

fig, axes = plt.subplots(sy, sx, figsize=(4 * sx, 5 * sy))
axes = axes.flatten()

for idx, (image_file, data) in enumerate(sorted(results.items(), key=lambda x: -x[1]['n_trials'])):
    ax = axes[idx]
    eyepos = data['eyepos']
    n_trials = data['n_trials']
    n_sessions = data['n_sessions']

    image = image_cache.get(image_file, None)
    if image is None:
        ax.text(0.5, 0.5, f'Image not cached\n{image_file}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{image_file}\n{n_trials} trials', fontsize=10)
        ax.axis('off')
        continue

    # Display image with fixation heatmap
    img_height, img_width = image.shape
    img_height_deg = img_height / 37.5  # ppd (display only)
    img_width_deg = img_width / 37.5

    extent_h = img_height_deg / 2
    extent_w = img_width_deg / 2

    ax.imshow(image, cmap='gray', origin='upper',
              extent=[-extent_w, extent_w, -extent_h, extent_h])

    # Fixation heatmap
    h, xedges, yedges = np.histogram2d(
        eyepos[:, 0], eyepos[:, 1],
        bins=50,
        range=[[-extent_w, extent_w], [-extent_h, extent_h]]
    )

    ax.imshow(h.T, extent=[-extent_w, extent_w, -extent_h, extent_h],
              origin='lower', cmap='hot', aspect='auto', alpha=0.6)

    ax.set_title(
        f'{image_file}\n{n_trials} trials, {n_sessions} sessions, {len(eyepos)} samples',
        fontsize=10
    )
    ax.set_xlabel('X (deg)')
    ax.set_ylabel('Y (deg)')

# Hide unused subplots
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Fixation Heatmaps - {len(results)} Images Across {len(sessions[:3])} Sessions',
             fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('../figures/backimage_fixation_heatmaps_multi_session.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Saved figure to ../figures/backimage_fixation_heatmaps_multi_session.png")
# %% Example usage - multiple sessions (optional)
# saccades_all, eyepos_fix_all, eyepos_sacc_all, n_trials_all, n_sess, image_file = \
#     get_saccades_and_fixations_across_sessions(model, sessions, image_file='Waterfall.jpg')

# %% TESTING SINGLE IMAGE HYBRID EYE TRACE GENERATION
# Checkpoint, we should be able to rerun analysis from here









#


# %% -----------------------------------------------------------------------------------
 # SINGLE NATURAL IMAGE HYBRID EYE TRACE GENERATION
 # methods to create hybrid eye traces with real fixational eye movements
 # from RSVP data and real saccade targets from backimage fixation data
 # to manipulate saccade frequency while keeping realistic eye movement statistics
 # -----------------------------------------------------------------------------------
# %% Load backimage fixation data
print("=" * 60)
print("HYBRID EYE TRACE ANALYSIS")
print("=" * 60)

# Load backimage fixation results via cache
backimage_results = load_backimage_fixation_results(
    model=model,
    sessions=sessions,
)

# Pick the image with most trials
image_file = max(backimage_results.items(), key=lambda x: x[1]['n_trials'])[0]
fixation_eyepos = backimage_results[image_file]['eyepos']

print(f"\nUsing image: {image_file}")
print(f"  Fixation samples available: {len(fixation_eyepos)}")





# %% -----------------------------------------------------------------------------------
# Workhouse function to create hybrid eye trace
#-----------------------------------------------------------------------------------
def create_hybrid_eye_trace(
    fixation_pool,
    saccade_targets,
    n_saccades=5,
    saccade_duration=6,
    total_duration=540,
    eye_scale=1.0,
    max_trim_frames=20,
    max_attempts=50,
    start_center=None,
    saccade_targets_seq=None,
    micro_bridge_frames=6    # short smooth bridge between stitched fixation bouts
):
    """
    Create a hybrid eye trace:
    - Fixations: sample real RSVP fixation bouts, zero-mean them, scale jitter by eye_scale,
      and re-center on the current fixation location.
    - Saccades: ballistic jumps to given targets (not scaled).
    - If a single bout cannot meet the target fixation length, stitch multiple bouts
      end-to-end with a short smooth bridge (microsaccade) between them, staying centered.

    Returns (hybrid_trace, saccade_mask, saccade_times, plan).
    """
    rng = np.random.default_rng()

    # Initial center
    if start_center is None:
        current_center = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)
    else:
        current_center = np.asarray(start_center, dtype=np.float32)
    initial_center = current_center.copy()

    # Precompute bout lengths
    bout_lengths = np.array([len(b) for b in fixation_pool], dtype=np.int32)

    def trim_bout(bout: np.ndarray, target_len: int) -> np.ndarray | None:
        """Allow small trimming from a longer bout to match target_len."""
        L = len(bout)
        if L == target_len:
            return bout
        if L > target_len and (L - target_len) <= max_trim_frames:
            start = rng.integers(0, L - target_len + 1)
            return bout[start:start + target_len]
        return None

    def compose_fixation_sequence(target_len: int) -> np.ndarray:
        """
        Build a fixation sequence of exact target_len by stitching multiple bouts.
        Each bout is zero-meaned, scaled by eye_scale, and re-centered at current_center.
        Insert a short smooth bridge between consecutive bouts to avoid jumps.
        """
        residual = int(target_len)
        parts = []
        sacc_mask_parts = []
        prev_last = None

        while residual > 0:
            # Reserve bridge frames if we already placed a segment
            avail_for_seg = residual if prev_last is None else max(residual - micro_bridge_frames, 1)

            # Rank candidates by |length - avail_for_seg|
            length_diff = np.abs(bout_lengths - avail_for_seg)
            ranked = np.argsort(length_diff)
            seg = None

            # Try top-K near candidates with trimming if needed
            k = min(50, len(ranked))
            for idx in rng.permutation(ranked[:k]):
                bout = fixation_pool[idx]
                if len(bout) == avail_for_seg:
                    seg = bout
                    break
                seg_try = trim_bout(bout, avail_for_seg)
                if seg_try is not None:
                    seg = seg_try
                    break

            # Fallback: random bouts + trimming
            if seg is None:
                for _ in range(max_attempts):
                    bout = fixation_pool[rng.integers(0, len(fixation_pool))]
                    if len(bout) == avail_for_seg:
                        seg = bout; break
                    seg_try = trim_bout(bout, avail_for_seg)
                    if seg_try is not None:
                        seg = seg_try; break

            # Last resort: take any bout and clamp to avail_for_seg
            if seg is None:
                bout = fixation_pool[rng.integers(0, len(fixation_pool))]
                if len(bout) >= avail_for_seg:
                    start = rng.integers(0, len(bout) - avail_for_seg + 1)
                    seg = bout[start:start + avail_for_seg]
                else:
                    # If even the shortest is too short, just take it and we’ll continue stitching
                    seg = bout

            # Zero-mean jitter -> scale -> re-center
            seg = seg.astype(np.float32)
            jitter = seg - seg.mean(axis=0, keepdims=True)
            seg_centered = current_center[None, :] + jitter * float(eye_scale)

            # If there is a previous segment, insert a short smooth bridge
            if prev_last is not None and residual > 0:
                bridge_len = min(micro_bridge_frames, residual)
                bridge = np.linspace(prev_last, seg_centered[0], num=bridge_len, dtype=np.float32)
                parts.append(bridge)
                sacc_mask_parts.append(np.zeros(bridge_len, dtype=bool))
                residual -= bridge_len

            # Append segment, truncating if necessary to respect residual
            take = min(len(seg_centered), residual)
            parts.append(seg_centered[:take])
            sacc_mask_parts.append(np.zeros(take, dtype=bool))
            residual -= take
            prev_last = parts[-1][-1]

        return np.vstack(parts), np.concatenate(sacc_mask_parts)

    hybrid_parts = []
    saccade_mask_parts = []
    saccade_times = []
    frame_idx = 0

    # Prepare saccade plan (for null matching)
    used_targets = []
    if saccade_targets_seq is not None:
        saccade_targets_seq = np.asarray(saccade_targets_seq, dtype=np.float32)
        assert len(saccade_targets_seq) == n_saccades, "saccade_targets_seq length must match n_saccades"

    # Compute per-fixation target length
    saccade_frames_total = int(n_saccades) * int(saccade_duration)
    fixation_frames_total = int(total_duration) - saccade_frames_total
    frames_per_fixation = max(10, fixation_frames_total // (n_saccades + 1))

    for i_fix in range(n_saccades + 1):
        remaining = total_duration - frame_idx
        reserve = (saccade_duration if i_fix < n_saccades else 0)
        target_len = int(min(frames_per_fixation, max(0, remaining - reserve)))
        if target_len <= 0:
            break

        # Compose fixation from one or more bouts with bridges (prefer single bout; stitch if needed)
        fix_seg, fix_mask = compose_fixation_sequence(target_len)
        hybrid_parts.append(fix_seg)
        saccade_mask_parts.append(fix_mask)
        frame_idx += len(fix_seg)

        # Insert saccade (not scaled) and update center
        if i_fix < n_saccades and frame_idx < total_duration:
            n_sacc_frames = int(min(saccade_duration, total_duration - frame_idx))
            if saccade_targets_seq is not None:
                sacc_target = saccade_targets_seq[i_fix].astype(np.float32)
            else:
                sacc_target = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)
                used_targets.append(sacc_target)

            start_pos = hybrid_parts[-1][-1]
            end_pos = sacc_target
            saccade_path = np.linspace(start_pos, end_pos, num=n_sacc_frames, dtype=np.float32)

            hybrid_parts.append(saccade_path)
            saccade_mask_parts.append(np.ones(n_sacc_frames, dtype=bool))
            saccade_times.append((frame_idx, frame_idx + n_sacc_frames))
            frame_idx += n_sacc_frames
            current_center = end_pos

    # Concatenate and clamp to total_duration
    hybrid_trace = np.vstack(hybrid_parts)
    saccade_mask = np.concatenate(saccade_mask_parts)
    if len(hybrid_trace) > total_duration:
        hybrid_trace = hybrid_trace[:total_duration]
        saccade_mask = saccade_mask[:total_duration]

    plan = {
        'start_center': initial_center,
        'saccade_targets_seq': (saccade_targets_seq if saccade_targets_seq is not None
                                else (np.stack(used_targets) if used_targets else np.empty((0,2), dtype=np.float32)))
    }
    return hybrid_trace.astype(np.float32), saccade_mask, saccade_times, plan

# %% Extract all fixation periods from FIXRSVP data into a pool -----
# %% Helper: build fixation_pool from FIXRSVP with correct units (deg) and contiguous bouts
def build_fixation_pool_from_fixrsvp(model, sessions, ppd=37.50476617, min_fix_frames=20, amp_thresh_deg=1.0):
    """
    Returns a list of contiguous fixation bouts (np.array[T,2] in degrees),
    extracted from FIXRSVP data without time compression.
    Detects pixel units and converts to deg by dividing by ppd.
    """
    fixation_pool = []

    for name in sessions:
        try:
            dataset_idx = model.names.index(name)
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)

            inds = torch.concatenate([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp')
            ], dim=0)
            if len(inds) == 0:
                continue

            dataset = train_data.shallow_copy()
            dataset.inds = inds
            dset_idx = inds[:,0].unique().item()

            # Original timeline eyepos for this slice
            eyepos_all = dataset.dsets[dset_idx]['eyepos'][:].numpy()  # (N_samples, 2)
            trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
            trials = np.unique(trial_inds[~np.isnan(trial_inds)])

            # Detect unit: pixels vs degrees. If typical magnitude >> 1, treat as pixels.
            median_amp = np.nanmedian(np.hypot(eyepos_all[:,0], eyepos_all[:,1]))
            if median_amp > 5.0:
                print(f"[{name}] eyepos appears in pixels (median={median_amp:.2f}); converting to degrees (/ppd={ppd:.2f})")
                eyepos_all = eyepos_all / ppd

            # Fixation mask in degrees
            fixation_mask_all = (np.hypot(eyepos_all[:,0], eyepos_all[:,1]) < amp_thresh_deg)

            # Extract contiguous fixation bouts per trial
            for t in trials:
                trial_mask = (trial_inds == t)
                trial_trace = eyepos_all[trial_mask]               # (T_trial, 2)
                trial_fix = fixation_mask_all[trial_mask].astype(bool)

                idx_true = np.where(trial_fix)[0]
                if idx_true.size == 0:
                    continue
                split_pts = np.where(np.diff(idx_true) != 1)[0] + 1
                runs = np.split(idx_true, split_pts)

                for run in runs:
                    if run.size >= min_fix_frames:
                        bout = trial_trace[run]                    # (T_bout, 2) deg
                        fixation_pool.append(bout)

        except Exception as e:
            print(f"Failed to load FIXRSVP from session {name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("FIXATION POOL (contiguous bouts, degrees) SUMMARY")
    print("="*60)
    if len(fixation_pool):
        lengths = [len(b) for b in fixation_pool]
        amps = [np.linalg.norm(b - b.mean(0), axis=1).max() for b in fixation_pool]
        print(f"  N bouts: {len(fixation_pool)}")
        print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, med={np.median(lengths):.1f}")
        print(f"  Max within-bout amp (deg): p50={np.percentile(amps,50):.2f}, p90={np.percentile(amps,90):.2f}, p99={np.percentile(amps,99):.2f}")
    else:
        print("  WARNING: fixation_pool is empty")
    return fixation_pool



# %% Extract all fixation periods from FIXRSVP data into a pool -----
print("\n" + "=" * 60)
print("BUILDING FIXATION POOL FROM FIXRSVP DATA (contiguous bouts, degrees)")
print("=" * 60)
rerun = False
fixation_pool = load_fixrsvp_fixation_pool(
    model=model,
    sessions=sessions,
    min_fix_frames=20,
    amp_thresh_deg=1.0,
    force_recompute=rerun,
)


# %% Load backimage fixation data and ensure degrees
print("=" * 60)
print("HYBRID EYE TRACE ANALYSIS")
print("=" * 60)

backimage_results = load_backimage_fixation_results(
    model=model,
    sessions=sessions,
)

# Pick the image with most trials
image_file = max(backimage_results.items(), key=lambda x: x[1]['n_trials'])[0]
fixation_eyepos = backimage_results[image_file]['eyepos'].astype(np.float32)

# Heuristic unit check for backimage eyepos; convert to degrees if needed
median_amp_bi = np.nanmedian(np.hypot(fixation_eyepos[:,0], fixation_eyepos[:,1]))
if median_amp_bi > 5.0:
    print(f"[BackImage:{image_file}] eyepos appears in pixels (median={median_amp_bi:.2f}); converting to degrees (/ppd={ppd:.2f})")
    fixation_eyepos = fixation_eyepos / ppd

print(f"\nUsing image: {image_file}")
print(f"  Fixation samples available: {len(fixation_eyepos)}")



# %% Test hybrid eye trace generation with the fixation pool
print("\n" + "=" * 60)
print("TESTING HYBRID EYE TRACE GENERATION")
print("=" * 60)
n_frames = 540
example_saccade_rates = [0, 2, 4]
example_eye_scales = [0.0, 1.0]  # visualize null vs real

# Add one extra column for a zoomed-in time series (column 4)
fig, axes = plt.subplots(len(example_saccade_rates), len(example_eye_scales) + 2,
                         figsize=(18, 10))

def get_fixation_windows(saccade_mask: np.ndarray):
    idx = np.where(~saccade_mask)[0]  # fixation frames
    if idx.size == 0:
        return []
    split_pts = np.where(np.diff(idx) != 1)[0] + 1
    runs = np.split(idx, split_pts)
    return [(r[0], r[-1] + 1) for r in runs]  # [start, end)

y_zoom = 0.5  # deg for zoomed time series

for i_sacc, sacc_rate in enumerate(example_saccade_rates):
    n_sacc = int(sacc_rate * (n_frames / 120))

    # Column indices
    col_scatter_0 = 0
    col_scatter_1 = 1
    col_timeseries = len(example_eye_scales)        # 3rd column
    col_zoom_timeseries = len(example_eye_scales) + 1  # 4th column

    # Scatter columns: real vs null for each eye_scale
    for i_eye, eye_scale in enumerate(example_eye_scales):
        # Real trace: scaled FEMs and re-centered at saccade landings
        real_trace, real_mask, real_sacc_times, plan = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=eye_scale
        )

        # Null trace: same saccade plan, no FEM jitter
        null_trace, null_mask, null_sacc_times, _ = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=0.0,
            start_center=plan['start_center'],
            saccade_targets_seq=plan['saccade_targets_seq']
        )

        # Plot 2D trajectory (overlay real vs null) in columns 1 and 2
        ax = axes[i_sacc, i_eye]  # i_eye = 0 or 1
        ax.scatter(real_trace[:, 0], real_trace[:, 1],
                   c=np.arange(len(real_trace)), cmap='viridis', s=2, alpha=0.6, label='real')
        ax.scatter(null_trace[:, 0], null_trace[:, 1],
                   c='gray', s=2, alpha=0.4, label='null')

        # Mark saccades (same for real and null)
        for s0, s1 in real_sacc_times:
            ax.plot(real_trace[s0:s1, 0], real_trace[s0:s1, 1], 'r-', lw=2, alpha=0.9)

        ax.set_title(f'Sacc={sacc_rate}Hz, Scale={eye_scale:.1f}x\n({len(real_sacc_times)} saccades)',
                     fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)

    # Column 3: full time series (real vs null) for both eye scales
    ax_ts = axes[i_sacc, col_timeseries]
    for eye_scale in example_eye_scales:
        real_trace, real_mask, real_sacc_times, plan = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=eye_scale
        )
        null_trace, null_mask, null_sacc_times, _ = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=0.0,
            start_center=plan['start_center'],
            saccade_targets_seq=plan['saccade_targets_seq']
        )

        ax_ts.plot(real_trace[:, 0], label=f'Real X ({eye_scale:.1f}x)', alpha=0.8, lw=1.2)
        ax_ts.plot(real_trace[:, 1], label=f'Real Y ({eye_scale:.1f}x)', alpha=0.8, lw=1.2, linestyle='-')
        ax_ts.plot(null_trace[:, 0], label='Null X', alpha=0.6, lw=1.0, linestyle='--', color='gray')
        ax_ts.plot(null_trace[:, 1], label='Null Y', alpha=0.6, lw=1.0, linestyle='--', color='gray')

    ax_ts.set_title(f'Time Series (Sacc={sacc_rate}Hz)', fontsize=10)
    ax_ts.set_xlabel('Frame')
    ax_ts.set_ylabel('Eye Position (deg)')
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    # Column 4: zoomed time series around longest fixation window (X only), axis limits only
    ax_zoomts = axes[i_sacc, col_zoom_timeseries]
    # Generate a single real/null pair at eye_scale=1.0 for zoom
    real_trace, real_mask, real_sacc_times, plan = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=1.0
    )
    null_trace, null_mask, _, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=0.0,
        start_center=plan['start_center'],
        saccade_targets_seq=plan['saccade_targets_seq']
    )
    fix_windows = get_fixation_windows(real_mask)
    if len(fix_windows) > 0:
        lengths = [e - s for (s, e) in fix_windows]
        s, e = fix_windows[int(np.argmax(lengths))]
        center_x = real_trace[s:e, 0].mean()
        ax_zoomts.plot(np.arange(s, e), real_trace[s:e, 0], label='Real X', color='#1f77b4', lw=1.2)
        ax_zoomts.plot(np.arange(s, e), null_trace[s:e, 0], label='Null X', color='gray', lw=1.0, linestyle='--', alpha=0.8)
        ax_zoomts.set_xlim(s, e)
        ax_zoomts.set_ylim(center_x - y_zoom, center_x + y_zoom)
        ax_zoomts.set_title(f'Zoomed Time Series X (±{y_zoom} deg)\nSacc={sacc_rate}Hz', fontsize=10)
    else:
        ax_zoomts.plot(real_trace[:, 0], label='Real X', color='#1f77b4', lw=1.2)
        ax_zoomts.plot(null_trace[:, 0], label='Null X', color='gray', lw=1.0, linestyle='--', alpha=0.8)
        cx = real_trace[:, 0].mean()
        ax_zoomts.set_ylim(cx - y_zoom, cx + y_zoom)
        ax_zoomts.set_title(f'Zoomed Time Series X (±{y_zoom} deg)\n(no saccades)', fontsize=10)

    ax_zoomts.set_xlabel('Frame')
    ax_zoomts.set_ylabel('X (deg)')
    ax_zoomts.grid(True, alpha=0.3)
    ax_zoomts.legend(fontsize=8)

plt.suptitle('Hybrid Eye Trace Examples (RSVP fixations centered at saccade landings)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_examples_real_fixations.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved hybrid eye trace examples")


# %% Hybrid eye trace now looks ok, next step is to generate reconstructed stimulus
# from these eye traces and analyze the effect of saccade rate and eye scale
# on the reconstructed stimulus quality (compared to ground truth static image).
# We want to do this for natural images, and take into account that the saccades
# will move the eye to the interesting parts of the image based on the backimage fixation data.
# Its therefore very important to use the same image for reconstruction as was used
# in the backimage fixation data to get realistic results.

# -----------------------------------------------------------------------------------













# %% DEBUG: Natural image eye traces & reconstructed stimulus (NO VIDEO)

print("\n" + "=" * 60)
print("DEBUG: NATURAL IMAGE (STATIC) EYE TRACE VISUALIZATION")
print("=" * 60)

# Use a stimulus stack built from the SAME BackImage used to draw fixation targets.
# This ensures saccade targets are matched to the natural image content.
n_frames = 540
max_T = 600
n_lags = 32
stim_len = max_T + n_lags + 1

image_cache = globals().get('image_cache', None)
if image_cache is None:
    image_cache = load_backimage_image_cache(
        model=model,
        sessions=sessions,
        results=backimage_results,
    )

backimage_image = image_cache.get(image_file)
if backimage_image is None:
    raise ValueError(f"Could not load backimage pixels for {image_file} from cache")

nat_full_stack = np.repeat(
    backimage_image[None, :, :].astype(np.float32),
    stim_len,
    axis=0,
)
print(f"Matched BackImage stack shape: {nat_full_stack.shape} (image_file={image_file})")

# Trajectory plots
example_saccade_rates = [0, 2, 4]  # Hz
example_eye_scales = [0.0, 1.0]

fig, axes = plt.subplots(len(example_saccade_rates), len(example_eye_scales) + 1,
                         figsize=(15, 10))

for i_sacc, sacc_rate in enumerate(example_saccade_rates):
    n_sacc = int(sacc_rate * (n_frames / 120))

    base_trace, base_mask, base_sacc_times, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,  # targets in degrees
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=1.0,
    )

    for i_eye, eye_scale in enumerate(example_eye_scales):
        hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)

        ax = axes[i_sacc, i_eye]
        ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
                   c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
        for sacc_start, sacc_end in base_sacc_times:
            ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
                    hybrid_trace_scaled[sacc_start:sacc_end, 1],
                    'r-', lw=2, alpha=0.8)
        ax.set_title(f'Sacc={sacc_rate}Hz, Scale={eye_scale:.1f}x\n({len(base_sacc_times)} saccades)',
                     fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)

    # time series for this saccade rate
    ax = axes[i_sacc, -1]
    for eye_scale in example_eye_scales:
        hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)
        ax.plot(hybrid_trace_scaled[:, 0], label=f'X (scale={eye_scale:.1f}x)', alpha=0.7, lw=1)
        ax.plot(hybrid_trace_scaled[:, 1] + 10, label=f'Y (scale={eye_scale:.1f}x)',
                alpha=0.7, lw=1, linestyle='--')

    ax.set_title(f'Time Series (Sacc={sacc_rate}Hz)', fontsize=10)
    ax.set_xlabel('Frame') 
    ax.set_ylabel('Eye Position (deg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hybrid Eye Trace Examples (Natural Image)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/nat_eye_trace_debug_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved trajectory visualization (natural image)")

# Reconstructed stimulus analysis (natural image, NO VIDEO)
print("\n" + "=" * 60)
print("DEBUG: NATURAL IMAGE RECONSTRUCTED STIMULUS ANALYSIS")
print("=" * 60)

conditions = [
    {'sacc_rate': 0, 'eye_scale': 0.0, 'label': 'nat_null_no_saccades_no_eye'},
    {'sacc_rate': 0, 'eye_scale': 1.0, 'label': 'nat_real_eye_no_saccades'},
    {'sacc_rate': 4, 'eye_scale': 1.0, 'label': 'nat_real_eye_4hz_saccades'},
]

# max_T and n_lags are defined above to ensure the stimulus stack is long enough.

for cond in conditions:
    sacc_rate = cond['sacc_rate']
    eye_scale = cond['eye_scale']
    label = cond['label']

    print(f"\nProcessing: {label}")
    n_sacc = int(sacc_rate * (n_frames / 120))

    base_trace, base_mask, sacc_times, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames
    )

    hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)[:n_frames]

    hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
    hybrid_padded[:len(hybrid_trace_scaled)] = hybrid_trace_scaled

    print(f"  Reconstructing stimulus...")
    y_real, _, eye_stim, _ = get_trial_stim_and_rates(
        hybrid_padded,
        nat_full_stack,          # use natural image stack
        out_size=(151, 151),
        n_lags=n_lags,
        scale=1.0,
        plot=False
    )

    frames = eye_stim[:, 0, -1, :, :].detach().cpu().numpy()
    print(f"  Frames shape: {frames.shape}")
    print(f"  Frame range: [{frames.min():.1f}, {frames.max():.1f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Mean reconstructed stimulus
    ax = axes[0]
    mean_stim = frames.mean(axis=0)
    im = ax.imshow(mean_stim, cmap='gray')
    ax.set_title(f'Mean Reconstructed Stimulus\n{label}', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Eye trajectory
    ax = axes[1]
    ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
               c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
    for sacc_start, sacc_end in sacc_times:
        ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
                hybrid_trace_scaled[sacc_start:sacc_end, 1],
                'r-', lw=2, alpha=0.8)
    ax.set_title('Eye Trajectory', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (deg)')
    ax.set_ylabel('Y (deg)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Sample frames (spatial profile)
    ax = axes[2]
    frame_indices = np.linspace(0, len(frames)-1, 5, dtype=int)
    for i, frame_idx in enumerate(frame_indices):
        ax.plot(frames[frame_idx], label=f'Frame {frame_idx}', alpha=0.7)
    ax.set_title('Sample Frames (Spatial Profile)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Space (pixels)')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Stimulus Analysis: {label}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/reconstructed_stim_analysis_{label}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Completed: {label}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE (Natural Image)")
print("=" * 60)
print("Generated artifacts:")
print("  - Trajectory visualization: nat_eye_trace_debug_trajectories.png")
print("  - Analysis plots: reconstructed_stim_analysis_nat_*.png")


#%%

# %% DEBUG: Zoomed-in eye trace examples (individual trials)

print("\n" + "=" * 60)
print("DEBUG: ZOOMED-IN EYE TRACE EXAMPLES (1-2 TRIALS)")
print("=" * 60)

n_frames = 540
example_conditions = [
    {'sacc_rate': 0, 'eye_scale': 1.0, 'label': 'real_eye_no_saccades'},
    {'sacc_rate': 4, 'eye_scale': 1.0, 'label': 'real_eye_4hz_saccades'},
]

for cond in example_conditions:
    sacc_rate = cond['sacc_rate']
    eye_scale = cond['eye_scale']
    label = cond['label']
    
    n_sacc = int(sacc_rate * (n_frames / 120))
    
    # Create ONE example trace
    base_trace, base_mask, sacc_times, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames
    )
    
    hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)
    
    print(f"\n{label}:")
    print(f"  Total duration: {len(hybrid_trace_scaled)} frames ({len(hybrid_trace_scaled)/120:.2f}s)")
    print(f"  Saccades: {len(sacc_times)}")
    print(f"  Eye position range: X=[{hybrid_trace_scaled[:,0].min():.2f}, {hybrid_trace_scaled[:,0].max():.2f}] deg")
    print(f"                      Y=[{hybrid_trace_scaled[:,1].min():.2f}, {hybrid_trace_scaled[:,1].max():.2f}] deg")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Full trial trajectory
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
               c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=2, alpha=0.6)
    for sacc_start, sacc_end in sacc_times:
        ax1.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
                hybrid_trace_scaled[sacc_start:sacc_end, 1],
                'r-', lw=2, alpha=0.8)
    ax1.set_title('Full Trial Trajectory', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X (deg)')
    ax1.set_ylabel('Y (deg)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Full trial time series
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(hybrid_trace_scaled[:, 0], label='X', alpha=0.7, lw=1)
    ax2.plot(hybrid_trace_scaled[:, 1], label='Y', alpha=0.7, lw=1)
    
    # Mark saccades
    for sacc_start, sacc_end in sacc_times:
        ax2.axvspan(sacc_start, sacc_end, alpha=0.2, color='red')
    
    ax2.set_title('Full Trial Time Series', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Eye Position (deg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Velocity profile
    ax3 = plt.subplot(3, 3, 3)
    velocity = np.sqrt(np.sum(np.diff(hybrid_trace_scaled, axis=0)**2, axis=1))
    ax3.plot(velocity, alpha=0.7, lw=1, color='purple')
    
    # Mark saccades
    for sacc_start, sacc_end in sacc_times:
        ax3.axvspan(sacc_start, sacc_end, alpha=0.2, color='red')
    
    ax3.set_title('Eye Velocity (deg/frame)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Velocity (deg/frame)')
    ax3.grid(True, alpha=0.3)
    
    # Panels 4-6: Zoom into first 3 saccades (or fixation period if no saccades)
    zoom_windows = []
    
    if len(sacc_times) > 0:
        # Create windows around first 3 saccades
        for i in range(min(3, len(sacc_times))):
            sacc_start, sacc_end = sacc_times[i]
            # Expand window by 50 frames before and after
            window_start = max(0, sacc_start - 50)
            window_end = min(len(hybrid_trace_scaled), sacc_end + 50)
            zoom_windows.append((window_start, window_end, f'Saccade {i+1}'))
    else:
        # No saccades: zoom into 3 equal fixation periods
        window_size = len(hybrid_trace_scaled) // 3
        for i in range(3):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            zoom_windows.append((window_start, window_end, f'Fixation {i+1}'))
    
    for panel_idx, (window_start, window_end, title) in enumerate(zoom_windows):
        ax = plt.subplot(3, 3, 4 + panel_idx)
        
        window_trace = hybrid_trace_scaled[window_start:window_end]
        ax.scatter(window_trace[:, 0], window_trace[:, 1],
                  c=np.arange(len(window_trace)), cmap='cool', s=10, alpha=0.7)
        
        # Mark saccade if in this window
        for sacc_start, sacc_end in sacc_times:
            if sacc_start >= window_start and sacc_end <= window_end:
                sacc_start_local = sacc_start - window_start
                sacc_end_local = sacc_end - window_start
                ax.plot(window_trace[sacc_start_local:sacc_end_local, 0],
                       window_trace[sacc_start_local:sacc_end_local, 1],
                       'r-', lw=2.5, alpha=0.9)
        
        ax.set_title(f'{title} (frames {window_start}-{window_end})', fontsize=10, fontweight='bold')
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)
    
    # Panels 7-9: Zoomed time series
    for panel_idx, (window_start, window_end, title) in enumerate(zoom_windows):
        ax = plt.subplot(3, 3, 7 + panel_idx)
        
        window_frames = np.arange(window_start, window_end)
        window_trace = hybrid_trace_scaled[window_start:window_end]
        
        ax.plot(window_frames - window_start, window_trace[:, 0], label='X', alpha=0.7, lw=1.5)
        ax.plot(window_frames - window_start, window_trace[:, 1], label='Y', alpha=0.7, lw=1.5)
        
        # Mark saccade
        for sacc_start, sacc_end in sacc_times:
            if sacc_start >= window_start and sacc_end <= window_end:
                ax.axvspan(sacc_start - window_start, sacc_end - window_start, 
                          alpha=0.2, color='red', label='Saccade')
        
        ax.set_title(f'{title} Time Series', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frame (within window)')
        ax.set_ylabel('Eye Position (deg)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Zoomed Eye Trace Analysis: {label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/eye_trace_zoomed_{label}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved: eye_trace_zoomed_{label}.png")

print("\n" + "=" * 60)
print("DEBUG COMPLETE (Zoomed Eye Traces)")
print("=" * 60)
print("Generated artifacts:")
print("  - eye_trace_zoomed_real_eye_no_saccades.png")
print("  - eye_trace_zoomed_real_eye_4hz_saccades.png")


# %% DEBUG: Visualize hybrid eye traces and reconstructed stimulus

print("\n" + "=" * 60)
print("DEBUG: HYBRID EYE TRACE VISUALIZATION")
print("=" * 60)

# Create a few example hybrid traces with different saccade rates
n_frames = 540
example_saccade_rates = [0, 2, 4]  # Hz
example_eye_scales = [0.0, 1.0]

fig, axes = plt.subplots(len(example_saccade_rates), len(example_eye_scales) + 1, 
                         figsize=(15, 10))

for i_sacc, sacc_rate in enumerate(example_saccade_rates):
    n_sacc = int(sacc_rate * (n_frames / 120))

    base_trace, base_mask, sacc_times, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=1.0,
    )
    
    for i_eye, eye_scale in enumerate(example_eye_scales):
        hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)
        
        # Plot 2D trajectory
        ax = axes[i_sacc, i_eye]
        ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1], 
                  c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
        
        # Mark saccades in red
        for sacc_start, sacc_end in sacc_times:
            ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0], 
                   hybrid_trace_scaled[sacc_start:sacc_end, 1], 
                   'r-', lw=2, alpha=0.8)
        
        ax.set_title(f'Sacc={sacc_rate}Hz, Scale={eye_scale:.1f}x\n({len(sacc_times)} saccades)', 
                    fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)
    
    # Plot time series for this saccade rate
    ax = axes[i_sacc, -1]
    
    for i_eye, eye_scale in enumerate(example_eye_scales):
        hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)
        
        # Plot X and Y separately
        ax.plot(hybrid_trace_scaled[:, 0], label=f'X (scale={eye_scale:.1f}x)', 
               alpha=0.7, lw=1)
        ax.plot(hybrid_trace_scaled[:, 1] + 10, label=f'Y (scale={eye_scale:.1f}x)', 
               alpha=0.7, lw=1, linestyle='--')
    
    ax.set_title(f'Time Series (Sacc={sacc_rate}Hz)', fontsize=10)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Eye Position (deg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hybrid Eye Trace Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_debug_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved trajectory visualization")

 # %% DEBUG: Create reconstructed stimulus videos for different conditions

# print("\n" + "=" * 60)
# print("DEBUG: RECONSTRUCTED STIMULUS VIDEOS")
# print("=" * 60)

# import imageio.v2 as imageio

# # Create 3 example conditions
# conditions = [
#     {'sacc_rate': 0, 'eye_scale': 0.0, 'label': 'null_no_saccades_no_eye'},
#     {'sacc_rate': 0, 'eye_scale': 1.0, 'label': 'real_eye_no_saccades'},
#     {'sacc_rate': 4, 'eye_scale': 1.0, 'label': 'real_eye_4hz_saccades'},
# ]

# max_T = 600
# n_lags = 32

# for cond in conditions:
#     sacc_rate = cond['sacc_rate']
#     eye_scale = cond['eye_scale']
#     label = cond['label']
    
#     print(f"\nProcessing: {label}")
    
#     n_sacc = int(sacc_rate * (n_frames / 120))
    
#     # Create hybrid trace
#     hybrid_trace, _, sacc_times, _ = create_hybrid_eye_trace(
#         fixation_pool=fixation_pool,
#         saccade_targets=fixation_eyepos,
#         n_saccades=n_sacc,
#         saccade_duration=6,
#         total_duration=n_frames
#     )
    
#     # Apply eye scale
#     mean_pos = hybrid_trace.mean(axis=0)
#     hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
#     hybrid_trace_scaled = hybrid_trace_scaled[:n_frames]
    
#     # Pad with NaNs
#     hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
#     hybrid_padded[:len(hybrid_trace_scaled)] = hybrid_trace_scaled
    
#     # Get reconstructed stimulus
#     print(f"  Reconstructing stimulus...")
#     y_real, _, eye_stim, _ = get_trial_stim_and_rates(
#         hybrid_padded,
#         full_stack,
#         out_size=(151, 151),
#         n_lags=n_lags,
#         scale=1.0,
#         plot=False
#     )
    
#     # Extract the reconstructed stimulus movie
#     # eye_stim shape: (T, 1, n_lags, H, W)
#     # We'll use the last lag (most recent) for visualization
#     frames = eye_stim[:, 0, -1, :, :].detach().cpu().numpy()
    
#     print(f"  Frames shape: {frames.shape}")
#     print(f"  Frame range: [{frames.min():.1f}, {frames.max():.1f}]")
    
#     # Normalize to 0-255
#     frames_norm = ((frames - frames.min()) / (frames.max() - frames.min() + 1e-6) * 255).astype(np.uint8)
    
#     # Save video
#     output_video = f'../figures/reconstructed_stim_{label}.mp4'
#     print(f"  Saving video to {output_video}...")
#     imageio.mimsave(output_video, frames_norm, fps=30, format='FFMPEG')
#     print(f"  ✓ Saved")
    
#     # Also create a static frame showing eye position overlay
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Panel 1: Mean reconstructed stimulus
#     ax = axes[0]
#     mean_stim = frames.mean(axis=0)
#     im = ax.imshow(mean_stim, cmap='gray')
#     ax.set_title(f'Mean Reconstructed Stimulus\n{label}', fontsize=11, fontweight='bold')
#     ax.axis('off')
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
#     # Panel 2: Eye position trajectory
#     ax = axes[1]
#     ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
#               c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
    
#     # Mark saccades
#     for sacc_start, sacc_end in sacc_times:
#         ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
#                hybrid_trace_scaled[sacc_start:sacc_end, 1],
#                'r-', lw=2, alpha=0.8)
    
#     ax.set_title('Eye Trajectory', fontsize=11, fontweight='bold')
#     ax.set_xlabel('X (deg)')
#     ax.set_ylabel('Y (deg)')
#     ax.grid(True, alpha=0.3)
    
#     # Panel 3: Sample frames over time
#     ax = axes[2]
#     frame_indices = np.linspace(0, len(frames)-1, 5, dtype=int)
#     for i, frame_idx in enumerate(frame_indices):
#         ax.plot(frames[frame_idx], label=f'Frame {frame_idx}', alpha=0.7)
    
#     ax.set_title('Sample Frames (Spatial Profile)', fontsize=11, fontweight='bold')
#     ax.set_xlabel('Space (pixels)')
#     ax.set_ylabel('Intensity')
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)
    
#     plt.suptitle(f'Stimulus Analysis: {label}', fontsize=12, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'../figures/reconstructed_stim_analysis_{label}.png', dpi=150, bbox_inches='tight')
#     plt.show()
    
#     print(f"✓ Completed: {label}")

# print("\n" + "=" * 60)
# print("DEBUG COMPLETE")
# print("=" * 60)
# print("Generated videos and analysis plots:")
# print("  - Trajectory visualization: hybrid_eye_trace_debug_trajectories.png")
# print("  - Stimulus videos: reconstructed_stim_*.mp4")
# print("  - Analysis plots: reconstructed_stim_analysis_*.png")





# %% Create stimulus - STATIC single image (no flashing)
# For a single static image, use frames_per_im = total_frames
# This way the same image is shown for the entire duration
n_frames = 540
max_T = 600
n_lags = 32
stim_len = max_T + n_lags + 1

image_cache = globals().get('image_cache', None)
if image_cache is None:
    image_cache = load_backimage_image_cache(
        model=model,
        sessions=sessions,
        results=backimage_results,
    )

backimage_image = image_cache.get(image_file)
if backimage_image is None:
    raise ValueError(f"Could not load backimage pixels for {image_file} from cache")

full_stack = np.repeat(
    backimage_image[None, :, :].astype(np.float32),
    stim_len,
    axis=0,
)

print(f"Stimulus stack shape: {full_stack.shape}")
print(f"(Single matched BackImage for entire trial: {image_file})")

# %% Minimal example: Single image, eye_scale=1.0, varying saccade frequencies
saccade_rates = [0, 2, 4, 8]  # Number of saccades per second (over 4.5s trial)
n_frames = 540
max_T = 600  # Pad to this length (like in your earlier code)

results_hybrid = {
    'saccade_rates': saccade_rates,
    'i_spikes': [],
    'i_rates': [],
    'I_t': [],
}

print("\n" + "=" * 60)
print("ANALYSIS: Saccade Frequency vs Spatial Information")
print("(Single static image, eye_scale=1.0)")
print("=" * 60)

for sacc_rate in saccade_rates:
    
    n_sacc = int(sacc_rate * (n_frames / 120))  # Convert to number of saccades per trial
    print(f"\nSaccade Rate: {sacc_rate} Hz, approximated to ({120*n_sacc/n_frames:.2f} Hz)")
    # Create hybrid trace using backimage saccade targets + backimage fixations
    hybrid_trace, _, sacc_times, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,  # ~50ms at 120 Hz
        total_duration=n_frames
    )
    
    # Pad to max_T with NaNs (required by get_trial_stim_and_rates)
    hybrid_padded = np.full((max_T, 2), np.nan)
    hybrid_padded[:len(hybrid_trace)] = hybrid_trace
    
    print(f"  Hybrid trace: {len(hybrid_trace)} valid frames, padded to {max_T}")
    print(f"  Saccades inserted: {len(sacc_times)}")
    
    # Compute rates (real eye trace with saccades)
    y_real, _, _, _ = get_trial_stim_and_rates(
        hybrid_padded,
        full_stack,
        out_size=(151, 151),
        n_lags=32,
        scale=1.0,
        plot=False
    )
    
    # Compute rates (null eye trace - central fixation only)
    null_trace = np.full((max_T, 2), np.nan)
    null_trace[:len(hybrid_trace), :] = 0.0  # Central fixation at (0, 0)
    
    y_null, _, _, _ = get_trial_stim_and_rates(
        null_trace,
        full_stack,
        out_size=(151, 151),
        n_lags=32,
        scale=1.0,
        plot=False
    )
    
    # Compute spatial info
    ispike_real, irate_real, I_t_real = spatial_ssi_population(y_real)
    ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)
    
    # Average over time to get scalar values
    ispike_real_val = ispike_real.mean().item() if isinstance(ispike_real, torch.Tensor) else ispike_real.mean()
    irate_real_val = irate_real.mean().item() if isinstance(irate_real, torch.Tensor) else irate_real.mean()
    ispike_null_val = ispike_null.mean().item() if isinstance(ispike_null, torch.Tensor) else ispike_null.mean()
    irate_null_val = irate_null.mean().item() if isinstance(irate_null, torch.Tensor) else irate_null.mean()
    
    results_hybrid['i_spikes'].append(ispike_real_val)
    results_hybrid['i_rates'].append(irate_real_val)
    results_hybrid['I_t'].append(I_t_real.mean().item() if isinstance(I_t_real, torch.Tensor) else I_t_real.mean())
    
    gain_spike = ispike_real_val / ispike_null_val if ispike_null_val > 0 else 0
    gain_rate = irate_real_val / irate_null_val if irate_null_val > 0 else 0
    
    print(f"  Spatial Info (bits/spike): real={ispike_real_val:.4f}, null={ispike_null_val:.4f}, gain={gain_spike:.2f}x")
    print(f"  Spatial Info (bits/sec): real={irate_real_val:.4f}, null={irate_null_val:.4f}, gain={gain_rate:.2f}x")
# %% Visualize results
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Bits per spike
ax = axes[0]
ax.plot(saccade_rates, results_hybrid['i_spikes'], 'o-', lw=2, ms=10, color='steelblue', label='Real eye trace')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Null (central fixation)')
ax.set_xlabel('Saccade Rate (Hz)', fontsize=12)
ax.set_ylabel('Spatial Information (bits/spike)', fontsize=12)
ax.set_title('Spatial Information vs Saccade Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Bits per second
ax = axes[1]
ax.plot(saccade_rates, results_hybrid['i_rates'], 's-', lw=2, ms=10, color='coral', label='Real eye trace')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Null (central fixation)')
ax.set_xlabel('Saccade Rate (Hz)', fontsize=12)
ax.set_ylabel('Spatial Information (bits/sec)', fontsize=12)
ax.set_title('Information Rate vs Saccade Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_saccade_frequency.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Saved figure to ../figures/hybrid_eye_trace_saccade_frequency.png")

# %% Save results
output_path = '../declan/hybrid_eye_trace_results.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results_hybrid, f)

print(f"✓ Saved results to {output_path}")

# %% -----------------------------------------------------------------------------------
#Full loop over eye scales and saccade rates
# % -----------------------------------------------------------------------------------
# % Full loop over eye scales and saccade rates
# %% Load backimage fixation data and get the actual image
print("=" * 60)
print("HYBRID EYE TRACE ANALYSIS - FULL PARAMETER SWEEP")
print("=" * 60)

# Load the precomputed backimage fixation results
with open('../declan/backimage_fixation_results.pkl', 'rb') as f:
    backimage_results = pickle.load(f)

# Pick the image with most trials
image_file = max(backimage_results.items(), key=lambda x: x[1]['n_trials'])[0]
fixation_eyepos = backimage_results[image_file]['eyepos']

print(f"\nUsing image: {image_file}")
print(f"  Fixation samples available: {len(fixation_eyepos)}")

# %% Load the actual backimage to use as stimulus via image cache
image_cache = load_backimage_image_cache(
    model=model,
    sessions=sessions,
    results=backimage_results,
)
backimage_image = image_cache.get(image_file)
if backimage_image is None:
    raise ValueError(f"Could not load image {image_file} from cache")

# Sanity check that degrees align with image coordinate assumptions
debug_eye_units_and_bounds(fixation_eyepos, img_hw=backimage_image.shape, ppd=ppd)


# %% Create stimulus stack from the actual backimage (NumPy, not torch)
# backimage_image is already a NumPy array
n_frames = 540
n_lags = 32  # must match the call to get_trial_stim_and_rates

# # Create stimulus stack from the actual backimage (NumPy), long enough for T+n_lags
full_stack = np.repeat(
    backimage_image[None, :, :].astype(np.float32), #backimage_image[np.newaxis, np.newaxis, ...].astype(np.float32),
    n_frames + n_lags,  # 540 + 32 = 572
    axis=0
)
img_height_deg = backimage_image.shape[0] / ppd  # assuming 30 pixels/deg
img_width_deg = backimage_image.shape[1] / ppd   # assuming 30 pixels/deg

print(f"Full stimulus stack shape: {full_stack.shape}")
print(f"(Single actual backimage for entire trial)")

# %% Parameter ranges
saccade_rates = [0, 0.25, 0.5, 1, 2, 4, 8, 16 ,32]  # Hz
# eye_scale_list = [0.0, 0.25, 0.5, 1.0, 2.0]  # Scale factor for fixational eye movements
#         frames_per_im_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
#         #frames_per_im_list = [1.88, 3.75, 7.5, 15, 30, 60]
eye_scale_list = [0.0] + list(np.exp(np.linspace(-2.75, np.log(2), 11))) # odd to include 1.0 in the middle

n_frames = 540
max_T = 600

# Storage
results_full = {
    'saccade_rates': saccade_rates,
    'eye_scale_list': eye_scale_list,
    'i_spikes': [],  # Will be (n_saccade_rates, n_eye_scales)
    'i_rates': [],   # Will be (n_saccade_rates, n_eye_scales)
    'I_t': [],       # Will be (n_saccade_rates, n_eye_scales)
    'image_file': image_file,
}

print("\n" + "=" * 60)
print("PARAMETER SWEEP: Saccade Rate × Eye Scale")
print(f"Using actual backimage: {image_file}")
print("=" * 60)

#%%
import tqdm as tqdm
for i_sacc, sacc_rate in enumerate(tqdm.tqdm(saccade_rates, desc="Saccade Rates")):
    i_spikes_eye_scale = []
    i_rates_eye_scale = []
    I_t_eye_scale = []

    n_sacc = int(sacc_rate * (n_frames / 120))

    # Generate a single "base" hybrid trace for this saccade rate.
    # We will rescale ONLY the fixation segments for each eye_scale.
    base_trace, base_mask, sacc_times, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=1.0,
    )

    for i_eye, eye_scale in enumerate(eye_scale_list):
        hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)[:n_frames]

        # Pad with NaNs (contract of get_trial_stim_and_rates)
        hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
        hybrid_padded[:len(hybrid_trace_scaled)] = hybrid_trace_scaled.astype(np.float32)

        if i_sacc == 0 and i_eye == 0:
            print(f"\n=== DEBUGGING INPUTS ===")
            print(f"full_stack shape: {full_stack.shape}")
            print(f"full_stack dtype: {full_stack.dtype}")
            print(
                f"full_stack min: {full_stack.min():.4f}, max: {full_stack.max():.4f}, mean: {full_stack.mean():.4f}"
            )
            print(f"full_stack contains NaN: {np.isnan(full_stack).any()}")
            print(f"hybrid_padded shape: {hybrid_padded.shape}")
            print(f"hybrid_padded valid samples: {np.sum(~np.isnan(hybrid_padded[:, 0]))}")
            print(
                f"Eye position range: X=[{hybrid_trace_scaled[:, 0].min():.2f}, {hybrid_trace_scaled[:, 0].max():.2f}], "
                f"Y=[{hybrid_trace_scaled[:, 1].min():.2f}, {hybrid_trace_scaled[:, 1].max():.2f}]"
            )
            print(f"Image dimensions (deg): {img_height_deg:.2f} x {img_width_deg:.2f}")

        y_real, _, eye_stim, _ = get_trial_stim_and_rates(
            hybrid_padded,
            full_stack,
            out_size=(151, 151),
            n_lags=n_lags,
            scale=1.0,
            plot=False,
        )

        if i_sacc == 0 and i_eye == 0:
            print(f"\n=== DEBUGGING RECONSTRUCTED STIMULUS ===")
            print(f"eye_stim shape: {eye_stim.shape}")
            has_nan = torch.isnan(eye_stim).any().item()
            if has_nan:
                print("eye_stim contains NaN: True")
            else:
                print(
                    f"eye_stim min: {eye_stim.min().item():.4f}, max: {eye_stim.max().item():.4f}, mean: {eye_stim.mean().item():.4f}"
                )
                print("eye_stim contains NaN: False")

        ispike_real, irate_real, I_t_real = spatial_ssi_population(y_real)
        ispike_real_val = float(ispike_real.mean().item())
        irate_real_val = float(irate_real.mean().item())
        I_t_real_val = float(I_t_real.mean().item())

        i_spikes_eye_scale.append(ispike_real_val)
        i_rates_eye_scale.append(irate_real_val)
        I_t_eye_scale.append(I_t_real_val)

        if i_eye == 0 or i_eye == len(eye_scale_list) - 1:
            print(
                f"  Sacc={sacc_rate}Hz, EyeScale={eye_scale:.2f}: bits/spike={ispike_real_val:.4f}, bits/sec={irate_real_val:.4f}"
            )

    # Store per-saccade-rate row
    results_full['i_spikes'].append(i_spikes_eye_scale)
    results_full['i_rates'].append(i_rates_eye_scale)
    results_full['I_t'].append(I_t_eye_scale)

# Convert to arrays for plotting/indexing
results_full['i_spikes'] = np.asarray(results_full['i_spikes'], dtype=np.float32)
results_full['i_rates'] = np.asarray(results_full['i_rates'], dtype=np.float32)
results_full['I_t'] = np.asarray(results_full['I_t'], dtype=np.float32)

# %% Save results
output_path = '../declan/hybrid_eye_trace_full_sweep_backimage4.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results_full, f)

print(f"\n✓ Saved results to {output_path}")

#%% load results (if needed)
import pickle
with open('../declan/hybrid_eye_trace_full_sweep_backimage4.pkl', 'rb') as f:
    results_full = pickle.load(f)
print("✓ Loaded results from ../declan/hybrid_eye_trace_full_sweep_backimage4.pkl")

# %% Visualize as heatmaps with the actual image in the title
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Show the image in the top panels
img_extent = 5.0  # degrees
ax_img_left = axes[0, 0]
ax_img_right = axes[0, 1]

img_height, img_width = backimage_image.shape
img_height_deg = img_height / 37.5
img_width_deg = img_width / 37.5
extent_h = img_height_deg / 2
extent_w = img_width_deg / 2

ax_img_left.imshow(backimage_image, cmap='gray', origin='upper',
                   extent=[-extent_w, extent_w, -extent_h, extent_h])
ax_img_left.set_title(f'Stimulus: {image_file}', fontsize=12, fontweight='bold')
ax_img_left.set_xlabel('X (deg)')
ax_img_left.set_ylabel('Y (deg)')

# Fixation heatmap overlay (no normalization)
h, xedges, yedges = np.histogram2d(
    fixation_eyepos[:, 0], fixation_eyepos[:, 1],
    bins=50,
    range=[[-extent_w, extent_w], [-extent_h, extent_h]]
)
ax_img_right.imshow(backimage_image, cmap='gray', origin='upper',
                    extent=[-extent_w, extent_w, -extent_h, extent_h])

if h.max() > 0:
    # Use log scale to make sparse counts visible; remove norm=LogNorm for linear
    im_h = ax_img_right.imshow(
        h.T,
        extent=[-extent_w, extent_w, -extent_h, extent_h],
        origin='lower',
        cmap='hot',
        aspect='auto',
        alpha=0.5,
        norm=mpl.colors.LogNorm(vmin=1, vmax=h.max())
    )
    plt.colorbar(im_h, ax=ax_img_right, fraction=0.046, pad=0.04, label='Fixation count')
else:
    ax_img_right.text(0.5, 0.5, 'No fixation samples', ha='center', va='center',
                      transform=ax_img_right.transAxes, color='r')

ax_img_right.set_title(f'Fixation Heatmap ({len(fixation_eyepos)} samples)', 
                       fontsize=12, fontweight='bold')
ax_img_right.set_xlabel('X (deg)')
ax_img_right.set_ylabel('Y (deg)')

# Heatmaps (raw, not normalized)
ispike = results_full['i_spikes']
irate  = results_full['i_rates']

# Panel 3: Bits per spike (raw)
ax = axes[1, 0]
sns.heatmap(
    ispike,
    xticklabels=[f"{x:.2f}" for x in eye_scale_list],
    yticklabels=[f"{x}" for x in saccade_rates],
    annot=True, fmt=".3f", cmap='viridis', ax=ax
)
ax.set_xlabel('Fixational Eye Scale', fontsize=12)
ax.set_ylabel('Saccade Rate (Hz)', fontsize=12)
ax.set_title('Spatial Information (bits/spike)', fontsize=13, fontweight='bold')

# Panel 4: Bits per second (raw)
ax = axes[1, 1]
sns.heatmap(
    irate,
    xticklabels=[f"{x:.2f}" for x in eye_scale_list],
    yticklabels=[f"{x}" for x in saccade_rates],
    annot=True, fmt=".3f", cmap='viridis', ax=ax
)
ax.set_xlabel('Fixational Eye Scale', fontsize=12)
ax.set_ylabel('Saccade Rate (Hz)', fontsize=12)
ax.set_title('Information Rate (bits/sec)', fontsize=13, fontweight='bold')

plt.suptitle(f'Hybrid Eye Traces on Real Backimage: {image_file}', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_heatmaps_backimage.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved figure to ../figures/hybrid_eye_trace_heatmaps_backimage.png")

# %% Additional visualization: Line plots showing interaction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Use raw (not normalized) data
ispike = results_full['i_spikes']
irate = results_full['i_rates']

# Panel 1: Effect of eye scale at different saccade rates
ax = axes[0]
for i_sacc, sacc_rate in enumerate(saccade_rates):
    ax.plot(eye_scale_list, ispike[i_sacc], 
            'o-', lw=2, ms=8, label=f'{sacc_rate} Hz')
ax.set_xlabel('Fixational Eye Scale', fontsize=12)
ax.set_ylabel('Spatial Info (bits/spike)', fontsize=12)
ax.set_title('Effect of Fixational Eye Movements', fontsize=13, fontweight='bold')
ax.legend(title='Saccade Rate')
ax.grid(True, alpha=0.3)

# Panel 2: Effect of saccade rate at different eye scales
ax = axes[1]
for i_eye, eye_scale in enumerate(eye_scale_list):
    ax.plot(saccade_rates, irate[:, i_eye],
            's-', lw=2, ms=8, label=f'{eye_scale:.2f}x')
ax.set_xlabel('Saccade Rate (Hz)', fontsize=12)
ax.set_ylabel('Info Rate (bits/sec)', fontsize=12)
ax.set_title('Effect of Saccade Frequency', fontsize=13, fontweight='bold')
ax.legend(title='Eye Scale')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_line_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved figure to ../figures/hybrid_eye_trace_line_plots.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Results shape: {results_full['i_spikes'].shape}")
print(f"  Saccade rates: {saccade_rates}")
print(f"  Eye scales: {eye_scale_list}")
# Find null and max conditions
null_idx = (0, 0)  # 0 Hz saccades, 0x eye scale
null_ispike = results_full['i_spikes'][null_idx]
null_irate = results_full['i_rates'][null_idx]
print(f"  Null condition (0 Hz, 0x): {null_ispike:.4f} bits/spike, {null_irate:.4f} bits/sec")

max_idx = np.unravel_index(results_full['i_spikes'].argmax(), results_full['i_spikes'].shape)
print(f"  Max condition: {results_full['i_spikes'].max():.4f} bits/spike")
print(f"    at ({saccade_rates[max_idx[0]]} Hz, {eye_scale_list[max_idx[1]]}x)")


# %% Overnight run function


# Ensure ppd is defined globally (prevents NameError in multiple cells/sections)
ppd = 37.50476617

# ...existing code...

# %% -----------------------------------------------------------------------------------
# Overnight sweep across ALL BackImage natural images (matched stimulus + matched targets)
# -----------------------------------------------------------------------------------
from pathlib import Path
import time
import pickle

def run_parameter_sweep_all_images(
    *,
    model,
    sessions,
    fixation_pool,
    backimage_results: dict,
    image_cache: dict,
    saccade_rates: list[float],
    eye_scale_list: list[float],
    n_frames: int = 540,
    max_T: int = 600,
    n_lags: int = 32,
    out_size: tuple[int, int] = (151, 151),
    stim_scale: float = 1.0,
    ppd: float = 37.50476617,
    save_dir: str | Path = "../declan/overnight_backimage_sweeps",
    resume: bool = True,
):
    """
    Runs saccade_rate × eye_scale sweep for every image in backimage_results.
    Saves ONE pickle per image as it completes, so it can resume safely.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sort images by availability (most trials first) for better early wins
    items = sorted(backimage_results.items(), key=lambda kv: -int(kv[1].get("n_trials", 0)))

    # For progress bars
    import tqdm as tqdm

    print(f"Sweep: {len(items)} images → saving to {save_dir}")

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for image_file, meta in tqdm.tqdm(items, desc="Images", unit="img"):
        safe_name = image_file.replace("/", "_")
        out_path = save_dir / f"sweep_{safe_name}.pkl"

        if resume and out_path.exists():
            n_skipped += 1
            continue

        try:
            # --- matched saccade targets (from this image) ---
            fixation_eyepos = np.asarray(meta["eyepos"], dtype=np.float32)

            # unit check (deg vs pixels)
            med_amp = np.nanmedian(np.hypot(fixation_eyepos[:, 0], fixation_eyepos[:, 1]))
            if med_amp > 5.0:
                fixation_eyepos = fixation_eyepos / float(ppd)

            # --- matched stimulus pixels (same image_file) ---
            backimage_image = image_cache.get(image_file)
            if backimage_image is None:
                raise ValueError(f"Image pixels not found in image_cache for {image_file!r}")

            stim_len = int(max_T + n_lags + 1)
            full_stack = np.repeat(
                backimage_image[None, :, :].astype(np.float32),
                stim_len,
                axis=0,
            )

            # For reporting
            img_height_deg = backimage_image.shape[0] / float(ppd)
            img_width_deg = backimage_image.shape[1] / float(ppd)

            # --- run sweep ---
            results = {
                "image_file": image_file,
                "n_trials": int(meta.get("n_trials", -1)),
                "n_sessions": int(meta.get("n_sessions", -1)),
                "img_hw": tuple(backimage_image.shape),
                "img_deg": (float(img_height_deg), float(img_width_deg)),
                "ppd": float(ppd),
                "n_frames": int(n_frames),
                "max_T": int(max_T),
                "n_lags": int(n_lags),
                "out_size": tuple(out_size),
                "stim_scale": float(stim_scale),
                "saccade_rates": list(saccade_rates),
                "eye_scale_list": list(eye_scale_list),
                "i_spikes": [],
                "i_rates": [],
                "I_t": [],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            for sacc_rate in saccade_rates:
                row_sp = []
                row_rt = []
                row_It = []

                n_sacc = int(float(sacc_rate) * (n_frames / 120.0))

                # One base trace per saccade rate; rescale fixations only per eye_scale
                base_trace, base_mask, sacc_times, _ = create_hybrid_eye_trace(
                    fixation_pool=fixation_pool,
                    saccade_targets=fixation_eyepos,
                    n_saccades=n_sacc,
                    saccade_duration=6,
                    total_duration=n_frames,
                    eye_scale=1.0,
                )

                for eye_scale in eye_scale_list:
                    hybrid_trace_scaled = rescale_fixations_only(base_trace, base_mask, eye_scale)[:n_frames]

                    hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
                    hybrid_padded[: len(hybrid_trace_scaled)] = hybrid_trace_scaled.astype(np.float32)

                    y_real, _, _, _ = get_trial_stim_and_rates(
                        hybrid_padded,
                        full_stack,
                        out_size=out_size,
                        n_lags=n_lags,
                        scale=stim_scale,
                        plot=False,
                    )

                    ispike_real, irate_real, I_t_real = spatial_ssi_population(y_real)

                    row_sp.append(float(ispike_real.mean().item()))
                    row_rt.append(float(irate_real.mean().item()))
                    row_It.append(float(I_t_real.mean().item()))

                results["i_spikes"].append(row_sp)
                results["i_rates"].append(row_rt)
                results["I_t"].append(row_It)

            results["i_spikes"] = np.asarray(results["i_spikes"], dtype=np.float32)
            results["i_rates"] = np.asarray(results["i_rates"], dtype=np.float32)
            results["I_t"] = np.asarray(results["I_t"], dtype=np.float32)

            # Save atomically
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(results, f)
            tmp.replace(out_path)

            n_done += 1

        except Exception as e:
            n_failed += 1
            # Persist failure info so you don’t re-fail forever on resume
            err_path = save_dir / f"sweep_{safe_name}.ERROR.txt"
            err_path.write_text(str(e))
            continue

    print(f"Done. completed={n_done}, skipped(existing)={n_skipped}, failed={n_failed}")
    return {"completed": n_done, "skipped": n_skipped, "failed": n_failed, "save_dir": str(save_dir)}

#%%
# # no filepath: (run in notebook cell)
# saccade_rates = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
# eye_scale_list = [0.0] + list(np.exp(np.linspace(-2.75, np.log(2), 11)))

# summary = run_parameter_sweep_all_images(
#     model=model,
#     sessions=sessions,
#     fixation_pool=fixation_pool,
#     backimage_results=backimage_results,
#     image_cache=image_cache,
#     saccade_rates=saccade_rates,
#     eye_scale_list=eye_scale_list,
#     n_frames=540,
#     max_T=600,
#     n_lags=32,
#     out_size=(151, 151),
#     stim_scale=1.0,
#     ppd=ppd,
#     save_dir="../declan/overnight_backimage_sweeps",
#     resume=True,
# )
# summary

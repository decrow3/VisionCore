#!/usr/bin/env python3
"""
translation_covariance_trajectories_figures.py

Trajectory-aware plots for FEM-induced population geometry:
  1) Image + eye trajectory (time-colored)
  2) Neural trajectory in 3D PCA (PC1-3) for a single stimulus
  3) Two images, same trace: trajectories in a shared 3D embedding
  4) One image, many traces: family of trajectories in PC1-2 (optionally PC3 as height)
  5) Eye xy vs neural coefficients a1-a2 (projection onto U_pca2 or U_grad2)

This script recomputes δr(t) = r_real(t) - r_null(t) per trace so that trajectories reflect
transformations rather than content offsets. It can optionally load Σ/U_pca2 from the
saved covariance products to define embeddings consistent with your TC pipeline.

References:
- translation_covariance.py for stimulus loading/model/readout and δy construction. :contentReference[oaicite:1]{index=1}
- translation_covariance_figures.py for figure output conventions. :contentReference[oaicite:2]{index=2}
"""

# %% Imports and path setup

import os
import sys
import argparse
import pickle
from typing import Tuple, Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt

import torch

# Repo root
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'scripts'))

# %% Project imports
# Project imports (same as translation_covariance.py)
from spatial_info import make_counterfactual_stim, get_spatial_readout, compute_rate_map_batched
from utils import get_model_and_dataset_configs

# Try to import DataYatesV1 backgrounds helper
try:
    from DataYatesV1.exp.support import get_backimage_directory
except Exception:
    get_backimage_directory = None

# -------------------------
# Utilities (mirrors TC)
# -------------------------

# %% Utilities

def get_free_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_and_readout() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Same logic as translation_covariance.py: load model and build readout from cached outputs.
    """
    model, _ = get_model_and_dataset_configs()
    device = get_free_device()
    model = model.to(device)

    # Try to load cached outputs for readout construction
    outputs_path_candidates = [
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs_mono.pkl'),
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs.pkl'),
    ]
    outputs = None
    for p in outputs_path_candidates:
        if os.path.exists(p):
            import dill
            with open(p, 'rb') as f:
                outputs = dill.load(f)
            break
    if outputs is None:
        raise RuntimeError(
            "Could not find cached outputs for readout construction. "
            "Expected scripts/mcfarland_outputs_mono.pkl (or mcfarland_outputs.pkl)."
        )
    readout = get_spatial_readout(model, outputs).to(device)
    return model, readout


def load_backimage_results(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _to_gray_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0)
    return arr


def ensure_min_trace_length(trace_arr: np.ndarray, min_len: int) -> np.ndarray:
    trace_arr = np.asarray(trace_arr, dtype=np.float32)
    if trace_arr.ndim == 1:
        trace_arr = trace_arr.reshape(-1, 2) if trace_arr.size % 2 == 0 else trace_arr[:-1].reshape(-1, 2)
    elif trace_arr.ndim == 2:
        if trace_arr.shape[1] != 2:
            trace_arr = trace_arr[:, :2] if trace_arr.shape[1] > 2 else trace_arr.T if trace_arr.shape[0] == 2 else trace_arr
    else:
        flat = trace_arr.reshape(-1)
        trace_arr = flat.reshape(-1, 2) if flat.size % 2 == 0 else flat[:-1].reshape(-1, 2)

    nan_rows = np.where(np.isnan(trace_arr).any(axis=1))[0]
    T_valid = nan_rows[0] if len(nan_rows) > 0 else len(trace_arr)
    trace_arr = trace_arr[:T_valid]

    if trace_arr.shape[0] >= min_len:
        return trace_arr
    base = trace_arr[:1] if trace_arr.shape[0] > 0 else np.zeros((1, 2), dtype=np.float32)
    pad_rows = min_len - trace_arr.shape[0]
    pad = np.repeat(base, pad_rows, axis=0)
    return np.vstack([trace_arr, pad])


def compress_to_population_vec(y_tensor) -> np.ndarray:
    """
    Compress [T,N,H,W] -> [T,N] by spatial mean, consistent with TC.
    """
    if isinstance(y_tensor, torch.Tensor):
        y_np = y_tensor.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_tensor)
    if y_np.ndim == 4:
        return y_np.mean(axis=(2, 3))
    return y_np


def get_trial_stim_and_rates(
    eyepos: np.ndarray,
    full_stack: np.ndarray,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    out_size: Tuple[int, int] = (151, 151),
    n_lags: int = 32,
    scale: float = 1.0,
):
    """
    Returns y_real, y_null, stim_real, stim_null.
    Mirrors translation_covariance.py.
    """
    eyepos_arr = np.asarray(eyepos, dtype=np.float32)
    if eyepos_arr.ndim == 1:
        eyepos_arr = eyepos_arr.reshape(-1, 2) if eyepos_arr.size % 2 == 0 else eyepos_arr[:-1].reshape(-1, 2)
    elif eyepos_arr.ndim == 2 and eyepos_arr.shape[1] != 2:
        eyepos_arr = eyepos_arr.T if eyepos_arr.shape[0] == 2 else eyepos_arr[:, :2]

    nan_rows = np.where(np.isnan(eyepos_arr).any(axis=1))[0]
    T_valid = nan_rows[0] if len(nan_rows) > 0 else len(eyepos_arr)
    eyepos_arr = eyepos_arr[:T_valid]

    eyepos_t = torch.from_numpy(eyepos_arr).float()
    null_eyepos = torch.zeros_like(eyepos_t) + eyepos_t.mean(0)

    eye_stim = make_counterfactual_stim(full_stack, eyepos_t, out_size=out_size, n_lags=n_lags, scale_factor=scale)
    eye_stim_null = make_counterfactual_stim(full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale)

    y = compute_rate_map_batched(model, readout, (eye_stim - 127.0) / 255.0)
    y_null = compute_rate_map_batched(model, readout, (eye_stim_null - 127.0) / 255.0)
    return y, y_null, eye_stim, eye_stim_null


def compute_delta_r(y, y_null) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (delta_tn, y_tn, mu_tn).
    delta_tn is the trajectory object you want to visualize.
    """
    y_tn = compress_to_population_vec(y)
    mu_tn = compress_to_population_vec(y_null)
    # align lengths defensively
    T = min(y_tn.shape[0], mu_tn.shape[0])
    y_tn = y_tn[:T]
    mu_tn = mu_tn[:T]
    return (y_tn - mu_tn), y_tn, mu_tn


def pca_basis_from_samples(X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA basis via SVD on centered samples.
    X: [S,N]
    Returns (V_k [N,k], mean [N,])
    """
    X = np.asarray(X, dtype=np.float32)
    m = np.nanmean(X, axis=0)
    Xc = X - m[None, :]
    mask = np.isfinite(Xc).all(axis=1)
    Xc = Xc[mask]
    if Xc.shape[0] < k + 2:
        raise RuntimeError("Not enough samples for PCA basis.")
    # SVD: Xc = U S Vt, columns of V are PCs
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :k]
    return V, m


def project_trajectory(delta_tn: np.ndarray, V: np.ndarray, mean_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project [T,N] onto V [N,k].
    If mean_vec provided, center delta_tn by mean_vec (usually not needed since delta is already residual).
    Returns Z [T,k].
    """
    X = np.asarray(delta_tn, dtype=np.float32)
    if mean_vec is not None:
        X = X - mean_vec[None, :]
    return X @ V


def _ensure_figdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _time_colorline(ax, x, y, c=None, lw=2):
    """
    Simple time-colored line: draw short segments with colormap.
    Uses matplotlib default colormap if c is None.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if c is None:
        c = np.linspace(0, 1, n)
    # draw segments
    for i in range(n - 1):
        ax.plot(x[i:i+2], y[i:i+2], c=plt.cm.viridis(c[i]), lw=lw)


def _time_colorline_3d(ax, x, y, z, c=None, lw=2):
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    n = len(x)
    if c is None:
        c = np.linspace(0, 1, n)
    for i in range(n - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], c=plt.cm.viridis(c[i]), lw=lw)


def find_datayates_root(args) -> Optional[str]:
    """
    Resolve the DataYatesV1 repository root without hardcoding.
    Priority:
      1) args.data_root, if provided
      2) ENV `DATAYATESV1_ROOT`
      3) common local candidates
    Returns path or None.
    """
    # 1) explicit arg
    if hasattr(args, "data_root") and args.data_root and os.path.isdir(args.data_root):
        return args.data_root

    # 2) environment variable
    env_root = os.environ.get("DATAYATESV1_ROOT")
    if env_root and os.path.isdir(env_root):
        return env_root

    # 3) common local candidates
    candidates = [
        os.path.join(ROOT, "..", "DataYatesV1"),
        os.path.join(os.path.expanduser("~"), "DataYatesV1"),
        "/home/declan/DataYatesV1/DataYatesV1",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def resolve_background_image(stim_key: str, data_root: Optional[str], backgrounds_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Try to load a background image for the given key.
    Search order:
      1) `backgrounds_dir` if provided
      2) DataYatesV1 helper `get_backimage_directory()` if available
      3) `<data_root>/exp/SupportData/Backgrounds` (legacy fallback)
    Matching strategy:
      - If `stim_key` is an absolute path, load it directly.
      - If `<bg_root>/<stim_key>` exists, load it.
      - Case-insensitive exact filename match.
      - Extension-insensitive match (basename without extension).
      - Fallback substring match.
    Returns grayscale float32 image array or None if not found.
    """
    # Absolute path direct load
    if os.path.isabs(stim_key) and os.path.isfile(stim_key):
        try:
            img = plt.imread(stim_key)
            return _to_gray_float32(img)
        except Exception:
            return None

    # Resolve backgrounds root
    bg_root = None
    if backgrounds_dir and os.path.isdir(backgrounds_dir):
        bg_root = backgrounds_dir
    elif get_backimage_directory is not None:
        try:
            _root = get_backimage_directory()
            if _root and os.path.isdir(str(_root)):
                bg_root = str(_root)
        except Exception:
            bg_root = None
    if not bg_root and data_root:
        cand = os.path.join(data_root, "exp", "SupportData", "Backgrounds")
        if os.path.isdir(cand):
            bg_root = cand
    if not bg_root:
        return None

    # Absolute path or direct join
    try_paths = []
    if os.path.isabs(stim_key):
        try_paths.append(stim_key)
    try_paths.append(os.path.join(bg_root, stim_key))
    for fp in try_paths:
        if os.path.isfile(fp):
            try:
                img = plt.imread(fp)
                return _to_gray_float32(img)
            except Exception:
                return None

    # Build matching targets
    target_name = os.path.basename(stim_key).lower()
    target_noext = os.path.splitext(target_name)[0]

    # Scan for matches
    candidates = []
    for dirpath, _, filenames in os.walk(bg_root):
        for fn in filenames:
            fn_low = fn.lower()
            fp = os.path.join(dirpath, fn)
            # exact case-insensitive
            if fn_low == target_name:
                candidates.append(fp)
                continue
            # extension-insensitive
            if os.path.splitext(fn_low)[0] == target_noext:
                candidates.append(fp)
                continue
            # substring fallback
            if target_noext and target_noext in fn_low:
                candidates.append(fp)

    if candidates:
        # Deduplicate while preserving order
        seen = set()
        for fp in candidates:
            if fp in seen:
                continue
            seen.add(fp)
            try:
                img = plt.imread(fp)
                return _to_gray_float32(img)
            except Exception:
                continue
    return None


# -------------------------
# Plot makers
# -------------------------

# %% Plot makers

def plot_image_with_trace(img_gray: np.ndarray, trace_xy: np.ndarray, outpath: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img_gray, cmap='gray', interpolation='nearest')
    # trace_xy is in degrees/pixels? Here we assume it is in stimulus coordinates used by make_counterfactual_stim.
    # For visualization, we just plot in trace coordinate system. If you want accurate overlay on pixels, add a mapping.
    # So this panel is mostly conceptual unless you define the mapping.
    ax.set_title(title)
    ax.set_axis_off()

    # Show the trace in its own coordinate system on top-right inset
    ax_in = fig.add_axes([0.62, 0.62, 0.33, 0.33])
    ax_in.set_title("Eye path", fontsize=9)
    _time_colorline(ax_in, trace_xy[:, 0], trace_xy[:, 1], lw=2)
    ax_in.set_xlabel("x", fontsize=8)
    ax_in.set_ylabel("y", fontsize=8)
    ax_in.tick_params(labelsize=8)
    ax_in.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_single_stim_trace_3d(delta_tn: np.ndarray, V3: np.ndarray, outpath: str, title: str = ""):
    Z = project_trajectory(delta_tn, V3)
    # pad if <3 dims
    if Z.shape[1] < 3:
        Z = np.pad(Z, ((0, 0), (0, 3 - Z.shape[1])), mode='constant')

    fig = plt.figure(figsize=(5.8, 5.0))
    ax = fig.add_subplot(111, projection='3d')
    _time_colorline_3d(ax, Z[:, 0], Z[:, 1], Z[:, 2], lw=2)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_two_images_same_trace_shared_embedding(
    deltaA: np.ndarray,
    deltaB: np.ndarray,
    outpath: str,
    title: str = "",
):
    # Shared PCA basis from concatenated samples (across time)
    X = np.vstack([deltaA, deltaB])
    V3, m = pca_basis_from_samples(X, k=3)
    ZA = project_trajectory(deltaA, V3, mean_vec=None)
    ZB = project_trajectory(deltaB, V3, mean_vec=None)

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    _time_colorline_3d(ax, ZA[:, 0], ZA[:, 1], ZA[:, 2], lw=2)
    _time_colorline_3d(ax, ZB[:, 0], ZB[:, 1], ZB[:, 2], lw=2)
    ax.set_title(title + "\n(two trajectories; time-colored per trajectory)")
    ax.set_xlabel("shared PC1")
    ax.set_ylabel("shared PC2")
    ax.set_zlabel("shared PC3")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_many_traces_one_stim_pc12(
    deltas: List[np.ndarray],
    V2: np.ndarray,
    outpath: str,
    title: str = "",
    highlight: int = 0,
):
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)

    for i, d in enumerate(deltas):
        Z = project_trajectory(d, V2)
        lw = 2.5 if i == highlight else 1.0
        # Use time-colored segments to show direction; for non-highlight, thin lines suffice
        _time_colorline(ax, Z[:, 0], Z[:, 1], lw=lw)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_eye_xy_vs_neural_a1a2(
    trace_xy: np.ndarray,
    delta_tn: np.ndarray,
    U2: np.ndarray,
    outpath: str,
    title: str = "",
):
    A = project_trajectory(delta_tn, U2)  # coefficients in the 2D plane

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axs[0].set_title("Retinal trajectory (xy)")
    _time_colorline(axs[0], trace_xy[:, 0], trace_xy[:, 1], lw=2)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True, alpha=0.2)

    axs[1].set_title("Neural coefficients (a1, a2)")
    _time_colorline(axs[1], A[:, 0], A[:, 1], lw=2)
    axs[1].set_xlabel("a1")
    axs[1].set_ylabel("a2")
    axs[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -------------------------
# Main driver
# -------------------------

# %% Main driver

def run(args):
    figdir = _ensure_figdir(args.outdir)

    # Load metadata
    backimage_results = load_backimage_results(args.results_pkl)
    all_keys = list(backimage_results.keys())
    if len(all_keys) == 0:
        raise RuntimeError("backimage_results is empty.")

    # Choose stimuli
    stimA = args.stimA or all_keys[0]
    if stimA not in backimage_results:
        raise RuntimeError(f"stimA '{stimA}' not found in backimage_results.")
    stimB = args.stimB
    if stimB is not None and stimB not in backimage_results:
        raise RuntimeError(f"stimB '{stimB}' not found in backimage_results.")

    # Load model/readout
    model, readout = load_model_and_readout()
    device = get_free_device()
    model = model.to(device)
    readout = readout.to(device)

    # Optional: load cov_results to reuse U_pca2 and/or Sigma-derived V3
    cov_results = {}
    cov_pkl = os.path.join(args.cov_results_dir, "all_cov_results.pkl")
    if os.path.exists(cov_pkl):
        with open(cov_pkl, "rb") as f:
            cov_results = pickle.load(f)

    data_root = find_datayates_root(args)
    backgrounds_dir = None
    if hasattr(args, "backgrounds_dir") and args.backgrounds_dir:
        if os.path.isdir(args.backgrounds_dir):
            backgrounds_dir = args.backgrounds_dir
        else:
            raise RuntimeError(f"--backgrounds-dir does not exist: {args.backgrounds_dir}")

    def get_image_and_traces(stim_key: str):
        entry = backimage_results[stim_key]
        img_gray = None
        if "image" in entry and entry["image"] is not None:
            img_gray = _to_gray_float32(entry["image"])
        else:
            # Attempt to resolve via explicit dir, DataYatesV1 helper, or data_root fallback
            img_gray = resolve_background_image(stim_key, data_root, backgrounds_dir=backgrounds_dir)
            if img_gray is None:
                raise RuntimeError(
                    f"Entry for {stim_key} lacks cached 'image' and could not be resolved. "
                    f"Provide --backgrounds-dir (preferred) or --data-root / DATAYATESV1_ROOT, and ensure the key matches a background filename."
                )
        traces = entry.get("eyepos", [])
        if len(traces) == 0:
            raise RuntimeError(f"No eyepos traces for {stim_key}.")
        return img_gray, traces

    imgA, tracesA = get_image_and_traces(stimA)
    traceA = ensure_min_trace_length(tracesA[args.trace_idx], args.n_lags)

    # Build null response once per stimulus (for y_null)
    def _compute_null_response(img_gray: np.ndarray) -> np.ndarray:
        T_null = 540
        null_trace = np.zeros((T_null, 2), dtype=np.float32)
        full_stack_null = np.repeat(img_gray[np.newaxis, ...], T_null + args.n_lags + 1, axis=0)
        y_null, _, _, _ = get_trial_stim_and_rates(null_trace, full_stack_null, model, readout,
                                                   out_size=tuple(args.out_size), n_lags=args.n_lags, scale=args.scale)
        return y_null

    y_null_A = _compute_null_response(imgA)

    def compute_delta_for_trace(img_gray: np.ndarray, trace_xy: np.ndarray, y_null_pre: np.ndarray) -> np.ndarray:
        T_stim = trace_xy.shape[0]
        Trep = T_stim + args.n_lags
        full_stack = np.repeat(img_gray[np.newaxis, ...], Trep, axis=0)
        y, _, _, _ = get_trial_stim_and_rates(trace_xy, full_stack, model, readout,
                                              out_size=tuple(args.out_size), n_lags=args.n_lags, scale=args.scale)

        delta_tn, _, _ = compute_delta_r(y, y_null_pre)
        # Align trace length to delta length for plotting
        T = min(delta_tn.shape[0], trace_xy.shape[0] + args.n_lags)
        return delta_tn[:T]

    # -------- Plot set for single stimulus A --------
    out_img_trace = os.path.join(figdir, "A_image_and_eye_path.png")
    plot_image_with_trace(imgA, traceA, out_img_trace, title=f"Stim A: {os.path.basename(stimA)}")
    print(f"Saved: {out_img_trace}")

    # Build deltas for A across many traces for a stable basis
    deltasA = []
    n_show = min(args.num_traces, len(tracesA))
    for i in range(n_show):
        tr = ensure_min_trace_length(tracesA[i], args.n_lags)
        deltasA.append(compute_delta_for_trace(imgA, tr, y_null_A))

    # Use the selected trace for single-trace plots
    deltaA = compute_delta_for_trace(imgA, traceA, y_null_A)

    # Basis for A: prefer Sigma eigenvectors (PCs) if available in cov_results; else PCA from samples of this trace
    if stimA in cov_results and "Sigma" in cov_results[stimA]:
        # Build V3 from Sigma eigvecs (top-3)
        Sigma = np.asarray(cov_results[stimA]["Sigma"])
        w, V = np.linalg.eigh(Sigma)
        order = np.argsort(w)[::-1]
        V3 = V[:, order[:3]]
        U2 = np.asarray(cov_results[stimA]["U_pca2"])
    else:
        # Pooled samples across traces for stability
        Xcat3 = np.vstack(deltasA)
        V3, _ = pca_basis_from_samples(Xcat3, k=3)
        U2 = V3[:, :2]

    out_neural_3d = os.path.join(figdir, "A_neural_trajectory_3D.png")
    plot_single_stim_trace_3d(deltaA, V3, out_neural_3d, title="Stim A: δr(t) in PC space (PC1–PC3)")
    print(f"Saved: {out_neural_3d}")

    out_eye_vs_a = os.path.join(figdir, "A_eye_xy_vs_neural_a1a2.png")
    plot_eye_xy_vs_neural_a1a2(traceA[:deltaA.shape[0]], deltaA, U2, out_eye_vs_a, title="Stim A: eye vs neural coefficients")
    print(f"Saved: {out_eye_vs_a}")

    # Define a stable V2 basis for multi-trace: PCA on concatenated δr samples
    Xcat = np.vstack(deltasA)
    V2, _ = pca_basis_from_samples(Xcat, k=2)

    out_many = os.path.join(figdir, "A_many_traces_PC12.png")
    plot_many_traces_one_stim_pc12(deltasA, V2, out_many, title=f"Stim A: many traces in shared PC1–PC2", highlight=0)
    print(f"Saved: {out_many}")

    # -------- Two stimuli, same trace (shared embedding) --------
    if stimB is not None:
        imgB, tracesB = get_image_and_traces(stimB)
        # Use the same physical traceA for both stimuli (core conceptual demo)
        # Null for B once, then same eye trace
        y_null_B = _compute_null_response(imgB)
        deltaB = compute_delta_for_trace(imgB, traceA, y_null_B)

        out_img_trace_B = os.path.join(figdir, "B_image_and_eye_path.png")
        plot_image_with_trace(imgB, traceA, out_img_trace_B, title=f"Stim B: {os.path.basename(stimB)}")

        out_shared = os.path.join(figdir, "AB_shared_embedding_same_trace.png")
        plot_two_images_same_trace_shared_embedding(
            deltaA=deltaA,
            deltaB=deltaB,
            outpath=out_shared,
            title=f"Same eye trace, two images: δr(t) in shared PCA space\nA={os.path.basename(stimA)}  B={os.path.basename(stimB)}",
        )
        print(f"Saved: {out_shared}")
    else:
        print("Skipping two-stimulus plot (stimB not provided)")

    print("\nSaved trajectory figures to:")
    for fn in sorted(os.listdir(figdir)):
        if fn.lower().endswith(".png"):
            print(" -", os.path.join(figdir, fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=os.path.join(ROOT, "declan", "translation_covariance", "trajectory_figures"))
    ap.add_argument("--results-pkl", type=str, default=os.path.join(ROOT, "declan", "backimage_fixation_results.pkl"))
    ap.add_argument("--cov-results-dir", type=str, default=os.path.join(ROOT, "declan", "translation_covariance"),
                    help="Optional: directory with all_cov_results.pkl to reuse U_pca2 / Sigma bases.")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Optional: DataYatesV1 repo root (env DATAYATESV1_ROOT also supported).")
    ap.add_argument("--backgrounds-dir", type=str, default=None,
                    help="Optional: direct path to SupportData/Backgrounds; overrides --data-root and DataYatesV1 helper.")
    ap.add_argument("--stimA", type=str, default=None, help="Filename key for stimulus A (must be in backimage_results).")
    ap.add_argument("--stimB", type=str, default=None, help="Filename key for stimulus B (optional).")
    ap.add_argument("--trace-idx", type=int, default=0, help="Which trace index to use for single-trace plots.")
    ap.add_argument("--num-traces", type=int, default=15, help="How many traces to show for multi-trace plot.")
    ap.add_argument("--n-lags", type=int, default=32)
    ap.add_argument("--out-size", type=int, nargs=2, default=(151, 151))
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    run(args)

# %% Interactive Defaults
# Tweak these values in VS Code and run the next cell.
_outdir = os.path.join(ROOT, "declan", "translation_covariance", "trajectory_figures")
_results_pkl = os.path.join(ROOT, "declan", "backimage_fixation_results.pkl")
_cov_results_dir = os.path.join(ROOT, "declan", "translation_covariance")
_data_root = None  # e.g., os.path.join(os.path.expanduser("~"), "DataYatesV1")
_backgrounds_dir = None  # e.g., os.path.join(ROOT, "data", "Backgrounds")

# Auto-resolve data root if not set
from types import SimpleNamespace as _NS
if _data_root is None:
    try:
        _data_root = find_datayates_root(_NS(data_root=None))
    except Exception:
        _data_root = None
print(f"[interactive] data_root={_data_root}")
print(f"[interactive] backgrounds_dir={_backgrounds_dir}")

# Example overrides (set to None or specific keys present in your results pkl)
_stimA = None  # e.g., "image_001.png"
_stimB = None  # optional second stimulus key
_trace_idx = 0
_num_traces = 15
_n_lags = 32
_out_size = (151, 151)
_scale = 1.0

interactive_args = _NS(
    outdir=_outdir,
    results_pkl=_results_pkl,
    cov_results_dir=_cov_results_dir,
    data_root=_data_root,
    backgrounds_dir=_backgrounds_dir,
    stimA=_stimA,
    stimB=_stimB,
    trace_idx=_trace_idx,
    num_traces=_num_traces,
    n_lags=_n_lags,
    out_size=_out_size,
    scale=_scale,
)

# %% Interactive Backgrounds
# Prefer explicit override; else use DataYatesV1 helper; else data_root fallback
if _backgrounds_dir and os.path.isdir(_backgrounds_dir):
    _bg_root = _backgrounds_dir
else:
    _bg_root = None
    if get_backimage_directory is not None:
        try:
            _dir = get_backimage_directory()
            if _dir and os.path.isdir(str(_dir)):
                _bg_root = str(_dir)
        except Exception:
            _bg_root = None
    if not _bg_root and _data_root:
        cand = os.path.join(_data_root, "exp", "SupportData", "Backgrounds")
        if os.path.isdir(cand):
            _bg_root = cand

if _bg_root and os.path.isdir(_bg_root):
    try:
        _bg_files = sorted([
            fn for fn in os.listdir(_bg_root)
            if os.path.isfile(os.path.join(_bg_root, fn))
        ])
        print(f"Found {len(_bg_files)} backgrounds in {_bg_root}. Examples:")
        for fn in _bg_files[:20]:
            print(" -", fn)
        # Tip: set `_stimA = _bg_files[0]` (or any shown filename) and re-run Interactive Defaults
    except Exception as _e:
        print(f"Error listing backgrounds: {_e}")
else:
    print("No backgrounds directory. Set _backgrounds_dir (preferred), or ensure DataYatesV1 is installed, or set _data_root / DATAYATESV1_ROOT.")

# %% Interactive Run (All Plots)
# Run the full pipeline with the interactive args defined above.
# This bypasses argparse and executes exactly like the CLI.
# You can re-run this cell after tweaking the Interactive Defaults.
run(interactive_args)

# %% Interactive Setup (for individual plot cells below)
# Run this cell first to set up data for individual plotting
import pickle
from types import SimpleNamespace as _NS

# Load results and model
backimage_results = load_backimage_results(interactive_args.results_pkl)
all_keys = list(backimage_results.keys())
stimA = interactive_args.stimA or all_keys[0]
stimB = interactive_args.stimB

model, readout = load_model_and_readout()
device = get_free_device()
model = model.to(device)
readout = readout.to(device)

# Load cov_results if available
cov_results = {}
cov_pkl = os.path.join(interactive_args.cov_results_dir, "all_cov_results.pkl")
if os.path.exists(cov_pkl):
    with open(cov_pkl, "rb") as f:
        cov_results = pickle.load(f)

data_root = find_datayates_root(interactive_args)
backgrounds_dir = interactive_args.backgrounds_dir

def get_image_and_traces(stim_key: str):
    entry = backimage_results[stim_key]
    img_gray = None
    if "image" in entry and entry["image"] is not None:
        img_gray = _to_gray_float32(entry["image"])
    else:
        img_gray = resolve_background_image(stim_key, data_root, backgrounds_dir=backgrounds_dir)
        if img_gray is None:
            raise RuntimeError(f"Could not resolve image for {stim_key}")
    traces = entry.get("eyepos", [])
    if len(traces) == 0:
        raise RuntimeError(f"No eyepos traces for {stim_key}.")
    return img_gray, traces

def _compute_null_response(img_gray: np.ndarray) -> np.ndarray:
    T_null = 540
    null_trace = np.zeros((T_null, 2), dtype=np.float32)
    full_stack_null = np.repeat(img_gray[np.newaxis, ...], T_null + interactive_args.n_lags + 1, axis=0)
    y_null, _, _, _ = get_trial_stim_and_rates(null_trace, full_stack_null, model, readout,
                                               out_size=tuple(interactive_args.out_size), n_lags=interactive_args.n_lags, scale=interactive_args.scale)
    return y_null

def compute_delta_for_trace(img_gray: np.ndarray, trace_xy: np.ndarray, y_null_pre: np.ndarray) -> np.ndarray:
    T_stim = trace_xy.shape[0]
    Trep = T_stim + interactive_args.n_lags
    full_stack = np.repeat(img_gray[np.newaxis, ...], Trep, axis=0)
    y, _, _, _ = get_trial_stim_and_rates(trace_xy, full_stack, model, readout,
                                          out_size=tuple(interactive_args.out_size), n_lags=interactive_args.n_lags, scale=interactive_args.scale)
    delta_tn, _, _ = compute_delta_r(y, y_null_pre)
    T = min(delta_tn.shape[0], trace_xy.shape[0] + interactive_args.n_lags)
    return delta_tn[:T]

# Load data
imgA, tracesA = get_image_and_traces(stimA)
traceA = ensure_min_trace_length(tracesA[interactive_args.trace_idx], interactive_args.n_lags)
y_null_A = _compute_null_response(imgA)

# Build deltas
deltasA = []
n_show = min(interactive_args.num_traces, len(tracesA))
for i in range(n_show):
    tr = ensure_min_trace_length(tracesA[i], interactive_args.n_lags)
    deltasA.append(compute_delta_for_trace(imgA, tr, y_null_A))

deltaA = compute_delta_for_trace(imgA, traceA, y_null_A)

# Compute basis
if stimA in cov_results and "Sigma" in cov_results[stimA]:
    Sigma = np.asarray(cov_results[stimA]["Sigma"])
    w, V = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]
    V3 = V[:, order[:3]]
    U2 = np.asarray(cov_results[stimA]["U_pca2"])
else:
    Xcat3 = np.vstack(deltasA)
    V3, _ = pca_basis_from_samples(Xcat3, k=3)
    U2 = V3[:, :2]

Xcat = np.vstack(deltasA)
V2, _ = pca_basis_from_samples(Xcat, k=2)

figdir = _ensure_figdir(interactive_args.outdir)
print(f"✓ Setup complete. stimA={stimA}, n_traces={n_show}")
print(f"  deltaA.shape={deltaA.shape}, V3.shape={V3.shape}, U2.shape={U2.shape}")

# %% Plot 1: Image + Eye Path
out_img_trace = os.path.join(figdir, "A_image_and_eye_path.png")
plot_image_with_trace(imgA, traceA, out_img_trace, title=f"Stim A: {os.path.basename(stimA)}")
print(f"Saved: {out_img_trace}")

# %% Plot 2: Neural Trajectory 3D (PC1-3)
out_neural_3d = os.path.join(figdir, "A_neural_trajectory_3D.png")
plot_single_stim_trace_3d(deltaA, V3, out_neural_3d, title="Stim A: δr(t) in PC space (PC1–PC3)")
print(f"Saved: {out_neural_3d}")

# %% Plot 3: Eye XY vs Neural Coefficients
out_eye_vs_a = os.path.join(figdir, "A_eye_xy_vs_neural_a1a2.png")
plot_eye_xy_vs_neural_a1a2(traceA[:deltaA.shape[0]], deltaA, U2, out_eye_vs_a, title="Stim A: eye vs neural coefficients")
print(f"Saved: {out_eye_vs_a}")

# %% Plot 4: Many Traces (PC1-2)
out_many = os.path.join(figdir, "A_many_traces_PC12.png")
plot_many_traces_one_stim_pc12(deltasA, V2, out_many, title=f"Stim A: many traces in shared PC1–PC2", highlight=0)
print(f"Saved: {out_many}")

# %% Plot 5: Two Stimuli Shared Embedding (optional)
if stimB is not None:
    imgB, tracesB = get_image_and_traces(stimB)
    y_null_B = _compute_null_response(imgB)
    deltaB = compute_delta_for_trace(imgB, traceA, y_null_B)
    
    out_img_trace_B = os.path.join(figdir, "B_image_and_eye_path.png")
    plot_image_with_trace(imgB, traceA, out_img_trace_B, title=f"Stim B: {os.path.basename(stimB)}")
    
    out_shared = os.path.join(figdir, "AB_shared_embedding_same_trace.png")
    plot_two_images_same_trace_shared_embedding(
        deltaA=deltaA, deltaB=deltaB, outpath=out_shared,
        title=f"Same eye trace, two images: δr(t) in shared PCA space\nA={os.path.basename(stimA)}  B={os.path.basename(stimB)}",
    )
    print(f"Saved: {out_shared}")
else:
    print("Skipping two-stimulus plot (stimB not provided). Set interactive_args.stimB or _stimB to generate.")

# %% Interactive Stimulus Picker
# Run this cell to list available stimulus keys
if 'backimage_results' in dir():
    _all_keys = list(backimage_results.keys())
else:
    import pickle
    _pkl = os.path.join(ROOT, 'declan', 'backimage_fixation_results.pkl')
    if os.path.exists(_pkl):
        with open(_pkl, 'rb') as _f:
            _all_keys = list(pickle.load(_f).keys())
    else:
        _all_keys = []

if _all_keys:
    print(f'Found {len(_all_keys)} stimulus keys. Examples:')
    for _k in _all_keys[:20]:
        print(f'  - {_k}')
    print("\nTo generate Plot 5 (two-stimulus), set _stimB in Interactive Defaults or:")
    print("  interactive_args.stimB = '<key_name>'")
    print("Then re-run 'Interactive Setup' and 'Plot 5' cells.")
else:
    print("No stimulus keys found. Run 'Interactive Setup' first.")
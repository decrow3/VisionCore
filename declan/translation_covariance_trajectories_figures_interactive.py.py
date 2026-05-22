#!/usr/bin/env python3
"""
translation_covariance_trajectories_interactive.py

Interactive, step-through plotting for FEM-induced population trajectories.

Design goals
- Works both as a CLI script and as an interactive “run cell-by-cell” file in VS Code / Spyder.
- Does NOT auto-run on import.
- Lets you step through each plot sequentially, inspecting intermediate arrays (img, trace, delta, PCs).

What it does (per your original intent)
1) Image + eye trajectory (time-colored, inset)
2) Neural trajectory in 3D PCA for a single stimulus
3) Two images, same trace: trajectories in a shared 3D embedding (optional)
4) One image, many traces: family of trajectories in PC1–PC2
5) Eye xy vs neural coefficients a1–a2

Key guardrails
- Uses residual activity δr(t)=r_real(t)-r_null(t) so plots emphasize motion-driven activity.
- Prefers loading U_pca2 / Sigma from your TC output (all_cov_results.pkl) if present.
- When background-image coordinate mapping is unknown, it does NOT pretend to overlay the trace on the image;
  instead it shows the image plus an inset with the eye path in its native coordinate system.

Usage (CLI)
python scripts/translation_covariance_trajectories_interactive.py \
  --stimA Colony_Bonnie.JPG --stimB Hawaii_trees.JPG \
  --backgrounds-dir /path/to/SupportData/Backgrounds --pause

Usage (interactive)
- Open in VS Code, run cells from top to bottom.
- Edit the “USER CONFIG” cell, then run the “RUN PIPELINE” cell.
"""

# %% Imports and path setup

import os
import sys
import argparse
import pickle
from typing import Tuple, Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch

# Repo root
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# %% Project imports (same as translation_covariance.py)
from spatial_info import make_counterfactual_stim, get_spatial_readout, compute_rate_map_batched
from utils import get_model_and_dataset_configs

# Optional DataYatesV1 helper
try:
    from DataYatesV1.exp.support import get_backimage_directory
except Exception:
    get_backimage_directory = None


# %% -------------------------
# Utilities
# -------------------------

def get_free_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_readout(
    outputs_candidates: Optional[List[str]] = None
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load model and construct readout from cached outputs (as in translation_covariance.py).
    """
    model, _ = get_model_and_dataset_configs()
    device = get_free_device()
    model = model.to(device)

    if outputs_candidates is None:
        outputs_candidates = [
            os.path.join(ROOT, "scripts", "mcfarland_outputs_mono.pkl"),
            os.path.join(ROOT, "scripts", "mcfarland_outputs.pkl"),
        ]

    outputs = None
    picked = None
    for p in outputs_candidates:
        if os.path.exists(p):
            import dill
            with open(p, "rb") as f:
                outputs = dill.load(f)
            picked = p
            break

    if outputs is None:
        raise RuntimeError(
            "Could not find cached outputs for readout construction.\n"
            f"Tried: {outputs_candidates}"
        )

    readout = get_spatial_readout(model, outputs).to(device)
    print(f"[load_model_and_readout] using cached outputs: {picked}")
    return model, readout


def load_pickle(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
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
    return np.clip(arr, 0.0, 255.0)


def ensure_trace_xy(trace_arr: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Normalize eyetrace shapes to [T,2], trim at NaNs, and ensure length >= n_lags.
    """
    trace_arr = np.asarray(trace_arr, dtype=np.float32)

    if trace_arr.ndim == 1:
        L = trace_arr.size - (trace_arr.size % 2)
        trace_arr = trace_arr[:L].reshape(-1, 2)
    elif trace_arr.ndim == 2:
        if trace_arr.shape[1] != 2:
            if trace_arr.shape[0] == 2:
                trace_arr = trace_arr.T
            else:
                trace_arr = trace_arr[:, :2]
    else:
        flat = trace_arr.reshape(-1)
        L = flat.size - (flat.size % 2)
        trace_arr = flat[:L].reshape(-1, 2)

    nan_rows = np.where(np.isnan(trace_arr).any(axis=1))[0]
    if len(nan_rows) > 0:
        trace_arr = trace_arr[:nan_rows[0]]

    if trace_arr.shape[0] < n_lags:
        pad = np.repeat(trace_arr[:1], n_lags - trace_arr.shape[0], axis=0) if trace_arr.shape[0] else np.zeros((n_lags, 2), np.float32)
        trace_arr = np.vstack([trace_arr, pad])

    return trace_arr


def compress_to_population_vec(y_tensor) -> np.ndarray:
    """
    Compress [T,N,H,W] -> [T,N] by spatial mean.
    """
    if isinstance(y_tensor, torch.Tensor):
        y_np = y_tensor.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_tensor)
    if y_np.ndim == 4:
        return y_np.mean(axis=(2, 3))
    if y_np.ndim == 2:
        return y_np
    raise ValueError(f"Unexpected y shape: {y_np.shape}")


def make_full_stack(img_gray: np.ndarray, T: int, n_lags: int) -> np.ndarray:
    """
    Build [T+n_lags, H, W] stack by repeating the static background image.
    """
    Trep = int(T) + int(n_lags)
    return np.repeat(img_gray[np.newaxis, ...], Trep, axis=0)


def get_trial_stim_and_rates(
    eyepos_xy: np.ndarray,
    full_stack: np.ndarray,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    out_size: Tuple[int, int],
    n_lags: int,
    scale: float,
):
    """
    Mirrors translation_covariance.py: make retinal movie + predict rates.
    Returns y_real and y_null (both in model output shape), plus stim tensors.
    """
    device = get_free_device()

    eyepos_xy = np.asarray(eyepos_xy, dtype=np.float32)
    eyepos_t = torch.from_numpy(eyepos_xy).float().to(device)

    null_eyepos = torch.zeros_like(eyepos_t) + eyepos_t.mean(0)

    eye_stim = make_counterfactual_stim(
        full_stack, eyepos_t, out_size=out_size, n_lags=n_lags, scale_factor=scale
    )
    eye_stim_null = make_counterfactual_stim(
        full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale
    )

    # model expects normalized input
    x = (eye_stim - 127.0) / 255.0
    x0 = (eye_stim_null - 127.0) / 255.0

    y = compute_rate_map_batched(model, readout, x)
    y_null = compute_rate_map_batched(model, readout, x0)

    return y, y_null, eye_stim, eye_stim_null


def compute_delta_r(y, y_null) -> np.ndarray:
    """
    Return δr(t,n) in [T,N].
    """
    y_tn = compress_to_population_vec(y)
    mu_tn = compress_to_population_vec(y_null)
    T = min(y_tn.shape[0], mu_tn.shape[0])
    return (y_tn[:T] - mu_tn[:T])


def pca_basis_from_samples(X: np.ndarray, k: int) -> np.ndarray:
    """
    PCA basis (top-k) from samples X [S,N] using SVD on centered samples.
    Returns V [N,k].
    """
    X = np.asarray(X, np.float32)
    m = np.mean(X, axis=0, keepdims=True)
    Xc = X - m
    # guard against NaNs
    ok = np.isfinite(Xc).all(axis=1)
    Xc = Xc[ok]
    if Xc.shape[0] < (k + 2):
        raise RuntimeError(f"Not enough samples for PCA: {Xc.shape[0]} < {k+2}")
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T[:, :k]


def project(X_tn: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Project [T,N] onto V [N,k] -> [T,k]
    """
    return np.asarray(X_tn, np.float32) @ np.asarray(V, np.float32)


def _time_colorline(ax, x, y, lw=2.0, cmap=plt.cm.viridis):
    x = np.asarray(x); y = np.asarray(y)
    t = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=cmap(t[i]), lw=lw)


def _time_colorline_3d(ax, x, y, z, lw=2.0, cmap=plt.cm.viridis):
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    t = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cmap(t[i]), lw=lw)


def pause_if_needed(pause: bool, msg: str = "Press Enter to continue..."):
    if pause:
        input(msg)


# %% -------------------------
# Background image resolution
# -------------------------

def find_datayates_root(data_root: Optional[str]) -> Optional[str]:
    """
    Resolve DataYatesV1 root:
      1) explicit data_root
      2) env DATAYATESV1_ROOT
      3) common candidates
    """
    if data_root and os.path.isdir(data_root):
        return data_root

    env_root = os.environ.get("DATAYATESV1_ROOT")
    if env_root and os.path.isdir(env_root):
        return env_root

    candidates = [
        os.path.join(ROOT, "..", "DataYatesV1"),
        os.path.join(os.path.expanduser("~"), "DataYatesV1"),
        "/home/declan/DataYatesV1/DataYatesV1",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def resolve_background_image(
    stim_key: str,
    backgrounds_dir: Optional[str],
    data_root: Optional[str],
) -> Optional[np.ndarray]:
    """
    Load image from:
      1) entry cache (not handled here)
      2) backgrounds_dir if provided
      3) DataYatesV1 get_backimage_directory() if available
      4) <data_root>/exp/SupportData/Backgrounds fallback

    Returns grayscale float32 or None.
    """
    # absolute path
    if os.path.isabs(stim_key) and os.path.isfile(stim_key):
        try:
            return _to_gray_float32(plt.imread(stim_key))
        except Exception:
            return None

    bg_root = None
    if backgrounds_dir and os.path.isdir(backgrounds_dir):
        bg_root = backgrounds_dir
    elif get_backimage_directory is not None:
        try:
            p = get_backimage_directory()
            if p and os.path.isdir(str(p)):
                bg_root = str(p)
        except Exception:
            bg_root = None
    if bg_root is None and data_root:
        cand = os.path.join(data_root, "exp", "SupportData", "Backgrounds")
        if os.path.isdir(cand):
            bg_root = cand

    if bg_root is None:
        return None

    # direct join
    direct = os.path.join(bg_root, stim_key)
    if os.path.isfile(direct):
        try:
            return _to_gray_float32(plt.imread(direct))
        except Exception:
            return None

    # case/extension-insensitive lookup
    target = os.path.basename(stim_key).lower()
    target_noext = os.path.splitext(target)[0]

    matches = []
    for dirpath, _, files in os.walk(bg_root):
        for fn in files:
            fn_low = fn.lower()
            fp = os.path.join(dirpath, fn)
            if fn_low == target:
                matches.append(fp)
            elif os.path.splitext(fn_low)[0] == target_noext:
                matches.append(fp)
            elif target_noext and (target_noext in fn_low):
                matches.append(fp)

    for fp in matches:
        try:
            return _to_gray_float32(plt.imread(fp))
        except Exception:
            continue

    return None


# %% -------------------------
# Plot functions
# -------------------------

def plot_image_and_eyepath(img_gray: np.ndarray, trace_xy: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_gray, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    ax_in = fig.add_axes([0.62, 0.62, 0.33, 0.33])
    ax_in.set_title("Eye path (native units)", fontsize=9)
    _time_colorline(ax_in, trace_xy[:, 0], trace_xy[:, 1], lw=2.0)
    ax_in.set_xlabel("x", fontsize=8)
    ax_in.set_ylabel("y", fontsize=8)
    ax_in.tick_params(labelsize=8)
    ax_in.grid(True, alpha=0.2)

    plt.show()
    return fig


def plot_neural_traj_3d(delta_tn: np.ndarray, V3: np.ndarray, title: str):
    Z = project(delta_tn, V3)
    if Z.shape[1] < 3:
        Z = np.pad(Z, ((0, 0), (0, 3 - Z.shape[1])), constant_values=0.0)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    _time_colorline_3d(ax, Z[:, 0], Z[:, 1], Z[:, 2], lw=2.0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()
    return fig


def plot_two_stims_shared_embedding(deltaA: np.ndarray, deltaB: np.ndarray, title: str):
    X = np.vstack([deltaA, deltaB])
    V3 = pca_basis_from_samples(X, k=3)
    ZA = project(deltaA, V3)
    ZB = project(deltaB, V3)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    _time_colorline_3d(ax, ZA[:, 0], ZA[:, 1], ZA[:, 2], lw=2.0)
    _time_colorline_3d(ax, ZB[:, 0], ZB[:, 1], ZB[:, 2], lw=2.0)
    ax.set_title(title + "\n(two trajectories, each time-colored)")
    ax.set_xlabel("shared PC1")
    ax.set_ylabel("shared PC2")
    ax.set_zlabel("shared PC3")
    plt.show()
    return fig


def plot_many_traces_pc12(deltas: List[np.ndarray], V2: np.ndarray, title: str, highlight_idx: int = 0):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)

    for i, d in enumerate(deltas):
        Z = project(d, V2)
        lw = 2.5 if i == highlight_idx else 1.0
        _time_colorline(ax, Z[:, 0], Z[:, 1], lw=lw)

    plt.show()
    return fig


def plot_eye_vs_neural_coeffs(trace_xy: np.ndarray, delta_tn: np.ndarray, U2: np.ndarray, title: str):
    A = project(delta_tn, U2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    axs[0].set_title("Eye trajectory (xy)")
    _time_colorline(axs[0], trace_xy[:, 0], trace_xy[:, 1], lw=2.0)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True, alpha=0.2)

    axs[1].set_title("Neural coefficients (a1, a2)")
    _time_colorline(axs[1], A[:, 0], A[:, 1], lw=2.0)
    axs[1].set_xlabel("a1")
    axs[1].set_ylabel("a2")
    axs[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    plt.show()
    return fig


# %% -------------------------
# Pipeline runner
# -------------------------

def compute_delta_for_trace(
    img_gray: np.ndarray,
    trace_xy: np.ndarray,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    out_size: Tuple[int, int],
    n_lags: int,
    scale: float,
    T_null: int = 540,
) -> np.ndarray:
    """
    Compute δr(t,n) for a single trace.
    Null response is computed from a stationary trace on the same image.
    """
    trace_xy = ensure_trace_xy(trace_xy, n_lags=n_lags)

    # null
    null_trace = np.zeros((T_null, 2), np.float32)
    stack_null = make_full_stack(img_gray, T=T_null, n_lags=n_lags)
    y0, y0null, *_ = get_trial_stim_and_rates(
        null_trace, stack_null, model, readout, out_size=out_size, n_lags=n_lags, scale=scale
    )
    # Note: y0null is null-of-null (same), but compute_delta_r expects y and y_null
    # We want y_null for the actual trace; simplest is: use y0 as y_null baseline.
    y_null = y0  # stationary baseline for this image

    # real trace
    stack = make_full_stack(img_gray, T=trace_xy.shape[0], n_lags=n_lags)
    y, _, *_ = get_trial_stim_and_rates(
        trace_xy, stack, model, readout, out_size=out_size, n_lags=n_lags, scale=scale
    )

    delta = compute_delta_r(y, y_null)

    # align time length for plotting: drop extra lag frames if desired
    # Here we keep whatever model produces; user can slice interactively.
    return delta


def load_basis_from_cov_results(cov_results: Dict, stim_key: str) -> Optional[Dict[str, np.ndarray]]:
    """
    If all_cov_results.pkl contains Sigma and U_pca2 for this stim_key, return them.
    """
    if not cov_results:
        return None
    if stim_key not in cov_results:
        return None
    d = cov_results[stim_key]
    if "Sigma" not in d or "U_pca2" not in d:
        return None
    Sigma = np.asarray(d["Sigma"])
    w, V = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]
    V3 = V[:, order[:3]]
    U2 = np.asarray(d["U_pca2"])
    return {"V3": V3, "U2": U2}


def run_interactive(
    results_pkl: str,
    cov_results_pkl: Optional[str],
    stimA: str,
    stimB: Optional[str],
    trace_idx: int,
    num_traces: int,
    out_size: Tuple[int, int],
    n_lags: int,
    scale: float,
    backgrounds_dir: Optional[str],
    data_root: Optional[str],
    pause: bool,
):
    """
    Runs the figure sequence with pauses between each figure.
    Returns a dict of intermediate objects for further inspection.
    """
    back = load_pickle(results_pkl)
    keys = list(back.keys())
    if stimA is None:
        stimA = keys[0]
    if stimA not in back:
        raise KeyError(f"stimA not in results_pkl keys: {stimA}")

    if stimB is not None and stimB not in back:
        raise KeyError(f"stimB not in results_pkl keys: {stimB}")

    # Resolve images + traces
    data_root = find_datayates_root(data_root)
    cov_results = load_pickle(cov_results_pkl) if (cov_results_pkl and os.path.exists(cov_results_pkl)) else {}

    def get_img_and_traces(stim_key: str):
        entry = back[stim_key]
        traces = entry.get("eyepos", [])
        if len(traces) == 0:
            raise RuntimeError(f"No eyepos in entry for {stim_key}")

        img_gray = None
        if "image" in entry and entry["image"] is not None:
            img_gray = _to_gray_float32(entry["image"])
        else:
            img_gray = resolve_background_image(stim_key, backgrounds_dir=backgrounds_dir, data_root=data_root)
            if img_gray is None:
                raise RuntimeError(
                    f"Could not resolve image for {stim_key}.\n"
                    f"Provide --backgrounds-dir or --data-root / DATAYATESV1_ROOT, or ensure entry['image'] exists."
                )
        return img_gray, traces

    imgA, tracesA = get_img_and_traces(stimA)
    traceA = ensure_trace_xy(tracesA[trace_idx], n_lags=n_lags)

    # Load model
    model, readout = load_model_and_readout()
    model.eval(); readout.eval()

    # --- Plot 1: image + eye path
    plot_image_and_eyepath(imgA, traceA, title=f"Stim A: {os.path.basename(stimA)}")
    pause_if_needed(pause, "Plot 1 shown. Enter to continue...")

    # Compute deltaA
    deltaA = compute_delta_for_trace(
        imgA, traceA, model, readout,
        out_size=out_size, n_lags=n_lags, scale=scale
    )
    print(f"[deltaA] shape={deltaA.shape} (T,N)")

    # Choose basis for A
    basisA = load_basis_from_cov_results(cov_results, stimA)
    if basisA is not None:
        V3 = basisA["V3"]
        U2 = basisA["U2"]
        print("[basis] using Sigma/U_pca2 from cov_results")
    else:
        V3 = pca_basis_from_samples(deltaA, k=3)
        U2 = V3[:, :2]
        print("[basis] using PCA from this trace's δr samples")

    # --- Plot 2: neural trajectory 3D
    plot_neural_traj_3d(deltaA, V3, title="Stim A: δr(t) in PC space (PC1–PC3)")
    pause_if_needed(pause, "Plot 2 shown. Enter to continue...")

    # --- Plot 5: eye vs neural coefficients (a1,a2)
    # slice eye trace to match delta length (best effort)
    Tplot = min(traceA.shape[0], deltaA.shape[0])
    plot_eye_vs_neural_coeffs(traceA[:Tplot], deltaA[:Tplot], U2, title="Stim A: eye xy vs neural (a1,a2)")
    pause_if_needed(pause, "Plot 5 shown. Enter to continue...")

    # --- Plot 4: many traces on stimA
    n_show = min(num_traces, len(tracesA))
    deltas = []
    for i in range(n_show):
        tr = ensure_trace_xy(tracesA[i], n_lags=n_lags)
        d = compute_delta_for_trace(imgA, tr, model, readout, out_size=out_size, n_lags=n_lags, scale=scale)
        deltas.append(d)
    Xcat = np.vstack(deltas)
    V2 = pca_basis_from_samples(Xcat, k=2)
    plot_many_traces_pc12(deltas, V2, title=f"Stim A: {n_show} traces in shared PC1–PC2", highlight_idx=0)
    pause_if_needed(pause, "Plot 4 shown. Enter to continue...")

    # --- Plot 3: two stimuli, same trace
    deltaB = None
    imgB = None
    if stimB is not None:
        imgB, _ = get_img_and_traces(stimB)
        plot_image_and_eyepath(imgB, traceA, title=f"Stim B: {os.path.basename(stimB)} (same trace)")
        pause_if_needed(pause, "Stim B image shown. Enter to continue...")

        deltaB = compute_delta_for_trace(
            imgB, traceA, model, readout,
            out_size=out_size, n_lags=n_lags, scale=scale
        )
        print(f"[deltaB] shape={deltaB.shape} (T,N)")
        T = min(deltaA.shape[0], deltaB.shape[0])
        plot_two_stims_shared_embedding(deltaA[:T], deltaB[:T], title=f"Same trace, two images\nA={os.path.basename(stimA)}  B={os.path.basename(stimB)}")
        pause_if_needed(pause, "Plot 3 shown. Enter to finish...")

    return {
        "stimA": stimA,
        "stimB": stimB,
        "imgA": imgA,
        "imgB": imgB,
        "traceA": traceA,
        "deltaA": deltaA,
        "deltaB": deltaB,
        "V3_A": V3,
        "U2_A": U2,
        "V2_many": V2,
    }


# %% -------------------------
# CLI entrypoint
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-pkl", type=str, default=os.path.join(ROOT, "declan", "backimage_fixation_results.pkl"))
    ap.add_argument("--cov-results-pkl", type=str, default=os.path.join(ROOT, "declan", "translation_covariance", "all_cov_results.pkl"))
    ap.add_argument("--stimA", type=str, default=None)
    ap.add_argument("--stimB", type=str, default=None)
    ap.add_argument("--trace-idx", type=int, default=0)
    ap.add_argument("--num-traces", type=int, default=15)
    ap.add_argument("--n-lags", type=int, default=32)
    ap.add_argument("--out-size", type=int, nargs=2, default=(151, 151))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--backgrounds-dir", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--pause", action="store_true", help="Pause between plots (interactive stepping).")
    return ap.parse_args()


def main():
    args = parse_args()
    run_interactive(
        results_pkl=args.results_pkl,
        cov_results_pkl=args.cov_results_pkl,
        stimA=args.stimA,
        stimB=args.stimB,
        trace_idx=args.trace_idx,
        num_traces=args.num_traces,
        out_size=tuple(args.out_size),
        n_lags=args.n_lags,
        scale=args.scale,
        backgrounds_dir=args.backgrounds_dir,
        data_root=args.data_root,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()


# %% =========================
# USER CONFIG (interactive)
# Edit these values, then run the “RUN PIPELINE” cell below.
# ===========================

# Paths
RESULTS_PKL = os.path.join(ROOT, "declan", "backimage_fixation_results.pkl")
COV_RESULTS_PKL = os.path.join(ROOT, "declan", "translation_covariance", "all_cov_results.pkl")

# Stimulus keys (must match keys in RESULTS_PKL)
STIM_A = None   # e.g. "Colony_Bonnie.JPG"
STIM_B = None   # optional, e.g. "Hawaii_trees.JPG"

# Data roots (only needed if entry['image'] is missing)
BACKGROUNDS_DIR = None  # preferred: "/path/to/SupportData/Backgrounds"
DATA_ROOT = None        # optional: DataYatesV1 repo root

# Analysis params
TRACE_IDX = 0
NUM_TRACES = 15
N_LAGS = 32
OUT_SIZE = (151, 151)
SCALE = 1.0

# Interactive stepping
PAUSE_BETWEEN_PLOTS = True


# %% RUN PIPELINE (interactive)
# Run this cell to step through plots sequentially.
# Returned dict contains intermediate objects for inspection.

# Uncomment to run in interactive mode:
# out = run_interactive(
#     results_pkl=RESULTS_PKL,
#     cov_results_pkl=COV_RESULTS_PKL if os.path.exists(COV_RESULTS_PKL) else None,
#     stimA=STIM_A,
#     stimB=STIM_B,
#     trace_idx=TRACE_IDX,
#     num_traces=NUM_TRACES,
#     out_size=OUT_SIZE,
#     n_lags=N_LAGS,
#     scale=SCALE,
#     backgrounds_dir=BACKGROUNDS_DIR,
#     data_root=DATA_ROOT,
#     pause=PAUSE_BETWEEN_PLOTS,
# )
# out.keys()
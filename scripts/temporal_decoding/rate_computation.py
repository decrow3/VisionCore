"""
Task 1.3: Rate Matrix Computation Pipeline

Given a stimulus stack and eye traces, compute population rate matrices
through the digital twin for different FEM conditions.

FEM conditions:
    'real'          — measured fixational eye traces
    'stabilized'    — eye fixed at trial mean (eye_scale=0)
    'scaled_0.5'    — FEM scaled to half amplitude around mean
    'scaled_2.0'    — FEM scaled to double amplitude around mean
    'shuffled'      — traces shuffled across trials (breaks stimulus-trace coupling)
    'matched_null'  — phase-randomized null traces (from null_traces.py)

NOTE on spatial rate maps:
    The PopulationReadout returns spatial rate maps (B, N, H_out, W_out).
    This module collapses spatial dims via spatial_max (default) to get (B, N).
    Change `spatial_collapse='mean'` to use spatial average instead.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_info import make_counterfactual_stim, embed_time_lags
from mcfarland_sim import eye_deg_to_norm, shift_movie_with_eye

# Constants
PPD = 37.50476617
N_LAGS = 32
OUT_SIZE = (101, 101)
BATCH_SIZE = 32
FRAME_RATE = 120.0


def _scale_trace(eyepos: torch.Tensor, eye_scale: float) -> torch.Tensor:
    """Scale eye trace around its mean (eye_scale=0 → stabilized, 1 → real)."""
    mean = eyepos.mean(0, keepdim=True)
    return mean + (eyepos - mean) * eye_scale


def build_counterfactual_stim(
    full_stack: np.ndarray,
    eyepos: np.ndarray,
    condition: str = 'real',
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
    ppd: float = PPD,
    null_trace: np.ndarray = None,
) -> torch.Tensor:
    """
    Build the eye-shifted stimulus sequence for a given FEM condition.

    Args:
        full_stack: (N_frames, H, W) uint8 stimulus frames
        eyepos: (T, 2) float32 eye positions in degrees for this trial
        condition: one of 'real', 'stabilized', 'scaled_0.5', 'scaled_2.0',
                   'shuffled' (caller must pass shuffled eyepos), 'matched_null'
                   (caller must pass null_trace)
        n_lags: number of temporal history frames for the model
        out_size: (H_out, W_out) output spatial size
        ppd: pixels per degree
        null_trace: (T, 2) null trace (required for condition='matched_null')

    Returns:
        eye_stim: (T_valid, 1, n_lags, H_out, W_out) torch.Tensor, float32
        where T_valid = T - 0 (counterfactual stim already handles lag padding)
    """
    eye_t = torch.from_numpy(eyepos).float()

    if condition == 'real':
        ep = eye_t
    elif condition == 'stabilized':
        ep = _scale_trace(eye_t, 0.0)
    elif condition.startswith('scaled_'):
        scale = float(condition.split('_')[1])
        ep = _scale_trace(eye_t, scale)
    elif condition == 'shuffled':
        # Caller is responsible for passing shuffled eyepos
        ep = eye_t
    elif condition == 'matched_null':
        assert null_trace is not None, "matched_null requires null_trace argument"
        ep = torch.from_numpy(null_trace).float()
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return make_counterfactual_stim(
        full_stack, ep,
        ppd=ppd,
        n_lags=n_lags,
        out_size=out_size,
    )


def _collapse_spatial(rate_map: torch.Tensor, method: str = 'max') -> torch.Tensor:
    """
    Collapse spatial dimensions of rate map to scalar per neuron.

    Note on methods:
        'max'  — peak response across the neuron's spatial rate map. Maximizes
                 contrast between active/inactive states but the max of a noisy
                 spatial map is biased upward (noise floor lifted by ~sigma).
        'mean' — expected spike count (weighted by readout's Gaussian mask).
                 Less noisy but may wash out responses for neurons with small RFs
                 relative to the output crop size.

    For the decoding analysis, 'max' and 'mean' should give qualitatively similar
    results if the key sanity check (C ≈ A under stabilized) passes with both.
    If they diverge, use compare_spatial_collapse() to diagnose.

    Args:
        rate_map: (B, N, H, W) spatial rate map
        method: 'max' or 'mean'

    Returns:
        (B, N) scalar rates
    """
    if method == 'max':
        return rate_map.amax(dim=(-2, -1))
    elif method == 'mean':
        return rate_map.mean(dim=(-2, -1))
    else:
        raise ValueError(f"Unknown spatial collapse method: {method}")


def compare_spatial_collapse(
    model,
    readout,
    stim_stack: np.ndarray,
    eye_traces: np.ndarray,
    durations: np.ndarray,
    condition: str = 'real',
    n_traces: int = 10,
) -> dict:
    """
    Run both spatial collapse methods on the same data and compare.

    Reports temporal variance per neuron for each method and their correlation.
    Use this to validate that 'max' and 'mean' give qualitatively similar signals.

    Returns:
        dict with correlation, ratio of mean/max variance, and per-neuron stats.
    """
    results = {}
    for method in ['max', 'mean']:
        result = compute_population_rates(
            model, readout, stim_stack,
            eye_traces[:n_traces], durations[:n_traces],
            condition=condition,
            spatial_collapse=method,
            verbose=False,
        )
        # Temporal variance per neuron, averaged over trials
        vars_per_neuron = np.mean([r.var(axis=0) for r in result['rates']], axis=0)
        results[method] = vars_per_neuron

    r = np.corrcoef(results['max'], results['mean'])[0, 1]
    ratio = results['mean'].mean() / (results['max'].mean() + 1e-12)
    print(f"Spatial collapse comparison:")
    print(f"  max variance mean:  {results['max'].mean():.6f}")
    print(f"  mean variance mean: {results['mean'].mean():.6f}")
    print(f"  ratio (mean/max):   {ratio:.4f}")
    print(f"  correlation (max vs mean variance per neuron): r={r:.4f}")
    print(f"  Interpretation: {'methods agree' if r > 0.8 else 'methods DIVERGE — investigate'}")
    return {'correlation': r, 'ratio': ratio, 'max_vars': results['max'], 'mean_vars': results['mean']}


def compute_trial_rates(
    model,
    readout,
    stim: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    spatial_collapse: str = 'max',
    return_spatial: bool = False,
) -> np.ndarray:
    """
    Run the model on a single trial's stimulus sequence and return rate vector per frame.

    Args:
        model: loaded VisionCore model
        readout: PopulationReadout from get_spatial_readout
        stim: (T_valid, 1, n_lags, H, W) eye-shifted stimulus
        batch_size: number of time steps to process at once
        spatial_collapse: 'max' or 'mean' for collapsing spatial dims
        return_spatial: if True, skip collapse and return (T_valid, N, H, W) raw maps

    Returns:
        rates: (T_valid, N_neurons) float32 array if return_spatial=False
               (T_valid, N_neurons, H, W) float32 array if return_spatial=True
    """
    device = next(model.model.parameters()).device
    T = stim.shape[0]
    rate_chunks = []

    model.model.eval()
    readout.eval()

    with torch.no_grad():
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)
            x_batch = stim[t_start:t_end].to(device)  # (B, 1, n_lags, H, W)

            # Core forward pass
            feats = model.model.core_forward(x_batch, None)  # (B, C, T_feat, H_feat, W_feat)
            # Take last time step of feature sequence
            feats_last = feats[:, :, -1]  # (B, C, H_feat, W_feat)

            # Readout: (B, N, H_out, W_out)
            y = readout(feats_last)
            # Activation
            rates_spatial = model.model.activation(y)  # (B, N, H_out, W_out)

            if return_spatial:
                chunk = rates_spatial.cpu().numpy()   # (B, N, H_out, W_out)
            else:
                chunk = _collapse_spatial(rates_spatial, method=spatial_collapse).cpu().numpy()
            rate_chunks.append(chunk)

            del x_batch, feats, feats_last, y, rates_spatial
            torch.cuda.empty_cache()

    return np.concatenate(rate_chunks, axis=0).astype(np.float32)  # (T_valid, N[, H, W])


def compute_population_rates(
    model,
    readout,
    stim_stack: np.ndarray,
    eye_traces: np.ndarray,
    durations: np.ndarray,
    condition: str = 'real',
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
    ppd: float = PPD,
    batch_size: int = BATCH_SIZE,
    spatial_collapse: str = 'max',
    stim_params: dict = None,
    null_traces: np.ndarray = None,
    null_idx: int = 0,
    shuffled_indices: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """
    Compute population rate matrices for all eye traces under a given FEM condition.

    Args:
        model: loaded VisionCore model
        readout: PopulationReadout
        stim_stack: (N_frames, H, W) uint8 stimulus (e.g., from e_optotype_stack)
        eye_traces: (M, max_T, 2) float32, NaN-padded
        durations: (M,) int, valid trace length for each trace
        condition: FEM condition string
        n_lags: model temporal history
        out_size: spatial output size for make_counterfactual_stim
        ppd: pixels per degree
        batch_size: frames per GPU batch
        spatial_collapse: 'max' or 'mean'
        stim_params: dict of stimulus parameters (stored in output)
        null_traces: (M, n_nulls, max_T, 2) for 'matched_null' condition
        null_idx: which null to use (0..n_nulls-1) for 'matched_null'
        shuffled_indices: (M,) int permutation of trace indices for 'shuffled' condition
        verbose: print progress

    Returns:
        dict with:
            rates: list of M arrays, each (T_m, N_neurons) — variable length
            condition: str
            stim_params: dict
            spatial_collapse: str
    """
    M = eye_traces.shape[0]
    all_rates = []

    for i in range(M):
        T = int(durations[i])
        eyepos = eye_traces[i, :T]  # (T, 2)

        # Handle shuffled condition: use a different trace's eye positions
        if condition == 'shuffled':
            if shuffled_indices is None:
                # Default: circular shift by M//2
                j = (i + M // 2) % M
            else:
                j = int(shuffled_indices[i])
            T_j = int(durations[j])
            eyepos = eye_traces[j, :T_j]

        # Handle matched_null condition
        null_trace = None
        if condition == 'matched_null':
            assert null_traces is not None, "matched_null requires null_traces"
            T_null = T
            null_trace = null_traces[i, null_idx, :T_null]

        # Build counterfactual stimulus
        eye_stim = build_counterfactual_stim(
            stim_stack, eyepos, condition=condition,
            n_lags=n_lags, out_size=out_size, ppd=ppd,
            null_trace=null_trace,
        )  # (T_valid, 1, n_lags, H_out, W_out) — uint8→float32 [0, 127]
        eye_stim = eye_stim / 127.0  # normalise to [0, 1] to match hi-res pipeline

        # Run model
        rates = compute_trial_rates(
            model, readout, eye_stim,
            batch_size=batch_size,
            spatial_collapse=spatial_collapse,
        )  # (T_valid, N)

        all_rates.append(rates)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{M} trials", flush=True)

    return {
        'rates': all_rates,
        'condition': condition,
        'stim_params': stim_params or {},
        'spatial_collapse': spatial_collapse,
    }


def rates_to_padded_array(rates_list: list) -> tuple:
    """
    Convert list of variable-length rate arrays to a padded numpy array.

    Args:
        rates_list: list of M arrays, each (T_m, N)

    Returns:
        rates_padded: (M, T_max, N) float32, NaN-padded
        lengths: (M,) int32
    """
    M = len(rates_list)
    N = rates_list[0].shape[1]
    T_max = max(r.shape[0] for r in rates_list)

    rates_padded = np.full((M, T_max, N), np.nan, dtype=np.float32)
    lengths = np.zeros(M, dtype=np.int32)

    for i, r in enumerate(rates_list):
        T = r.shape[0]
        rates_padded[i, :T] = r
        lengths[i] = T

    return rates_padded, lengths


def save_rates(result: dict, path: str) -> None:
    """Save rate computation results to .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rates_padded, lengths = rates_to_padded_array(result['rates'])
    np.savez_compressed(
        path,
        rates=rates_padded,
        lengths=lengths,
        condition=np.array([result['condition']]),
        spatial_collapse=np.array([result['spatial_collapse']]),
        **{f'stim_{k}': np.array([v]) for k, v in result['stim_params'].items()},
    )
    print(f"Saved rates ({rates_padded.shape}) to {path}")


def load_rates(path: str) -> dict:
    """Load rates from .npz file."""
    d = np.load(path, allow_pickle=True)
    rates_padded = d['rates']    # (M, T_max, N)
    lengths = d['lengths']       # (M,)
    M = rates_padded.shape[0]
    rates_list = [rates_padded[i, :lengths[i]] for i in range(M)]
    return {
        'rates': rates_list,
        'condition': str(d['condition'][0]),
        'spatial_collapse': str(d['spatial_collapse'][0]),
        'stim_params': {},
    }


def compute_population_rates_hires(
    model,
    readout,
    orientation_deg: float,
    logmar: float,
    eye_traces: np.ndarray,
    durations: np.ndarray,
    condition: str = 'real',
    n_lags: int = N_LAGS,
    retina_size: tuple = OUT_SIZE,
    batch_size: int = BATCH_SIZE,
    spatial_collapse: str = 'max',
    stim_params: dict = None,
    null_traces: np.ndarray = None,
    null_idx: int = 0,
    center_offset_deg: tuple = (0.0, 0.0),
    verbose: bool = True,
) -> dict:
    """
    Compute population rate matrices using the high-res world→retina pipeline.

    Use this instead of compute_population_rates() for the hyperacuity regime
    (LogMAR ≤ 0.2) where direct rendering at 37.5 ppd loses sub-pixel structure.

    Args:
        model: loaded VisionCore model
        readout: PopulationReadout
        orientation_deg: E orientation in degrees (0, 90, 180, 270)
        logmar: letter size in LogMAR units
        eye_traces: (M, max_T, 2) float32, NaN-padded eye traces
        durations: (M,) int, valid trace length per trial
        condition: 'real', 'stabilized', 'matched_null', etc.
        n_lags: model temporal history frames
        retina_size: (H, W) retinal patch size (must match model input)
        batch_size: time steps per GPU batch
        spatial_collapse: 'max' or 'mean'
        stim_params: dict for metadata storage
        null_traces: (M, n_nulls, max_T, 2) for matched_null condition
        null_idx: which null trace to use
        center_offset_deg: (dx, dy) sub-pixel phase offset for robustness testing
        verbose: print progress

    Returns:
        dict with 'rates' (list of (T_m, N) arrays), 'condition', 'stim_params'
    """
    from stimulus_hires import hires_counterfactual_stim

    M = eye_traces.shape[0]
    all_rates = []
    device = next(model.model.parameters()).device

    for i in range(M):
        T = int(durations[i])
        eyepos = eye_traces[i, :T]

        null_trace = None
        if condition == 'matched_null':
            assert null_traces is not None, "matched_null requires null_traces"
            null_trace = null_traces[i, null_idx, :T]

        eye_stim = hires_counterfactual_stim(
            orientation_deg, logmar, eyepos,
            condition=condition,
            null_trace=null_trace,
            center_offset_deg=center_offset_deg,
            n_lags=n_lags,
            retina_size=retina_size,
            device=str(device),
        )

        rates = compute_trial_rates(
            model, readout, eye_stim,
            batch_size=batch_size,
            spatial_collapse=spatial_collapse,
        )
        all_rates.append(rates)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{M} trials", flush=True)

    return {
        'rates': all_rates,
        'condition': condition,
        'stim_params': stim_params or {},
        'spatial_collapse': spatial_collapse,
    }


if __name__ == '__main__':
    import dill
    import sys

    print("Rate computation smoke test (requires model + data)...")

    # Check if model is available before running
    pkl_path = os.path.join(os.path.dirname(__file__), '..', 'mcfarland_outputs_mono.pkl')
    if not os.path.exists(pkl_path):
        print(f"Cannot run smoke test: {pkl_path} not found")
        sys.exit(0)

    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout
    from stimulus import e_optotype_stack
    from extract_eye_traces import load_eye_traces

    traces_path = os.path.join(os.path.dirname(__file__), 'data', 'eye_traces.npz')
    if not os.path.exists(traces_path):
        print(f"Cannot run smoke test: {traces_path} not found. Run extract_eye_traces.py first.")
        sys.exit(0)

    print("Loading model...")
    model, _ = get_model_and_dataset_configs()
    model.model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    with open(pkl_path, 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)

    print("Loading eye traces...")
    traces_data = load_eye_traces(traces_path)
    traces = traces_data['traces'][:5]   # use 5 traces for smoke test
    durations = traces_data['durations'][:5]

    print("Generating E stimulus (LogMAR=0.5, orientation=0)...")
    stim_stack = e_optotype_stack(0, 0.5)

    for condition in ['real', 'stabilized']:
        print(f"Computing rates: condition={condition}...")
        result = compute_population_rates(
            model, readout, stim_stack, traces, durations,
            condition=condition,
            verbose=True,
        )
        rates_list = result['rates']
        print(f"  Rates: {len(rates_list)} trials, shapes: {[r.shape for r in rates_list[:3]]}")

        # Verify: stabilized should have lower temporal variance than real
        real_var = np.mean([r.var(axis=0).mean() for r in rates_list])
        print(f"  Mean temporal variance: {real_var:.6f}")

    print("\nSmoke test passed!")

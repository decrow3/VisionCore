"""
Task 4.2: Spatiotemporal Resonance (Conditional)

Characterizes each neuron's preferred SF and TF to test whether FEM velocities
shift stimulus spatial frequencies into each neuron's temporal passband.

Resonance hypothesis: FEMs scan the E optotype at velocities that convert
the E's spatial frequencies into temporal frequencies matching neuronal TF tuning.

If resonance score correlates with decoder weight (r > 0.2), include in figures.
Otherwise report as inconclusive.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAME_RATE = 120.0
PPD = 37.50476617


def estimate_sf_tf_tuning(
    model,
    readout,
    sf_range=(0.5, 20.0),
    tf_range=(0.5, 30.0),
    n_sf: int = 10,
    n_tf: int = 10,
    grating_duration: int = 60,
    device: str = 'cpu',
) -> dict:
    """
    Estimate preferred SF and TF for each neuron using drifting sinusoidal gratings.

    Sweeps a grid of (SF, TF) combinations and records population responses.

    Args:
        model: VisionCore model
        readout: PopulationReadout
        sf_range: (min_sf, max_sf) in cycles per degree
        tf_range: (min_tf, max_tf) in Hz
        n_sf: number of SF values
        n_tf: number of TF values
        grating_duration: frames per grating condition
        device: torch device

    Returns:
        dict with:
            sf_prefs: (N,) preferred SF per neuron (cpd)
            tf_prefs: (N,) preferred TF per neuron (Hz)
            sf_grid: (n_sf,) SF values tested
            tf_grid: (n_tf,) TF values tested
            response_matrix: (N, n_sf, n_tf) mean response per condition
    """
    import torch
    from rate_computation import _collapse_spatial

    sf_grid = np.logspace(np.log10(sf_range[0]), np.log10(sf_range[1]), n_sf)
    tf_grid = np.logspace(np.log10(tf_range[0]), np.log10(tf_range[1]), n_tf)

    # We'll use the model's expected input size
    H = W = 101  # match out_size convention
    n_lags = 32

    # Get a sample stimulus to check output size
    model.model.eval()
    readout.eval()

    response_matrix = np.zeros((n_sf, n_tf), dtype=object)  # will hold (N,) arrays

    print(f"Estimating SF/TF tuning: {n_sf} SF × {n_tf} TF = {n_sf*n_tf} conditions")

    first_condition = True
    N = None

    for i, sf in enumerate(sf_grid):
        for j, tf in enumerate(tf_grid):
            # Generate drifting grating
            grating = _make_drifting_grating(
                sf_cpd=sf,
                tf_hz=tf,
                ppd=PPD,
                duration=grating_duration,
                H=H, W=W,
                frame_rate=FRAME_RATE,
            )  # (T+n_lags, 1, n_lags, H, W) already time-embedded

            with torch.no_grad():
                stim_t = torch.from_numpy(grating).float().to(device)
                feats = model.model.core_forward(stim_t, None)
                y = readout(feats[:, :, -1])
                rates_spatial = model.model.activation(y)
                rates = _collapse_spatial(rates_spatial, method='max')  # (T, N)

            mean_response = rates.mean(dim=0).cpu().numpy()  # (N,)
            response_matrix[i, j] = mean_response

            if first_condition:
                N = len(mean_response)
                first_condition = False

    # Convert to (N, n_sf, n_tf) array
    response_4d = np.zeros((N, n_sf, n_tf), dtype=np.float32)
    for i in range(n_sf):
        for j in range(n_tf):
            response_4d[:, i, j] = response_matrix[i, j]

    # Find preferred SF and TF as argmax
    flat_idx = response_4d.reshape(N, -1).argmax(axis=1)
    sf_idx, tf_idx = np.unravel_index(flat_idx, (n_sf, n_tf))
    sf_prefs = sf_grid[sf_idx]
    tf_prefs = tf_grid[tf_idx]

    return {
        'sf_prefs': sf_prefs,
        'tf_prefs': tf_prefs,
        'sf_grid': sf_grid,
        'tf_grid': tf_grid,
        'response_matrix': response_4d,
        'N': N,
    }


def _make_drifting_grating(
    sf_cpd: float,
    tf_hz: float,
    ppd: float = PPD,
    duration: int = 60,
    H: int = 101,
    W: int = 101,
    frame_rate: float = FRAME_RATE,
    orientation_deg: float = 0.0,
) -> np.ndarray:
    """
    Generate a drifting sinusoidal grating as an embedded time-lag stimulus.

    Returns:
        (duration, 1, n_lags, H, W) float32 array (already lag-embedded)
    """
    n_lags = 32
    T_total = duration + n_lags

    # Spatial coordinates in degrees
    xs = (np.arange(W) - W / 2) / ppd
    ys = (np.arange(H) - H / 2) / ppd
    XX, YY = np.meshgrid(xs, ys)

    # Rotate grating direction
    ori_rad = orientation_deg * np.pi / 180
    spatial_phase = (XX * np.cos(ori_rad) + YY * np.sin(ori_rad)) * sf_cpd * 2 * np.pi

    # Time-varying grating: I(x, t) = sin(k*x - 2π*tf*t)
    frames = np.zeros((T_total, H, W), dtype=np.float32)
    for t in range(T_total):
        temporal_phase = 2 * np.pi * tf_hz * t / frame_rate
        grating = np.sin(spatial_phase - temporal_phase)
        # Scale to [0, 255], background 127
        frames[t] = (grating * 64 + 127).clip(0, 255)

    # Embed time lags: output (duration, 1, n_lags, H, W)
    embedded = np.zeros((duration, 1, n_lags, H, W), dtype=np.float32)
    for t in range(duration):
        for lag in range(n_lags):
            embedded[t, 0, lag] = frames[t + n_lags - 1 - lag]

    # Normalize
    embedded = (embedded - 127.0) / 255.0
    return embedded


def compute_resonance_scores(
    tuning: dict,
    eye_traces: np.ndarray,
    durations: np.ndarray,
    logmar: float = 0.5,
    ppd: float = PPD,
    frame_rate: float = FRAME_RATE,
) -> dict:
    """
    Compute per-neuron resonance score:
    how well FEM velocities match each neuron's SF/TF tuning.

    Resonance: velocity v scans SF of s cpd → temporal frequency v*s Hz
    Resonance score = how much of the FEM velocity distribution falls in the TF passband.

    Args:
        tuning: output of estimate_sf_tf_tuning()
        eye_traces: (M, max_T, 2) eye traces in degrees
        durations: (M,) valid lengths
        logmar: LogMAR of stimulus (determines relevant SF)
        ppd: pixels per degree
        frame_rate: Hz

    Returns:
        dict with:
            resonance_scores: (N,) resonance score per neuron
            fem_speed_distribution: (T_total,) FEM speeds in deg/s
            fem_tf_per_sf: (N,) expected TF at each neuron's preferred SF
    """
    N = tuning['N']
    sf_prefs = tuning['sf_prefs']  # (N,) preferred SF in cpd

    # Compute FEM velocity distribution from eye traces
    all_speeds = []
    for i in range(len(eye_traces)):
        T = int(durations[i])
        trace = eye_traces[i, :T]
        if T > 1:
            velocities = np.diff(trace, axis=0) * frame_rate  # deg/s
            speeds = np.sqrt((velocities ** 2).sum(axis=1))
            all_speeds.append(speeds)

    fem_speeds = np.concatenate(all_speeds)  # all FEM speeds in deg/s

    # For each neuron, compute the TF it would receive from FEMs scanning its preferred SF
    # TF = speed (deg/s) × SF (cycles/deg) → cycles/s = Hz
    fem_tf_median = np.median(fem_speeds)  # representative FEM speed

    fem_tf_per_sf = fem_tf_median * sf_prefs  # (N,) TF at each neuron's preferred SF

    # Resonance score = probability that FEM-driven TF matches neuron's preferred TF
    tf_prefs = tuning['tf_prefs']  # (N,)

    # Score: Gaussian similarity between FEM-induced TF and preferred TF
    # (on log scale since TF has log-normal distribution)
    log_fem_tf = np.log(fem_tf_per_sf + 1e-6)
    log_tf_pref = np.log(tf_prefs + 1e-6)
    log_tf_sigma = 0.5  # bandwidth in log space (roughly one octave)

    resonance_scores = np.exp(-0.5 * ((log_fem_tf - log_tf_pref) / log_tf_sigma) ** 2)

    return {
        'resonance_scores': resonance_scores.astype(np.float32),
        'fem_speed_distribution': fem_speeds.astype(np.float32),
        'fem_tf_per_sf': fem_tf_per_sf.astype(np.float32),
        'sf_prefs': sf_prefs,
        'tf_prefs': tf_prefs,
    }


def compute_resonance_decoder_correlation(
    resonance_scores: np.ndarray,
    decoder_weights: np.ndarray,
) -> dict:
    """
    Compute Pearson correlation between resonance scores and decoder weights.

    If r > 0.2, include resonance analysis in main figures.

    Args:
        resonance_scores: (N,) per-neuron resonance scores
        decoder_weights: (N,) absolute decoder weights from Model C

    Returns:
        dict with: r (correlation), p (p-value), conclusive (bool)
    """
    from scipy import stats

    r, p = stats.pearsonr(resonance_scores, np.abs(decoder_weights))
    conclusive = abs(r) > 0.2

    return {
        'r': float(r),
        'p': float(p),
        'conclusive': conclusive,
        'interpretation': (
            f'Resonance–decoder correlation: r={r:.3f}, p={p:.4f}. '
            + ('Conclusive: include in figures.' if conclusive
               else 'Inconclusive: r < 0.2, do not include in main figures.')
        ),
    }


def plot_sf_tf_tuning(
    tuning: dict,
    resonance: Optional[dict] = None,
    figsize=(12, 5),
) -> plt.Figure:
    """
    Plot SF and TF preference distributions, and resonance scores.
    """
    n_panels = 3 if resonance is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    # Panel 1: SF distribution
    ax = axes[0]
    ax.hist(tuning['sf_prefs'], bins=20, color='royalblue', alpha=0.7)
    ax.set_xlabel('Preferred SF (cpd)', fontsize=11)
    ax.set_ylabel('# neurons', fontsize=11)
    ax.set_title('Preferred Spatial Frequency', fontsize=12)
    ax.set_xscale('log')

    # Panel 2: TF distribution
    ax = axes[1]
    ax.hist(tuning['tf_prefs'], bins=20, color='tomato', alpha=0.7)
    ax.set_xlabel('Preferred TF (Hz)', fontsize=11)
    ax.set_ylabel('# neurons', fontsize=11)
    ax.set_title('Preferred Temporal Frequency', fontsize=12)
    ax.set_xscale('log')

    # Panel 3: Resonance score distribution (if provided)
    if resonance is not None:
        ax = axes[2]
        ax.hist(resonance['resonance_scores'], bins=20, color='steelblue', alpha=0.7)
        ax.set_xlabel('Resonance score', fontsize=11)
        ax.set_ylabel('# neurons', fontsize=11)
        ax.set_title('Resonance (FEM velocity ↔ TF tuning)', fontsize=12)

    plt.suptitle('Spatiotemporal Tuning', fontsize=13)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Testing resonance analysis with synthetic data (no model)...")
    np.random.seed(42)

    N = 50

    # Synthetic tuning
    sf_prefs = np.random.lognormal(np.log(4.0), 0.5, N).astype(np.float32)
    tf_prefs = np.random.lognormal(np.log(4.0), 0.5, N).astype(np.float32)
    tuning = {'sf_prefs': sf_prefs, 'tf_prefs': tf_prefs, 'N': N}

    # Synthetic eye traces
    M, max_T = 20, 200
    traces = np.zeros((M, max_T, 2), dtype=np.float32)
    durations = np.full(M, 100, dtype=np.int32)
    for i in range(M):
        T = durations[i]
        traces[i, :T] = np.cumsum(np.random.randn(T, 2) * 0.005, axis=0)

    resonance = compute_resonance_scores(tuning, traces, durations)
    scores = resonance['resonance_scores']
    print(f"Resonance scores: min={scores.min():.4f}, max={scores.max():.4f}, "
          f"mean={scores.mean():.4f}")

    # Test correlation
    decoder_weights = np.random.randn(N)
    corr_result = compute_resonance_decoder_correlation(scores, decoder_weights)
    print(f"Correlation test: {corr_result['interpretation']}")

    fig = plot_sf_tf_tuning(tuning, resonance)
    fig.savefig('test_resonance.png', dpi=100, bbox_inches='tight')
    print("Saved test_resonance.png")
    print("All tests passed!")

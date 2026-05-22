"""
Single-trial spatial map inspector.

Recomputes the full (T, N, H, W) spatial rate maps for one real-FEM and one
stabilized trial at a given LogMAR value, then visualizes:
  - imshow panels of the (H, W) map at multiple time points
  - Three time series per neuron: amax(map), argmax position, center of mass

Usage:
    python inspect_maps.py [--logmar 0.0] [--orientation 0] [--n_neurons 3]
                           [--trace_idx 0]

Output:
    figures/diag_spatial_maps_lm{logmar}_ori{ori}_trial{idx}.png
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
EYE_TRACES_PATH = os.path.join(DATA_DIR, 'eye_traces.npz')
PKL_PATH = os.path.join(SCRIPT_DIR, '..', 'mcfarland_outputs_mono.pkl')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

N_TIME_PANELS = 6  # number of time snapshots to show


def load_model_readout(device):
    import dill
    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout
    model, _ = get_model_and_dataset_configs(mode='standard')
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    model = model.to(device)
    with open(PKL_PATH, 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)
    return model, readout


def get_maps_for_trial(model, readout, orientation_deg, logmar, eyepos,
                        condition, null_trace=None, device='cpu'):
    """Build stimulus and compute spatial maps for one trial."""
    from stimulus_hires import hires_counterfactual_stim
    from rate_computation import compute_trial_rates

    stim = hires_counterfactual_stim(
        orientation_deg=orientation_deg,
        logmar=logmar,
        eyepos=eyepos,
        condition=condition,
        null_trace=null_trace,
        device=device,
    )  # (T_valid, 1, n_lags, H, W)

    maps = compute_trial_rates(model, readout, stim.to(device),
                               return_spatial=True)  # (T, N, H, W)
    return maps


def center_of_mass(map2d):
    """
    Compute (y, x) center of mass of a 2D array.
    map2d: (H, W) non-negative values
    Returns (y_com, x_com) in pixel coordinates.
    """
    H, W = map2d.shape
    ys = np.arange(H, dtype=float)
    xs = np.arange(W, dtype=float)
    total = map2d.sum() + 1e-12
    y_com = (map2d.sum(axis=1) * ys).sum() / total
    x_com = (map2d.sum(axis=0) * xs).sum() / total
    return y_com, x_com


def select_neurons(maps_real, n_neurons):
    """
    Select top n_neurons by variance of amax across time.
    maps_real: (T, N, H, W)
    """
    amax_over_time = maps_real.max(axis=(-2, -1))  # (T, N)
    variances = amax_over_time.var(axis=0)          # (N,)
    return np.argsort(variances)[-n_neurons:][::-1]


def compute_time_series(maps, neuron_idx):
    """
    From (T, N, H, W) maps, compute three time series for one neuron:
      - amax: max activation value
      - argmax_x, argmax_y: position of max activation
      - com_x, com_y: center of mass
    """
    T = maps.shape[0]
    neuron_maps = maps[:, neuron_idx]  # (T, H, W)

    amax_ts = neuron_maps.max(axis=(-2, -1))  # (T,)

    argmax_flat = neuron_maps.reshape(T, -1).argmax(axis=1)
    H, W = neuron_maps.shape[1], neuron_maps.shape[2]
    argmax_y = argmax_flat // W
    argmax_x = argmax_flat % W

    com_y = np.zeros(T)
    com_x = np.zeros(T)
    for t in range(T):
        m = np.clip(neuron_maps[t], 0, None)
        com_y[t], com_x[t] = center_of_mass(m)

    return {
        'amax': amax_ts,
        'argmax_x': argmax_x.astype(float),
        'argmax_y': argmax_y.astype(float),
        'com_x': com_x,
        'com_y': com_y,
    }


def plot_neuron(neuron_idx, maps_real, maps_stab, cond_label_stab, T_frames, fig_path):
    """
    Produce a figure for one neuron with:
    - Row 1: time snapshots (real FEM)
    - Row 2: time snapshots (stabilized)
    - Row 3: time series (amax, argmax_y, com_y) for both conditions
    """
    n_panels = N_TIME_PANELS
    T_real = maps_real.shape[0]
    T_stab = maps_stab.shape[0]
    T_min = min(T_real, T_stab)

    time_indices = np.linspace(0, T_min - 1, n_panels, dtype=int)

    fig = plt.figure(figsize=(3 * n_panels, 12))
    gs = gridspec.GridSpec(4, n_panels, figure=fig, hspace=0.4, wspace=0.3)

    vmax_real = maps_real[:, neuron_idx].max()
    vmax_stab = maps_stab[:, neuron_idx].max()
    vmax = max(vmax_real, vmax_stab, 1e-6)

    # Real FEM snapshots
    for col, t in enumerate(time_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(maps_real[t, neuron_idx], cmap='hot', vmin=0, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(f't={t}', fontsize=7)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('real FEM', fontsize=8)

    # Stabilized snapshots
    for col, t in enumerate(time_indices):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(maps_stab[t, neuron_idx], cmap='hot', vmin=0, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(f't={t}', fontsize=7)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(cond_label_stab, fontsize=8)

    # Time series: amax
    ts_real = compute_time_series(maps_real, neuron_idx)
    ts_stab = compute_time_series(maps_stab, neuron_idx)
    t_ax = np.arange(T_min)

    ax_amax = fig.add_subplot(gs[2, :])
    ax_amax.plot(t_ax, ts_real['amax'][:T_min], 'royalblue', label='real FEM', linewidth=1)
    ax_amax.plot(t_ax, ts_stab['amax'][:T_min], 'tomato', label=cond_label_stab, linewidth=1)
    ax_amax.set_ylabel('amax activation')
    ax_amax.set_title(f'Neuron {neuron_idx} — amax over time')
    ax_amax.legend(fontsize=8)
    ax_amax.grid(True, alpha=0.3)

    # Time series: com_y and argmax_y
    ax_pos = fig.add_subplot(gs[3, :])
    ax_pos.plot(t_ax, ts_real['com_y'][:T_min], 'royalblue', label='real FEM (CoM y)', linewidth=1)
    ax_pos.plot(t_ax, ts_stab['com_y'][:T_min], 'tomato', label=f'{cond_label_stab} (CoM y)',
                linewidth=1)
    ax_pos.plot(t_ax, ts_real['argmax_y'][:T_min], 'royalblue', linestyle='--',
                alpha=0.5, label='real FEM (argmax y)', linewidth=0.8)
    ax_pos.plot(t_ax, ts_stab['argmax_y'][:T_min], 'tomato', linestyle='--',
                alpha=0.5, label=f'{cond_label_stab} (argmax y)', linewidth=0.8)
    ax_pos.set_ylabel('Activation position (y, px)')
    ax_pos.set_xlabel('Time frame')
    ax_pos.set_title('Position over time: center of mass (solid) and argmax (dashed)')
    ax_pos.legend(fontsize=7, ncol=2)
    ax_pos.grid(True, alpha=0.3)

    fig.suptitle(f'Spatial map dynamics — Neuron {neuron_idx}', fontsize=12)

    fig.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Spatial map inspector')
    parser.add_argument('--logmar', type=float, default=0.0)
    parser.add_argument('--orientation', type=int, default=0)
    parser.add_argument('--n_neurons', type=int, default=3,
                        help='Number of top-variance neurons to plot')
    parser.add_argument('--trace_idx', type=int, default=0,
                        help='Eye trace index to use for this single-trial plot')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"LogMAR={args.logmar}, orientation={args.orientation}, "
          f"trace_idx={args.trace_idx}, n_neurons={args.n_neurons}")

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    td = load_eye_traces(EYE_TRACES_PATH)
    print_summary(td)
    eye_traces = td['traces']
    durations = td['durations']

    idx = args.trace_idx
    dur = int(durations[idx])
    eyepos = eye_traces[idx, :dur]
    print(f"Trial {idx}: {dur} frames, eye range "
          f"x=[{eyepos[:,0].min():.3f},{eyepos[:,0].max():.3f}] "
          f"y=[{eyepos[:,1].min():.3f},{eyepos[:,1].max():.3f}] deg")

    # Load model
    print("Loading model...")
    model, readout = load_model_readout(device)

    # Compute real-FEM maps
    print("Computing real-FEM spatial maps...")
    maps_real = get_maps_for_trial(
        model, readout, args.orientation, args.logmar, eyepos,
        condition='real', device=device,
    )
    print(f"  Maps shape: {maps_real.shape}  (T, N, H_map, W_map)")

    # Compute stabilized maps
    print("Computing stabilized spatial maps...")
    maps_stab = get_maps_for_trial(
        model, readout, args.orientation, args.logmar, eyepos,
        condition='stabilized', device=device,
    )

    # Select top-variance neurons
    top_neurons = select_neurons(maps_real, args.n_neurons)
    print(f"Top {args.n_neurons} neurons by amax variance: {top_neurons}")

    # Time series summary
    T_min = min(maps_real.shape[0], maps_stab.shape[0])
    T_frames = np.linspace(0, T_min - 1, N_TIME_PANELS, dtype=int)

    lm_str = f'{args.logmar:.2f}'.replace('-', 'neg')

    print("\n=== Position variance (CoM y) across time ===")
    print(f"{'Neuron':>8}  {'real CoM_y var':>16}  {'stab CoM_y var':>16}  {'ratio':>8}")
    print('-' * 55)
    for n in top_neurons:
        ts_r = compute_time_series(maps_real, n)
        ts_s = compute_time_series(maps_stab, n)
        var_r = ts_r['com_y'][:T_min].var()
        var_s = ts_s['com_y'][:T_min].var()
        ratio = var_r / (var_s + 1e-12)
        print(f"{n:>8}  {var_r:>16.4f}  {var_s:>16.4f}  {ratio:>8.2f}")

    # Plot each neuron
    print("\nGenerating figures...")
    for n in top_neurons:
        fig_path = os.path.join(
            FIGURES_DIR,
            f'diag_spatial_maps_lm{lm_str}_ori{args.orientation}_trial{idx}_n{n}.png'
        )
        plot_neuron(n, maps_real, maps_stab, 'stabilized', T_frames, fig_path)

    print("\nDone.")


if __name__ == '__main__':
    main()

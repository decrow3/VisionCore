"""
Step-shift sensitivity test.

Tests whether the pipeline is sensitive to a controlled spatial shift of the E
stimulus, and whether amax spatial collapse preserves or discards that sensitivity.

Tests sensitivity at two levels separately:
  1. World/model-input level: does the shift survive HiResRetina resampling?
  2. Model-response level: does the model respond differently, and does amax kill it?

Usage:
    python step_shift_test.py [--logmar 0.0] [--n_traces 5] [--orientation 0]

Output:
    figures/diag_step_shift_lm{logmar}.png
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
EYE_TRACES_PATH = os.path.join(DATA_DIR, 'eye_traces.npz')
PKL_PATH = os.path.join(SCRIPT_DIR, '..', 'mcfarland_outputs_mono.pkl')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Shift magnitudes to test (in arcmin)
SHIFT_ARCMIN = [0.5, 1.0, 2.0]


def render_world_image(orientation_deg, logmar, center_offset_deg, device='cpu'):
    """
    Render the world image for one (orientation, logmar, offset) and return
    it as a (H, W) float32 numpy array. Used to verify the shift is applied
    before any retinal resampling.
    """
    from stimulus_hires import HiResERenderer, WORLD_PPD, WORLD_SIZE, BLUR_SIGMA
    renderer = HiResERenderer(ppd=WORLD_PPD, canvas_size=WORLD_SIZE,
                              blur_sigma=BLUR_SIGMA).to(device)
    renderer.eval()
    with torch.no_grad():
        world = renderer(orientation_deg, logmar, center_offset_deg)  # (1,1,H,W)
    return world[0, 0].cpu().numpy()


def world_image_diff_stats(orientation_deg, logmar, shift_arcmin, device='cpu'):
    """
    Compute diff between reference and shifted world images BEFORE retinal resampling.
    This is the first rung of the debugging ladder: if this is zero, the offset
    is not being applied in the renderer.
    """
    from stimulus_hires import WORLD_PPD
    shift_deg = shift_arcmin / 60.0
    world_ref = render_world_image(orientation_deg, logmar, (0.0, 0.0), device)
    world_shift = render_world_image(orientation_deg, logmar, (shift_deg, 0.0), device)
    diff = np.abs(world_shift - world_ref)
    return {
        'max_diff': float(diff.max()),
        'mean_diff': float(diff.mean()),
        'world_ref': world_ref,
        'world_shift': world_shift,
        'diff_image': world_shift - world_ref,
        'shift_world_px': shift_deg * WORLD_PPD,
    }


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


def make_stabilized_stim(orientation_deg, logmar, eye_traces, durations,
                         center_offset_deg=(0.0, 0.0), n_traces=5, device='cpu'):
    """Build stabilized-condition stimulus sequences for a small set of traces."""
    from stimulus_hires import hires_counterfactual_stim
    stims = []
    for i in range(min(n_traces, len(eye_traces))):
        dur = int(durations[i])
        eyepos = eye_traces[i, :dur]
        stim = hires_counterfactual_stim(
            orientation_deg=orientation_deg,
            logmar=logmar,
            eyepos=eyepos,
            condition='stabilized',
            center_offset_deg=center_offset_deg,
            device=device,
        )  # (T_valid, 1, n_lags, H, W)
        stims.append(stim)
    return stims


def get_spatial_maps(model, readout, stims, device):
    """
    Run model on a list of stim tensors, return spatial maps (T, N, H, W) per trial.
    Uses return_spatial=True.
    """
    from rate_computation import compute_trial_rates
    all_maps = []
    for stim in stims:
        maps = compute_trial_rates(model, readout, stim.to(device),
                                   return_spatial=True)  # (T, N, H, W)
        all_maps.append(maps)
    return all_maps  # list of (T_i, N, H, W)


def collapse_max(maps):
    """(T, N, H, W) → (T, N)"""
    return maps.max(axis=-1).max(axis=-1)


def collapse_mean(maps):
    """(T, N, H, W) → (T, N)"""
    return maps.mean(axis=-1).mean(axis=-1)


def collapse_com(maps):
    """
    (T, N, H, W) → (T, N*2): y-CoM and x-CoM per neuron concatenated.

    This is the "known-preserved" control: a spatial shift should move the
    centre of mass proportionally regardless of peak value changes. If CoM
    sensitivity is high but amax sensitivity is low, amax is the bottleneck.
    """
    T, N, H, W = maps.shape
    ys = np.arange(H, dtype=float)
    xs = np.arange(W, dtype=float)
    out = np.zeros((T, N * 2), dtype=float)
    for t in range(T):
        for n in range(N):
            m = np.clip(maps[t, n], 0, None)
            total = m.sum() + 1e-12
            out[t, n] = (m.sum(axis=1) * ys).sum() / total       # y-CoM
            out[t, N + n] = (m.sum(axis=0) * xs).sum() / total   # x-CoM
    return out


def map_diff_stats(maps_ref, maps_shift):
    """
    Compute mean |map_shift - map_ref| across neurons and space, per time frame.
    Returns time series of shape (T_min,).
    """
    T = min(m.shape[0] for m in maps_ref + maps_shift)
    diffs = []
    for mref, msh in zip(maps_ref, maps_shift):
        d = np.abs(msh[:T] - mref[:T])  # (T, N, H, W)
        diffs.append(d.mean(axis=(1, 2, 3)))  # (T,)
    return np.stack(diffs, axis=0).mean(axis=0)  # (T,)


def collapsed_diff_stats(maps_ref, maps_shift, collapse_fn):
    """Mean |collapse(shift) - collapse(ref)| over neurons and trials, per time."""
    T = min(m.shape[0] for m in maps_ref + maps_shift)
    diffs = []
    for mref, msh in zip(maps_ref, maps_shift):
        cr = collapse_fn(mref[:T])   # (T, N)
        cs = collapse_fn(msh[:T])    # (T, N)
        diffs.append(np.abs(cs - cr).mean(axis=1))  # (T,)
    return np.stack(diffs, axis=0).mean(axis=0)


def retinal_input_diff(stims_ref, stims_shift):
    """
    Measure difference in model INPUT (last lag of each time step) between
    reference and shifted stimuli. Returns mean |shift - ref| per time frame.
    """
    diffs = []
    T = min(s.shape[0] for s in stims_ref + stims_shift)
    for sref, ssh in zip(stims_ref, stims_shift):
        # stim: (T_valid, 1, n_lags, H, W) — last lag is the current frame
        fr = sref[:T, 0, -1].numpy()   # (T, H, W)
        fs = ssh[:T, 0, -1].numpy()    # (T, H, W)
        diffs.append(np.abs(fs - fr).mean(axis=(1, 2)))  # (T,)
    return np.stack(diffs, axis=0).mean(axis=0)


def representative_neuron_diff_image(maps_ref, maps_shift, neuron_idx, t_frame):
    """
    Return the spatial difference map (H, W) for one neuron and time frame,
    averaged across trials.
    """
    T = min(m.shape[0] for m in maps_ref + maps_shift)
    t = min(t_frame, T - 1)
    diffs = []
    for mref, msh in zip(maps_ref, maps_shift):
        diffs.append(msh[t, neuron_idx] - mref[t, neuron_idx])  # (H, W)
    return np.stack(diffs, axis=0).mean(axis=0)


def pick_representative_neuron(maps_ref, maps_shift):
    """Pick the neuron with the largest mean absolute response change."""
    T = min(m.shape[0] for m in maps_ref + maps_shift)
    delta = np.zeros(maps_ref[0].shape[1])  # (N,)
    for mref, msh in zip(maps_ref, maps_shift):
        delta += np.abs(msh[:T] - mref[:T]).mean(axis=(0, 2, 3))
    return int(np.argmax(delta))


def main():
    parser = argparse.ArgumentParser(description='Step-shift sensitivity test')
    parser.add_argument('--logmar', type=float, default=0.0)
    parser.add_argument('--orientation', type=int, default=0)
    parser.add_argument('--n_traces', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"LogMAR: {args.logmar}, orientation: {args.orientation}, n_traces: {args.n_traces}")

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    td = load_eye_traces(EYE_TRACES_PATH)
    print_summary(td)
    eye_traces = td['traces']
    durations = td['durations']

    # Load model
    print("Loading model...")
    model, readout = load_model_readout(device)

    # Reference (no shift)
    print("Building reference stimuli (offset=0)...")
    stims_ref = make_stabilized_stim(
        args.orientation, args.logmar, eye_traces, durations,
        center_offset_deg=(0.0, 0.0), n_traces=args.n_traces, device=device,
    )
    print(f"  Reference stim shape: {stims_ref[0].shape}")

    print("Computing reference spatial maps...")
    maps_ref = get_spatial_maps(model, readout, stims_ref, device)
    N = maps_ref[0].shape[1]
    H_map, W_map = maps_ref[0].shape[2], maps_ref[0].shape[3]
    H_ret, W_ret = stims_ref[0].shape[-2], stims_ref[0].shape[-1]
    print(f"  Maps shape: {maps_ref[0].shape}  (T, N={N}, H={H_map}, W={W_map})")

    # For each shift magnitude
    from stimulus_hires import WORLD_PPD, RETINA_PPD
    all_shift_results = {}
    for shift_arcmin in SHIFT_ARCMIN:
        shift_deg = shift_arcmin / 60.0
        shift_world_px = shift_deg * WORLD_PPD
        shift_retina_px = shift_deg * RETINA_PPD
        print(f"\n--- Shift: {shift_arcmin} arcmin = {shift_deg:.4f} deg ---")
        print(f"  Nominal: {shift_world_px:.2f} px @ world ({WORLD_PPD} ppd), "
              f"{shift_retina_px:.2f} px @ retinal ({RETINA_PPD:.1f} ppd)")

        # ── Rung 0: world image diff (BEFORE retinal resampling) ──────────────
        # If this is zero, center_offset_deg is not being applied in the renderer.
        wd = world_image_diff_stats(args.orientation, args.logmar, shift_arcmin, device)
        print(f"  [Rung 0] World image diff:  max={wd['max_diff']:.5f}, "
              f"mean={wd['mean_diff']:.7f}", flush=True)
        if wd['max_diff'] < 1e-6:
            print(f"  *** WARNING: world image diff is zero — "
                  f"center_offset_deg is NOT being applied in renderer ***")

        # ── Rung 1: retinal input diff (after resampling, before model) ───────
        stims_shift = make_stabilized_stim(
            args.orientation, args.logmar, eye_traces, durations,
            center_offset_deg=(shift_deg, 0.0), n_traces=args.n_traces, device=device,
        )
        input_diff_ts = retinal_input_diff(stims_ref, stims_shift)
        print(f"  [Rung 1] Retinal input diff: mean={input_diff_ts.mean():.5f}  "
              f"(range [{input_diff_ts.min():.5f}, {input_diff_ts.max():.5f}])")
        if input_diff_ts.mean() < 1e-6 and wd['max_diff'] > 1e-6:
            print(f"  *** WARNING: world diff exists but retinal diff is zero — "
                  f"shift is lost in HiResRetina resampling ***")

        # ── Rung 2: model spatial map diff ────────────────────────────────────
        maps_shift = get_spatial_maps(model, readout, stims_shift, device)
        neuron_idx = pick_representative_neuron(maps_ref, maps_shift)
        map_diff_ts = map_diff_stats(maps_ref, maps_shift)
        print(f"  [Rung 2] Map diff (pre-collapse): mean={map_diff_ts.mean():.6f}  "
              f"neuron={neuron_idx}")
        if map_diff_ts.mean() < 1e-8 and input_diff_ts.mean() > 1e-6:
            print(f"  *** WARNING: retinal diff exists but model map diff is zero — "
                  f"model is insensitive at this scale ***")

        # ── Rung 3: collapse comparison (raw → CoM → mean → amax) ─────────────
        max_diff_ts = collapsed_diff_stats(maps_ref, maps_shift, collapse_max)
        mean_diff_ts = collapsed_diff_stats(maps_ref, maps_shift, collapse_mean)
        com_diff_ts = collapsed_diff_stats(maps_ref, maps_shift, collapse_com)
        pre = map_diff_ts.mean() + 1e-12
        print(f"  [Rung 3] raw (pre-collapse): {map_diff_ts.mean():.6f}")
        print(f"           CoM:  {com_diff_ts.mean():.6f}  (ratio={com_diff_ts.mean()/pre:.3f})")
        print(f"           mean: {mean_diff_ts.mean():.6f}  (ratio={mean_diff_ts.mean()/pre:.3f})")
        print(f"           amax: {max_diff_ts.mean():.6f}  (ratio={max_diff_ts.mean()/pre:.3f})")

        T_mid = maps_ref[0].shape[0] // 2
        diff_image = representative_neuron_diff_image(maps_ref, maps_shift, neuron_idx, T_mid)

        all_shift_results[shift_arcmin] = {
            'shift_world_px': shift_world_px,
            'shift_retina_px': shift_retina_px,
            'world_max_diff': wd['max_diff'],
            'world_diff_image': wd['diff_image'],
            'input_diff_ts': input_diff_ts,
            'map_diff_ts': map_diff_ts,
            'max_diff_ts': max_diff_ts,
            'mean_diff_ts': mean_diff_ts,
            'com_diff_ts': com_diff_ts,
            'diff_image': diff_image,
            'neuron_idx': neuron_idx,
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_shifts = len(SHIFT_ARCMIN)
    fig, axes = plt.subplots(n_shifts, 4, figsize=(18, 4 * n_shifts))
    if n_shifts == 1:
        axes = axes[np.newaxis, :]

    for row, shift_arcmin in enumerate(SHIFT_ARCMIN):
        r = all_shift_results[shift_arcmin]
        ax = axes[row]

        # Panel 1: world image diff (Rung 0) — shift before retinal resampling
        T_min = min(s.shape[0] for s in stims_ref)
        T_frame = T_min // 2
        world_diff = r['world_diff_image']
        vmax_w = max(np.abs(world_diff).max(), 1e-6)
        im = ax[0].imshow(world_diff, cmap='RdBu_r', vmin=-vmax_w, vmax=vmax_w)
        ax[0].set_title(f'World image diff (Rung 0)\n{shift_arcmin} arcmin = '
                        f'{r["shift_world_px"]:.1f} px\nmax={r["world_max_diff"]:.5f}',
                        fontsize=8)
        plt.colorbar(im, ax=ax[0])

        # Panel 2: spatial map diff image for representative neuron
        vmax = np.abs(r['diff_image']).max()
        im2 = ax[1].imshow(r['diff_image'], cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax)
        ax[1].set_title(f'Map diff: neuron {r["neuron_idx"]}\n'
                        f'(map shift - map ref, t={T_frame})', fontsize=8)
        plt.colorbar(im2, ax=ax[1])

        # Panel 3: time series of pre-collapse vs post-collapse diff
        t_ax = np.arange(len(r['map_diff_ts']))
        ax[2].plot(t_ax, r['map_diff_ts'], 'k-', label='raw (pre-collapse)', linewidth=1.5)
        ax[2].plot(t_ax, r['com_diff_ts'], 'm--', label='CoM', linewidth=1.5)
        ax[2].plot(t_ax, r['mean_diff_ts'], 'b--', label='mean', linewidth=1.5)
        ax[2].plot(t_ax, r['max_diff_ts'], 'r--', label='amax', linewidth=1.5)
        ax[2].plot(t_ax[:len(r['input_diff_ts'])], r['input_diff_ts'],
                   'g:', label='retinal input', linewidth=1.5)
        ax[2].set_title(f'Mean |diff| over time\n(shift={shift_arcmin} arcmin)', fontsize=8)
        ax[2].set_xlabel('Time frame')
        ax[2].set_ylabel('Mean |diff|')
        ax[2].legend(fontsize=7)
        ax[2].grid(True, alpha=0.3)

        # Panel 4: bar chart — sensitivity ratio (after collapse / pre-collapse)
        # raw→CoM→mean→amax is the information-preservation chain
        pre = r['map_diff_ts'].mean()
        ratios = {
            'CoM / pre': r['com_diff_ts'].mean() / (pre + 1e-12),
            'mean / pre': r['mean_diff_ts'].mean() / (pre + 1e-12),
            'amax / pre': r['max_diff_ts'].mean() / (pre + 1e-12),
        }
        bars = ax[3].bar(list(ratios.keys()), list(ratios.values()),
                          color=['mediumorchid', 'steelblue', 'tomato'])
        ax[3].axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
        ax[3].set_ylim([0, max(1.5, max(ratios.values()) * 1.2)])
        ax[3].set_title(f'Sensitivity ratio\n(after/before collapse)', fontsize=8)
        ax[3].set_ylabel('Ratio')
        for bar, (k, v) in zip(bars, ratios.items()):
            ax[3].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.02, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=8)

    lm_str = f'{args.logmar:.2f}'.replace('-', 'neg')
    fig.suptitle(f'Step-shift sensitivity test  |  LogMAR={args.logmar:.2f}, '
                 f'ori={args.orientation}°, n_traces={args.n_traces}',
                 fontsize=11, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f'diag_step_shift_lm{lm_str}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n=== Summary (information-preservation chain: raw → CoM → mean → amax) ===")
    print(f"{'Shift (arcmin)':>16}  {'World px':>9}  {'WorldDiff':>10}  "
          f"{'RetinalDiff':>12}  {'MapDiff':>9}  {'CoMRatio':>9}  {'meanRatio':>9}  {'amaxRatio':>9}")
    print('-' * 108)
    for shift_arcmin in SHIFT_ARCMIN:
        r = all_shift_results[shift_arcmin]
        pre = r['map_diff_ts'].mean() + 1e-12
        print(f"{shift_arcmin:>16.1f}  {r['shift_world_px']:>9.2f}  "
              f"{r['world_max_diff']:>10.5f}  "
              f"{r['input_diff_ts'].mean():>12.5f}  "
              f"{r['map_diff_ts'].mean():>9.6f}  "
              f"{r['com_diff_ts'].mean()/pre:>9.3f}  "
              f"{r['mean_diff_ts'].mean()/pre:>9.3f}  "
              f"{r['max_diff_ts'].mean()/pre:>9.3f}")


if __name__ == '__main__':
    main()

"""
Collapse-comparison pilot.

Compares three spatial collapse modes — max, mean, flat+PCA — for the same
small set of trials at a single LogMAR value. If C ≈ A only under max but not
under mean or flat+PCA, the original null is a representational artifact.

Usage:
    python collapse_comparison.py [--logmar 0.0] [--n_traces 20]
                                   [--n_splits 3] [--n_pca_flat 50]

Output:
    Printed accuracy table + figures/diag_collapse_comparison_lm{logmar}.png
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

ORIENTATIONS = [0, 90, 180, 270]
CONDITIONS = ['real', 'stabilized']


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


def compute_rates_for_collapse(model, readout, orientation_deg, logmar,
                                eye_traces, durations, n_traces, collapse_mode,
                                condition, null_traces_arr, pca_flat, device):
    """
    Compute rates for one (orientation, condition) with a given collapse mode.

    For 'max' and 'mean': uses compute_trial_rates directly with spatial_collapse.
    For 'flat': uses return_spatial=True, then flattens (T, N, H, W) → (T, N*H*W),
                projects via pca_flat (pre-fitted or None to return raw flat).

    Returns list of (T_valid, n_features) arrays.
    """
    from stimulus_hires import hires_counterfactual_stim
    from rate_computation import compute_trial_rates

    rates_list = []
    for i in range(n_traces):
        dur = int(durations[i])
        eyepos = eye_traces[i, :dur]

        null_trace = None
        if condition == 'matched_null' and null_traces_arr is not None:
            null_trace = null_traces_arr[0, i, :dur]

        stim = hires_counterfactual_stim(
            orientation_deg=orientation_deg,
            logmar=logmar,
            eyepos=eyepos,
            condition=condition,
            null_trace=null_trace,
            device=device,
        )

        if collapse_mode in ('max', 'mean'):
            rates = compute_trial_rates(
                model, readout, stim.to(device),
                spatial_collapse=collapse_mode,
                return_spatial=False,
            )  # (T, N)
        elif collapse_mode == 'amax_com':
            # amax + center-of-mass x/y per neuron → (T, 3N)
            maps = compute_trial_rates(
                model, readout, stim.to(device),
                return_spatial=True,
            )  # (T, N, H, W)
            T_m, N_m, H_m, W_m = maps.shape
            amax_vals = maps.max(axis=-1).max(axis=-1)    # (T, N)
            # CoM: weighted average of pixel coordinates
            ys = np.arange(H_m, dtype=np.float32)
            xs = np.arange(W_m, dtype=np.float32)
            m_clip = np.clip(maps, 0, None)               # (T, N, H, W)
            total = m_clip.sum(axis=(-2, -1)) + 1e-12     # (T, N)
            com_y = (m_clip * ys[None, None, :, None]).sum(axis=(-2, -1)) / total  # (T, N)
            com_x = (m_clip * xs[None, None, None, :]).sum(axis=(-2, -1)) / total  # (T, N)
            rates = np.concatenate([amax_vals, com_y, com_x], axis=1)  # (T, 3N)
        else:  # 'flat'
            maps = compute_trial_rates(
                model, readout, stim.to(device),
                return_spatial=True,
            )  # (T, N, H, W)
            T_m, N_m, H_m, W_m = maps.shape
            flat = maps.reshape(T_m, N_m * H_m * W_m)
            if pca_flat is not None:
                rates = pca_flat.transform(flat)  # (T, n_pca)
            else:
                rates = flat  # (T, N*H*W) — large!

        rates_list.append(rates)

    return rates_list


def fit_flat_pca(model, readout, logmar, eye_traces, durations, n_traces,
                 condition, device, n_components):
    """
    Fit PCA on flat spatial maps from a few trials (train set).
    Returns fitted PCA object.
    """
    from sklearn.decomposition import PCA
    from stimulus_hires import hires_counterfactual_stim
    from rate_computation import compute_trial_rates

    all_flat = []
    n_fit = min(n_traces, 5)  # use first 5 traces to fit PCA
    for i in range(n_fit):
        dur = int(durations[i])
        eyepos = eye_traces[i, :dur]
        stim = hires_counterfactual_stim(
            orientation_deg=0,  # use one orientation for PCA fit
            logmar=logmar,
            eyepos=eyepos,
            condition=condition,
            device=device,
        )
        maps = compute_trial_rates(model, readout, stim.to(device),
                                   return_spatial=True)  # (T, N, H, W)
        T, N, H, W = maps.shape
        all_flat.append(maps.reshape(T, N * H * W))

    all_flat = np.concatenate(all_flat, axis=0)  # (T_total, N*H*W)
    n_comp = min(n_components, min(all_flat.shape))
    pca = PCA(n_components=n_comp)
    pca.fit(all_flat)
    print(f"  Flat+PCA: fitted {n_comp} components on {all_flat.shape[0]} frames, "
          f"dim {all_flat.shape[1]} → {n_comp}")
    return pca


def run_decoding_for_mode(model, readout, logmar, eye_traces, durations,
                           n_traces, condition, collapse_mode, pca_flat,
                           n_splits, n_pca_C, device):
    """
    Compute rates for all orientations under given collapse mode, then run
    the full decoding ladder (models A and C).
    Returns ladder result dict.
    """
    from decoding import run_decoding_ladder

    rates_by_stim = {}
    for ori in ORIENTATIONS:
        rates = compute_rates_for_collapse(
            model, readout, ori, logmar, eye_traces, durations,
            n_traces, collapse_mode, condition, None, pca_flat, device,
        )
        rates_by_stim[f'ori{ori}'] = rates

    ladder = run_decoding_ladder(
        rates_by_stim,
        models=['A', 'C'],
        n_splits=n_splits,
        n_components_C=n_pca_C,
        verbose=False,
    )
    return ladder


def main():
    parser = argparse.ArgumentParser(description='Spatial collapse mode comparison')
    parser.add_argument('--logmar', type=float, default=0.0)
    parser.add_argument('--n_traces', type=int, default=20)
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--n_pca_flat', type=int, default=50,
                        help='PCA components for flat spatial map')
    parser.add_argument('--n_pca_C', type=int, default=20,
                        help='PCA components for Model C temporal residual')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"LogMAR={args.logmar}, n_traces={args.n_traces}, "
          f"n_splits={args.n_splits}, n_pca_flat={args.n_pca_flat}")

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    td = load_eye_traces(EYE_TRACES_PATH)
    print_summary(td)
    eye_traces = td['traces']
    durations = td['durations']

    # Load model
    print("Loading model...")
    model, readout = load_model_readout(device)

    # Pre-fit PCA for flat+PCA mode
    print("\nFitting flat+PCA on stabilized condition...")
    pca_flat = fit_flat_pca(model, readout, args.logmar, eye_traces, durations,
                             args.n_traces, 'stabilized', device, args.n_pca_flat)

    # Run all combinations
    collapse_modes = ['max', 'mean', 'amax_com', 'flat']
    results = {}

    for cond in CONDITIONS:
        for collapse in collapse_modes:
            pf = pca_flat if collapse == 'flat' else None
            print(f"\n--- {cond}, collapse={collapse} ---", flush=True)
            ladder = run_decoding_for_mode(
                model, readout, args.logmar, eye_traces, durations,
                args.n_traces, cond, collapse, pf,
                args.n_splits, args.n_pca_C, device,
            )
            results[(cond, collapse)] = ladder
            print(f"  A: {ladder['A']['mean_acc']:.4f} ± {ladder['A']['std_acc']:.4f}  "
                  f"C: {ladder['C']['mean_acc']:.4f} ± {ladder['C']['std_acc']:.4f}  "
                  f"ΔC-A: {ladder['C']['mean_acc'] - ladder['A']['mean_acc']:+.4f}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Collapse':>10}  {'Condition':>12}  {'Model A':>9}  {'Model C':>9}  {'C - A':>8}")
    print('-' * 55)
    for cond in CONDITIONS:
        for collapse in collapse_modes:
            r = results[(cond, collapse)]
            a = r['A']['mean_acc']
            c = r['C']['mean_acc']
            print(f"{collapse:>10}  {cond:>12}  {a:>9.4f}  {c:>9.4f}  {c-a:>+8.4f}")

    # Interpretation guidance
    print(f"\n{'='*70}")
    print("Interpretation:")
    any_rescue = False
    for cond in CONDITIONS:
        for collapse in collapse_modes:
            r = results[(cond, collapse)]
            delta = r['C']['mean_acc'] - r['A']['mean_acc']
            if delta > 0.02:
                print(f"  ✓ {collapse}/{cond}: C > A by {delta:+.4f} — temporal structure IS useful here")
                any_rescue = True
    if not any_rescue:
        print("  All collapse modes give C ≈ A — null is more robust across representations")

    max_rescues = [results[(cond, 'flat')]['C']['mean_acc'] - results[(cond, 'max')]['C']['mean_acc']
                   for cond in CONDITIONS]
    if max(max_rescues) > 0.03:
        print("  flat+PCA gives higher accuracy than max — spatial structure matters, "
              "amax is discarding signal")

    com_vs_max_A = [results[(cond, 'amax_com')]['A']['mean_acc'] - results[(cond, 'max')]['A']['mean_acc']
                    for cond in CONDITIONS]
    com_vs_max_C = [results[(cond, 'amax_com')]['C']['mean_acc'] - results[(cond, 'max')]['C']['mean_acc']
                    for cond in CONDITIONS]
    if max(com_vs_max_A) > 0.02:
        print(f"  amax+CoM improves Model A by >{max(com_vs_max_A):.3f} — spatial position IS orientation-discriminating")
    if max(com_vs_max_C) > 0.02:
        print(f"  amax+CoM improves Model C by >{max(com_vs_max_C):.3f} — position dynamics help temporal decoding")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(collapse_modes))
    width = 0.35

    for ci, cond in enumerate(CONDITIONS):
        ax = axes[ci]
        acc_A = [results[(cond, m)]['A']['mean_acc'] for m in collapse_modes]
        acc_C = [results[(cond, m)]['C']['mean_acc'] for m in collapse_modes]
        std_A = [results[(cond, m)]['A']['std_acc'] for m in collapse_modes]
        std_C = [results[(cond, m)]['C']['std_acc'] for m in collapse_modes]

        bars_A = ax.bar(x - width/2, acc_A, width, label='Model A', color='steelblue',
                         yerr=std_A, capsize=4)
        bars_C = ax.bar(x + width/2, acc_C, width, label='Model C', color='tomato',
                         yerr=std_C, capsize=4)
        ax.axhline(0.625, color='k', linestyle='--', alpha=0.5, label='Threshold (62.5%)')
        ax.axhline(0.25, color='gray', linestyle=':', alpha=0.3, label='Chance')
        ax.set_xticks(x)
        ax.set_xticklabels(collapse_modes)
        ax.set_ylabel('Decoding accuracy')
        ax.set_ylim([0.0, 1.1])
        ax.set_title(f'{cond}  (LogMAR={args.logmar:.2f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    lm_str = f'{args.logmar:.2f}'.replace('-', 'neg')
    fig.suptitle(f'Collapse mode comparison — LogMAR={args.logmar:.2f}, '
                 f'n_traces={args.n_traces}', fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f'diag_collapse_comparison_lm{lm_str}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

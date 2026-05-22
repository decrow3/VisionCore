"""
Robustness test for the flat/stabilized C > A anomaly.

In the collapse_comparison pilot, the flat+PCA collapse on stabilized trials
showed C > A by +0.117 (Model A=0.274, C=0.391), while all other
collapse/condition combinations gave C ≈ A. Both values were near chance (0.25).

This could be a real finding (spatial structure in stabilized responses contains
temporal information at chance-level orientation representation) OR an artifact of:
  1. Too few PCA components → underfits, noise structure leaks
  2. Too many PCA components → overfits flat 511k-dim space to 80 trials
  3. PCA fitted on orientation-0 stabilized only (5 traces) → not representative
  4. Too few CV splits → high variance in accuracy estimate
  5. High dimensionality relative to n_trials → overfitting in logistic regression

This script sweeps over all these axes and reports the C-A gap for each setting.

Usage:
    python flat_stabilized_robustness.py [--n_traces 20] [--logmar 0.0]

Output:
    Printed table + figures/diag_flat_stab_robustness_lm{logmar}.png
"""
import os
import sys
import argparse
import itertools
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


def compute_flat_maps_all_oris(model, readout, logmar, eye_traces, durations,
                                n_traces, device):
    """
    Compute flat (T, N*H*W) spatial maps for all orientations.
    Returns dict: ori → list of (T, N*H*W) arrays.
    This is the expensive step — compute once, sweep parameters later.
    """
    from stimulus_hires import hires_counterfactual_stim
    from rate_computation import compute_trial_rates

    flat_by_ori = {}
    for ori in ORIENTATIONS:
        print(f"  Orientation {ori}°...", flush=True)
        flat_list = []
        for i in range(n_traces):
            dur = int(durations[i])
            eyepos = eye_traces[i, :dur]
            stim = hires_counterfactual_stim(
                orientation_deg=ori,
                logmar=logmar,
                eyepos=eyepos,
                condition='stabilized',
                device=device,
            )
            maps = compute_trial_rates(model, readout, stim.to(device),
                                       return_spatial=True)  # (T, N, H, W)
            T_m, N_m, H_m, W_m = maps.shape
            flat_list.append(maps.reshape(T_m, N_m * H_m * W_m))
        flat_by_ori[ori] = flat_list
    return flat_by_ori


def run_flat_with_pca_params(flat_by_ori, n_pca_flat, n_splits, n_pca_C,
                              pca_fit_oris, pca_fit_n_traces, rng_seed=0):
    """
    Given pre-computed flat maps, fit PCA with specified parameters and decode.

    Args:
        flat_by_ori: dict ori → list of (T, N*H*W)
        n_pca_flat: spatial PCA components to retain
        n_splits: GroupKFold CV splits
        n_pca_C: temporal residual PCA components for Model C
        pca_fit_oris: list of orientations to use when fitting spatial PCA
        pca_fit_n_traces: number of traces (per orientation) to use for PCA fit

    Returns:
        dict with 'A' and 'C' mean_acc, std_acc
    """
    from sklearn.decomposition import PCA
    from decoding import run_decoding_ladder

    # ── Fit spatial PCA ──
    fit_data = []
    for ori in pca_fit_oris:
        flat_list = flat_by_ori[ori]
        n_fit = min(pca_fit_n_traces, len(flat_list))
        for i in range(n_fit):
            fit_data.append(flat_list[i])  # (T, D)

    all_fit = np.concatenate(fit_data, axis=0)  # (T_total, D)
    n_comp = min(n_pca_flat, min(all_fit.shape))
    pca = PCA(n_components=n_comp, random_state=rng_seed)
    pca.fit(all_fit)

    # ── Project flat maps → (T, n_pca_flat) ──
    rates_by_stim = {}
    for ori in ORIENTATIONS:
        proj_list = []
        for flat in flat_by_ori[ori]:
            proj_list.append(pca.transform(flat).astype(np.float32))
        rates_by_stim[f'ori{ori}'] = proj_list

    # ── Decode ──
    ladder = run_decoding_ladder(
        rates_by_stim,
        models=['A', 'C'],
        n_splits=n_splits,
        n_components_C=n_pca_C,
        verbose=False,
    )
    return ladder


def main():
    parser = argparse.ArgumentParser(description='flat/stabilized robustness sweep')
    parser.add_argument('--logmar', type=float, default=0.0)
    parser.add_argument('--n_traces', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"LogMAR={args.logmar}, n_traces={args.n_traces}")

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    td = load_eye_traces(EYE_TRACES_PATH)
    print_summary(td)
    eye_traces = td['traces']
    durations = td['durations']

    # Load model
    print("Loading model...")
    model, readout = load_model_readout(device)

    # ── Compute flat maps once (expensive) ──
    print("\nComputing flat spatial maps for all orientations (stabilized)...")
    flat_by_ori = compute_flat_maps_all_oris(
        model, readout, args.logmar, eye_traces, durations,
        args.n_traces, device,
    )
    T_example = flat_by_ori[0][0].shape[0]
    D_flat = flat_by_ori[0][0].shape[1]
    print(f"  Flat dim: {D_flat}, example trial length: {T_example} frames")

    # ── Parameter sweep ──
    # Axis 1: n_pca_flat
    n_pca_options = [5, 10, 20, 30, 50, 100]
    n_pca_options = [p for p in n_pca_options if p <= min(D_flat, args.n_traces * len(ORIENTATIONS) * T_example)]

    # Axis 2: n_splits
    n_splits_options = [3, 5]
    # Cap n_splits by available traces per class
    n_trials_per_class = args.n_traces
    n_splits_options = [s for s in n_splits_options if s <= n_trials_per_class]

    # Axis 3: PCA fitting subset (orientation subset)
    pca_ori_subsets = [
        ([0], 'ori0_only'),
        ([0, 180], 'ori0+180'),
        ([0, 90, 180, 270], 'all_oris'),
    ]

    # Axis 4: PCA fit n_traces (1-5 traces per ori)
    pca_fit_trace_options = [2, 5, 10]
    pca_fit_trace_options = [t for t in pca_fit_trace_options if t <= args.n_traces]

    # Fixed: n_pca_C=20 throughout (the temporal component count)
    n_pca_C = 20

    print(f"\n{'='*80}")
    print(f"Parameter sweep — flat/stabilized C > A robustness")
    print(f"n_pca_options={n_pca_options}")
    print(f"n_splits_options={n_splits_options}")
    print(f"pca_ori_subsets={[s for _, s in pca_ori_subsets]}")
    print(f"pca_fit_trace_options={pca_fit_trace_options}")
    print(f"{'='*80}\n")

    results = []

    # ── Sweep 1: n_pca_flat × n_splits, canonical PCA subset (all oris, 5 traces) ──
    print("Sweep 1: n_pca_flat × n_splits  (PCA fit: all oris, 5 traces)")
    print(f"{'n_pca':>8}  {'n_splits':>9}  {'A acc':>8}  {'C acc':>8}  {'C-A':>8}  {'flag':>6}")
    print('-' * 55)
    sweep1 = []
    for n_pca, n_splits in itertools.product(n_pca_options, n_splits_options):
        r = run_flat_with_pca_params(
            flat_by_ori, n_pca, n_splits, n_pca_C,
            pca_fit_oris=ORIENTATIONS, pca_fit_n_traces=min(5, args.n_traces),
        )
        acc_A = r['A']['mean_acc']
        acc_C = r['C']['mean_acc']
        delta = acc_C - acc_A
        flag = '⚠️ ' if delta > 0.05 else 'ok'
        print(f"{n_pca:>8}  {n_splits:>9}  {acc_A:>8.4f}  {acc_C:>8.4f}  {delta:>+8.4f}  {flag:>6}")
        sweep1.append({'n_pca': n_pca, 'n_splits': n_splits, 'acc_A': acc_A,
                       'acc_C': acc_C, 'delta': delta})
        results.append({'sweep': 1, 'n_pca': n_pca, 'n_splits': n_splits,
                        'pca_oris': 'all', 'pca_n_traces': 5,
                        'acc_A': acc_A, 'acc_C': acc_C, 'delta': delta})

    # ── Sweep 2: PCA orientation subset (fixed n_pca=50, n_splits=3) ──
    print(f"\nSweep 2: PCA orientation subset  (n_pca=50, n_splits=3)")
    n_pca_fixed = min(50, max(n_pca_options))
    n_splits_fixed = min(3, max(n_splits_options))
    print(f"{'ori_subset':>15}  {'A acc':>8}  {'C acc':>8}  {'C-A':>8}  {'flag':>6}")
    print('-' * 50)
    for pca_oris, ori_label in pca_ori_subsets:
        r = run_flat_with_pca_params(
            flat_by_ori, n_pca_fixed, n_splits_fixed, n_pca_C,
            pca_fit_oris=pca_oris, pca_fit_n_traces=min(5, args.n_traces),
        )
        acc_A = r['A']['mean_acc']
        acc_C = r['C']['mean_acc']
        delta = acc_C - acc_A
        flag = '⚠️ ' if delta > 0.05 else 'ok'
        print(f"{ori_label:>15}  {acc_A:>8.4f}  {acc_C:>8.4f}  {delta:>+8.4f}  {flag:>6}")
        results.append({'sweep': 2, 'n_pca': n_pca_fixed, 'n_splits': n_splits_fixed,
                        'pca_oris': ori_label, 'pca_n_traces': 5,
                        'acc_A': acc_A, 'acc_C': acc_C, 'delta': delta})

    # ── Sweep 3: PCA fit sample size ──
    print(f"\nSweep 3: PCA fit trace count  (n_pca=50, n_splits=3, all oris)")
    print(f"{'pca_n_traces':>14}  {'A acc':>8}  {'C acc':>8}  {'C-A':>8}  {'flag':>6}")
    print('-' * 50)
    for pca_n_traces in pca_fit_trace_options:
        r = run_flat_with_pca_params(
            flat_by_ori, n_pca_fixed, n_splits_fixed, n_pca_C,
            pca_fit_oris=ORIENTATIONS, pca_fit_n_traces=pca_n_traces,
        )
        acc_A = r['A']['mean_acc']
        acc_C = r['C']['mean_acc']
        delta = acc_C - acc_A
        flag = '⚠️ ' if delta > 0.05 else 'ok'
        print(f"{pca_n_traces:>14}  {acc_A:>8.4f}  {acc_C:>8.4f}  {delta:>+8.4f}  {flag:>6}")
        results.append({'sweep': 3, 'n_pca': n_pca_fixed, 'n_splits': n_splits_fixed,
                        'pca_oris': 'all', 'pca_n_traces': pca_n_traces,
                        'acc_A': acc_A, 'acc_C': acc_C, 'delta': delta})

    # ── Interpretation ──
    all_deltas = [r['delta'] for r in results]
    print(f"\n{'='*70}")
    print(f"Summary across all {len(results)} conditions:")
    print(f"  C-A range: [{min(all_deltas):+.4f}, {max(all_deltas):+.4f}]")
    print(f"  Mean C-A: {np.mean(all_deltas):+.4f}")
    print(f"  Fraction with C-A > 0.05: "
          f"{sum(d > 0.05 for d in all_deltas)}/{len(all_deltas)}")
    print(f"  Fraction with C-A < -0.01: "
          f"{sum(d < -0.01 for d in all_deltas)}/{len(all_deltas)}")

    if max(all_deltas) < 0.05:
        print("\n  VERDICT: C ≈ A across all settings — flat/stabilized anomaly is an artifact")
        print("           (likely variance from small n and imperfectly fitted spatial PCA)")
    elif min(all_deltas) > 0.03:
        print("\n  VERDICT: C > A is robust to PCA/split choices — genuine signal in temporal structure")
    else:
        print("\n  VERDICT: Mixed results — gap is parameter-sensitive, likely unstable artifact")
        # Identify what drives the gap
        high_delta = [r for r in results if r['delta'] > 0.05]
        low_delta = [r for r in results if r['delta'] < 0.01]
        if high_delta:
            n_pcas_high = sorted(set(r['n_pca'] for r in high_delta))
            print(f"  Gap > 0.05 seen at n_pca: {n_pcas_high}")
        if low_delta:
            n_pcas_low = sorted(set(r['n_pca'] for r in low_delta))
            print(f"  Gap < 0.01 seen at n_pca: {n_pcas_low}")

    # ── Plot: Sweep 1 heatmap ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: heatmap of C-A over n_pca × n_splits
    ax = axes[0]
    mat = np.zeros((len(n_pca_options), len(n_splits_options)))
    for r in sweep1:
        i = n_pca_options.index(r['n_pca'])
        j = n_splits_options.index(r['n_splits'])
        mat[i, j] = r['delta']
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-0.15, vmax=0.15,
                   origin='lower')
    ax.set_xticks(range(len(n_splits_options)))
    ax.set_xticklabels(n_splits_options)
    ax.set_yticks(range(len(n_pca_options)))
    ax.set_yticklabels(n_pca_options)
    ax.set_xlabel('n_splits')
    ax.set_ylabel('n_pca_flat')
    ax.set_title('Sweep 1: C-A gap\n(flat/stabilized, all oris, 5 traces)')
    plt.colorbar(im, ax=ax, label='C - A accuracy')
    for i in range(len(n_pca_options)):
        for j in range(len(n_splits_options)):
            ax.text(j, i, f'{mat[i,j]:+.3f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(mat[i, j]) > 0.08 else 'black')

    # Panel 2: bar chart of C-A for different PCA orientation subsets (Sweep 2)
    ax = axes[1]
    sweep2_results = [r for r in results if r['sweep'] == 2]
    labels2 = [r['pca_oris'] for r in sweep2_results]
    deltas2 = [r['delta'] for r in sweep2_results]
    colors2 = ['tomato' if d > 0.05 else 'steelblue' for d in deltas2]
    bars = ax.bar(range(len(labels2)), deltas2, color=colors2)
    ax.axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='0.05 threshold')
    ax.axhline(0.0, color='k', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(labels2)))
    ax.set_xticklabels(labels2, rotation=20, fontsize=8)
    ax.set_ylabel('C - A accuracy')
    ax.set_title(f'Sweep 2: PCA orientation subset\n(n_pca={n_pca_fixed}, n_splits={n_splits_fixed})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 3: bar chart of C-A for different PCA trace counts (Sweep 3)
    ax = axes[2]
    sweep3_results = [r for r in results if r['sweep'] == 3]
    labels3 = [f"{r['pca_n_traces']} traces" for r in sweep3_results]
    deltas3 = [r['delta'] for r in sweep3_results]
    colors3 = ['tomato' if d > 0.05 else 'steelblue' for d in deltas3]
    ax.bar(range(len(labels3)), deltas3, color=colors3)
    ax.axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='0.05 threshold')
    ax.axhline(0.0, color='k', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(labels3)))
    ax.set_xticklabels(labels3, fontsize=8)
    ax.set_ylabel('C - A accuracy')
    ax.set_title(f'Sweep 3: PCA fit sample size\n(n_pca={n_pca_fixed}, n_splits={n_splits_fixed})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    lm_str = f'{args.logmar:.2f}'.replace('-', 'neg')
    fig.suptitle(f'flat/stabilized C-A robustness — LogMAR={args.logmar:.2f}, '
                 f'n_traces={args.n_traces}', fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f'diag_flat_stab_robustness_lm{lm_str}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

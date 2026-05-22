"""
Task 3: Dual-regime neurometric sweep with corrected pipeline.

Resolved regime:    LogMAR ∈ {1.0, 0.8, 0.6, 0.5, 0.4}
  → Lo-res pipeline (e_optotype_stack), E gap > 1px @37.5ppd
  → Uses cached rates where available (471 traces)

Hyperacuity regime: LogMAR ∈ {0.3, 0.2, 0.1, 0.0, -0.1, -0.15, -0.2, -0.25, -0.3}
  → Hi-res pipeline (stimulus_hires.py), E rendered at 120ppd
  → Recomputes all rates (old lo-res rates deleted)

Models: A (time-avg rate) and C (mean-rate + temporal-residual PCA)
  → Additive: C ≥ A by construction (decoding.py v2)

Conditions: real, stabilized, matched_null

Key output: ΔLogMAR = threshold(stabilized, A) - threshold(real, C)
"""
import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RATES_DIR = os.path.join(DATA_DIR, 'rates')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
EYE_TRACES_PATH = os.path.join(DATA_DIR, 'eye_traces.npz')
PKL_PATH = os.path.join(SCRIPT_DIR, '..', 'mcfarland_outputs_mono.pkl')

for d in [RATES_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Two-regime LogMAR grids
LOGMAR_RESOLVED = [1.0, 0.8, 0.6, 0.5, 0.4]
LOGMAR_HYPERACUITY = [0.3, 0.2, 0.1, 0.0, -0.1, -0.15, -0.2, -0.25, -0.3]
LOGMAR_FULL = LOGMAR_RESOLVED + LOGMAR_HYPERACUITY
ORIENTATIONS = [0, 90, 180, 270]
CONDITIONS = ['real', 'stabilized', 'matched_null']

# LogMAR below which hi-res pipeline is used (lo-res gap becomes sub-pixel at ~0.18)
HIRES_THRESHOLD = 0.35


def load_model_and_readout(device=None):
    import dill, torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model (device={device})...")
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


def get_rate_cache_path(logmar, ori, cond, hires=False):
    prefix = 'rates_hires' if hires else 'rates'
    return os.path.join(RATES_DIR, f'{prefix}_lm{logmar:.2f}_ori{ori}_{cond}.npz')


def compute_and_cache_rates(model, readout, logmar, ori, cond,
                             traces, durations, null_traces_arr,
                             force=False, verbose=True):
    """Compute and cache rates for one (logmar, ori, cond) combination."""
    use_hires = (logmar < HIRES_THRESHOLD)
    cache_path = get_rate_cache_path(logmar, ori, cond, hires=use_hires)

    if os.path.exists(cache_path) and not force:
        if verbose:
            print(f"    [cached] {os.path.basename(cache_path)}", flush=True)
        from rate_computation import load_rates
        return load_rates(cache_path)['rates']

    if verbose:
        pipeline = 'hi-res' if use_hires else 'lo-res'
        print(f"    Computing ({pipeline}): LM={logmar:.2f}, ori={ori}, {cond}...", flush=True)

    from rate_computation import (compute_population_rates,
                                   compute_population_rates_hires, save_rates)

    if use_hires:
        result = compute_population_rates_hires(
            model, readout, ori, logmar, traces, durations,
            condition=cond,
            null_traces=null_traces_arr if cond == 'matched_null' else None,
            stim_params={'logmar': logmar, 'orientation': ori},
            verbose=False,
        )
    else:
        from stimulus import e_optotype_stack
        stim_stack = e_optotype_stack(ori, logmar)
        result = compute_population_rates(
            model, readout, stim_stack, traces, durations,
            condition=cond,
            null_traces=null_traces_arr if cond == 'matched_null' else None,
            stim_params={'logmar': logmar, 'orientation': ori},
            verbose=False,
        )

    save_rates(result, cache_path)
    return result['rates']


def main():
    parser = argparse.ArgumentParser(description='Dual-regime neurometric sweep')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_pca_C', type=int, default=50,
                        help='PCA components for Model C temporal residual')
    parser.add_argument('--no_matched_null', action='store_true')
    parser.add_argument('--force', action='store_true', help='Recompute cached rates')
    parser.add_argument('--resolved_only', action='store_true')
    parser.add_argument('--hyperacuity_only', action='store_true')
    parser.add_argument('--n_traces', type=int, default=None, help='Limit traces (debug)')
    args = parser.parse_args()

    conditions = CONDITIONS if not args.no_matched_null else ['real', 'stabilized']

    if args.resolved_only:
        logmar_values = LOGMAR_RESOLVED
    elif args.hyperacuity_only:
        logmar_values = LOGMAR_HYPERACUITY
    else:
        logmar_values = LOGMAR_FULL

    print(f"LogMAR values: {logmar_values}")
    print(f"Conditions: {conditions}")
    print(f"Hi-res threshold: LogMAR < {HIRES_THRESHOLD}")
    print()

    # Load model
    model, readout = load_model_and_readout()

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    td = load_eye_traces(EYE_TRACES_PATH)
    print_summary(td)

    traces = td['traces']
    durations = td['durations']

    if args.n_traces is not None:
        idx = np.random.choice(len(traces), min(args.n_traces, len(traces)), replace=False)
        traces = traces[idx]
        durations = durations[idx]
        print(f"  Using {len(traces)} traces")

    # Generate null traces if needed
    null_traces_arr = None
    if 'matched_null' in conditions:
        from null_traces import generate_phase_randomized_traces
        print("Generating null traces...")
        null_traces_arr = generate_phase_randomized_traces(traces, n_nulls=5, seed=42)

    # Run neurometric sweep
    from neurometric import fit_neurometric_threshold, plot_neurometric_curves, save_neurometric_results
    from decoding import run_decoding_ladder

    results = {
        'logmar_values': logmar_values,
        'conditions': conditions,
        'models': ['A', 'C'],
        'accuracy': {},
        'accuracy_std': {},
        'threshold': {},
        'pipeline': {},  # track which pipeline used at each LogMAR
    }
    for cond in conditions:
        for mod in ['A', 'C']:
            results['accuracy'][(cond, mod)] = []
            results['accuracy_std'][(cond, mod)] = []

    for logmar in logmar_values:
        use_hires = (logmar < HIRES_THRESHOLD)
        print(f"\n=== LogMAR = {logmar:.2f} ({'hi-res' if use_hires else 'lo-res'}) ===",
              flush=True)
        results['pipeline'][logmar] = 'hires' if use_hires else 'lores'

        for cond in conditions:
            rates_by_stim = {}
            for ori in ORIENTATIONS:
                rates = compute_and_cache_rates(
                    model, readout, logmar, ori, cond,
                    traces, durations, null_traces_arr,
                    force=args.force, verbose=True,
                )
                rates_by_stim[f'ori{ori}'] = rates

            print(f"  Decoding: {cond}...", flush=True)
            ladder = run_decoding_ladder(
                rates_by_stim,
                models=['A', 'C'],
                n_splits=args.n_splits,
                n_components_C=args.n_pca_C,
                verbose=False,
            )

            for mod in ['A', 'C']:
                r = ladder[mod]
                results['accuracy'][(cond, mod)].append(r['mean_acc'])
                results['accuracy_std'][(cond, mod)].append(r['std_acc'])
                print(f"    {cond} Model {mod}: {r['mean_acc']:.4f} ± {r['std_acc']:.4f}  "
                      f"(gain: {r['mean_acc']-ladder['A']['mean_acc']:+.4f})", flush=True)

    # Convert to arrays and compute thresholds
    lm_arr = np.array(logmar_values)
    for cond in conditions:
        for mod in ['A', 'C']:
            acc = np.array(results['accuracy'][(cond, mod)])
            std = np.array(results['accuracy_std'][(cond, mod)])
            results['accuracy'][(cond, mod)] = acc
            results['accuracy_std'][(cond, mod)] = std
            thr = fit_neurometric_threshold(lm_arr, acc)
            results['threshold'][(cond, mod)] = thr

    t_stab_A = results['threshold'].get(('stabilized', 'A'))
    t_real_C = results['threshold'].get(('real', 'C'))
    if t_stab_A is not None and t_real_C is not None:
        delta_lm = t_stab_A - t_real_C
        results['delta_logmar'] = delta_lm
        print(f"\n=== ΔLogMAR = {delta_lm:.3f} ===")
        print(f"  Threshold(stabilized, A) = {t_stab_A:.3f}")
        print(f"  Threshold(real, C)       = {t_real_C:.3f}")
    else:
        results['delta_logmar'] = None
        print(f"\n  Thresholds: stab_A={t_stab_A}, real_C={t_real_C}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, 'neurometric_dual_regime.npz')
    save_neurometric_results(results, output_path)
    print(f"\nSaved: {output_path}")

    # Plot
    import matplotlib.pyplot as plt
    from plotting import save_figure

    fig = plot_neurometric_curves(results)
    save_figure(fig, os.path.join(FIGURES_DIR, 'fig_neurometric_dual_regime.png'))
    plt.close(fig)

    # Also plot with pipeline annotation
    fig2, ax = plt.subplots(figsize=(10, 6))
    style_map = {
        ('real', 'A'):         ('cornflowerblue', '--', 'o'),
        ('real', 'C'):         ('royalblue', '-', 'o'),
        ('stabilized', 'A'):   ('salmon', '--', 's'),
        ('stabilized', 'C'):   ('tomato', '-', 's'),
        ('matched_null', 'A'): ('silver', '--', '^'),
        ('matched_null', 'C'): ('gray', '-', '^'),
    }
    lm_sort = np.argsort(lm_arr)
    lm_sorted = lm_arr[lm_sort]

    for cond in conditions:
        for mod in ['A', 'C']:
            key = (cond, mod)
            if key not in style_map:
                continue
            color, ls, marker = style_map[key]
            acc = results['accuracy'][key][lm_sort]
            std = results['accuracy_std'][key][lm_sort]
            label = f'{cond} Model {mod}'
            ax.plot(lm_sorted, acc, color=color, linestyle=ls,
                    marker=marker, markersize=6, label=label)
            ax.fill_between(lm_sorted, acc-std, acc+std, alpha=0.12, color=color)
            thr = results['threshold'].get(key)
            if thr is not None:
                ax.axvline(thr, color=color, linestyle=':', alpha=0.7, linewidth=0.8)

    # Mark regime boundary
    ax.axvline(HIRES_THRESHOLD, color='purple', linestyle='--', alpha=0.4,
               label=f'Lo-res/Hi-res split (LM={HIRES_THRESHOLD})')
    ax.axhline(0.625, color='k', linestyle='--', alpha=0.5, label='Threshold criterion')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.3, label='Chance')

    dlogmar = results.get('delta_logmar')
    title = 'Dual-Regime Neurometric Curves (Corrected Pipeline)'
    if dlogmar is not None:
        title += f'\nΔLogMAR = {dlogmar:.3f}'
    ax.set_title(title)
    ax.set_xlabel('LogMAR')
    ax.set_ylabel('Decoding accuracy')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.1, 1.05])

    save_figure(fig2, os.path.join(FIGURES_DIR, 'fig_neurometric_dual_regime_annotated.png'))
    plt.close(fig2)
    print(f"Saved figures.")


if __name__ == '__main__':
    main()

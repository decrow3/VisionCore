"""
Fine-grained neurometric sweep around the threshold (LogMAR 0.0 to -0.10).

The full run showed all conditions at ceiling at 0.0 and at chance at -0.10.
This script fills in -0.02, -0.04, -0.06, -0.08 to characterize the
psychometric slope and detect any FEM benefit at sub-threshold sizes.

Conditions: real, stabilized, matched_null
Models: A, C
LogMAR grid: [0.0, -0.02, -0.04, -0.06, -0.08, -0.10]  (uses cached -0.10 and 0.0)
"""
import os
import sys
import argparse
import numpy as np
import pickle

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

# Fine-grained LogMAR grid spanning the threshold
LOGMAR_FINE = [0.02, 0.00, -0.02, -0.04, -0.06, -0.08, -0.10]
ORIENTATIONS = [0, 90, 180, 270]
CONDITIONS = ['real', 'stabilized', 'matched_null']


def main():
    parser = argparse.ArgumentParser(description='Fine-grained neurometric sweep')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_pca_C', type=int, default=30)
    parser.add_argument('--no_matched_null', action='store_true',
                        help='Skip matched_null condition (saves time)')
    parser.add_argument('--force', action='store_true', help='Recompute cached files')
    parser.add_argument('--mode', type=str, default='standard')
    args = parser.parse_args()

    conditions = CONDITIONS if not args.no_matched_null else ['real', 'stabilized']
    output_path = os.path.join(RESULTS_DIR, 'neurometric_fine.npz')

    # Load model and readout
    import dill, torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model (mode={args.mode}, device={device})...")
    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout
    model, _ = get_model_and_dataset_configs(mode=args.mode)
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    model = model.to(device)
    print(f"Loading readout from {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    traces_data = load_eye_traces(EYE_TRACES_PATH)
    print_summary(traces_data)
    traces = traces_data['traces']
    durations = traces_data['durations']

    # Generate null traces if needed
    null_traces_arr = None
    if 'matched_null' in conditions:
        from null_traces import generate_phase_randomized_traces
        print("Generating null traces...")
        null_traces_arr = generate_phase_randomized_traces(traces, n_nulls=5, seed=42)

    # Run fine-grained neurometric sweep
    from neurometric import compute_neurometric_curve, save_neurometric_results, \
        fit_neurometric_threshold, plot_neurometric_curves
    from plotting import save_figure
    import matplotlib.pyplot as plt

    print(f"\nRunning fine-grained neurometric sweep...")
    print(f"  LogMAR values: {LOGMAR_FINE}")
    print(f"  Conditions: {conditions}")

    neuro_results = compute_neurometric_curve(
        model, readout, traces, durations,
        logmar_values=LOGMAR_FINE,
        orientations=ORIENTATIONS,
        conditions=conditions,
        models_to_run=('A', 'C'),
        n_splits=args.n_splits,
        n_pca_C=args.n_pca_C,
        null_traces=null_traces_arr if 'matched_null' in conditions else None,
        cache_dir=RATES_DIR,
        verbose=True,
    )

    # Save results
    save_neurometric_results(neuro_results, output_path)
    print(f"\nSaved: {output_path}")

    # Print key results
    print("\n=== Fine-grained neurometric results ===")
    lm = np.array(neuro_results['logmar_values'])
    for cond in conditions:
        for mod in ('A', 'C'):
            if ('accuracy' in neuro_results and
                    (cond, mod) in neuro_results['accuracy']):
                acc = neuro_results['accuracy'][(cond, mod)]
                thr = neuro_results['threshold'].get((cond, mod))
                print(f"  {cond:15s} Model {mod}: threshold={thr}")
                for lv, av in zip(lm, acc):
                    print(f"    LogMAR {lv:+.2f}: {av:.4f}")

    print(f"\n  ΔLogMAR = {neuro_results.get('delta_logmar')}")

    # Plot
    fig = plot_neurometric_curves(neuro_results)
    fig_path = os.path.join(FIGURES_DIR, 'fig_neurometric_fine.png')
    save_figure(fig, fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Also make a zoomed-in plot showing just the fine range
    fig2, ax = plt.subplots(figsize=(8, 5))
    style_map = {
        ('real', 'A'):       ('cornflowerblue', '--', 'o', 'Real FEM, Model A'),
        ('real', 'C'):       ('royalblue', '-', 'o', 'Real FEM, Model C'),
        ('stabilized', 'A'): ('salmon', '--', 's', 'Stabilized, Model A'),
        ('stabilized', 'C'): ('tomato', '-', 's', 'Stabilized, Model C'),
        ('matched_null', 'A'): ('silver', '--', '^', 'Null FEM, Model A'),
        ('matched_null', 'C'): ('gray', '-', '^', 'Null FEM, Model C'),
    }
    lm_sorted_idx = np.argsort(lm)
    lm_sorted = lm[lm_sorted_idx]

    for cond in conditions:
        for mod in ('A', 'C'):
            key = (cond, mod)
            if key not in style_map:
                continue
            if ('accuracy' not in neuro_results or
                    key not in neuro_results['accuracy']):
                continue
            color, ls, marker, label = style_map[key]
            acc = neuro_results['accuracy'][key]
            std = neuro_results['accuracy_std'][key]
            acc_sorted = acc[lm_sorted_idx]
            std_sorted = std[lm_sorted_idx]
            ax.plot(lm_sorted, acc_sorted, color=color, linestyle=ls,
                    marker=marker, markersize=6, label=label)
            ax.fill_between(lm_sorted, acc_sorted - std_sorted,
                            acc_sorted + std_sorted, alpha=0.15, color=color)
            thr = neuro_results['threshold'].get(key)
            if thr is not None:
                ax.axvline(thr, color=color, linestyle=':', alpha=0.7, linewidth=0.8)

    ax.axhline(0.625, color='black', linestyle='--', alpha=0.5, label='Threshold criterion')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.3, label='Chance (4AFC)')
    ax.set_xlabel('LogMAR (negative = below 20/20)')
    ax.set_ylabel('Decoding accuracy')
    ax.set_title('Fine-grained Neurometric Curves\n(threshold region)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([lm_sorted[0] - 0.01, lm_sorted[-1] + 0.01])
    ax.set_ylim([0.0, 1.1])

    dlogmar = neuro_results.get('delta_logmar')
    if dlogmar is not None:
        ax.set_title(f'Fine-grained Neurometric Curves\nΔLogMAR = {dlogmar:.3f}')

    fig2_path = os.path.join(FIGURES_DIR, 'fig_neurometric_fine_zoom.png')
    save_figure(fig2, fig2_path)
    plt.close(fig2)
    print(f"Saved: {fig2_path}")


if __name__ == '__main__':
    main()

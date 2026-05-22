"""
Task 2.4: Neurometric Curves

Sweeps LogMAR values and plots decoding accuracy vs. stimulus size for each
FEM condition and model. Defines neural acuity threshold as the LogMAR where
accuracy reaches 62.5% (midpoint of chance-to-ceiling for 4AFC).

This is the most computationally expensive task. Use --reduced for fast testing.

Usage:
    python neurometric.py [--reduced] [--n_traces 50] [--output results/neurometric.npz]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logmar grid
LOGMAR_FULL = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.1, -0.2, -0.3]
LOGMAR_REDUCED = [1.0, 0.6, 0.2, 0.0, -0.2]
ORIENTATIONS = [0, 90, 180, 270]
CONDITIONS_FULL = ['real', 'stabilized', 'matched_null']
CONDITIONS_REDUCED = ['real', 'stabilized']
THRESHOLD_CRITERION = 0.625  # midpoint between chance (0.25) and ceiling (1.0)


def fit_neurometric_threshold(
    logmar_values: np.ndarray,
    accuracies: np.ndarray,
    criterion: float = THRESHOLD_CRITERION,
) -> Optional[float]:
    """
    Find the LogMAR where accuracy crosses the threshold criterion.

    Uses linear interpolation between adjacent sampled points.
    Returns None if the curve never reaches the criterion.

    Args:
        logmar_values: (n,) LogMAR values (any order)
        accuracies: (n,) mean decoding accuracies at each LogMAR
        criterion: accuracy threshold (default 0.625 for 4AFC)

    Returns:
        threshold_logmar: LogMAR at criterion, or None
    """
    logmar_values = np.asarray(logmar_values, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)

    if logmar_values.ndim != 1 or accuracies.ndim != 1 or logmar_values.shape[0] != accuracies.shape[0]:
        raise ValueError("logmar_values and accuracies must be 1D arrays of the same length")

    # Sort by LogMAR ascending (hard -> easy).
    sort_idx = np.argsort(logmar_values)
    lm = logmar_values[sort_idx]
    acc = accuracies[sort_idx]

    if not np.isfinite(acc).all():
        # Upstream should avoid NaNs, but be defensive.
        finite = np.isfinite(acc)
        lm = lm[finite]
        acc = acc[finite]
        if lm.size == 0:
            return None

    if acc.max() < criterion:
        return None  # never reaches criterion
    if acc.min() >= criterion:
        return float(lm.min())  # always above criterion (threshold is beyond hard end)

    y = acc - float(criterion)

    # Find the first crossing from below->above as we go from hard->easy.
    # This yields the hardest LogMAR at which performance reaches criterion.
    for i in range(len(lm) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if y0 < 0.0 and y1 >= 0.0:
            x0, x1 = float(lm[i]), float(lm[i + 1])
            if y1 == y0:
                return x1
            t = (0.0 - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)

    # If we get here, the curve dipped below criterion but never rose back above
    # within the sampled range; report the easiest end.
    return float(lm.max())


def compute_neurometric_curve(
    model,
    readout,
    eye_traces: np.ndarray,
    durations: np.ndarray,
    logmar_values: list = LOGMAR_REDUCED,
    orientations: list = ORIENTATIONS,
    conditions: list = CONDITIONS_REDUCED,
    models_to_run: tuple = ('A', 'C'),
    n_splits: int = 5,
    n_pca_C: int = 30,
    null_traces: np.ndarray = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute neurometric curves: decoding accuracy vs. LogMAR for each condition.

    Caches intermediate rate matrices to avoid recomputation.

    Args:
        model: loaded VisionCore model
        readout: PopulationReadout
        eye_traces: (M, max_T, 2) eye trace array
        durations: (M,) valid trace lengths
        logmar_values: list of LogMAR values to evaluate
        orientations: list of orientations (E opening directions)
        conditions: list of FEM conditions
        models_to_run: which decoder models ('A', 'C')
        n_splits: CV folds
        n_pca_C: PCA components for Model C
        null_traces: (M, n_nulls, max_T, 2) for matched_null condition
        cache_dir: directory to cache rate matrices (None = no caching)
        verbose: print progress

    Returns:
        dict with:
            logmar_values: list
            conditions: list
            models: list
            accuracy: dict (condition, model) → (n_logmar,) mean accuracy
            accuracy_std: dict (condition, model) → (n_logmar,) std accuracy
            threshold: dict (condition, model) → threshold LogMAR or None
            delta_logmar: threshold(stabilized, A) - threshold(real, C)
    """
    from stimulus import e_optotype_stack
    from rate_computation import compute_population_rates, save_rates, load_rates
    from decoding import run_decoding_ladder

    results = {
        'logmar_values': logmar_values,
        'conditions': conditions,
        'models': list(models_to_run),
        'accuracy': {},
        'accuracy_std': {},
        'threshold': {},
    }

    # Accumulate per-logmar per-condition per-model accuracies
    for cond in conditions:
        for mod in models_to_run:
            results['accuracy'][(cond, mod)] = []
            results['accuracy_std'][(cond, mod)] = []

    for logmar in logmar_values:
        if verbose:
            print(f"\n=== LogMAR = {logmar:.2f} ===")

        # Load or compute rate matrices for all orientations under each condition
        rates_by_stim_per_condition = {cond: {} for cond in conditions}

        for ori in orientations:
            stim_stack = e_optotype_stack(ori, logmar)
            stim_id = f'ori{ori}'

            for cond in conditions:
                # Check cache
                if cache_dir is not None:
                    cache_path = os.path.join(
                        cache_dir, f'rates_lm{logmar:.2f}_ori{ori}_{cond}.npz'
                    )
                    if os.path.exists(cache_path):
                        if verbose:
                            print(f"  Loading cached: {os.path.basename(cache_path)}")
                        loaded = load_rates(cache_path)
                        rates_by_stim_per_condition[cond][stim_id] = loaded['rates']
                        continue

                if verbose:
                    print(f"  Computing rates: LogMAR={logmar:.2f}, ori={ori}, {cond}...")

                result = compute_population_rates(
                    model, readout, stim_stack, eye_traces, durations,
                    condition=cond,
                    null_traces=null_traces if cond == 'matched_null' else None,
                    stim_params={'logmar': logmar, 'orientation': ori},
                    verbose=False,
                )

                rates_by_stim_per_condition[cond][stim_id] = result['rates']

                if cache_dir is not None:
                    os.makedirs(cache_dir, exist_ok=True)
                    save_rates(result, cache_path)

        # Run decoding for each condition
        for cond in conditions:
            if verbose:
                print(f"  Decoding: condition={cond}...")

            ladder_results = run_decoding_ladder(
                rates_by_stim_per_condition[cond],
                models=list(models_to_run),
                n_splits=n_splits,
                n_components_C=n_pca_C,
                verbose=False,
            )

            for mod in models_to_run:
                if mod in ladder_results:
                    results['accuracy'][(cond, mod)].append(
                        ladder_results[mod]['mean_acc']
                    )
                    results['accuracy_std'][(cond, mod)].append(
                        ladder_results[mod]['std_acc']
                    )

    # Convert to arrays and compute thresholds
    lm_arr = np.array(logmar_values)
    for cond in conditions:
        for mod in models_to_run:
            acc = np.array(results['accuracy'][(cond, mod)])
            std = np.array(results['accuracy_std'][(cond, mod)])
            results['accuracy'][(cond, mod)] = acc
            results['accuracy_std'][(cond, mod)] = std
            results['threshold'][(cond, mod)] = fit_neurometric_threshold(lm_arr, acc)

    # Compute ΔLogMAR: threshold(stabilized, A) - threshold(real, C)
    t_stab_A = results['threshold'].get(('stabilized', 'A'))
    t_real_C = results['threshold'].get(('real', 'C'))
    if t_stab_A is not None and t_real_C is not None:
        results['delta_logmar'] = t_stab_A - t_real_C
        if verbose:
            print(f"\n=== ΔLogMAR = {results['delta_logmar']:.3f} ===")
            print(f"  Threshold(stabilized, A) = {t_stab_A:.3f}")
            print(f"  Threshold(real, C)       = {t_real_C:.3f}")
    else:
        results['delta_logmar'] = None

    return results


def plot_neurometric_curves(
    results: dict,
    chance: float = 0.25,
    figsize=(9, 6),
) -> plt.Figure:
    """
    Plot neurometric curves with threshold markers.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    style_map = {
        ('real', 'C'):         ('royalblue', '-', 'Real FEM, Model C'),
        ('real', 'A'):         ('cornflowerblue', '--', 'Real FEM, Model A'),
        ('stabilized', 'C'):   ('tomato', '-', 'Stabilized, Model C'),
        ('stabilized', 'A'):   ('firebrick', '--', 'Stabilized, Model A'),
        ('matched_null', 'C'): ('gray', '-', 'Null FEM, Model C'),
        ('matched_null', 'A'): ('silver', '--', 'Null FEM, Model A'),
    }

    lm_arr = np.array(results['logmar_values'])
    sort_idx = np.argsort(lm_arr)
    lm_sorted = lm_arr[sort_idx]

    for (cond, mod), acc_arr in results['accuracy'].items():
        key = (cond, mod)
        color, ls, label = style_map.get(key, ('black', '-', f'{cond}, {mod}'))
        acc_sorted = np.array(acc_arr)[sort_idx]
        std_sorted = np.array(results['accuracy_std'][key])[sort_idx]

        ax.plot(lm_sorted, acc_sorted, ls, color=color, label=label, linewidth=2)
        ax.fill_between(lm_sorted,
                        acc_sorted - std_sorted,
                        acc_sorted + std_sorted,
                        alpha=0.15, color=color)

        # Mark threshold
        threshold = results['threshold'].get(key)
        if threshold is not None:
            ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=1)

    ax.axhline(y=chance, color='black', linestyle='--', alpha=0.4, label='chance (25%)')
    ax.axhline(y=THRESHOLD_CRITERION, color='black', linestyle=':', alpha=0.4,
               label=f'threshold criterion ({THRESHOLD_CRITERION:.0%})')

    # ΔLogMAR annotation
    if results.get('delta_logmar') is not None:
        ax.text(0.02, 0.97,
                f'ΔLogMAR = {results["delta_logmar"]:.3f}',
                transform=ax.transAxes, fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('LogMAR', fontsize=12)
    ax.set_ylabel('Decoding accuracy', fontsize=12)
    ax.set_title('Neurometric Curves: E Orientation Discrimination', fontsize=13)
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.01, 0.90))
    ax.set_ylim([chance - 0.05, 1.05])
    ax.invert_xaxis()  # small LogMAR (fine) on right, large (coarse) on left
    ax.grid(True, alpha=0.3)

    return fig


def save_neurometric_results(results: dict, path: str) -> None:
    """Save neurometric results to .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_dict = {
        'logmar_values': np.array(results['logmar_values']),
        'conditions': np.array(results['conditions']),
        'models': np.array(results['models']),
        'delta_logmar': np.array([results.get('delta_logmar', np.nan)]),
    }
    for (cond, mod), arr in results['accuracy'].items():
        save_dict[f'acc_{cond}_{mod}'] = np.array(arr)
        save_dict[f'std_{cond}_{mod}'] = np.array(results['accuracy_std'][(cond, mod)])
        threshold = results['threshold'].get((cond, mod))
        save_dict[f'threshold_{cond}_{mod}'] = np.array(
            [threshold if threshold is not None else np.nan]
        )
    np.savez_compressed(path, **save_dict)
    print(f"Saved neurometric results to {path}")


def load_neurometric_results(path: str) -> dict:
    """Load neurometric results from .npz file."""
    d = np.load(path, allow_pickle=True)
    logmar_values = d['logmar_values'].tolist()
    conditions = d['conditions'].tolist()
    models = d['models'].tolist()

    results = {
        'logmar_values': logmar_values,
        'conditions': conditions,
        'models': models,
        'accuracy': {},
        'accuracy_std': {},
        'threshold': {},
        'delta_logmar': float(d['delta_logmar'][0]),
    }

    for cond in conditions:
        for mod in models:
            key = (cond, mod)
            k_acc = f'acc_{cond}_{mod}'
            k_std = f'std_{cond}_{mod}'
            k_thr = f'threshold_{cond}_{mod}'
            if k_acc in d:
                results['accuracy'][key] = d[k_acc]
                results['accuracy_std'][key] = d[k_std]
                thr = float(d[k_thr][0])
                results['threshold'][key] = None if np.isnan(thr) else thr

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute neurometric curves')
    parser.add_argument('--reduced', action='store_true',
                        help='Use reduced LogMAR grid and conditions for faster testing')
    parser.add_argument('--n_traces', type=int, default=None,
                        help='Number of eye traces to use (None = all)')
    parser.add_argument('--output', default=os.path.join(
        os.path.dirname(__file__), 'data', 'results', 'neurometric.npz'))
    parser.add_argument('--cache_dir', default=os.path.join(
        os.path.dirname(__file__), 'data', 'rates'))
    parser.add_argument('--mode', default='standard')
    args = parser.parse_args()

    import dill
    import torch
    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout
    from extract_eye_traces import load_eye_traces

    traces_path = os.path.join(os.path.dirname(__file__), 'data', 'eye_traces.npz')
    if not os.path.exists(traces_path):
        print(f"ERROR: {traces_path} not found. Run extract_eye_traces.py first.")
        sys.exit(1)

    print("Loading model...")
    model, _ = get_model_and_dataset_configs(mode=args.mode)
    model.model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    pkl_path = os.path.join(os.path.dirname(__file__), '..', 'mcfarland_outputs_mono.pkl')
    with open(pkl_path, 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)

    traces_data = load_eye_traces(traces_path)
    traces = traces_data['traces']
    durations = traces_data['durations']

    if args.n_traces is not None:
        traces = traces[:args.n_traces]
        durations = durations[:args.n_traces]

    logmar_values = LOGMAR_REDUCED if args.reduced else LOGMAR_FULL
    conditions = CONDITIONS_REDUCED if args.reduced else CONDITIONS_FULL

    print(f"Running neurometric sweep: {len(logmar_values)} LogMAR values, "
          f"{len(conditions)} conditions, {len(traces)} traces")

    results = compute_neurometric_curve(
        model, readout,
        traces, durations,
        logmar_values=logmar_values,
        conditions=conditions,
        cache_dir=args.cache_dir,
        verbose=True,
    )

    save_neurometric_results(results, args.output)

    fig = plot_neurometric_curves(results)
    fig_path = args.output.replace('.npz', '.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")

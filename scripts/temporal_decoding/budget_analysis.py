"""
Task 3.3: Trace-Budget Stratification

Bins real eye traces by oculomotor budget (RMS displacement) and tests
whether the temporal coding gain (Model C − Model A) scales with movement.

Key prediction: larger FEMs should produce greater temporal coding benefit.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bin_traces_by_budget(
    traces: np.ndarray,
    durations: np.ndarray,
    rms: np.ndarray,
    n_bins: int = 3,
    bin_labels: list = None,
) -> dict:
    """
    Bin traces into equal-count groups based on RMS displacement.

    Args:
        traces: (M, max_T, 2) eye traces
        durations: (M,) valid trace lengths
        rms: (M,) RMS displacement values
        n_bins: number of bins (default 3 = terciles)
        bin_labels: optional list of bin names

    Returns:
        dict with bin label → dict with 'traces', 'durations', 'rms', 'indices'
    """
    if bin_labels is None:
        bin_labels = ['low', 'medium', 'high'][:n_bins]
        if n_bins > 3:
            bin_labels = [f'bin_{i}' for i in range(n_bins)]

    M = len(rms)
    bin_edges = np.percentile(rms, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10  # include max

    bins = {}
    for i in range(n_bins):
        label = bin_labels[i]
        lo, hi = bin_edges[i], bin_edges[i + 1]
        idx = np.where((rms >= lo) & (rms < hi))[0]
        bins[label] = {
            'indices': idx,
            'traces': traces[idx],
            'durations': durations[idx],
            'rms': rms[idx],
            'rms_mean': float(rms[idx].mean()),
            'rms_range': (float(lo), float(hi)),
            'n': len(idx),
        }

    return bins


def compute_budget_gain(
    rates_by_stim_by_bin: dict,
    n_splits: int = 5,
    n_components_C: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Compute decoding gain (Model C − Model A accuracy) for each trace budget bin.

    Args:
        rates_by_stim_by_bin: dict bin_label → dict stim_id → (M_bin, T, N) rates
        n_splits: CV folds
        n_components_C: PCA components for Model C
        verbose: print progress

    Returns:
        dict with bin_label → dict with:
            acc_A, std_A: Model A accuracy ± std
            acc_C, std_C: Model C accuracy ± std
            gain: acc_C - acc_A
            gain_std: propagated uncertainty
            n: number of traces in bin
            rms_mean: mean RMS in bin
    """
    from decoding import run_decoding_ladder

    results = {}
    for bin_label, rates_by_stim in rates_by_stim_by_bin.items():
        M_bin = next(iter(rates_by_stim.values())).shape[0]
        if M_bin < n_splits:
            print(f"  Bin '{bin_label}': only {M_bin} traces, skipping (need ≥ {n_splits})")
            continue

        if verbose:
            print(f"\n  Bin '{bin_label}': {M_bin} traces")

        ladder = run_decoding_ladder(
            {sid: [rates_by_stim[sid][i] for i in range(M_bin)]
             for sid in rates_by_stim},
            models=['A', 'C'],
            n_splits=min(n_splits, M_bin),
            n_components_C=n_components_C,
            verbose=False,
        )

        acc_A = ladder['A']['mean_acc']
        std_A = ladder['A']['std_acc']
        acc_C = ladder['C']['mean_acc']
        std_C = ladder['C']['std_acc']
        gain = acc_C - acc_A
        gain_std = np.sqrt(std_A ** 2 + std_C ** 2)

        results[bin_label] = {
            'acc_A': acc_A, 'std_A': std_A,
            'acc_C': acc_C, 'std_C': std_C,
            'gain': gain, 'gain_std': gain_std,
        }

        if verbose:
            print(f"    Model A: {acc_A:.3f} ± {std_A:.3f}")
            print(f"    Model C: {acc_C:.3f} ± {std_C:.3f}")
            print(f"    Gain C−A: {gain:+.3f} ± {gain_std:.3f}")

    return results


def run_budget_stratification(
    model,
    readout,
    logmar: float,
    traces: np.ndarray,
    durations: np.ndarray,
    rms: np.ndarray,
    orientations: list = None,
    n_bins: int = 3,
    n_splits: int = 5,
    n_components_C: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Full budget stratification pipeline: bin traces, compute rates, decode.

    Args:
        model: loaded VisionCore model
        readout: PopulationReadout
        logmar: LogMAR value for E stimulus (generates all 4 orientation variants)
        traces: (M, max_T, 2) eye traces
        durations: (M,) valid lengths
        rms: (M,) RMS displacement
        orientations: list of orientation angles (default [0, 90, 180, 270])
        n_bins: number of budget bins
        n_splits: CV folds
        n_components_C: PCA components for Model C
        verbose: print progress

    Returns:
        dict with:
            bins_metadata: output of bin_traces_by_budget (without rates)
            gain_by_bin: output of compute_budget_gain
            rms_means: list of mean RMS per bin
            gains: list of gain per bin
            trend: 'monotonic_increase', 'monotonic_decrease', 'peak', 'flat'
    """
    from rate_computation import compute_population_rates
    from stimulus import e_optotype_stack

    if orientations is None:
        orientations = [0, 90, 180, 270]

    # Step 1: Bin traces
    bins = bin_traces_by_budget(traces, durations, rms, n_bins=n_bins)
    if verbose:
        for label, b in bins.items():
            print(f"  Bin '{label}': n={b['n']}, "
                  f"RMS={b['rms_mean']:.4f} deg "
                  f"({b['rms_range'][0]:.4f}–{b['rms_range'][1]:.4f})")

    # Step 2: Compute rates per bin for each stimulus orientation
    # Each orientation is a different stimulus class — must generate its own stack
    rates_by_stim_by_bin = {label: {} for label in bins}

    for ori in orientations:
        stim_id = f'ori{ori}'
        stim_stack_ori = e_optotype_stack(ori, logmar)  # generate per-orientation stack
        if verbose:
            print(f"\nComputing rates for orientation {ori}°...")

        for bin_label, bin_data in bins.items():
            bin_traces = bin_data['traces']
            bin_durations = bin_data['durations']
            M_bin = len(bin_traces)

            if M_bin == 0:
                continue

            result = compute_population_rates(
                model, readout, stim_stack_ori, bin_traces, bin_durations,
                condition='real', verbose=False,
            )

            # Convert to uniform array (truncate to min length)
            min_T = min(r.shape[0] for r in result['rates'])
            rates_arr = np.stack([r[:min_T] for r in result['rates']])  # (M_bin, T, N)
            rates_by_stim_by_bin[bin_label][stim_id] = rates_arr

    # Step 3: Decode and compute gains
    if verbose:
        print("\nDecoding per bin...")
    gain_results = compute_budget_gain(
        rates_by_stim_by_bin,
        n_splits=n_splits,
        n_components_C=n_components_C,
        verbose=verbose,
    )

    # Collect ordered results
    bin_labels = sorted(bins.keys())
    rms_means = [bins[l]['rms_mean'] for l in bin_labels if l in gain_results]
    gains = [gain_results[l]['gain'] for l in bin_labels if l in gain_results]
    ns = [bins[l]['n'] for l in bin_labels if l in gain_results]

    # Classify trend
    trend = 'flat'
    if len(gains) >= 2:
        diffs = np.diff(gains)
        if all(d > 0 for d in diffs):
            trend = 'monotonic_increase'
        elif all(d < 0 for d in diffs):
            trend = 'monotonic_decrease'
        elif len(gains) >= 3 and gains[1] > gains[0] and gains[1] > gains[-1]:
            trend = 'peak'

    if verbose:
        print(f"\nTrend: {trend}")

    return {
        'bins_metadata': {l: {k: v for k, v in bins[l].items() if k != 'traces'}
                          for l in bins},
        'gain_by_bin': gain_results,
        'rms_means': rms_means,
        'gains': gains,
        'ns': ns,
        'trend': trend,
        'bin_labels': [l for l in bin_labels if l in gain_results],
    }


def plot_budget_gain(
    stratification_result: dict,
    chance: float = 0.25,
    figsize=(8, 5),
) -> plt.Figure:
    """Plot decoding gain vs. oculomotor budget bin."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    bin_labels = stratification_result['bin_labels']
    gain_results = stratification_result['gain_by_bin']
    rms_means = stratification_result['rms_means']

    # Panel 1: Absolute accuracy per model
    x = np.arange(len(bin_labels))
    width = 0.35
    ax = axes[0]
    acc_A = [gain_results[l]['acc_A'] for l in bin_labels]
    acc_C = [gain_results[l]['acc_C'] for l in bin_labels]
    std_A = [gain_results[l]['std_A'] for l in bin_labels]
    std_C = [gain_results[l]['std_C'] for l in bin_labels]

    ax.bar(x - width/2, acc_A, width, yerr=std_A, label='Model A',
           color='tomato', alpha=0.8, capsize=4)
    ax.bar(x + width/2, acc_C, width, yerr=std_C, label='Model C',
           color='royalblue', alpha=0.8, capsize=4)
    ax.axhline(y=chance, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(RMS={r:.3f}°)" for l, r in zip(bin_labels, rms_means)])
    ax.set_ylabel('Decoding accuracy')
    ax.set_title('Accuracy by Oculomotor Budget')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # Panel 2: Gain (C − A)
    ax = axes[1]
    gains = [gain_results[l]['gain'] for l in bin_labels]
    gain_stds = [gain_results[l]['gain_std'] for l in bin_labels]
    ns = stratification_result['ns']

    bars = ax.bar(x, gains, yerr=gain_stds, color='steelblue', alpha=0.8, capsize=4)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(bin_labels, ns)])
    ax.set_ylabel('Gain (Model C − Model A)')
    ax.set_title(f'Temporal Coding Gain vs. Budget\n(trend: {stratification_result["trend"]})')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate gain values
    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{gain:+.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Testing budget stratification with synthetic data...")
    np.random.seed(42)

    M = 90
    rms = np.random.exponential(0.03, M).astype(np.float32)
    durations = np.random.randint(80, 300, M)

    bins = bin_traces_by_budget(
        np.zeros((M, 300, 2), dtype=np.float32),  # dummy traces
        durations, rms, n_bins=3
    )

    for label, b in bins.items():
        print(f"  Bin '{label}': n={b['n']}, RMS={b['rms_mean']:.4f}")
        assert b['n'] > 0, f"Empty bin: {label}"
    print("Binning: OK")

    # Test compute_budget_gain with synthetic rate data
    K, T, N = 4, 60, 10
    stim_ids = ['ori0', 'ori90', 'ori180', 'ori270']

    # Make rates_by_stim_by_bin where high-budget bins have better temporal structure
    rates_by_stim_by_bin = {}
    for bin_label in bins:
        M_bin = bins[bin_label]['n']
        rms_bin = bins[bin_label]['rms_mean']
        rates_by_stim_by_bin[bin_label] = {}
        for k, sid in enumerate(stim_ids):
            # Temporal modulation proportional to RMS
            signal = np.zeros((M_bin, T, N))
            signal[:, :, k] = 1.0 + rms_bin * 10 * np.sin(
                np.linspace(0, 2*np.pi, T))[np.newaxis]
            noise = np.random.randn(M_bin, T, N) * 0.5
            rates_by_stim_by_bin[bin_label][sid] = (signal + noise).astype(np.float32)

    gain_results = compute_budget_gain(
        rates_by_stim_by_bin,
        n_splits=3, n_components_C=10, verbose=True,
    )

    rms_means = [bins[l]['rms_mean'] for l in sorted(bins.keys()) if l in gain_results]
    gains = [gain_results[l]['gain'] for l in sorted(bins.keys()) if l in gain_results]

    fig = plot_budget_gain({
        'bin_labels': sorted(gain_results.keys()),
        'gain_by_bin': gain_results,
        'rms_means': rms_means,
        'gains': gains,
        'ns': [bins[l]['n'] for l in sorted(bins.keys()) if l in gain_results],
        'trend': 'monotonic_increase',
    })
    fig.savefig('test_budget_analysis.png', dpi=100, bbox_inches='tight')
    print("Saved test_budget_analysis.png")
    print("All tests passed!")

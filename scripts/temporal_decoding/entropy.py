"""
Task 2.5: Sequential Entropy Reduction (Ideal Observer)

Implements a Bayesian ideal observer that updates a posterior over stimulus
classes as the temporal response unfolds. Measures how quickly the system
accumulates evidence under different FEM conditions.

Under Gaussian class-conditional likelihoods, this is tractable with ~130
neurons and 4 classes.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAME_RATE = 120.0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def fit_class_gaussians(
    rates_by_stim: dict,
    train_mask: np.ndarray,
    diagonal_cov: bool = True,
    regularization: float = 1e-4,
) -> dict:
    """
    Fit class-conditional Gaussian parameters from training data.

    Args:
        rates_by_stim: dict stim_id → (M, T, N) rate arrays (all trials)
        train_mask: (M,) bool mask selecting training traces
        diagonal_cov: use diagonal covariance (faster, fewer parameters)
        regularization: ridge term added to covariance diagonal

    Returns:
        dict with:
            class_means: dict stim_id → (T, N) class mean trajectory
            sigma_inv: (N, N) or (N,) inverse shared covariance
            stim_ids: list of stim_id in order
    """
    stim_ids = sorted(rates_by_stim.keys())
    T = next(iter(rates_by_stim.values())).shape[1]
    N = next(iter(rates_by_stim.values())).shape[2]

    class_means = {}
    for sid in stim_ids:
        X_train = rates_by_stim[sid][train_mask]  # (M_train, T, N)
        class_means[sid] = X_train.mean(axis=0)   # (T, N)

    # Pooled residual covariance (shared across classes, averaged over time)
    all_residuals = []
    for sid in stim_ids:
        X_train = rates_by_stim[sid][train_mask]
        residuals = X_train - class_means[sid][np.newaxis]  # (M_train, T, N)
        all_residuals.append(residuals.reshape(-1, N))

    all_res = np.concatenate(all_residuals, axis=0)  # (M_total*T, N)

    if diagonal_cov:
        # Diagonal: use per-neuron variance
        var = all_res.var(axis=0) + regularization  # (N,)
        sigma_inv = 1.0 / var  # (N,) element-wise inverse
    else:
        # Full covariance: regularized
        cov = all_res.T @ all_res / len(all_res)  # (N, N)
        cov += regularization * np.eye(N)
        sigma_inv = np.linalg.inv(cov)  # (N, N)

    return {
        'class_means': class_means,
        'sigma_inv': sigma_inv,
        'stim_ids': stim_ids,
        'diagonal_cov': diagonal_cov,
        'N': N,
        'T': T,
    }


def sequential_posterior(
    test_rates: np.ndarray,
    class_params: dict,
) -> tuple:
    """
    Compute sequential posterior entropy for a single test trial.

    At each time t, the observer updates:
        log p(S=k | r_{1:t}) += log N(r(t) | mu_k(t), Sigma)

    Args:
        test_rates: (T, N) rate trajectory for the test trial
        class_params: output of fit_class_gaussians()

    Returns:
        entropy_over_time: (T,) posterior entropy H(S | r_{1:t}) in bits
        posteriors_over_time: (T, K) posterior distribution at each time step
    """
    stim_ids = class_params['stim_ids']
    class_means = class_params['class_means']
    sigma_inv = class_params['sigma_inv']
    diagonal_cov = class_params['diagonal_cov']
    K = len(stim_ids)
    T = test_rates.shape[0]

    log_posterior = np.zeros(K)  # log p(S=k), uniform prior log(1/K)
    entropy_over_time = np.zeros(T)
    posteriors_over_time = np.zeros((T, K))

    for t in range(T):
        r_t = test_rates[t]  # (N,)

        for i, sid in enumerate(stim_ids):
            diff = r_t - class_means[sid][t]  # (N,)
            if diagonal_cov:
                # Diagonal Gaussian log-likelihood: -0.5 * sum(diff^2 / var)
                log_posterior[i] += -0.5 * np.sum(diff ** 2 * sigma_inv)
            else:
                # Full Gaussian: -0.5 * diff^T Sigma^{-1} diff
                log_posterior[i] += -0.5 * diff @ sigma_inv @ diff

        posterior = _softmax(log_posterior)
        posteriors_over_time[t] = posterior

        # Shannon entropy in bits
        entropy = -np.sum(posterior * np.log2(posterior + 1e-12))
        entropy_over_time[t] = entropy

    return entropy_over_time, posteriors_over_time


def compute_sequential_entropy(
    rates_by_stim: dict,
    n_splits: int = 5,
    diagonal_cov: bool = True,
    regularization: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Compute mean sequential entropy across held-out test trials.

    Uses grouped cross-validation (by trace index) consistent with the
    decoding ladder.

    Args:
        rates_by_stim: dict stim_id → (M, T, N) rate arrays
        n_splits: CV folds (grouped by trace index)
        diagonal_cov: use diagonal covariance approximation
        regularization: covariance regularization
        verbose: print progress

    Returns:
        dict with:
            entropy_mean: (T,) mean posterior entropy over test trials
            entropy_std: (T,) std of entropy across test trials
            entropy_per_trial: list of (T,) arrays (one per test trial)
            entropy_reduction_rate: bits per 100 ms (from linear fit)
            time_to_criterion: time (ms) to reach H < 0.5 bits
    """
    from sklearn.model_selection import GroupKFold

    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)
    M = rates_by_stim[stim_ids[0]].shape[0]
    T = rates_by_stim[stim_ids[0]].shape[1]

    # Build group arrays for CV
    groups = np.arange(M, dtype=int)

    # Collect test entropy over all folds
    all_entropies = []

    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((M, 1))  # dummy X for split

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        # Create boolean masks
        train_mask = np.zeros(M, dtype=bool)
        train_mask[train_idx] = True

        # Fit class Gaussians on training fold
        params = fit_class_gaussians(
            rates_by_stim, train_mask,
            diagonal_cov=diagonal_cov,
            regularization=regularization,
        )

        # Evaluate on test traces for all stimuli
        for sid in stim_ids:
            for i in test_idx:
                test_rates = rates_by_stim[sid][i]  # (T, N)
                entropy, _ = sequential_posterior(test_rates, params)
                all_entropies.append(entropy)

        if verbose:
            print(f"  Fold {fold+1}/{n_splits}: processed {len(test_idx)*K} test trials")

    all_entropies = np.stack(all_entropies, axis=0)  # (N_test_total, T)

    entropy_mean = all_entropies.mean(axis=0)   # (T,)
    entropy_std = all_entropies.std(axis=0)      # (T,)

    # Compute entropy reduction rate (bits per 100 ms)
    # Fit linear regression on the first 50% of the entropy trace
    t_ms = np.arange(T) / FRAME_RATE * 1000
    half_T = T // 2
    if half_T > 2:
        coeffs = np.polyfit(t_ms[:half_T], entropy_mean[:half_T], 1)
        entropy_reduction_rate = -coeffs[0] * 100  # bits per 100 ms (negative slope = reduction)
    else:
        entropy_reduction_rate = 0.0

    # Time to reach H < 0.5 bits criterion
    below_criterion = np.where(entropy_mean < 0.5)[0]
    time_to_criterion = float(t_ms[below_criterion[0]]) if len(below_criterion) > 0 else None

    return {
        'entropy_mean': entropy_mean,
        'entropy_std': entropy_std,
        'entropy_per_trial': all_entropies,
        'time_ms': t_ms,
        'entropy_reduction_rate_bits_per_100ms': entropy_reduction_rate,
        'time_to_criterion_ms': time_to_criterion,
        'H_max': float(np.log2(K)),  # = 2 bits for 4 classes
    }


def plot_sequential_entropy(
    results_by_condition: dict,
    figsize=(9, 5),
    colors: dict = None,
) -> plt.Figure:
    """
    Plot posterior entropy vs. time for each FEM condition.

    Args:
        results_by_condition: dict condition → output of compute_sequential_entropy()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = {
            'real': 'royalblue',
            'stabilized': 'tomato',
            'matched_null': 'gray',
        }

    H_max = None
    for condition, data in results_by_condition.items():
        color = colors.get(condition, 'black')
        t_ms = data['time_ms']
        H_mean = data['entropy_mean']
        H_std = data['entropy_std']
        H_max = data['H_max']

        reduction_rate = data['entropy_reduction_rate_bits_per_100ms']
        t_crit = data['time_to_criterion_ms']
        label = f"{condition} ({reduction_rate:.3f} bits/100ms)"
        if t_crit is not None:
            label += f", t_crit={t_crit:.0f}ms"

        ax.plot(t_ms, H_mean, color=color, label=label, linewidth=2)
        ax.fill_between(t_ms, H_mean - H_std, H_mean + H_std,
                        alpha=0.2, color=color)

    if H_max is not None:
        ax.axhline(y=H_max, color='black', linestyle='--', alpha=0.4,
                   label=f'prior entropy ({H_max:.2f} bits)')
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.4,
               label='criterion (0.5 bits)')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Posterior entropy H(S | r₁:ₜ) (bits)', fontsize=12)
    ax.set_title('Sequential Evidence Accumulation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig


if __name__ == '__main__':
    print("Testing sequential entropy with synthetic data...")
    np.random.seed(42)

    K, M, T, N = 4, 30, 80, 10
    stim_ids = ['ori0', 'ori90', 'ori180', 'ori270']

    # Create distinguishable class-conditional distributions
    # Each class has a different mean rate for one neuron
    rates_by_stim = {}
    for k, sid in enumerate(stim_ids):
        base_rates = np.ones((M, T, N)) * 0.5
        # Increasing signal over time for neuron k
        signal = np.zeros((T, N))
        signal[:, k] = np.linspace(0, 2.0, T)  # growing signal
        base_rates += signal[np.newaxis]
        noise = np.random.randn(M, T, N) * 0.3
        rates_by_stim[sid] = (base_rates + noise).astype(np.float32)

    results = compute_sequential_entropy(rates_by_stim, n_splits=3, verbose=True)

    print(f"\nEntropy at t=0: {results['entropy_mean'][0]:.3f} bits "
          f"(expected ~{results['H_max']:.3f})")
    print(f"Entropy at t={T}: {results['entropy_mean'][-1]:.3f} bits")
    print(f"Entropy reduction rate: {results['entropy_reduction_rate_bits_per_100ms']:.4f} bits/100ms")
    print(f"Time to H<0.5: {results['time_to_criterion_ms']} ms")

    assert results['entropy_mean'][0] > results['entropy_mean'][-1], \
        "Entropy should decrease over time"
    print("Entropy decreases over time: OK")

    # Compare two conditions
    results_by_cond = {'real': results}

    # Make a weaker condition (less signal)
    rates_weak = {}
    for k, sid in enumerate(stim_ids):
        base_rates = np.ones((M, T, N)) * 0.5
        signal = np.zeros((T, N))
        signal[:, k] = np.linspace(0, 0.2, T)  # weak signal
        base_rates += signal[np.newaxis]
        noise = np.random.randn(M, T, N) * 0.3
        rates_weak[sid] = (base_rates + noise).astype(np.float32)

    results_weak = compute_sequential_entropy(rates_weak, n_splits=3, verbose=False)
    results_by_cond['stabilized'] = results_weak

    fig = plot_sequential_entropy(results_by_cond)
    fig.savefig('test_entropy.png', dpi=100, bbox_inches='tight')
    print("Saved test_entropy.png")
    print("All tests passed!")

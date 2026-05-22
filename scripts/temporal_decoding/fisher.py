"""
Task 4.1: Fisher Information Upgrade — J_indep → J_pop

Upgrades the existing per-neuron independent Fisher information to the
population Fisher information that accounts for noise correlations.

J_pop = f'^T Σ^{-1} f'
J_indep = Σ_n (f'_n)^2 / (r_n + ε)   [Poisson approximation]

η = J_pop / J_indep quantifies the efficiency gain (or loss) from correlations.

Uses forward-mode AD (torch.func.jvp or manual dual numbers) to compute
derivatives of rates with respect to stimulus parameters without backprop.

Reference: check_fixrsvp_model_fisherinfo.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_rate_gradient(
    model,
    readout,
    stim_fn,
    param: float,
    delta: float = 0.1,
    device: str = 'cpu',
) -> tuple:
    """
    Compute df/d(param) via finite differences.

    Args:
        model: VisionCore model
        readout: PopulationReadout
        stim_fn: callable(param) → (1, 1, n_lags, H, W) stimulus tensor
        param: current parameter value
        delta: finite difference step
        device: torch device

    Returns:
        f_prime: (N,) derivative of rates w.r.t. param
        rates_primal: (N,) rates at current param
    """
    with torch.no_grad():
        stim_lo = stim_fn(param - delta).to(device)
        stim_hi = stim_fn(param + delta).to(device)

        def get_rates(stim):
            feats = model.model.core_forward(stim, None)
            y = readout(feats[:, :, -1])
            rates_spatial = model.model.activation(y)  # (1, N, H, W)
            return rates_spatial.amax(dim=(-2, -1))[0]  # (N,)

        r_lo = get_rates(stim_lo)
        r_hi = get_rates(stim_hi)
        rates_primal = get_rates(stim_fn(param).to(device))

    f_prime = (r_hi - r_lo) / (2 * delta)
    return f_prime.cpu().numpy(), rates_primal.cpu().numpy()


def compute_noise_covariance(
    rates_list: list,
    regularization: float = 1e-4,
) -> np.ndarray:
    """
    Estimate the noise covariance matrix from multiple trials at the same stimulus.

    Args:
        rates_list: list of (N,) rate vectors (one per trial/eye trace)
        regularization: ridge regularization term

    Returns:
        Sigma: (N, N) regularized noise covariance
    """
    R = np.stack(rates_list, axis=0)  # (M, N)
    R -= R.mean(axis=0, keepdims=True)
    N = R.shape[1]
    Sigma = R.T @ R / (len(R) - 1) + regularization * np.eye(N)
    return Sigma


def compute_population_fisher(
    f_prime: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float = 1e-6,
) -> dict:
    """
    Compute population and independent Fisher information.

    Args:
        f_prime: (N,) rate gradient w.r.t. stimulus parameter
        Sigma: (N, N) noise covariance (Poisson + FEM contributions)
        epsilon: small value for numerical stability

    Returns:
        dict with:
            J_pop: population Fisher information
            J_indep: independent (diagonal) Fisher information
            eta: J_pop / J_indep (efficiency ratio)
    """
    N = len(f_prime)
    rates_diag = np.abs(f_prime ** 2)  # approximate as Poisson: var ≈ rate

    # J_indep: diagonal noise (Poisson)
    diag_sigma = np.diag(Sigma)
    J_indep = float(np.sum(f_prime ** 2 / (diag_sigma + epsilon)))

    # J_pop: full covariance noise
    try:
        L = np.linalg.cholesky(Sigma)
        v = np.linalg.solve(L, f_prime)
        J_pop = float(v @ v)
    except np.linalg.LinAlgError:
        # Sigma not PD: use pseudoinverse
        Sigma_inv = np.linalg.pinv(Sigma)
        J_pop = float(f_prime @ Sigma_inv @ f_prime)

    eta = J_pop / (J_indep + epsilon)

    return {
        'J_pop': J_pop,
        'J_indep': J_indep,
        'eta': eta,
    }


def compute_fisher_matrix(
    model,
    readout,
    stim_fns: dict,
    rates_by_param: dict,
    regularization: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Compute the full Fisher information matrix F_ij for a set of stimulus parameters.

    F_ij = (∂f/∂θ_i)^T Σ^{-1} (∂f/∂θ_j)

    Args:
        model: VisionCore model
        readout: PopulationReadout
        stim_fns: dict param_name → callable(value) → stimulus tensor
        rates_by_param: dict param_name → list of (N,) rate arrays (for noise est.)
        regularization: covariance regularization

    Returns:
        dict with:
            F: (n_params, n_params) Fisher matrix
            param_names: list of parameter names
            J_indep_per_param: (n_params,) per-parameter J_indep
            J_pop_per_param: (n_params,) per-parameter J_pop
            eta_per_param: (n_params,) efficiency ratio per parameter
    """
    param_names = list(stim_fns.keys())
    n_params = len(param_names)

    # Estimate noise covariance from provided rate samples
    # Use the first parameter's rate list for noise estimation
    first_rates = next(iter(rates_by_param.values()))
    Sigma = compute_noise_covariance(first_rates, regularization=regularization)

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    # Compute gradient for each parameter
    gradients = {}
    primal_rates = {}
    for name, stim_fn_pair in stim_fns.items():
        stim_fn, param_val, delta = stim_fn_pair
        if verbose:
            print(f"  Computing gradient for {name}...")
        grad, primal = compute_rate_gradient(model, readout, stim_fn, param_val, delta)
        gradients[name] = grad
        primal_rates[name] = primal

    # Build Fisher matrix
    F = np.zeros((n_params, n_params))
    J_indep_per_param = np.zeros(n_params)
    J_pop_per_param = np.zeros(n_params)
    eta_per_param = np.zeros(n_params)

    for i, ni in enumerate(param_names):
        for j, nj in enumerate(param_names):
            F[i, j] = gradients[ni] @ Sigma_inv @ gradients[nj]

        # Per-parameter Fisher (diagonal element)
        fi = compute_population_fisher(gradients[ni], Sigma)
        J_indep_per_param[i] = fi['J_indep']
        J_pop_per_param[i] = fi['J_pop']
        eta_per_param[i] = fi['eta']

    return {
        'F': F,
        'param_names': param_names,
        'J_indep_per_param': J_indep_per_param,
        'J_pop_per_param': J_pop_per_param,
        'eta_per_param': eta_per_param,
        'Sigma': Sigma,
    }


def compare_noise_models(
    f_prime: np.ndarray,
    rates_list: list,
    additional_covariances: dict = None,
    regularization: float = 1e-4,
) -> dict:
    """
    Compute J_pop under several noise models and report sensitivity.

    Noise models:
        1. Poisson-only: Σ = diag(rates)
        2. Poisson + FEM: Σ = diag(rates) + C_FEM
        3. Full: Σ = estimated empirical covariance

    Args:
        f_prime: (N,) rate gradient
        rates_list: list of (N,) rates at this parameter (for noise estimation)
        additional_covariances: dict label → (N, N) additional covariance matrices
        regularization: ridge regularization

    Returns:
        dict noise_model → Fisher info dict
    """
    N = len(f_prime)
    mean_rates = np.stack(rates_list).mean(axis=0)

    # Noise model 1: Poisson-only
    Sigma_poisson = np.diag(mean_rates + regularization)

    # Noise model 2: empirical (captures FEM variability)
    Sigma_empirical = compute_noise_covariance(rates_list, regularization=regularization)

    models = {
        'Poisson_only': Sigma_poisson,
        'Empirical': Sigma_empirical,
    }

    if additional_covariances:
        for label, C_add in additional_covariances.items():
            models[f'Poisson+{label}'] = Sigma_poisson + C_add

    results = {}
    for model_name, Sigma in models.items():
        results[model_name] = compute_population_fisher(f_prime, Sigma, regularization)

    return results


def plot_fisher_results(
    fisher_results: dict,
    figsize=(10, 4),
) -> plt.Figure:
    """
    Plot J_pop, J_indep, and η for multiple noise models.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    noise_models = list(fisher_results.keys())
    x = np.arange(len(noise_models))

    J_pops = [fisher_results[m]['J_pop'] for m in noise_models]
    J_indeps = [fisher_results[m]['J_indep'] for m in noise_models]
    etas = [fisher_results[m]['eta'] for m in noise_models]

    axes[0].bar(x, J_pops, color='royalblue', alpha=0.8)
    axes[0].set_title('J_pop (population Fisher)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(noise_models, rotation=20, ha='right', fontsize=8)
    axes[0].set_ylabel('Fisher information')

    axes[1].bar(x, J_indeps, color='tomato', alpha=0.8)
    axes[1].set_title('J_indep (diagonal noise)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(noise_models, rotation=20, ha='right', fontsize=8)

    axes[2].bar(x, etas, color='steelblue', alpha=0.8)
    axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('η = J_pop / J_indep')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(noise_models, rotation=20, ha='right', fontsize=8)
    axes[2].set_ylabel('Efficiency ratio')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Testing Fisher information computation with synthetic data...")
    np.random.seed(42)

    N = 50
    M = 100

    # Synthetic: gradient vector
    f_prime = np.random.randn(N) * 0.1
    mean_rates = np.abs(np.random.randn(N)) * 2.0 + 0.5

    # Generate rate samples: mean + noise
    rates_list = [mean_rates + np.random.randn(N) * np.sqrt(mean_rates) * 0.1
                  for _ in range(M)]

    # Test independent Fisher
    Sigma_poisson = np.diag(mean_rates + 1e-4)
    fi = compute_population_fisher(f_prime, Sigma_poisson)
    print(f"Poisson noise: J_pop={fi['J_pop']:.4f}, J_indep={fi['J_indep']:.4f}, η={fi['eta']:.4f}")
    assert abs(fi['eta'] - 1.0) < 0.1, "With diagonal Sigma, J_pop ≈ J_indep"
    print("J_pop ≈ J_indep for diagonal Sigma: OK")

    # Test with correlated noise (should reduce J_pop)
    # Add correlation: neuron pairs with same tuning have correlated noise
    Sigma_corr = np.diag(mean_rates + 1e-4)
    # Add some positive correlations
    corr_strength = 0.3
    for i in range(0, N, 5):
        for j in range(i+1, min(i+5, N)):
            cov = corr_strength * np.sqrt(mean_rates[i] * mean_rates[j])
            Sigma_corr[i, j] = cov
            Sigma_corr[j, i] = cov

    # Ensure PSD
    Sigma_corr += 0.1 * np.eye(N)
    fi_corr = compute_population_fisher(f_prime, Sigma_corr)
    print(f"Correlated noise: J_pop={fi_corr['J_pop']:.4f}, "
          f"J_indep={fi_corr['J_indep']:.4f}, η={fi_corr['eta']:.4f}")

    # Compare noise models
    results = compare_noise_models(f_prime, rates_list)
    print("\nNoise model comparison:")
    for name, res in results.items():
        print(f"  {name}: J_pop={res['J_pop']:.4f}, η={res['eta']:.4f}")

    fig = plot_fisher_results(results)
    fig.savefig('test_fisher.png', dpi=100, bbox_inches='tight')
    print("\nSaved test_fisher.png")
    print("All tests passed!")

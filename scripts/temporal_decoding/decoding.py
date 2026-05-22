"""
Tasks 2.1–2.2: Decoding Infrastructure — Models A, B, C, D

Implements the ablation ladder with grouped cross-validation:
    Model A: time-averaged rate (spatial code baseline)
    Model B: [Model A features] + PCA of class-mean temporal residual
    Model C: [Model A features] + PCA of single-trial temporal residual
    Model D: residual covariance features (second-order temporal structure)

Models B and C are additive over A by construction — PCA is fitted on the
mean-subtracted temporal residual (X - X.mean(axis=time)), so the mean-rate
information is preserved in the concatenated Model A features and PCA can only
capture structure that is orthogonal to the mean rate. This guarantees B ≥ A
and C ≥ A at every cross-validation fold.

Cross-validation is grouped by eye trace ID to ensure generalization
across eye movements (not just across time bins within a trace).
"""
import os
import sys
import numpy as np
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Data preparation helpers ────────────────────────────────────────────────

def prepare_rate_tensors(
    rates_by_stim: dict,
    min_length: Optional[int] = None,
    target_length: Optional[int] = None,
    verbose: bool = True,
) -> tuple:
    """
    Convert per-stimulus rate lists to arrays, truncate to common length.

    Args:
        rates_by_stim: dict mapping stim_id → list of (T_m, N) rate arrays
        min_length: truncate each trial to this many frames (None = use minimum)
        target_length: if set, drop trials shorter than this and use exactly
                       target_length frames (overrides min_length)
        verbose: print trial-length distribution stats

    Returns:
        X_by_stim: dict stim_id → (M, T_use, N) float32 array
        T_use: common temporal length used
        N: number of neurons
    """
    all_lengths = [r.shape[0] for stim_rates in rates_by_stim.values()
                   for r in stim_rates]

    if verbose:
        print(f"  Trial lengths: min={min(all_lengths)}, "
              f"median={int(np.median(all_lengths))}, "
              f"p10={int(np.percentile(all_lengths, 10))}, "
              f"p90={int(np.percentile(all_lengths, 90))}, "
              f"n_trials={len(all_lengths)}")

    if target_length is not None:
        # Drop trials shorter than target_length
        filtered = {sid: [r for r in rlist if r.shape[0] >= target_length]
                    for sid, rlist in rates_by_stim.items()}
        n_dropped = len(all_lengths) - sum(len(v) for v in filtered.values())
        if n_dropped > 0 and verbose:
            print(f"  target_length={target_length}: dropped {n_dropped}/{len(all_lengths)} short trials")
        rates_by_stim = filtered
        T_use = target_length
    else:
        T_use = min(all_lengths) if min_length is None else min(min_length, min(all_lengths))

    X_by_stim = {}
    for stim_id, rate_list in rates_by_stim.items():
        X_by_stim[stim_id] = np.stack([r[:T_use] for r in rate_list], axis=0)

    N = next(iter(X_by_stim.values())).shape[2]
    return X_by_stim, T_use, N


def build_classifier_dataset(X_by_stim: dict) -> tuple:
    """
    Flatten per-stimulus arrays into (X, y, groups) for sklearn.

    Args:
        X_by_stim: dict stim_id → (M, ...) feature arrays (already projected)

    Returns:
        X: (M_total, n_features) array
        y: (M_total,) int labels
        groups: (M_total,) int — trace index within stim (used for grouped CV)
    """
    X_list, y_list, g_list = [], [], []
    stim_ids = sorted(X_by_stim.keys())

    for label, stim_id in enumerate(stim_ids):
        arr = X_by_stim[stim_id]  # (M, ...)
        M = arr.shape[0]
        flat = arr.reshape(M, -1)
        X_list.append(flat)
        y_list.append(np.full(M, label, dtype=int))
        g_list.append(np.arange(M, dtype=int))  # trace index = group

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X, y, groups


def decode_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    use_mlp: bool = False,
) -> tuple:
    """
    Grouped cross-validated linear decoding.

    Groups = eye trace IDs (split by trace, not by time or trial).

    Returns:
        mean_accuracy, std_accuracy, per_fold_accuracies
    """
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        if use_mlp:
            clf = MLPClassifier(
                hidden_layer_sizes=(64,), activation='relu',
                max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1,
            )
        else:
            clf = LogisticRegression(
                C=C, max_iter=2000, solver='lbfgs',
                random_state=42,
            )
        clf.fit(X_tr, y[train_idx])
        accuracies.append(clf.score(X_te, y[test_idx]))

    accuracies = np.array(accuracies)
    return float(accuracies.mean()), float(accuracies.std()), accuracies


# ─── Model A: time-averaged rate ─────────────────────────────────────────────

def extract_model_A(X_by_stim: dict) -> dict:
    """
    Model A: time-averaged rate per trial.
    X_A = mean over time → (M, N)
    """
    return {stim_id: X.mean(axis=1) for stim_id, X in X_by_stim.items()}


# ─── Model B: class-mean temporal residual subspace + Model A ─────────────────

def fit_model_B_subspace(
    X_by_stim: dict,
    n_components: Optional[int] = None,
    equalize_trace_distributions: bool = True,
) -> PCA:
    """
    Fit the Model B subspace from training data (temporal residual).

    Computes class-mean temporal residuals (after subtracting per-trial, per-neuron
    mean rate), then fits PCA on those residual means. The subspace U_B captures
    systematic temporal dynamics orthogonal to the mean rate.

    When used via extract_model_B, the final features are [Model_A || pca_B_projection],
    ensuring B ≥ A by construction (the logistic regression can always set temporal
    projection weights to zero and recover Model A performance).

    Args:
        X_by_stim: dict stim_id → (M, T, N) training rates (TRAINING FOLD ONLY)
        n_components: PCA components (default = K-1 for K classes)
        equalize_trace_distributions: if True, subsample to equal M per class

    Returns:
        Fitted PCA object (acts on flattened temporal residuals of shape T*N)
    """
    K = len(X_by_stim)
    if n_components is None:
        n_components = min(K - 1, 20)

    if equalize_trace_distributions:
        min_M = min(X.shape[0] for X in X_by_stim.values())
        X_by_stim = {sid: X[:min_M] for sid, X in X_by_stim.items()}

    class_mean_residuals = []
    for stim_id in sorted(X_by_stim.keys()):
        X = X_by_stim[stim_id]  # (M, T, N)
        # Subtract per-trial, per-neuron mean rate to get temporal residuals
        X_residual = X - X.mean(axis=1, keepdims=True)  # (M, T, N)
        # Class-mean residual trajectory
        class_mean_residual = X_residual.mean(axis=0)   # (T, N)
        class_mean_residuals.append(class_mean_residual.flatten())  # (T*N,)

    means_matrix = np.stack(class_mean_residuals, axis=0)  # (K, T*N)

    pca = PCA(n_components=n_components)
    pca.fit(means_matrix)
    return pca


def extract_model_B(X_by_stim: dict, pca_B: PCA) -> dict:
    """
    Extract Model B features: [Model_A features || temporal-residual PCA projection].

    Concatenates the time-averaged rate (Model A) with the projection of the
    mean-subtracted temporal residual onto the B subspace. This ensures B ≥ A.

    Returns (M, N + d_B) features per stimulus class.
    """
    result = {}
    for stim_id, X in X_by_stim.items():
        M, T, N = X.shape
        X_A = X.mean(axis=1)                           # (M, N) — time-averaged rate
        X_residual = X - X.mean(axis=1, keepdims=True)  # (M, T, N) — temporal residual
        X_B_proj = pca_B.transform(X_residual.reshape(M, T * N))  # (M, d_B)
        result[stim_id] = np.concatenate([X_A, X_B_proj], axis=1)  # (M, N + d_B)
    return result


# ─── Model C: full single-trial temporal residual subspace + Model A ──────────

def fit_model_C_subspace(
    X_by_stim: dict,
    n_components: int = 50,
) -> PCA:
    """
    Fit the Model C subspace from all training temporal residuals.

    Fits PCA on the mean-subtracted temporal residuals of all training trials.
    This captures single-trial temporal modulations beyond the mean rate.

    When used via extract_model_C, the final features are [Model_A || pca_C_projection],
    ensuring C ≥ A by construction.

    Args:
        X_by_stim: dict stim_id → (M, T, N) training rates (TRAINING FOLD ONLY)
        n_components: number of PCA components to retain

    Returns:
        Fitted PCA object (acts on flattened temporal residuals of shape T*N)
    """
    all_residuals = []
    for X in X_by_stim.values():
        M, T, N = X.shape
        X_residual = X - X.mean(axis=1, keepdims=True)  # subtract per-trial mean rate
        all_residuals.append(X_residual.reshape(M, T * N))

    all_flat = np.concatenate(all_residuals, axis=0)  # (M_total, T*N)

    # Cap components to avoid overfitting: need n_components << M_total.
    # Rule: at most M_total // 4 components so the logistic regression has
    # at least 4× more samples than features.
    M_total = all_flat.shape[0]
    n_components_safe = min(n_components, min(all_flat.shape), M_total // 4)

    pca = PCA(n_components=max(1, n_components_safe))
    pca.fit(all_flat)
    return pca


def extract_model_C(X_by_stim: dict, pca_C: PCA) -> dict:
    """
    Extract Model C features: [Model_A features || temporal-residual PCA projection].

    Concatenates the time-averaged rate (Model A) with the projection of the
    mean-subtracted temporal residual onto the C subspace. This ensures C ≥ A.

    Returns (M, N + d_C) features per stimulus class.
    """
    result = {}
    for stim_id, X in X_by_stim.items():
        M, T, N = X.shape
        X_A = X.mean(axis=1)                           # (M, N) — time-averaged rate
        X_residual = X - X.mean(axis=1, keepdims=True)  # (M, T, N) — temporal residual
        X_C_proj = pca_C.transform(X_residual.reshape(M, T * N))  # (M, d_C)
        result[stim_id] = np.concatenate([X_A, X_C_proj], axis=1)  # (M, N + d_C)
    return result


# ─── Model D: residual covariance features ────────────────────────────────────

def compute_residual_covariance_features(
    trajectory: np.ndarray,
    pca_B: PCA,
    n_lags: int = 5,
) -> np.ndarray:
    """
    Compute lagged cross-covariance of the residual after projecting out B subspace.

    Args:
        trajectory: (T, N) single-trial rate trajectory
        pca_B: fitted PCA (Model B subspace, trained on temporal residuals)
        n_lags: number of temporal lags for cross-covariance

    Returns:
        (n_features,) feature vector (upper triangle of lagged covariance matrices)
    """
    T, N = trajectory.shape
    # Subtract per-neuron mean rate first (consistent with new B/C convention)
    traj_residual = trajectory - trajectory.mean(axis=0, keepdims=True)  # (T, N)
    traj_flat = traj_residual.flatten()

    b_proj = pca_B.inverse_transform(pca_B.transform(traj_flat.reshape(1, -1)))
    residual = (traj_flat - b_proj.flatten()).reshape(T, N)

    cov_features = []
    for lag in range(n_lags + 1):
        if lag == 0:
            C = residual.T @ residual / T
        else:
            C = residual[lag:].T @ residual[:-lag] / (T - lag)
        cov_features.append(C[np.triu_indices(N)])

    return np.concatenate(cov_features)


def compute_autocov_features(
    trajectory: np.ndarray,
    pca_B: PCA,
    n_lags: int = 10,
) -> np.ndarray:
    """
    D2: within-neuron autocovariance only (no cross-neuron terms).
    Tests if the D gain requires population-level interactions.
    """
    T, N = trajectory.shape
    # Subtract per-neuron mean rate first (consistent with new B/C convention)
    traj_residual = trajectory - trajectory.mean(axis=0, keepdims=True)
    traj_flat = traj_residual.flatten()
    b_proj = pca_B.inverse_transform(pca_B.transform(traj_flat.reshape(1, -1)))
    residual = (traj_flat - b_proj.flatten()).reshape(T, N)

    features = []
    for lag in range(1, n_lags + 1):
        autocov = np.mean(residual[lag:] * residual[:-lag], axis=0)  # (N,)
        features.append(autocov)
    return np.concatenate(features)


def compute_shuffled_D_features(
    trajectory: np.ndarray,
    pca_B: PCA,
    n_lags: int = 5,
    rng=None,
) -> np.ndarray:
    """
    Negative control: shuffle time independently per neuron, then compute D features.
    """
    rng = rng or np.random.default_rng()
    T, N = trajectory.shape
    shuffled = trajectory.copy()
    for n in range(N):
        rng.shuffle(shuffled[:, n])
    return compute_residual_covariance_features(shuffled, pca_B, n_lags)


def extract_model_D(
    X_by_stim: dict,
    pca_B: PCA,
    n_lags_cov: int = 5,
    n_pca_D: int = 30,
    pca_D: Optional[PCA] = None,
    shuffled: bool = False,
    rng=None,
) -> tuple:
    """
    Extract Model D features for all trials.

    Returns B features concatenated with PCA-reduced residual covariance features.

    Args:
        X_by_stim: dict stim_id → (M, T, N)
        pca_B: fitted B subspace PCA
        n_lags_cov: temporal lags for cross-covariance
        n_pca_D: PCA components to keep for covariance features (fitted if pca_D is None)
        pca_D: fitted PCA for covariance dim reduction (fitted on training data if None)
        shuffled: if True, use time-shuffled negative control

    Returns:
        features_by_stim: dict stim_id → (M, d_B + d_D) features
        pca_D: fitted PCA for covariance features (for reuse in test set)
    """
    rng = rng or np.random.default_rng()

    # Step 1: Extract raw covariance features for all trials
    raw_cov_by_stim = {}
    for stim_id, X in X_by_stim.items():
        M, T, N = X.shape
        raw_feats = []
        for i in range(M):
            if shuffled:
                feat = compute_shuffled_D_features(X[i], pca_B, n_lags_cov, rng)
            else:
                feat = compute_residual_covariance_features(X[i], pca_B, n_lags_cov)
            raw_feats.append(feat)
        raw_cov_by_stim[stim_id] = np.stack(raw_feats, axis=0)  # (M, n_cov_feat)

    # Step 2: Fit PCA on covariance features if not provided (training fold only)
    if pca_D is None:
        all_cov = np.concatenate(list(raw_cov_by_stim.values()), axis=0)
        n_components = min(n_pca_D, min(all_cov.shape))
        pca_D = PCA(n_components=n_components)
        pca_D.fit(all_cov)

    # Step 3: Combine B features + reduced covariance features
    B_features = extract_model_B(X_by_stim, pca_B)
    features_by_stim = {}
    for stim_id in X_by_stim:
        cov_reduced = pca_D.transform(raw_cov_by_stim[stim_id])  # (M, d_D)
        features_by_stim[stim_id] = np.concatenate(
            [B_features[stim_id], cov_reduced], axis=1
        )

    return features_by_stim, pca_D


# ─── Main ablation ladder ─────────────────────────────────────────────────────

def run_decoding_ladder(
    rates_by_stim: dict,
    models: list = ('A', 'B', 'C', 'D'),
    n_splits: int = 5,
    C_logistic: float = 1.0,
    n_components_C: int = 50,
    n_lags_cov: int = 5,
    run_mlp_control: bool = False,
    target_length: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full ablation ladder with grouped cross-validation.

    Args:
        rates_by_stim: dict mapping stim_id → list of (T_m, N) rate arrays
        models: which models to run ('A', 'B', 'C', 'D')
        n_splits: number of CV folds (grouped by trace index)
        C_logistic: L2 regularization strength for logistic regression
        n_components_C: PCA components for Model C
        n_lags_cov: temporal lags for Model D covariance
        run_mlp_control: also run MLP on Model C features as a control
        target_length: drop trials shorter than this and truncate to exactly
                       this many frames (None = truncate to shortest trial)
        verbose: print progress

    Returns:
        dict with keys 'A', 'B', 'C', 'D', each a dict with:
            'mean_acc': float
            'std_acc': float
            'fold_acc': (n_splits,) array
    """
    X_by_stim, T_use, N = prepare_rate_tensors(rates_by_stim, target_length=target_length,
                                                verbose=verbose)
    stim_ids = sorted(X_by_stim.keys())
    K = len(stim_ids)

    results = {}

    # We need to do the PCA fitting inside each CV fold
    gkf = GroupKFold(n_splits=n_splits)

    # Build the group labels (trace index per trial, consistent across stimuli)
    M_per_stim = {sid: X_by_stim[sid].shape[0] for sid in stim_ids}
    M_min = min(M_per_stim.values())
    # Truncate to equal M across stimuli (ensures valid grouped CV)
    X_by_stim = {sid: X_by_stim[sid][:M_min] for sid in stim_ids}

    # groups = trace index (same index = same eye trace used for all stimulus classes)
    groups = np.arange(M_min, dtype=int)

    for model_name in models:
        if verbose:
            print(f"\nRunning Model {model_name}...")

        fold_accuracies = []

        # Enumerate folds using Model A labels for consistent splitting
        X_A_all = np.concatenate([X_by_stim[sid].mean(axis=1) for sid in stim_ids], axis=0)
        y_all = np.concatenate([np.full(M_min, i) for i in range(K)], axis=0)
        groups_all = np.tile(groups, K)

        for fold, (train_idx_all, test_idx_all) in enumerate(
            gkf.split(X_A_all, y_all, groups=groups_all)
        ):
            # Convert global indices back to per-stim train/test trace indices
            train_traces = train_idx_all[train_idx_all < M_min]
            test_traces = test_idx_all[test_idx_all < M_min]

            X_train = {sid: X_by_stim[sid][train_traces] for sid in stim_ids}
            X_test = {sid: X_by_stim[sid][test_traces] for sid in stim_ids}

            # Extract features for this model (PCA fitted on training fold only)
            if model_name == 'A':
                X_tr = extract_model_A(X_train)
                X_te = extract_model_A(X_test)

            elif model_name == 'B':
                pca_B = fit_model_B_subspace(X_train)
                X_tr = extract_model_B(X_train, pca_B)
                X_te = extract_model_B(X_test, pca_B)

            elif model_name == 'C':
                pca_C = fit_model_C_subspace(X_train, n_components=n_components_C)
                X_tr = extract_model_C(X_train, pca_C)
                X_te = extract_model_C(X_test, pca_C)

            elif model_name == 'D':
                pca_B = fit_model_B_subspace(X_train)
                X_tr_D, pca_D = extract_model_D(X_train, pca_B, n_lags_cov=n_lags_cov)
                X_te_D, _ = extract_model_D(X_test, pca_B, n_lags_cov=n_lags_cov, pca_D=pca_D)
                X_tr, X_te = X_tr_D, X_te_D

            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Build classifier dataset
            X_tr_flat, y_tr, _ = build_classifier_dataset(X_tr)
            X_te_flat, y_te, _ = build_classifier_dataset(X_te)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_flat)
            X_te_s = scaler.transform(X_te_flat)

            clf = LogisticRegression(
                C=C_logistic, max_iter=2000,
                solver='lbfgs', random_state=42,
            )
            clf.fit(X_tr_s, y_tr)
            acc = clf.score(X_te_s, y_te)
            fold_accuracies.append(acc)

            if verbose:
                print(f"  Fold {fold+1}/{n_splits}: acc={acc:.3f}")

        fold_accuracies = np.array(fold_accuracies)
        results[model_name] = {
            'mean_acc': float(fold_accuracies.mean()),
            'std_acc': float(fold_accuracies.std()),
            'fold_acc': fold_accuracies,
        }

        if verbose:
            print(f"  Model {model_name}: {fold_accuracies.mean():.3f} ± {fold_accuracies.std():.3f}")

    # MLP control on Model C features
    if run_mlp_control and 'C' in models:
        if verbose:
            print("\nRunning MLP control on Model C features...")
        pca_C = fit_model_C_subspace(X_by_stim, n_components=n_components_C)
        X_C_all = extract_model_C(X_by_stim, pca_C)
        X_flat, y, groups_flat = build_classifier_dataset(X_C_all)
        mean_acc, std_acc, fold_acc = decode_with_cv(
            X_flat, y, groups_flat, n_splits=n_splits, use_mlp=True
        )
        results['C_mlp'] = {
            'mean_acc': mean_acc, 'std_acc': std_acc, 'fold_acc': fold_acc
        }
        if verbose:
            print(f"  Model C (MLP): {mean_acc:.3f} ± {std_acc:.3f}")

    return results


def print_ladder_results(results: dict, chance: float = 0.25) -> None:
    """Pretty-print ablation ladder results."""
    print("\n=== Decoding Ladder Results ===")
    print(f"Chance level: {chance:.2%}")
    print(f"{'Model':<10} {'Accuracy':>10} {'±':>4} {'Std':>8}")
    print("-" * 35)
    for model_name in ['A', 'B', 'C', 'D', 'C_mlp']:
        if model_name not in results:
            continue
        r = results[model_name]
        gain = r['mean_acc'] - chance
        print(f"  {model_name:<8} {r['mean_acc']:.3f} ({gain:+.3f})   ± {r['std_acc']:.3f}")
    print()


if __name__ == '__main__':
    print("Testing decoding infrastructure with synthetic data...")
    np.random.seed(42)

    # Synthetic data: 4 stimuli, 40 traces each, 60 time steps, 10 neurons
    K, M, T, N = 4, 40, 60, 10

    # Create discriminable rate patterns per stimulus
    rates_by_stim = {}
    stim_ids = ['ori0', 'ori90', 'ori180', 'ori270']
    for k, sid in enumerate(stim_ids):
        # Each stimulus has a different mean rate pattern
        signal = np.zeros((1, T, N))
        signal[0, :, k % N] = 1.0 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, T))
        noise = np.random.randn(M, T, N) * 0.3
        rates_by_stim[sid] = [signal[0] + noise[i] for i in range(M)]

    results = run_decoding_ladder(
        rates_by_stim,
        models=['A', 'B', 'C'],
        n_splits=4,
        verbose=True,
    )
    print_ladder_results(results)

    # Verify expected ordering
    assert results['A']['mean_acc'] > 0.3, "Model A should beat chance"
    print("Sanity check passed: Model A > chance")
    print("All tests passed!")

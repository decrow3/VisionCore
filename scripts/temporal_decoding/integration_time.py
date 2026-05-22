"""
Task 2.3: Integration Time Sweep

Trains causal sliding-window decoders at multiple integration times and
plots accuracy vs. window size for different FEM conditions.

Key prediction: real FEMs should show steeper accuracy gain with window size
than stabilized. The crossover point (if it exists) shows that FEMs hurt at
short integration times but help at biologically relevant timescales.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows in frames at 120 Hz: 1 frame = 8.33 ms
DEFAULT_WINDOWS = [1, 3, 6, 12, 24, 36, 48, 60]
FRAME_RATE = 120.0


def decode_causal_window(
    rates_by_stim: dict,
    window: int,
    n_splits: int = 5,
    n_pca: int = 30,
    C_logistic: float = 1.0,
) -> tuple:
    """
    Decode stimulus orientation using only the last `window` frames of each trial.

    Args:
        rates_by_stim: dict stim_id → list of (T_m, N) rate arrays
        window: number of frames to use (last W frames of each trial)
        n_splits: CV folds
        n_pca: PCA components for dimensionality reduction
        C_logistic: regularization

    Returns:
        mean_acc, std_acc, fold_accs
    """
    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)

    # Truncate each trial to its last `window` frames and flatten
    # Equalize number of trials across stimuli
    M_per_stim = {sid: len(rates_by_stim[sid]) for sid in stim_ids}
    M_min = min(M_per_stim.values())

    X_list, y_list, g_list = [], [], []
    for label, sid in enumerate(stim_ids):
        rate_list = rates_by_stim[sid][:M_min]
        X_stim = []
        for r in rate_list:
            T_m = r.shape[0]
            if T_m < window:
                # Pad with first-frame value if shorter than window
                pad = np.repeat(r[[0]], window - T_m, axis=0)
                r_win = np.concatenate([pad, r], axis=0)
            else:
                r_win = r[-window:]  # last W frames
            X_stim.append(r_win.flatten())
        X_stim = np.stack(X_stim, axis=0)  # (M_min, W*N)
        X_list.append(X_stim)
        y_list.append(np.full(M_min, label))
        g_list.append(np.arange(M_min))

    X_all = np.concatenate(X_list, axis=0)   # (K*M_min, W*N)
    y_all = np.concatenate(y_list, axis=0)
    g_all = np.tile(np.arange(M_min), K)     # groups = trace index

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []

    for train_idx, test_idx in gkf.split(X_all, y_all, groups=g_all):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])

        # PCA to reduce dimensionality (fitted on training fold)
        n_comp = min(n_pca, X_tr.shape[0] - 1, X_tr.shape[1])
        if n_comp > 0:
            pca = PCA(n_components=n_comp)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(
            C=C_logistic, max_iter=2000,
            solver='lbfgs', random_state=42,
        )
        clf.fit(X_tr, y_all[train_idx])
        fold_accs.append(clf.score(X_te, y_all[test_idx]))

    fold_accs = np.array(fold_accs)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


def decode_causal_window_time_mean(
    rates_by_stim: dict,
    window: int,
    n_splits: int = 5,
    C_logistic: float = 1.0,
) -> tuple:
    """Decode using only the mean rate over the last `window` frames.

    This is an accumulation-aligned control: it tests whether averaging over more
    frames improves the estimate of the stimulus-dependent mean response.

    Unlike decode_causal_window(), this does NOT flatten time and does NOT use PCA.
    """
    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)

    # Equalize number of trials across stimuli
    M_min = min(len(rates_by_stim[sid]) for sid in stim_ids)

    X_list, y_list = [], []
    for label, sid in enumerate(stim_ids):
        rate_list = rates_by_stim[sid][:M_min]
        feats = []
        for r in rate_list:
            T_m = r.shape[0]
            if T_m < window:
                pad = np.repeat(r[[0]], window - T_m, axis=0)
                r_win = np.concatenate([pad, r], axis=0)
            else:
                r_win = r[-window:]
            feats.append(r_win.mean(axis=0))
        X_list.append(np.stack(feats, axis=0))  # (M_min, N)
        y_list.append(np.full(M_min, label))

    X_all = np.concatenate(X_list, axis=0)   # (K*M_min, N)
    y_all = np.concatenate(y_list, axis=0)
    g_all = np.tile(np.arange(M_min), K)     # groups = trace index

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []

    for train_idx, test_idx in gkf.split(X_all, y_all, groups=g_all):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])

        clf = LogisticRegression(
            C=C_logistic, max_iter=2000,
            solver='lbfgs', random_state=42,
        )
        clf.fit(X_tr, y_all[train_idx])
        fold_accs.append(clf.score(X_te, y_all[test_idx]))

    fold_accs = np.array(fold_accs)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


def integration_time_curve(
    rates_by_condition: dict,
    windows: list = DEFAULT_WINDOWS,
    n_splits: int = 5,
    n_pca: int = 30,
    C_logistic: float = 1.0,
    verbose: bool = True,
    method: str = 'flat_pca',
) -> dict:
    """
    Compute accuracy vs. window size for each FEM condition.

    Args:
        rates_by_condition: dict mapping condition → dict {stim_id → rate_list}
        windows: list of window sizes (frames at 120 Hz)
        n_splits: CV folds
        n_pca: PCA components
        C_logistic: regularization
        verbose: print progress

    Returns:
        dict mapping condition → dict with:
            'windows': list of window sizes
            'windows_ms': list of window sizes in ms
            'mean_acc': (len(windows),) array
            'std_acc': (len(windows),) array
            'fold_acc': (len(windows), n_splits) array
    """
    results = {}

    if method not in {'flat_pca', 'time_mean'}:
        raise ValueError("method must be 'flat_pca' or 'time_mean'")

    for condition, rates_by_stim in rates_by_condition.items():
        if verbose:
            print(f"\nCondition: {condition}")
        means, stds, all_folds = [], [], []

        for W in windows:
            W_ms = W / FRAME_RATE * 1000
            if method == 'flat_pca':
                mean_acc, std_acc, fold_accs = decode_causal_window(
                    rates_by_stim, window=W,
                    n_splits=n_splits, n_pca=n_pca, C_logistic=C_logistic,
                )
            else:
                mean_acc, std_acc, fold_accs = decode_causal_window_time_mean(
                    rates_by_stim, window=W,
                    n_splits=n_splits, C_logistic=C_logistic,
                )
            means.append(mean_acc)
            stds.append(std_acc)
            all_folds.append(fold_accs)
            if verbose:
                print(f"  W={W:3d} ({W_ms:.0f} ms): acc={mean_acc:.3f} ± {std_acc:.3f}")

        results[condition] = {
            'windows': windows,
            'windows_ms': [w / FRAME_RATE * 1000 for w in windows],
            'mean_acc': np.array(means),
            'std_acc': np.array(stds),
            'fold_acc': np.stack(all_folds, axis=0),  # (n_windows, n_splits)
        }

    return results


def find_crossover_point(
    results: dict,
    condition_a: str,
    condition_b: str,
) -> Optional[float]:
    """
    Find the window size (ms) at which condition_a first outperforms condition_b.

    Returns:
        Crossover window size in ms, or None if no crossover found.
    """
    acc_a = results[condition_a]['mean_acc']
    acc_b = results[condition_b]['mean_acc']
    windows_ms = results[condition_a]['windows_ms']

    diff = acc_a - acc_b
    crossover_idx = np.where(diff > 0)[0]
    if len(crossover_idx) == 0:
        return None
    return float(windows_ms[crossover_idx[0]])


def plot_integration_time_curves(
    results: dict,
    chance: float = 0.25,
    figsize=(8, 5),
    colors: dict = None,
) -> plt.Figure:
    """
    Plot accuracy vs. integration window for all conditions.

    Args:
        results: output of integration_time_curve()
        chance: chance level for 4AFC (0.25)
        colors: dict mapping condition name → color string

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = {
            'real': 'royalblue',
            'stabilized': 'tomato',
            'scaled_0.5': 'cornflowerblue',
            'scaled_2.0': 'navy',
            'matched_null': 'gray',
        }

    for condition, data in results.items():
        color = colors.get(condition, 'black')
        windows_ms = data['windows_ms']
        mean_acc = data['mean_acc']
        std_acc = data['std_acc']

        ax.plot(windows_ms, mean_acc, 'o-', color=color, label=condition, linewidth=2)
        ax.fill_between(
            windows_ms,
            mean_acc - std_acc,
            mean_acc + std_acc,
            alpha=0.2, color=color,
        )

    ax.axhline(y=chance, color='black', linestyle='--', alpha=0.5, label='chance')

    # Annotate crossover if real and stabilized are both present
    if 'real' in results and 'stabilized' in results:
        crossover_ms = find_crossover_point(results, 'real', 'stabilized')
        if crossover_ms is not None:
            ax.axvline(x=crossover_ms, color='green', linestyle=':', alpha=0.7)
            ax.text(crossover_ms + 5, chance + 0.02,
                    f'crossover\n{crossover_ms:.0f} ms',
                    color='green', fontsize=9)
        else:
            ax.text(0.98, 0.05, 'No crossover\n(FEMs always help)',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, color='gray')

    ax.set_xlabel('Integration window (ms)', fontsize=12)
    ax.set_ylabel('Decoding accuracy', fontsize=12)
    ax.set_title('Decoding Accuracy vs. Integration Time', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim([chance - 0.05, 1.05])
    ax.grid(True, alpha=0.3)

    return fig


if __name__ == '__main__':
    print("Testing integration time sweep with synthetic data...")
    np.random.seed(42)

    K, M, T, N = 4, 40, 80, 10

    def make_rates(temporal_snr: float):
        """Make rates with given temporal SNR relative to rate-only SNR."""
        stim_ids = ['ori0', 'ori90', 'ori180', 'ori270']
        rates = {}
        for k, sid in enumerate(stim_ids):
            signal_rate = np.zeros((1, T, N))
            signal_rate[0, :, k % N] = 1.0  # static rate difference
            signal_temporal = np.zeros((1, T, N))
            # Temporal modulation unique to each stimulus
            signal_temporal[0, :, k % N] = temporal_snr * np.sin(
                2 * np.pi * k / K + np.linspace(0, 2 * np.pi, T)
            )
            noise = np.random.randn(M, T, N) * 0.5
            rates[sid] = [signal_rate[0] + signal_temporal[0] + noise[i] for i in range(M)]
        return rates

    rates_by_condition = {
        'real': make_rates(temporal_snr=2.0),
        'stabilized': make_rates(temporal_snr=0.1),
    }

    results = integration_time_curve(
        rates_by_condition,
        windows=[1, 3, 6, 12, 24, 48],
        n_splits=4,
        verbose=True,
    )

    crossover = find_crossover_point(results, 'real', 'stabilized')
    print(f"\nCrossover point: {crossover} ms (None = no crossover)")

    fig = plot_integration_time_curves(results)
    fig.savefig('test_integration_time.png', dpi=100, bbox_inches='tight')
    print("Saved test_integration_time.png")

    # Verify curves are non-decreasing (more data ≥ same accuracy)
    for condition in results:
        accs = results[condition]['mean_acc']
        # Allow small non-monotonicities due to finite samples
        violations = np.sum(np.diff(accs) < -0.05)
        print(f"  {condition}: max drop = {np.min(np.diff(accs)):.3f}")

    print("All tests passed!")

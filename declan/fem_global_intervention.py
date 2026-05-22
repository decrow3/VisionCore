"""
Priority 2: Global FEM Subspace Intervention (revised)
=====================================================

Implements the revised causal test after Priority 1 established that the
per-orientation FEM subspaces U_FEM^k are effectively orientation-invariant.

For each requested LogMAR and condition:
1. Load cached E-optotype rate files
2. Time-average each trial to rbar
3. Fit a pooled global FEM subspace U_FEM from within-orientation covariance
4. Project U_FEM out of each trial mean: rbar_clean = rbar - U U^T rbar
5. Rerun D1 decoding before and after the ablation
6. Save a compact numeric summary and a small figure

Also reports a cheap C_signal eigenspectrum diagnostic to help interpret the
alignment change across LogMARs.

Usage
-----
python declan/fem_global_intervention.py
python declan/fem_global_intervention.py --logmars -0.20,-0.40 --condition real
python declan/fem_global_intervention.py --logmars -0.20,-0.40 --condition stabilized
python declan/fem_global_intervention.py --rate_file_tag allhires_fresh --hires_threshold 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RATES_DIR = os.path.join(REPO_DIR, 'scripts', 'temporal_decoding', 'data', 'rates')

ORIENTATIONS = [0, 90, 180, 270]
ORI_KEYS = [f'ori{o}' for o in ORIENTATIONS]
CHANCE = 1.0 / len(ORI_KEYS)


@dataclass(frozen=True)
class InterventionResult:
    logmar: float
    condition: str
    d1_acc: float
    d1_std: float
    d1_clean_acc: float
    d1_clean_std: float
    d1_delta: float
    alpha: float
    alpha_chance: float
    top_signal_eigvals: np.ndarray
    top_fem_eigvals: np.ndarray
    n_trials_per_orientation: int
    n_neurons: int


def _normalize_tag(tag: str) -> str:
    tag = str(tag or '').strip()
    if tag and not tag.startswith('_'):
        tag = '_' + tag
    return tag


def _load_rates_list(path: str) -> list[np.ndarray]:
    d = np.load(path, allow_pickle=True)
    rates_padded = d['rates']
    lengths = d['lengths'].astype(int)
    return [rates_padded[i, :lengths[i]] for i in range(rates_padded.shape[0])]


def load_rates(logmar: float, condition: str, rates_dir: str,
               hires_threshold: float = 2.0, rate_file_tag: str = '') -> dict[str, list[np.ndarray]]:
    use_hires = float(logmar) < float(hires_threshold)
    prefix = 'rates_hires' if use_hires else 'rates'
    tag = _normalize_tag(rate_file_tag)
    result: dict[str, list[np.ndarray]] = {}
    for ori in ORIENTATIONS:
        fname = f'{prefix}_lm{logmar:.2f}_ori{ori}_{condition}{tag}.npz'
        path = os.path.join(rates_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing cached rates: {path}')
        result[f'ori{ori}'] = _load_rates_list(path)
    return result


def time_average(rates_by_stim: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([r.mean(axis=0) for r in rate_list]).astype(np.float64)
        for key, rate_list in rates_by_stim.items()
    }


def equalise_trials(ravg_by_stim: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    m_min = min(v.shape[0] for v in ravg_by_stim.values())
    return {k: v[:m_min] for k, v in ravg_by_stim.items()}


def compute_signal_covariance(ravg_by_stim: dict[str, np.ndarray]) -> np.ndarray:
    means = np.stack([ravg_by_stim[k].mean(axis=0) for k in ORI_KEYS], axis=0)
    means = means - means.mean(axis=0, keepdims=True)
    c_signal = (means.T @ means) / max(means.shape[0] - 1, 1)
    return (c_signal + c_signal.T) / 2


def compute_pooled_fem_covariance(ravg_by_stim: dict[str, np.ndarray]) -> np.ndarray:
    residuals = []
    for key in ORI_KEYS:
        r = ravg_by_stim[key]
        r_centered = r - r.mean(axis=0, keepdims=True)
        residuals.append(r_centered)
    all_res = np.concatenate(residuals, axis=0)
    all_res = all_res - all_res.mean(axis=0, keepdims=True)
    c_fem = (all_res.T @ all_res) / max(all_res.shape[0] - 1, 1)
    return (c_fem + c_fem.T) / 2


def top_eigspace(cov: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs[:, -n_components:]


def alignment_fraction(c_signal: np.ndarray, u_fem: np.ndarray) -> tuple[float, float]:
    alpha = float(np.trace(u_fem.T @ c_signal @ u_fem)) / (float(np.trace(c_signal)) + 1e-12)
    alpha_chance = float(u_fem.shape[1] / c_signal.shape[0])
    return alpha, alpha_chance


def project_out_subspace(ravg_by_stim: dict[str, np.ndarray], u_remove: np.ndarray) -> dict[str, np.ndarray]:
    p_remove = u_remove @ u_remove.T
    return {k: r - (r @ p_remove.T) for k, r in ravg_by_stim.items()}


def _cv_splits(m_min: int, n_splits: int):
    groups = np.arange(m_min)
    x_dummy = np.zeros((len(ORI_KEYS) * m_min, 1))
    y_dummy = np.repeat(np.arange(len(ORI_KEYS)), m_min)
    g_dummy = np.tile(groups, len(ORI_KEYS))
    gkf = GroupKFold(n_splits=n_splits)
    for tr_all, te_all in gkf.split(x_dummy, y_dummy, groups=g_dummy):
        train_tr = tr_all[tr_all < m_min]
        test_tr = te_all[te_all < m_min]
        yield train_tr, test_tr


def run_d1_decoder(ravg_by_stim: dict[str, np.ndarray], n_splits: int = 5) -> tuple[float, float, np.ndarray]:
    m_min = min(v.shape[0] for v in ravg_by_stim.values())
    fold_accs = []
    for train_tr, test_tr in _cv_splits(m_min, n_splits):
        x_tr = np.concatenate([ravg_by_stim[k][train_tr] for k in ORI_KEYS], axis=0)
        y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
        x_te = np.concatenate([ravg_by_stim[k][test_tr] for k in ORI_KEYS], axis=0)
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(x_tr), y_tr)
        fold_accs.append(clf.score(sc.transform(x_te), y_te))

    fold_accs = np.asarray(fold_accs, dtype=float)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


def run_d1_decoder_with_intervention(ravg_by_stim: dict[str, np.ndarray],
                                     n_fem_components: int,
                                     n_splits: int = 5) -> tuple[float, float, np.ndarray]:
    """
    D1 decoder after projecting out U_FEM, with U_FEM fit inside each CV fold.

    U_FEM is estimated from training traces only, then applied to both train and
    test before decoding. This avoids leaking test-set data into the subspace
    estimate, which would bias the cleaned accuracy upward at -0.20 (where removal
    of a well-estimated subspace should help most).
    """
    m_min = min(v.shape[0] for v in ravg_by_stim.values())
    fold_accs = []
    for train_tr, test_tr in _cv_splits(m_min, n_splits):
        ravg_train = {k: ravg_by_stim[k][train_tr] for k in ORI_KEYS}
        ravg_test = {k: ravg_by_stim[k][test_tr] for k in ORI_KEYS}

        # Fit U_FEM on training fold only
        c_fem_fold = compute_pooled_fem_covariance(ravg_train)
        _, u_fem_fold = top_eigspace(c_fem_fold, n_fem_components)

        # Project out from both folds using the training-fold subspace
        ravg_train_clean = project_out_subspace(ravg_train, u_fem_fold)
        ravg_test_clean = project_out_subspace(ravg_test, u_fem_fold)

        x_tr = np.concatenate([ravg_train_clean[k] for k in ORI_KEYS], axis=0)
        y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
        x_te = np.concatenate([ravg_test_clean[k] for k in ORI_KEYS], axis=0)
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(x_tr), y_tr)
        fold_accs.append(clf.score(sc.transform(x_te), y_te))

    fold_accs = np.asarray(fold_accs, dtype=float)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


def analyse_one(logmar: float, condition: str, rates_dir: str, rate_file_tag: str,
                hires_threshold: float, n_fem_components: int, n_splits: int) -> InterventionResult:
    print(f'\n{"=" * 60}')
    print(f'Priority 2 intervention | logmar={logmar:+.2f} | condition={condition}')
    print(f'{"=" * 60}')

    rates = load_rates(
        logmar=logmar,
        condition=condition,
        rates_dir=rates_dir,
        hires_threshold=hires_threshold,
        rate_file_tag=rate_file_tag,
    )
    ravg = equalise_trials(time_average(rates))
    m_min, n_neurons = next(iter(ravg.values())).shape
    print(f'  Using M={m_min} trials/orientation, N={n_neurons} neurons')

    c_signal = compute_signal_covariance(ravg)
    c_fem = compute_pooled_fem_covariance(ravg)
    eig_signal, _ = top_eigspace(c_signal, n_fem_components)
    eig_fem, u_fem = top_eigspace(c_fem, n_fem_components)
    alpha, alpha_chance = alignment_fraction(c_signal, u_fem)

    print(f'  alpha={alpha:.4f}  chance={alpha_chance:.4f}  xchance={alpha / alpha_chance:.2f}')
    print(f'  top signal eigvals: {np.round(eig_signal[-n_fem_components:][::-1], 6)}')
    print(f'  top FEM eigvals:    {np.round(eig_fem[-n_fem_components:][::-1], 6)}')

    d1_acc, d1_std, d1_folds = run_d1_decoder(ravg, n_splits=n_splits)
    # U_FEM is fit inside each fold to avoid leaking test traces into the subspace estimate.
    d1_clean_acc, d1_clean_std, d1_clean_folds = run_d1_decoder_with_intervention(
        ravg, n_fem_components=n_fem_components, n_splits=n_splits
    )
    d1_delta = d1_clean_acc - d1_acc

    print(f'  D1 original: {d1_acc:.3f} ± {d1_std:.3f}  folds={np.round(d1_folds, 3)}')
    print(f'  D1 cleaned : {d1_clean_acc:.3f} ± {d1_clean_std:.3f}  folds={np.round(d1_clean_folds, 3)}')
    print(f'  Δ cleaned-original = {d1_delta:+.3f}')

    return InterventionResult(
        logmar=float(logmar),
        condition=str(condition),
        d1_acc=d1_acc,
        d1_std=d1_std,
        d1_clean_acc=d1_clean_acc,
        d1_clean_std=d1_clean_std,
        d1_delta=d1_delta,
        alpha=alpha,
        alpha_chance=alpha_chance,
        top_signal_eigvals=eig_signal[-n_fem_components:][::-1].copy(),
        top_fem_eigvals=eig_fem[-n_fem_components:][::-1].copy(),
        n_trials_per_orientation=int(m_min),
        n_neurons=int(n_neurons),
    )


def save_results(results: list[InterventionResult], out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f'{stem}.npz')
    save_dict: dict[str, np.ndarray] = {}
    for r in results:
        tag = f'lm{r.logmar:+.2f}'.replace('+', '')
        save_dict[f'{tag}_condition'] = np.asarray([r.condition], dtype=object)
        save_dict[f'{tag}_d1_acc'] = np.asarray([r.d1_acc], dtype=float)
        save_dict[f'{tag}_d1_std'] = np.asarray([r.d1_std], dtype=float)
        save_dict[f'{tag}_d1_clean_acc'] = np.asarray([r.d1_clean_acc], dtype=float)
        save_dict[f'{tag}_d1_clean_std'] = np.asarray([r.d1_clean_std], dtype=float)
        save_dict[f'{tag}_d1_delta'] = np.asarray([r.d1_delta], dtype=float)
        save_dict[f'{tag}_alpha'] = np.asarray([r.alpha], dtype=float)
        save_dict[f'{tag}_alpha_chance'] = np.asarray([r.alpha_chance], dtype=float)
        save_dict[f'{tag}_top_signal_eigvals'] = np.asarray(r.top_signal_eigvals, dtype=float)
        save_dict[f'{tag}_top_fem_eigvals'] = np.asarray(r.top_fem_eigvals, dtype=float)
        save_dict[f'{tag}_n_trials_per_orientation'] = np.asarray([r.n_trials_per_orientation], dtype=int)
        save_dict[f'{tag}_n_neurons'] = np.asarray([r.n_neurons], dtype=int)
    np.savez(out_npz, **save_dict)
    print(f'Results saved: {out_npz}')

    fig = make_figure(results)
    out_png = os.path.join(out_dir, f'{stem}.png')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {out_png}')


def make_figure(results: list[InterventionResult]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    labels = [f'{r.logmar:+.2f}' for r in results]
    x = np.arange(len(results))

    axes[0].bar(x - 0.18, [r.d1_acc for r in results], width=0.36, yerr=[r.d1_std for r in results],
                capsize=3, label='D1 original', color='#4d4d4d')
    axes[0].bar(x + 0.18, [r.d1_clean_acc for r in results], width=0.36,
                yerr=[r.d1_clean_std for r in results], capsize=3, label='D1 after removing U_FEM',
                color='#1f77b4')
    axes[0].axhline(CHANCE, color='k', linestyle='--', linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('LogMAR')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Priority 2 intervention')
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, [r.d1_delta for r in results], color='#2ca02c')
    axes[1].axhline(0.0, color='k', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('LogMAR')
    axes[1].set_ylabel('Δ accuracy')
    axes[1].set_title('D1 cleaned - original')

    axes[2].bar(x - 0.18, [r.alpha for r in results], width=0.36, label='alpha', color='#9467bd')
    axes[2].bar(x + 0.18, [r.alpha_chance for r in results], width=0.36, label='chance', color='#c7c7c7')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('LogMAR')
    axes[2].set_ylabel('Alignment fraction')
    axes[2].set_title('Alpha diagnostic')
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser(description='Revised global FEM subspace intervention (Priority 2)')
    p.add_argument('--logmars', type=str, default='-0.20,-0.40',
                   help='Comma-separated LogMAR values (default: -0.20,-0.40)')
    p.add_argument('--condition', type=str, default='real', choices=['real', 'stabilized'],
                   help='Condition to analyse (default: real)')
    p.add_argument('--rates_dir', type=str, default=RATES_DIR)
    p.add_argument('--rate_file_tag', type=str, default='',
                   help="Optional cache file tag appended to filenames (e.g. 'allhires_fresh' -> *_allhires_fresh.npz). Default empty = no tag.")
    p.add_argument('--hires_threshold', type=float, default=2.0,
                   help='Threshold used only for cache routing. 2.0 forces hires for this sweep.')
    p.add_argument('--n_fem_components', type=int, default=2)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(SCRIPT_DIR, 'fem_global_intervention_results'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logmars = [float(x.strip()) for x in str(args.logmars).split(',') if x.strip()]

    print('Global FEM subspace intervention (revised Priority 2)')
    print(f'LogMARs: {logmars}')
    print(f'Condition: {args.condition}')
    print(f'Rates dir: {args.rates_dir}')
    print(f'Rate file tag: {args.rate_file_tag}')

    results = [
        analyse_one(
            logmar=lm,
            condition=str(args.condition),
            rates_dir=str(args.rates_dir),
            rate_file_tag=str(args.rate_file_tag),
            hires_threshold=float(args.hires_threshold),
            n_fem_components=int(args.n_fem_components),
            n_splits=int(args.n_splits),
        )
        for lm in logmars
    ]

    stem = f'fem_global_intervention_{args.condition}'
    save_results(results, out_dir=str(args.out_dir), stem=stem)

    print('\nSummary')
    for r in results:
        print(
            f'  LM {r.logmar:+.2f} | D1={r.d1_acc:.3f} -> {r.d1_clean_acc:.3f} '
            f'(Δ={r.d1_delta:+.3f}) | alpha={r.alpha:.4f} '
            f'(chance={r.alpha_chance:.4f})'
        )


if __name__ == '__main__':
    main()

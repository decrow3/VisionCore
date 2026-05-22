"""
Differential FEM covariance intervention
=======================================

Isolate the covariance component that is stronger in the real FEM condition than
in the stabilized condition, then test whether removing that differential
subspace changes D1 decoding.

This is the natural follow-up to the pooled U_FEM intervention: if pooled
ablation helps both conditions, the removed subspace is shared positional
variance. The differential covariance C_real - C_stabilized targets the part
that is unique to dynamic FEMs.

Usage
-----
python declan/fem_differential_intervention.py \
  --logmars=-0.20,-0.40 \
  --rate_file_tag allhires_fresh \
  --hires_threshold 2.0
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fem_global_intervention import (  # noqa: E402
    CHANCE,
    ORI_KEYS,
    RATES_DIR,
    _cv_splits,
    compute_pooled_fem_covariance,
    equalise_trials,
    load_rates,
    project_out_subspace,
    run_d1_decoder,
    time_average,
)


@dataclass(frozen=True)
class DifferentialResult:
    logmar: float
    n_components_used: int
    positive_eigvals: np.ndarray
    d1_real: float
    d1_real_std: float
    d1_real_clean: float
    d1_real_clean_std: float
    d1_real_delta: float
    d1_stab: float
    d1_stab_std: float
    d1_stab_clean: float
    d1_stab_clean_std: float
    d1_stab_delta: float


def positive_eigspace(cov: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2)
    pos_idx = np.where(eigvals > 1e-12)[0]
    if pos_idx.size == 0:
        return np.zeros(0, dtype=float), np.zeros((cov.shape[0], 0), dtype=float)
    keep_idx = pos_idx[-n_components:]
    return eigvals[keep_idx], eigvecs[:, keep_idx]


def run_d1_decoder_with_fixed_subspace(
    ravg_by_stim: dict[str, np.ndarray],
    u_remove: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, float, np.ndarray]:
    if u_remove.shape[1] == 0:
        return run_d1_decoder(ravg_by_stim, n_splits=n_splits)

    m_min = min(v.shape[0] for v in ravg_by_stim.values())
    fold_accs = []
    for train_tr, test_tr in _cv_splits(m_min, n_splits):
        ravg_train = {k: ravg_by_stim[k][train_tr] for k in ORI_KEYS}
        ravg_test = {k: ravg_by_stim[k][test_tr] for k in ORI_KEYS}

        ravg_train_clean = project_out_subspace(ravg_train, u_remove)
        ravg_test_clean = project_out_subspace(ravg_test, u_remove)

        fold_mean, _, fold_acc = run_d1_decoder(
            {k: np.concatenate([ravg_train_clean[k], ravg_test_clean[k]], axis=0) for k in ORI_KEYS},
            n_splits=2,
        )
        # Avoid nesting a second CV protocol with different grouping assumptions.
        # Rebuild the exact train/test split using the same D1 classifier recipe.
        x_tr = np.concatenate([ravg_train_clean[k] for k in ORI_KEYS], axis=0)
        y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
        x_te = np.concatenate([ravg_test_clean[k] for k in ORI_KEYS], axis=0)
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(x_tr), y_tr)
        fold_accs.append(clf.score(sc.transform(x_te), y_te))

    fold_accs = np.asarray(fold_accs, dtype=float)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


def run_d1_decoder_with_differential_intervention(
    ravg_real: dict[str, np.ndarray],
    ravg_stab: dict[str, np.ndarray],
    n_components: int,
    n_splits: int,
) -> tuple[tuple[float, float, np.ndarray], tuple[float, float, np.ndarray], np.ndarray]:
    m_min = min(min(v.shape[0] for v in ravg_real.values()), min(v.shape[0] for v in ravg_stab.values()))
    fold_real = []
    fold_stab = []
    eigvals_used = []

    for train_tr, test_tr in _cv_splits(m_min, n_splits):
        real_train = {k: ravg_real[k][train_tr] for k in ORI_KEYS}
        real_test = {k: ravg_real[k][test_tr] for k in ORI_KEYS}
        stab_train = {k: ravg_stab[k][train_tr] for k in ORI_KEYS}
        stab_test = {k: ravg_stab[k][test_tr] for k in ORI_KEYS}

        c_real = compute_pooled_fem_covariance(real_train)
        c_stab = compute_pooled_fem_covariance(stab_train)
        eigvals_diff, u_diff = positive_eigspace(c_real - c_stab, n_components)
        eigvals_used.append(eigvals_diff)

        real_train_clean = project_out_subspace(real_train, u_diff)
        real_test_clean = project_out_subspace(real_test, u_diff)
        stab_train_clean = project_out_subspace(stab_train, u_diff)
        stab_test_clean = project_out_subspace(stab_test, u_diff)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        def _score(train_dict: dict[str, np.ndarray], test_dict: dict[str, np.ndarray]) -> float:
            x_tr = np.concatenate([train_dict[k] for k in ORI_KEYS], axis=0)
            y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
            x_te = np.concatenate([test_dict[k] for k in ORI_KEYS], axis=0)
            y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])
            sc = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
            clf.fit(sc.fit_transform(x_tr), y_tr)
            return float(clf.score(sc.transform(x_te), y_te))

        fold_real.append(_score(real_train_clean, real_test_clean))
        fold_stab.append(_score(stab_train_clean, stab_test_clean))

    fold_real = np.asarray(fold_real, dtype=float)
    fold_stab = np.asarray(fold_stab, dtype=float)
    padded = np.full((len(eigvals_used), n_components), np.nan, dtype=float)
    for i, ev in enumerate(eigvals_used):
        padded[i, :ev.shape[0]] = ev[::-1]

    return (
        (float(fold_real.mean()), float(fold_real.std()), fold_real),
        (float(fold_stab.mean()), float(fold_stab.std()), fold_stab),
        padded,
    )


def analyse_one(
    logmar: float,
    rates_dir: str,
    rate_file_tag: str,
    hires_threshold: float,
    n_components: int,
    n_splits: int,
) -> DifferentialResult:
    print(f'\n{"=" * 60}')
    print(f'Differential intervention | logmar={logmar:+.2f}')
    print(f'{"=" * 60}')

    real_rates = equalise_trials(time_average(load_rates(
        logmar=logmar,
        condition='real',
        rates_dir=rates_dir,
        hires_threshold=hires_threshold,
        rate_file_tag=rate_file_tag,
    )))
    stab_rates = equalise_trials(time_average(load_rates(
        logmar=logmar,
        condition='stabilized',
        rates_dir=rates_dir,
        hires_threshold=hires_threshold,
        rate_file_tag=rate_file_tag,
    )))

    m_real, n_neurons = next(iter(real_rates.values())).shape
    m_stab, _ = next(iter(stab_rates.values())).shape
    m_min = min(m_real, m_stab)
    real_rates = {k: v[:m_min] for k, v in real_rates.items()}
    stab_rates = {k: v[:m_min] for k, v in stab_rates.items()}
    print(f'  Using M={m_min} trials/orientation, N={n_neurons} neurons')

    d1_real, d1_real_std, d1_real_folds = run_d1_decoder(real_rates, n_splits=n_splits)
    d1_stab, d1_stab_std, d1_stab_folds = run_d1_decoder(stab_rates, n_splits=n_splits)
    (d1_real_clean, d1_real_clean_std, d1_real_clean_folds), (d1_stab_clean, d1_stab_clean_std, d1_stab_clean_folds), diff_eigs = run_d1_decoder_with_differential_intervention(
        real_rates,
        stab_rates,
        n_components=n_components,
        n_splits=n_splits,
    )

    mean_pos_eigs = np.nanmean(diff_eigs, axis=0)
    n_used = int(np.sum(np.isfinite(mean_pos_eigs)))
    print(f'  mean positive eigvals(C_real - C_stab): {np.round(mean_pos_eigs[np.isfinite(mean_pos_eigs)], 6)}')
    print(f'  Real baseline: {d1_real:.3f} ± {d1_real_std:.3f}  folds={np.round(d1_real_folds, 3)}')
    print(f'  Real cleaned : {d1_real_clean:.3f} ± {d1_real_clean_std:.3f}  folds={np.round(d1_real_clean_folds, 3)}')
    print(f'  Real delta   : {d1_real_clean - d1_real:+.3f}')
    print(f'  Stab baseline: {d1_stab:.3f} ± {d1_stab_std:.3f}  folds={np.round(d1_stab_folds, 3)}')
    print(f'  Stab cleaned : {d1_stab_clean:.3f} ± {d1_stab_clean_std:.3f}  folds={np.round(d1_stab_clean_folds, 3)}')
    print(f'  Stab delta   : {d1_stab_clean - d1_stab:+.3f}')

    return DifferentialResult(
        logmar=float(logmar),
        n_components_used=n_used,
        positive_eigvals=mean_pos_eigs[np.isfinite(mean_pos_eigs)].copy(),
        d1_real=d1_real,
        d1_real_std=d1_real_std,
        d1_real_clean=d1_real_clean,
        d1_real_clean_std=d1_real_clean_std,
        d1_real_delta=d1_real_clean - d1_real,
        d1_stab=d1_stab,
        d1_stab_std=d1_stab_std,
        d1_stab_clean=d1_stab_clean,
        d1_stab_clean_std=d1_stab_clean_std,
        d1_stab_delta=d1_stab_clean - d1_stab,
    )


def make_figure(results: list[DifferentialResult]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    x = np.arange(len(results))
    labels = [f'{r.logmar:+.2f}' for r in results]

    axes[0].bar(x - 0.18, [r.d1_real_delta for r in results], width=0.36, label='real', color='#d62728')
    axes[0].bar(x + 0.18, [r.d1_stab_delta for r in results], width=0.36, label='stabilized', color='#1f77b4')
    axes[0].axhline(0.0, color='k', linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('LogMAR')
    axes[0].set_ylabel('Δ accuracy after diff ablation')
    axes[0].set_title('Differential subspace effect')
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - 0.18, [r.d1_real for r in results], width=0.36, label='real base', color='#8c564b')
    axes[1].bar(x + 0.18, [r.d1_real_clean for r in results], width=0.36, label='real cleaned', color='#ff9896')
    axes[1].axhline(CHANCE, color='k', linestyle='--', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('LogMAR')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Real condition')
    axes[1].legend(frameon=False, fontsize=8)

    max_dim = max((r.positive_eigvals.shape[0] for r in results), default=0)
    for i, r in enumerate(results):
        if r.positive_eigvals.shape[0] == 0:
            continue
        axes[2].plot(np.arange(1, r.positive_eigvals.shape[0] + 1), r.positive_eigvals, marker='o', label=f'{r.logmar:+.2f}')
    axes[2].set_xlabel('Differential component')
    axes[2].set_ylabel('Positive eigenvalue')
    axes[2].set_title('eig(C_real - C_stab)')
    if max_dim > 0:
        axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def save_results(results: list[DifferentialResult], out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f'{stem}.npz')
    save_dict: dict[str, np.ndarray] = {}
    for r in results:
        tag = f'lm{r.logmar:+.2f}'.replace('+', '')
        save_dict[f'{tag}_n_components_used'] = np.asarray([r.n_components_used], dtype=int)
        save_dict[f'{tag}_positive_eigvals'] = np.asarray(r.positive_eigvals, dtype=float)
        save_dict[f'{tag}_d1_real'] = np.asarray([r.d1_real], dtype=float)
        save_dict[f'{tag}_d1_real_std'] = np.asarray([r.d1_real_std], dtype=float)
        save_dict[f'{tag}_d1_real_clean'] = np.asarray([r.d1_real_clean], dtype=float)
        save_dict[f'{tag}_d1_real_clean_std'] = np.asarray([r.d1_real_clean_std], dtype=float)
        save_dict[f'{tag}_d1_real_delta'] = np.asarray([r.d1_real_delta], dtype=float)
        save_dict[f'{tag}_d1_stab'] = np.asarray([r.d1_stab], dtype=float)
        save_dict[f'{tag}_d1_stab_std'] = np.asarray([r.d1_stab_std], dtype=float)
        save_dict[f'{tag}_d1_stab_clean'] = np.asarray([r.d1_stab_clean], dtype=float)
        save_dict[f'{tag}_d1_stab_clean_std'] = np.asarray([r.d1_stab_clean_std], dtype=float)
        save_dict[f'{tag}_d1_stab_delta'] = np.asarray([r.d1_stab_delta], dtype=float)
    np.savez(out_npz, **save_dict)
    print(f'Results saved: {out_npz}')

    fig = make_figure(results)
    out_png = os.path.join(out_dir, f'{stem}.png')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {out_png}')


def parse_args():
    p = argparse.ArgumentParser(description='Differential real-minus-stabilized FEM covariance intervention')
    p.add_argument('--logmars', type=str, default='-0.20,-0.40')
    p.add_argument('--rates_dir', type=str, default=RATES_DIR)
    p.add_argument('--rate_file_tag', type=str, default='allhires_fresh')
    p.add_argument('--hires_threshold', type=float, default=2.0)
    p.add_argument('--n_components', type=int, default=2)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--out_dir', type=str, default=os.path.join(SCRIPT_DIR, 'fem_differential_intervention_results'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logmars = [float(x.strip()) for x in str(args.logmars).split(',') if x.strip()]

    print('Differential FEM covariance intervention')
    print(f'LogMARs: {logmars}')
    print(f'Rates dir: {args.rates_dir}')
    print(f'Rate file tag: {args.rate_file_tag}')

    results = [
        analyse_one(
            logmar=lm,
            rates_dir=str(args.rates_dir),
            rate_file_tag=str(args.rate_file_tag),
            hires_threshold=float(args.hires_threshold),
            n_components=int(args.n_components),
            n_splits=int(args.n_splits),
        )
        for lm in logmars
    ]

    save_results(results, out_dir=str(args.out_dir), stem='fem_differential_intervention')

    print('\nSummary')
    for r in results:
        print(
            f'  LM {r.logmar:+.2f} | real Δ={r.d1_real_delta:+.3f} | '
            f'stab Δ={r.d1_stab_delta:+.3f} | eig+={np.round(r.positive_eigvals, 6)}'
        )


if __name__ == '__main__':
    main()

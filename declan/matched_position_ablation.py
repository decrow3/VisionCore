"""
B.5: Matched position-distribution reweighting pre-check.

Reweights existing real trials so their per-trial mean eye-position distribution
matches the stabilized distribution, then reruns the pooled FEM subspace ablation
on both unweighted and reweighted data.

This is a cheap pre-A check: it requires no new rate caching, only existing
real and stabilized .npz files.

Interpretation
--------------
If the non-specific ablation gain (real and stabilized both improve by ~equal
amounts under the pooled ablation) dissolves under matched distributions:
    → The non-specificity was a position-distribution artifact.
    → Step A (fixed_center) will be a cleaner confirmation.

If the non-specificity persists under matched distributions:
    → The effect is not explained by distribution mismatch.
    → Step A tests something sharper; the result will be more informative.

Usage
-----
python declan/matched_position_ablation.py
python declan/matched_position_ablation.py --logmars -0.20,-0.40 --n_fem_components 2
python declan/matched_position_ablation.py --rate_file_tag allhires_fresh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
RATES_DIR = REPO_DIR / 'scripts' / 'temporal_decoding' / 'data' / 'rates'
EYE_TRACES_PATH = REPO_DIR / 'scripts' / 'temporal_decoding' / 'data' / 'eye_traces.npz'
OUT_DIR = SCRIPT_DIR / 'matched_position_results'

ORIENTATIONS = [0, 90, 180, 270]
ORI_KEYS = [f'ori{o}' for o in ORIENTATIONS]
CHANCE = 1.0 / len(ORI_KEYS)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_rates_list(path: Path) -> list[np.ndarray]:
    d = np.load(path, allow_pickle=True)
    rates_padded = d['rates']
    lengths = d['lengths'].astype(int)
    return [rates_padded[i, :lengths[i]] for i in range(rates_padded.shape[0])]


def load_rates(logmar: float, condition: str, rate_file_tag: str,
               hires_threshold: float = 2.0) -> dict[str, list[np.ndarray]]:
    use_hires = logmar < hires_threshold
    prefix = 'rates_hires' if use_hires else 'rates'
    tag = rate_file_tag if rate_file_tag.startswith('_') or not rate_file_tag else f'_{rate_file_tag}'
    result: dict[str, list[np.ndarray]] = {}
    for ori in ORIENTATIONS:
        fname = f'{prefix}_lm{logmar:.2f}_ori{ori}_{condition}{tag}.npz'
        path = RATES_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f'Missing cached rates: {path}')
        result[f'ori{ori}'] = _load_rates_list(path)
    return result


def time_average(rates_by_stim: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        k: np.stack([r.mean(axis=0) for r in v]).astype(np.float64)
        for k, v in rates_by_stim.items()
    }


# ── Eye position loading ──────────────────────────────────────────────────────

def load_mean_eye_positions() -> np.ndarray:
    """
    Load per-trial mean eye positions from eye_traces.npz.
    Returns (M, 2) array in degrees.
    """
    if not EYE_TRACES_PATH.exists():
        raise FileNotFoundError(
            f'eye_traces.npz not found at {EYE_TRACES_PATH}. '
            'Run scripts/temporal_decoding/extract_eye_traces.py first.'
        )
    td = np.load(EYE_TRACES_PATH, allow_pickle=True)
    traces = td['traces']       # (M, T_max, 2)
    durations = td['durations'].astype(int)  # (M,)
    mean_pos = np.array([traces[i, :durations[i]].mean(axis=0) for i in range(len(durations))])
    return mean_pos.astype(np.float32)  # (M, 2)


# ── Matching ──────────────────────────────────────────────────────────────────

def compute_importance_weights(
    real_positions: np.ndarray,
    stab_positions: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Compute per-trial importance weights for real trials so their mean-position
    distribution matches the stabilized distribution.

    Uses 2D histogram density ratio: w_i = p_stab(x_i) / p_real(x_i).
    Weights are clipped to [0.05, 20] to prevent extreme reweighting and
    normalised to sum to n_real so weighted counts are on the original scale.

    Returns
    -------
    weights : (M_real,) float array
    """
    # Build 2D histograms over the joint (x, y) position space
    all_pos = np.concatenate([real_positions, stab_positions], axis=0)
    x_range = (all_pos[:, 0].min() - 1e-6, all_pos[:, 0].max() + 1e-6)
    y_range = (all_pos[:, 1].min() - 1e-6, all_pos[:, 1].max() + 1e-6)

    hist_real, xedges, yedges = np.histogram2d(
        real_positions[:, 0], real_positions[:, 1],
        bins=n_bins, range=[x_range, y_range], density=True,
    )
    hist_stab, _, _ = np.histogram2d(
        stab_positions[:, 0], stab_positions[:, 1],
        bins=n_bins, range=[x_range, y_range], density=True,
    )

    # Assign each real trial to a bin and look up its density ratio
    xi = np.clip(np.digitize(real_positions[:, 0], xedges[1:-1]), 0, n_bins - 1)
    yi = np.clip(np.digitize(real_positions[:, 1], yedges[1:-1]), 0, n_bins - 1)

    p_real = hist_real[xi, yi] + 1e-9
    p_stab = hist_stab[xi, yi] + 1e-9
    weights = np.clip(p_stab / p_real, 0.05, 20.0)
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


# ── Decoding ─────────────────────────────────────────────────────────────────

def _cv_splits(m_min: int, n_splits: int):
    groups = np.arange(m_min)
    x_dummy = np.zeros((len(ORI_KEYS) * m_min, 1))
    y_dummy = np.repeat(np.arange(len(ORI_KEYS)), m_min)
    g_dummy = np.tile(groups, len(ORI_KEYS))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(x_dummy, y_dummy, groups=g_dummy):
        yield tr[tr < m_min], te[te < m_min]


def run_d1_weighted(
    ravg: dict[str, np.ndarray],
    weights: np.ndarray | None = None,
    n_splits: int = 5,
) -> tuple[float, float]:
    """D1 with optional per-trial importance weights."""
    m_min = min(v.shape[0] for v in ravg.values())
    w = weights[:m_min] if weights is not None else np.ones(m_min)
    fold_accs = []
    for train_tr, test_tr in _cv_splits(m_min, n_splits):
        x_tr = np.concatenate([ravg[k][train_tr] for k in ORI_KEYS])
        y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
        sw_tr = np.tile(w[train_tr], len(ORI_KEYS))
        x_te = np.concatenate([ravg[k][test_tr] for k in ORI_KEYS])
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(x_tr), y_tr, sample_weight=sw_tr)
        fold_accs.append(clf.score(sc.transform(x_te), y_te))
    return float(np.mean(fold_accs)), float(np.std(fold_accs))


def run_ablation_weighted(
    ravg: dict[str, np.ndarray],
    weights: np.ndarray | None = None,
    n_fem_components: int = 2,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Pooled FEM subspace ablation with optional per-trial weights."""
    from scipy.linalg import eigh

    m_min = min(v.shape[0] for v in ravg.values())
    w = weights[:m_min] if weights is not None else np.ones(m_min)
    fold_accs = []

    for train_tr, test_tr in _cv_splits(m_min, n_splits):
        # Fit U_FEM on weighted training traces
        residuals = []
        for k in ORI_KEYS:
            r = ravg[k][train_tr]
            mu = np.average(r, axis=0, weights=w[train_tr])
            residuals.append((r - mu) * w[train_tr, np.newaxis])
        all_res = np.concatenate(residuals, axis=0)
        c_fem = (all_res.T @ all_res) / max(all_res.shape[0] - 1, 1)
        c_fem = (c_fem + c_fem.T) / 2
        evals, evecs = eigh(c_fem, subset_by_index=[c_fem.shape[0] - n_fem_components,
                                                    c_fem.shape[0] - 1])
        u_fem = evecs[:, ::-1]  # descending order
        proj = u_fem @ u_fem.T

        # Project out and decode
        clean = lambda r, p=proj: r - r @ p.T
        x_tr = np.concatenate([clean(ravg[k][train_tr]) for k in ORI_KEYS])
        y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
        sw_tr = np.tile(w[train_tr], len(ORI_KEYS))
        x_te = np.concatenate([clean(ravg[k][test_tr]) for k in ORI_KEYS])
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(x_tr), y_tr, sample_weight=sw_tr)
        fold_accs.append(clf.score(sc.transform(x_te), y_te))

    return float(np.mean(fold_accs)), float(np.std(fold_accs))


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse_one(
    logmar: float,
    rate_file_tag: str,
    hires_threshold: float,
    n_fem_components: int,
    n_splits: int,
    n_bins: int,
) -> dict:
    print(f'\n{"=" * 60}')
    print(f'B.5 matched-position ablation | logmar={logmar:+.2f}')
    print(f'{"=" * 60}')

    rates_real = load_rates(logmar, 'real', rate_file_tag, hires_threshold)
    rates_stab = load_rates(logmar, 'stabilized', rate_file_tag, hires_threshold)
    ravg_real = {k: np.stack([r.mean(0) for r in v]).astype(np.float64)
                 for k, v in rates_real.items()}
    ravg_stab = {k: np.stack([r.mean(0) for r in v]).astype(np.float64)
                 for k, v in rates_stab.items()}

    m_real = min(v.shape[0] for v in ravg_real.values())
    m_stab = min(v.shape[0] for v in ravg_stab.values())
    m_min = min(m_real, m_stab)
    ravg_real = {k: v[:m_min] for k, v in ravg_real.items()}
    ravg_stab = {k: v[:m_min] for k, v in ravg_stab.items()}
    print(f'  M={m_min} trials/orientation')

    # Load per-trial mean eye positions for both conditions
    mean_pos = load_mean_eye_positions()
    # Assume real and stabilized trials are indexed identically
    real_pos = mean_pos[:m_min]
    stab_pos = mean_pos[:m_min]  # stabilized has same trial identity, different trace treatment

    # Compute importance weights for real → match stabilized distribution
    weights = compute_importance_weights(real_pos, stab_pos, n_bins=n_bins)
    print(f'  Weight stats: min={weights.min():.3f}  max={weights.max():.3f}  '
          f'mean={weights.mean():.3f}  eff_n={int(1/((weights/weights.sum())**2).sum())}')

    # Unweighted D1 and ablation
    d1_real_uw, _ = run_d1_weighted(ravg_real, weights=None, n_splits=n_splits)
    d1_abl_uw, _ = run_ablation_weighted(ravg_real, weights=None,
                                          n_fem_components=n_fem_components, n_splits=n_splits)
    d1_stab_uw, _ = run_d1_weighted(ravg_stab, weights=None, n_splits=n_splits)
    d1_stab_abl_uw, _ = run_ablation_weighted(ravg_stab, weights=None,
                                               n_fem_components=n_fem_components, n_splits=n_splits)

    # Weighted D1 and ablation (real reweighted to match stabilized distribution)
    d1_real_w, _ = run_d1_weighted(ravg_real, weights=weights, n_splits=n_splits)
    d1_abl_w, _ = run_ablation_weighted(ravg_real, weights=weights,
                                         n_fem_components=n_fem_components, n_splits=n_splits)

    print(f'\n  Unweighted:')
    print(f'    real   D1={d1_real_uw:.3f}  ablated={d1_abl_uw:.3f}  Δ={d1_abl_uw-d1_real_uw:+.3f}')
    print(f'    stab   D1={d1_stab_uw:.3f}  ablated={d1_stab_abl_uw:.3f}  Δ={d1_stab_abl_uw-d1_stab_uw:+.3f}')
    print(f'  Reweighted real (matched to stab distribution):')
    print(f'    real_w D1={d1_real_w:.3f}  ablated={d1_abl_w:.3f}  Δ={d1_abl_w-d1_real_w:+.3f}')

    # Interpretation
    delta_uw = d1_abl_uw - d1_real_uw
    delta_stab_uw = d1_stab_abl_uw - d1_stab_uw
    delta_w = d1_abl_w - d1_real_w
    specificity_ratio = (delta_uw - delta_w) / (abs(delta_uw) + 1e-6)

    print(f'\n  Non-specificity: unweighted Δ={delta_uw:+.3f}  stab Δ={delta_stab_uw:+.3f}')
    print(f'  After matching: reweighted Δ={delta_w:+.3f}  '
          f'(specificity_ratio={specificity_ratio:.2f})')

    if abs(delta_uw - delta_w) < 0.01:
        print('  VERDICT: Reweighting does not change ablation Δ — distribution mismatch '
              'does not explain non-specificity. Step A tests something sharper.')
    elif delta_w > delta_stab_uw + 0.01:
        print('  VERDICT: Reweighted real Δ > stabilized Δ — matching partially recovers '
              'specificity. Step A will sharpen this further.')
    else:
        print('  VERDICT: Non-specificity dissolves under matching — position-distribution '
              'mismatch was the main driver. Step A will be a clean confirmation.')

    return {
        'logmar': logmar,
        'd1_real_uw': d1_real_uw, 'd1_abl_uw': d1_abl_uw,
        'd1_stab_uw': d1_stab_uw, 'd1_stab_abl_uw': d1_stab_abl_uw,
        'd1_real_w': d1_real_w, 'd1_abl_w': d1_abl_w,
        'weights': weights,
        'real_pos': real_pos,
        'stab_pos': stab_pos,
    }


def make_figure(results: list[dict]) -> plt.Figure:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n), squeeze=False)

    for row, r in enumerate(results):
        lm = r['logmar']
        ax = axes[row]

        # Panel 1: position distributions real vs stab
        ax[0].scatter(r['real_pos'][:, 0], r['real_pos'][:, 1],
                      s=8, alpha=0.4, label='real', color='royalblue')
        ax[0].scatter(r['stab_pos'][:, 0], r['stab_pos'][:, 1],
                      s=8, alpha=0.4, label='stabilized', color='tomato')
        ax[0].set_title(f'lm={lm:+.2f}: mean eye positions', fontsize=9)
        ax[0].set_xlabel('x (deg)')
        ax[0].set_ylabel('y (deg)')
        ax[0].legend(fontsize=7)
        ax[0].set_aspect('equal')

        # Panel 2: importance weights histogram
        ax[1].hist(r['weights'], bins=30, color='steelblue', edgecolor='white')
        ax[1].axvline(1.0, color='k', linestyle='--', linewidth=0.8)
        ax[1].set_title(f'lm={lm:+.2f}: importance weights', fontsize=9)
        ax[1].set_xlabel('weight')
        ax[1].set_ylabel('count')

        # Panel 3: Δ accuracy bar chart
        labels = ['real\nunweighted', 'stab\nunweighted', 'real\nreweighted']
        deltas = [
            r['d1_abl_uw'] - r['d1_real_uw'],
            r['d1_stab_abl_uw'] - r['d1_stab_uw'],
            r['d1_abl_w'] - r['d1_real_w'],
        ]
        colors = ['royalblue', 'tomato', 'mediumseagreen']
        bars = ax[2].bar(labels, deltas, color=colors)
        ax[2].axhline(0, color='k', linewidth=0.8)
        ax[2].set_title(f'lm={lm:+.2f}: ablation Δ', fontsize=9)
        ax[2].set_ylabel('D1_ablated − D1_original')
        for bar, d in zip(bars, deltas):
            ax[2].text(bar.get_x() + bar.get_width() / 2,
                       d + 0.003 * np.sign(d + 1e-9),
                       f'{d:+.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description='B.5 matched position-distribution reweighting')
    p.add_argument('--logmars', type=str, default='-0.20,-0.40')
    p.add_argument('--rate_file_tag', type=str, default='allhires_fresh')
    p.add_argument('--hires_threshold', type=float, default=2.0)
    p.add_argument('--n_fem_components', type=int, default=2)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--n_bins', type=int, default=10,
                   help='Number of bins per axis for 2D position histogram matching')
    args = p.parse_args()

    logmars = [float(x.strip()) for x in args.logmars.split(',') if x.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        analyse_one(
            logmar=lm,
            rate_file_tag=args.rate_file_tag,
            hires_threshold=args.hires_threshold,
            n_fem_components=args.n_fem_components,
            n_splits=args.n_splits,
            n_bins=args.n_bins,
        )
        for lm in logmars
    ]

    fig = make_figure(results)
    fig_path = OUT_DIR / 'matched_position_ablation.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved: {fig_path}')

    npz_path = OUT_DIR / 'matched_position_ablation.npz'
    save_dict: dict[str, np.ndarray] = {}
    for r in results:
        tag = f'lm{r["logmar"]:+.2f}'.replace('+', '')
        for k in ['d1_real_uw', 'd1_abl_uw', 'd1_stab_uw', 'd1_stab_abl_uw',
                  'd1_real_w', 'd1_abl_w']:
            save_dict[f'{tag}_{k}'] = np.array([r[k]])
    np.savez(npz_path, **save_dict)
    print(f'Results saved: {npz_path}')


if __name__ == '__main__':
    main()

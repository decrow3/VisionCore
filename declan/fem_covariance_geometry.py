"""
Priority 1: Per-orientation FEM Subspace Geometry and Second-Order Decoding
============================================================================

Tests whether FEM-induced covariance rotates with E orientation, and whether
that rotation carries decodable orientation information — especially in the
hyperacuity regime where mean-rate decoding (D1) is degraded.

Steps
-----
1  Per-orientation FEM subspaces  U_FEM^k for each orientation k
2  Subspace rotation check         pairwise principal angles between U_FEM^k
3  Signal alignment per orientation α^k = tr(U_FEM^k^T C_signal U_FEM^k) / tr(C_signal)
4  Second-order decoder            classify by which subspace best captures δr = r̄ − μ̄
5  Combined decoder                D1 + FEM projection features vs D1 alone

All steps are pure linear algebra on cached rate files — no model or GPU needed.

Usage
-----
python declan/fem_covariance_geometry.py
python declan/fem_covariance_geometry.py --logmars -0.20,-0.40
python declan/fem_covariance_geometry.py --condition stabilized  # sanity check
python declan/fem_covariance_geometry.py --n_fem_components 2   # default
python declan/fem_covariance_geometry.py --n_splits 10
python declan/fem_covariance_geometry.py --out_dir /path/to/out
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RATES_DIR = os.path.join(REPO_DIR, 'scripts', 'temporal_decoding', 'data', 'rates')

ORIENTATIONS = [0, 90, 180, 270]
ORI_KEYS = [f'ori{o}' for o in ORIENTATIONS]
CHANCE = 1.0 / len(ORIENTATIONS)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_rates_list(path: str) -> list:
    """Load .npz rate cache. Returns list of (T_i, N) float32 arrays."""
    d = np.load(path, allow_pickle=True)
    rates_padded = d['rates']
    lengths = d['lengths'].astype(int)
    return [rates_padded[i, :lengths[i]] for i in range(rates_padded.shape[0])]


def load_rates(logmar: float, condition: str, hires_threshold: float = 2.0,
               rates_dir: str = RATES_DIR) -> dict:
    """
    Load all four orientations for one LogMAR / condition.
    Returns dict ori_key -> list of (T_i, N) arrays.
    Raises FileNotFoundError early with a clear message if any file is missing.
    """
    use_hires = logmar < hires_threshold
    prefix = 'rates_hires' if use_hires else 'rates'
    result = {}
    for ori in ORIENTATIONS:
        fname = f'{prefix}_lm{logmar:.2f}_ori{ori}_{condition}.npz'
        path = os.path.join(rates_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached rates: {path}\n"
                "Run cache_eoptotype_rates.py first."
            )
        result[f'ori{ori}'] = _load_rates_list(path)
    n_trials = {k: len(v) for k, v in result.items()}
    n_neurons = result[ORI_KEYS[0]][0].shape[1]
    print(f"  Loaded: lm={logmar:+.2f} {condition:12s} | "
          f"N={n_neurons} neurons | trials/ori: {list(n_trials.values())}")
    return result


def time_average(rates_by_stim: dict) -> dict:
    """list of (T_i, N) -> (M, N) time-averaged rates per orientation."""
    return {k: np.stack([r.mean(axis=0) for r in v]).astype(np.float64)
            for k, v in rates_by_stim.items()}


def equalise_trials(ravg_by_stim: dict) -> dict:
    """Truncate all orientations to the same number of trials (min across oris)."""
    M_min = min(v.shape[0] for v in ravg_by_stim.values())
    return {k: v[:M_min] for k, v in ravg_by_stim.items()}


# ── Step 1: Per-orientation FEM subspaces ────────────────────────────────────

def fit_fem_subspaces(ravg_by_stim: dict, n_components: int = 2) -> dict:
    """
    For each orientation k compute C_FEM^k = Cov[r̄_k] across eye traces,
    then extract the top n_components eigenvectors U_FEM^k.

    ravg_by_stim : dict ori_key -> (M, N)
    Returns      : dict ori_key -> {'C_FEM':(N,N), 'U_FEM':(N,d), 'eigvals':(N,)}
    """
    result = {}
    for key, R in ravg_by_stim.items():
        R_c = R - R.mean(axis=0, keepdims=True)
        C = (R_c.T @ R_c) / max(R.shape[0] - 1, 1)
        C = (C + C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(C)
        U = eigvecs[:, -n_components:]   # (N, d) top eigenvectors
        result[key] = {'C_FEM': C, 'U_FEM': U, 'eigvals': eigvals}
    return result


# ── Step 2: Subspace rotation check ─────────────────────────────────────────

def principal_angle_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Cosines of principal angles between column spaces of U1 (N,d1) and U2 (N,d2).
    Returns d=min(d1,d2) values in [0,1]; 1 = identical direction.
    """
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    sv = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.clip(sv, 0.0, 1.0)


def subspace_overlap_matrix(fem_subspaces: dict) -> np.ndarray:
    """
    4×4 matrix of mean squared cosines between all pairs of U_FEM^k.
    Diagonal = 1 by definition.
    """
    n = len(ORI_KEYS)
    mat = np.eye(n)
    for i, ki in enumerate(ORI_KEYS):
        for j, kj in enumerate(ORI_KEYS):
            if i >= j:
                continue
            cos = principal_angle_cosines(
                fem_subspaces[ki]['U_FEM'],
                fem_subspaces[kj]['U_FEM'],
            )
            overlap = float(np.mean(cos ** 2))
            mat[i, j] = overlap
            mat[j, i] = overlap
    return mat


# ── Step 3: Signal alignment per orientation ─────────────────────────────────

def compute_signal_covariance(ravg_by_stim: dict) -> np.ndarray:
    """
    C_signal = covariance of per-orientation mean vectors across the 4 orientations.
    Returns (N, N).
    """
    means = np.stack([v.mean(axis=0) for v in ravg_by_stim.values()])  # (K, N)
    means -= means.mean(axis=0, keepdims=True)
    C = (means.T @ means) / max(means.shape[0] - 1, 1)
    return (C + C.T) / 2


def alignment_fractions(fem_subspaces: dict, C_signal: np.ndarray) -> tuple:
    """
    α^k = tr(U_FEM^k^T C_signal U_FEM^k) / tr(C_signal) for each orientation k.
    Returns (dict ori_key->float, float chance_level).
    """
    denom = float(np.trace(C_signal))
    d = next(iter(fem_subspaces.values()))['U_FEM'].shape[1]
    N = C_signal.shape[0]
    alphas = {}
    for key, fs in fem_subspaces.items():
        U = fs['U_FEM']
        alphas[key] = float(np.trace(U.T @ C_signal @ U)) / (denom + 1e-12)
    return alphas, float(d / N)


# ── CV helpers ────────────────────────────────────────────────────────────────

def _cv_splits(M_min: int, n_splits: int):
    """
    Yield (train_traces, test_traces) index arrays for grouped CV.
    Groups = trace indices; same trace used for all orientations, so splits
    are consistent across orientations.
    """
    groups = np.arange(M_min)
    # Dummy X/y just to drive GroupKFold
    X_dummy = np.zeros((len(ORI_KEYS) * M_min, 1))
    y_dummy = np.repeat(np.arange(len(ORI_KEYS)), M_min)
    g_dummy = np.tile(groups, len(ORI_KEYS))

    gkf = GroupKFold(n_splits=n_splits)
    for tr_all, te_all in gkf.split(X_dummy, y_dummy, groups=g_dummy):
        # Because groups tile cleanly, trace indices from ori0 stratum
        # are identical to trace indices from every other stratum (just offset).
        train_traces = tr_all[tr_all < M_min]
        test_traces = te_all[te_all < M_min]
        yield train_traces, test_traces


# ── Step 4: Second-order (covariance geometry) decoder ───────────────────────

def run_covariance_decoder(ravg: dict, n_splits: int = 5,
                           n_fem_components: int = 2) -> tuple:
    """
    Classify orientation by which U_FEM^k best captures δr = r̄ − μ̄_train.
    Assign k* = argmax_k ‖(U_FEM^k)^T δr‖² (equivalent to argmin residual
    for orthonormal U, but numerically cleaner).
    FEM subspaces fitted on training fold only.

    Returns (mean_acc, std_acc, fold_accs).
    """
    M_min = min(v.shape[0] for v in ravg.values())
    fold_accs = []

    for train_tr, test_tr in _cv_splits(M_min, n_splits):
        ravg_tr = {k: ravg[k][train_tr] for k in ORI_KEYS}
        ravg_te = {k: ravg[k][test_tr] for k in ORI_KEYS}

        # Fit subspaces and grand mean from training fold only
        fem_tr = fit_fem_subspaces(ravg_tr, n_components=n_fem_components)
        mu_grand = np.concatenate(list(ravg_tr.values()), axis=0).mean(axis=0)  # (N,)

        # Stack test data: (4*M_te, N), labels (4*M_te,)
        X_te = np.concatenate([ravg_te[k] for k in ORI_KEYS], axis=0)
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        delta_r = X_te - mu_grand  # (4*M_te, N)

        # For each candidate orientation, compute ‖(U_FEM^k)^T δr‖² — shape (4*M_te,)
        # Stack into scores matrix (4*M_te, 4)
        scores = np.column_stack([
            np.sum((delta_r @ fem_tr[k]['U_FEM']) ** 2, axis=1)
            for k in ORI_KEYS
        ])
        y_pred = np.argmax(scores, axis=1)
        fold_accs.append(float(np.mean(y_pred == y_te)))

    fold_accs = np.array(fold_accs)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


# ── Step 4b: D1 (mean-rate) decoder ──────────────────────────────────────────

def run_d1_decoder(ravg: dict, n_splits: int = 5) -> tuple:
    """
    Model A / D1: logistic regression on time-averaged rate vector.
    Returns (mean_acc, std_acc, fold_accs).
    """
    M_min = min(v.shape[0] for v in ravg.values())
    fold_accs = []

    for train_tr, test_tr in _cv_splits(M_min, n_splits):
        X_tr = np.concatenate([ravg[k][train_tr] for k in ORI_KEYS], axis=0)
        y_tr = np.repeat(np.arange(len(ORI_KEYS)), train_tr.shape[0])
        X_te = np.concatenate([ravg[k][test_tr] for k in ORI_KEYS], axis=0)
        y_te = np.repeat(np.arange(len(ORI_KEYS)), test_tr.shape[0])

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(X_tr), y_tr)
        fold_accs.append(clf.score(sc.transform(X_te), y_te))

    fold_accs = np.array(fold_accs)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


# ── Step 5: Combined decoder (D1 + FEM geometry features) ────────────────────

def run_combined_decoder(ravg: dict, n_splits: int = 5,
                         n_fem_components: int = 2) -> tuple:
    """
    Classify from [r̄ (N) || (U_FEM^k)^T r̄ for k=0..3 (4*d features)].
    FEM subspaces fitted on training fold only.
    If combined accuracy > D1, FEM geometry adds information beyond mean rate.

    Returns (mean_acc, std_acc, fold_accs).
    """
    M_min = min(v.shape[0] for v in ravg.values())
    fold_accs = []

    for train_tr, test_tr in _cv_splits(M_min, n_splits):
        ravg_tr = {k: ravg[k][train_tr] for k in ORI_KEYS}
        ravg_te = {k: ravg[k][test_tr] for k in ORI_KEYS}

        fem_tr = fit_fem_subspaces(ravg_tr, n_components=n_fem_components)

        def _make_features(ravg_dict: dict) -> tuple:
            """Return (X_feat, y) for the combined decoder."""
            feat_list, y_list = [], []
            for label, key in enumerate(ORI_KEYS):
                R = ravg_dict[key]  # (M, N)
                # FEM projections: concatenate 2D coordinates for all 4 orientations
                proj = np.concatenate(
                    [R @ fem_tr[ki]['U_FEM'] for ki in ORI_KEYS], axis=1
                )  # (M, 4*d)
                feat_list.append(np.concatenate([R, proj], axis=1))
                y_list.append(np.full(R.shape[0], label, dtype=int))
            return np.concatenate(feat_list), np.concatenate(y_list)

        X_tr, y_tr = _make_features(ravg_tr)
        X_te, y_te = _make_features(ravg_te)

        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        clf.fit(sc.fit_transform(X_tr), y_tr)
        fold_accs.append(clf.score(sc.transform(X_te), y_te))

    fold_accs = np.array(fold_accs)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


# ── Per-LogMAR analysis ───────────────────────────────────────────────────────

def analyse_logmar(logmar: float, condition: str, n_splits: int,
                   n_fem_components: int, rates_dir: str) -> dict:
    """Run all five steps for one LogMAR / condition. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"LogMAR {logmar:+.2f}  condition={condition}")
    print('='*60)

    rates = load_rates(logmar, condition, rates_dir=rates_dir)
    ravg = equalise_trials(time_average(rates))
    M, N = next(iter(ravg.values())).shape
    print(f"  Using M={M} trials per orientation, N={N} neurons")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    print("\nStep 1: Per-orientation FEM subspaces")
    fem = fit_fem_subspaces(ravg, n_components=n_fem_components)
    for key, fs in fem.items():
        top2 = fs['eigvals'][-n_fem_components:][::-1]
        total = fs['eigvals'].sum()
        pct = 100 * top2.sum() / (total + 1e-12)
        print(f"  {key}: top-{n_fem_components} eigvals = {top2}, "
              f"variance explained = {pct:.1f}%")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    print("\nStep 2: Subspace rotation check")
    overlap = subspace_overlap_matrix(fem)
    off_diag = overlap[np.triu_indices(len(ORI_KEYS), k=1)]
    print(f"  Overlap matrix (mean sq cosine):")
    header = "          " + "  ".join(f"{k:>8}" for k in ORI_KEYS)
    print(f"  {header}")
    for i, ki in enumerate(ORI_KEYS):
        row = "  ".join(f"{overlap[i,j]:8.4f}" for j in range(len(ORI_KEYS)))
        print(f"  {ki:>8}: {row}")
    print(f"  Off-diagonal mean  = {off_diag.mean():.4f}  "
          f"(1.0 = identical, 0.0 = orthogonal)")
    print(f"  Off-diagonal range = [{off_diag.min():.4f}, {off_diag.max():.4f}]")

    # ── Step 3 ───────────────────────────────────────────────────────────────
    print("\nStep 3: Signal alignment per orientation")
    C_signal = compute_signal_covariance(ravg)
    alphas, chance = alignment_fractions(fem, C_signal)
    print(f"  Chance level α_chance = {chance:.4f}  (= {n_fem_components}/N)")
    for key, a in alphas.items():
        print(f"  {key}: α = {a:.4f}  (×chance = {a/chance:.2f})")
    print(f"  Mean α = {np.mean(list(alphas.values())):.4f}")

    # ── Steps 4 & 5: decoders ────────────────────────────────────────────────
    print(f"\nStep 4: Second-order (covariance geometry) decoder  [n_splits={n_splits}]")
    cov_acc, cov_std, cov_folds = run_covariance_decoder(ravg, n_splits, n_fem_components)
    print(f"  Covariance decoder: {cov_acc:.3f} ± {cov_std:.3f}  "
          f"(folds: {cov_folds.round(3)})")

    print(f"\nStep 4b: D1 (mean-rate) decoder")
    d1_acc, d1_std, d1_folds = run_d1_decoder(ravg, n_splits)
    print(f"  D1 decoder:         {d1_acc:.3f} ± {d1_std:.3f}  "
          f"(folds: {d1_folds.round(3)})")

    print(f"\nStep 5: Combined decoder (D1 + FEM geometry features)")
    comb_acc, comb_std, comb_folds = run_combined_decoder(ravg, n_splits, n_fem_components)
    print(f"  Combined decoder:   {comb_acc:.3f} ± {comb_std:.3f}  "
          f"(folds: {comb_folds.round(3)})")

    gain_cov = cov_acc - d1_acc
    gain_comb = comb_acc - d1_acc
    print(f"\n  Covariance decoder − D1 = {gain_cov:+.3f}")
    print(f"  Combined decoder − D1   = {gain_comb:+.3f}")

    return {
        'logmar': logmar,
        'condition': condition,
        'M': M, 'N': N,
        'fem_subspaces': fem,
        'overlap_matrix': overlap,
        'off_diag_mean': float(off_diag.mean()),
        'C_signal': C_signal,
        'alphas': alphas,
        'alpha_chance': chance,
        'd1_acc': d1_acc, 'd1_std': d1_std, 'd1_folds': d1_folds,
        'cov_acc': cov_acc, 'cov_std': cov_std, 'cov_folds': cov_folds,
        'comb_acc': comb_acc, 'comb_std': comb_std, 'comb_folds': comb_folds,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results: list) -> plt.Figure:
    """
    2-row × 3-col GridSpec figure:
      Row 0: overlap matrix LM0 | overlap matrix LM1 | FEM eigenvalue spectra
      Row 1: alignment fractions α^k | decoder comparison | (empty)
    """
    n_lm = len(results)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('FEM Covariance Geometry — Priority 1', fontsize=13, fontweight='bold')

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: overlap matrices ───────────────────────────────────────────────
    ovlp_axes = [fig.add_subplot(gs[0, col]) for col in range(min(n_lm, 2))]
    for i, (ax, r) in enumerate(zip(ovlp_axes, results)):
        im = ax.imshow(r['overlap_matrix'], vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(len(ORI_KEYS)))
        ax.set_yticks(range(len(ORI_KEYS)))
        ax.set_xticklabels([str(o) for o in ORIENTATIONS], fontsize=8)
        ax.set_yticklabels([str(o) for o in ORIENTATIONS], fontsize=8)
        ax.set_title(f'Subspace overlap\nLM={r["logmar"]:+.2f}  {r["condition"]}',
                     fontsize=9)
        for ii in range(len(ORI_KEYS)):
            for jj in range(len(ORI_KEYS)):
                ax.text(jj, ii, f'{r["overlap_matrix"][ii, jj]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if r['overlap_matrix'][ii, jj] > 0.6 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ── Row 0 col 2: FEM eigenvalue spectra ──────────────────────────────────
    ax_eig = fig.add_subplot(gs[0, 2])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--']
    n_show = 20
    for j, r in enumerate(results):
        ls = linestyles[j % len(linestyles)]
        for ci, key in enumerate(ORI_KEYS):
            ev = r['fem_subspaces'][key]['eigvals']
            ev_desc = np.maximum(ev[-n_show:][::-1], 1e-12)
            ax_eig.plot(np.arange(1, n_show + 1), ev_desc,
                        color=colors[ci], linestyle=ls, alpha=0.75, linewidth=1.2,
                        label=f'{key}' if j == 0 else None)
    ax_eig.set_yscale('log')
    ax_eig.set_xlabel('Component', fontsize=9)
    ax_eig.set_ylabel('Eigenvalue', fontsize=9)
    ax_eig.set_title('FEM eigenvalue spectra\n(solid=LM0, dashed=LM1)', fontsize=9)
    ax_eig.legend(fontsize=7, ncol=2)
    ax_eig.grid(True, alpha=0.3)

    # ── Row 1 col 0: alignment fractions ─────────────────────────────────────
    ax_alpha = fig.add_subplot(gs[1, 0])
    x = np.arange(len(ORI_KEYS))
    width = 0.35
    for j, r in enumerate(results):
        vals = [r['alphas'][k] for k in ORI_KEYS]
        offset = (j - 0.5 * (n_lm - 1)) * width
        ax_alpha.bar(x + offset, vals, width, label=f'LM={r["logmar"]:+.2f}', alpha=0.8)
    ax_alpha.axhline(results[0]['alpha_chance'], color='k', linestyle='--',
                     linewidth=1, label=f'chance={results[0]["alpha_chance"]:.4f}')
    ax_alpha.set_xticks(x)
    ax_alpha.set_xticklabels([str(o) + '°' for o in ORIENTATIONS], fontsize=9)
    ax_alpha.set_ylabel('Alignment fraction α', fontsize=9)
    ax_alpha.set_title('Signal alignment per orientation', fontsize=9)
    ax_alpha.legend(fontsize=8)
    ax_alpha.grid(True, axis='y', alpha=0.3)

    # ── Row 1 col 1: decoder comparison ──────────────────────────────────────
    ax_dec = fig.add_subplot(gs[1, 1])
    decoder_labels = ['D1\n(mean rate)', 'Cov\n(2nd order)', 'Combined\n(D1+cov)']
    x_dec = np.arange(len(decoder_labels))
    width_d = 0.25
    for j, r in enumerate(results):
        accs = [r['d1_acc'], r['cov_acc'], r['comb_acc']]
        stds = [r['d1_std'], r['cov_std'], r['comb_std']]
        offset = (j - 0.5 * (n_lm - 1)) * width_d
        ax_dec.bar(x_dec + offset, accs, width_d, yerr=stds, capsize=3,
                   label=f'LM={r["logmar"]:+.2f} {r["condition"]}', alpha=0.8)
    ax_dec.axhline(CHANCE, color='k', linestyle='--', linewidth=1,
                   label=f'chance={CHANCE:.2f}')
    ax_dec.set_xticks(x_dec)
    ax_dec.set_xticklabels(decoder_labels, fontsize=9)
    ax_dec.set_ylabel('Accuracy', fontsize=9)
    ax_dec.set_ylim(0, 1.05)
    ax_dec.set_title('Decoder comparison\n(grouped CV)', fontsize=9)
    ax_dec.legend(fontsize=8)
    ax_dec.grid(True, axis='y', alpha=0.3)

    # ── Row 1 col 2: off-diagonal overlap summary bar ─────────────────────────
    ax_ovlp = fig.add_subplot(gs[1, 2])
    lm_labels = [f'LM={r["logmar"]:+.2f}\n{r["condition"]}' for r in results]
    off_diag_means = [r['off_diag_mean'] for r in results]
    bars = ax_ovlp.bar(lm_labels, off_diag_means, color='steelblue', alpha=0.8)
    ax_ovlp.axhline(1.0, color='k', linestyle='--', linewidth=1, label='identical (1.0)')
    ax_ovlp.set_ylim(0, 1.1)
    ax_ovlp.set_ylabel('Mean off-diagonal overlap\n(mean sq cosine)', fontsize=9)
    ax_ovlp.set_title('FEM subspace rotation\n(lower = more rotation)', fontsize=9)
    for bar, val in zip(bars, off_diag_means):
        ax_ovlp.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax_ovlp.legend(fontsize=8)
    ax_ovlp.grid(True, axis='y', alpha=0.3)

    return fig


# ── Summary printout ──────────────────────────────────────────────────────────

def print_summary(results: list) -> None:
    """Print a concise interpretation table."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'LogMAR':>8}  {'cond':>12}  {'D1':>6}  {'Cov':>6}  {'Comb':>6}  "
    header += f"{'Cov-D1':>7}  {'Comb-D1':>8}  {'α_mean':>7}  {'OvlpOff':>8}"
    print(header)
    print("-" * 70)
    for r in results:
        alpha_mean = float(np.mean(list(r['alphas'].values())))
        print(
            f"{r['logmar']:>8.2f}  {r['condition']:>12s}  "
            f"{r['d1_acc']:>6.3f}  {r['cov_acc']:>6.3f}  {r['comb_acc']:>6.3f}  "
            f"{r['cov_acc']-r['d1_acc']:>+7.3f}  "
            f"{r['comb_acc']-r['d1_acc']:>+8.3f}  "
            f"{alpha_mean:>7.4f}  "
            f"{r['off_diag_mean']:>8.4f}"
        )
    print()
    print("Columns:")
    print("  Cov-D1   : covariance geometry decoder accuracy minus D1")
    print("             >0 means second-order geometry alone beats mean rate")
    print("  Comb-D1  : combined (D1 + FEM features) minus D1 alone")
    print("             >0 means FEM geometry adds information beyond D1")
    print("  α_mean   : mean signal alignment of FEM subspace across orientations")
    print("             > chance means FEM noise falls in signal directions")
    print("  OvlpOff  : mean off-diagonal subspace overlap (mean sq cosine)")
    print("             < 1.0 means FEM subspaces rotate with orientation")
    print()

    # Interpretation
    for r in results:
        lm = r['logmar']
        alpha_mean = float(np.mean(list(r['alphas'].values())))
        cov_gain = r['cov_acc'] - r['d1_acc']
        comb_gain = r['comb_acc'] - r['d1_acc']
        ovlp = r['off_diag_mean']
        print(f"LogMAR {lm:+.2f} ({r['condition']}):")
        rotates = ovlp < 0.9
        print(f"  Subspace rotation: {'YES' if rotates else 'NO / WEAK'}  "
              f"(off-diag overlap={ovlp:.3f})")
        print(f"  Alignment vs chance: α_mean={alpha_mean:.4f}  "
              f"chance={r['alpha_chance']:.4f}  "
              f"ratio={alpha_mean/r['alpha_chance']:.2f}x")
        if comb_gain > 0.01:
            print(f"  Combined decoder adds {comb_gain:+.3f} over D1 — "
                  "FEM geometry carries orientation info beyond mean rate")
        elif comb_gain > 0:
            print(f"  Combined decoder adds {comb_gain:+.3f} over D1 — marginal")
        else:
            print(f"  Combined decoder: {comb_gain:+.3f} vs D1 — no additional info")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='FEM covariance geometry analysis (Priority 1)')
    p.add_argument('--logmars', type=str, default='-0.20,-0.40',
                   help='Comma-separated LogMAR values (default: -0.20,-0.40)')
    p.add_argument('--condition', type=str, default='real',
                   help='FEM condition: real | stabilized (default: real)')
    p.add_argument('--n_splits', type=int, default=5,
                   help='CV folds (default: 5)')
    p.add_argument('--n_fem_components', type=int, default=2,
                   help='FEM subspace dimensionality (default: 2)')
    p.add_argument('--rates_dir', type=str, default=RATES_DIR,
                   help='Directory containing cached rate .npz files')
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(SCRIPT_DIR, 'fem_covariance_geometry_results'),
                   help='Output directory for figure and results .npz')
    p.add_argument('--no_figure', action='store_true',
                   help='Skip figure generation')
    return p.parse_args()


def main():
    args = parse_args()

    logmars = [float(x) for x in args.logmars.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"FEM Covariance Geometry Analysis")
    print(f"LogMARs: {logmars}  condition: {args.condition}")
    print(f"n_splits={args.n_splits}  n_fem_components={args.n_fem_components}")
    print(f"Rates dir: {args.rates_dir}")
    print(f"Output dir: {args.out_dir}")

    results = []
    for lm in logmars:
        r = analyse_logmar(
            logmar=lm,
            condition=args.condition,
            n_splits=args.n_splits,
            n_fem_components=args.n_fem_components,
            rates_dir=args.rates_dir,
        )
        results.append(r)

    print_summary(results)

    # Save numeric results (exclude large matrices to keep file small)
    out_npz = os.path.join(args.out_dir, f'fem_geometry_{args.condition}.npz')
    save_dict = {}
    for r in results:
        tag = f'lm{r["logmar"]:+.2f}'
        save_dict.update({
            f'{tag}_d1_acc': r['d1_acc'],
            f'{tag}_d1_folds': r['d1_folds'],
            f'{tag}_cov_acc': r['cov_acc'],
            f'{tag}_cov_folds': r['cov_folds'],
            f'{tag}_comb_acc': r['comb_acc'],
            f'{tag}_comb_folds': r['comb_folds'],
            f'{tag}_overlap_matrix': r['overlap_matrix'],
            f'{tag}_alphas': np.array([r['alphas'][k] for k in ORI_KEYS]),
            f'{tag}_alpha_chance': r['alpha_chance'],
            f'{tag}_off_diag_mean': r['off_diag_mean'],
        })
    np.savez(out_npz, **save_dict)
    print(f"Results saved: {out_npz}")

    if not args.no_figure:
        fig = make_figure(results)
        out_fig = os.path.join(args.out_dir, f'fem_geometry_{args.condition}.png')
        fig.savefig(out_fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved: {out_fig}")

    print("\nDone.")


if __name__ == '__main__':
    main()

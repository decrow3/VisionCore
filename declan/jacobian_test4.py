"""
Test 4: Full Mechanistic Model  C_FEM ≈ J · Σ_eye · Jᵀ
=======================================================

Given that Test 3 alignment is ~0.40–0.60 (genuine Jacobian structure, but not
complete), this test asks: how much of C_FEM is quantitatively explained by the
first-order pushforward model, and what is the structure of the residual?

Inputs expected (saved by Test 3 / existing translation_covariance pipeline):
  - J_per_condition:  dict[(orientation, logmar)] -> np.ndarray [N, 2]
  - C_FEM_per_condition: dict[(orientation, logmar)] -> np.ndarray [N, N]
  - U_pca2_per_condition: same keys -> np.ndarray [N, 2]
  - eye_traces: np.ndarray [M, T, 2]  (full traces, not trial means)

Outputs:
  - Per-condition: C_predicted, residual, eigenvalue ratios, residual rank
  - Summary table across conditions
  - Figures: predicted vs empirical eigenspectra, residual heatmaps
"""

import numpy as np
from numpy.linalg import eigh, matrix_rank, norm
from scipy.linalg import subspace_angles
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 0.  Helpers (reuse from translation_covariance where possible)
# ---------------------------------------------------------------------------

def alignment_score(U1: np.ndarray, U2: np.ndarray) -> float:
    """Mean cos² of principal angles between two 2D subspaces."""
    th = subspace_angles(U1, U2)
    return float(np.mean(np.cos(th) ** 2))


def capture_fraction(U: np.ndarray, Sigma: np.ndarray) -> float:
    """Fraction of Sigma's variance captured by subspace U."""
    num = np.trace(U.T @ Sigma @ U)
    den = np.trace(Sigma)
    return float(num / den) if den > 0 and np.isfinite(den) else np.nan


def top2_pca(Sigma: np.ndarray):
    """Top-2 eigenvectors of a symmetric matrix (descending order)."""
    w, V = eigh(Sigma)
    idx = np.argsort(w)[::-1]
    return w[idx][:2], V[:, idx][:, :2]


def frobenius_residual(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F / ||A||_F"""
    return float(norm(A - B, 'fro') / (norm(A, 'fro') + 1e-12))


def effective_rank(M: np.ndarray, threshold: float = 0.01) -> int:
    """Number of eigenvalues > threshold * max eigenvalue."""
    w = np.abs(np.linalg.eigvalsh(M))
    return int(np.sum(w > threshold * w.max()))


# ---------------------------------------------------------------------------
# 1.  Σ_eye estimation — three variants
# ---------------------------------------------------------------------------

def estimate_sigma_eye(eye_traces: np.ndarray,
                       durations: np.ndarray = None,
                       gru_window: int = 8) -> dict:
    """
    Estimate the effective eye-position covariance under three assumptions.

    Parameters
    ----------
    eye_traces : [M, T_max, 2]  NaN-padded traces (degrees)
    durations  : [M] int  valid frame count per trace (if None, uses T_max for all)
    gru_window : int  GRU integration window in frames (for exponential weighting)

    Returns
    -------
    dict with keys:
        'trial_mean'   : [2,2] covariance of per-trace mean positions
        'per_frame'    : [2,2] covariance of all per-frame deviations (pooled M×T_i)
        'gru_weighted' : [2,2] per-frame cov weighted by exponential decay toward
                          the most-recent frame (approximates GRU temporal integration)
    """
    M, T_max, _ = eye_traces.shape
    if durations is None:
        durations = np.full(M, T_max, dtype=int)

    # 1a. Trial-mean: covariance of each trace's centroid across M trials
    trial_means = np.stack([
        eye_traces[i, :int(durations[i])].mean(0) for i in range(M)
    ])                                                  # [M, 2]
    Sigma_trial = np.cov(trial_means.T)                 # [2, 2]

    # 1b. Per-frame deviations: pool all M×T_i valid (frame, 2) pairs,
    #     each deviation taken relative to that trace's centroid.
    dev_list = []
    for i in range(M):
        ep = eye_traces[i, :int(durations[i])]          # [T_i, 2]
        dev_list.append(ep - ep.mean(0, keepdims=True))
    dev_flat = np.concatenate(dev_list, axis=0)         # [sum(T_i), 2]
    Sigma_frame = np.cov(dev_flat.T)                    # [2, 2]

    # 1c. GRU-decay-weighted: exponential kernel with τ = gru_window frames,
    #     most weight on the most recent frame.  Per trace, compute the
    #     kernel-weighted mean and then the weighted outer-product covariance.
    tau = float(gru_window)
    Sigma_gru = np.zeros((2, 2))
    total_weight = 0.0
    for i in range(M):
        T_i = int(durations[i])
        ep = eye_traces[i, :T_i]                        # [T_i, 2]
        # Weight vector: lag 0 (most recent) = 1, lag k = exp(-k/tau)
        lags = np.arange(T_i - 1, -1, -1, dtype=float) # [T_i] — 0 is current frame
        w = np.exp(-lags / tau)
        w /= w.sum()
        mu_w = (w[:, None] * ep).sum(0)                 # weighted mean
        d = ep - mu_w[None, :]                          # [T_i, 2]
        Sigma_gru += (w[:, None] * d).T @ d             # weighted outer product
        total_weight += 1.0
    Sigma_gru /= total_weight                           # average over traces

    return {
        'trial_mean':   Sigma_trial,
        'per_frame':    Sigma_frame,
        'gru_weighted': Sigma_gru,
    }


# ---------------------------------------------------------------------------
# 2.  Per-condition mechanistic model evaluation
# ---------------------------------------------------------------------------

def evaluate_mechanistic_model(
    J: np.ndarray,
    C_FEM: np.ndarray,
    U_pca2: np.ndarray,
    Sigma_eye_variants: dict,
    label: str = '',
) -> dict:
    """
    For one (orientation, logmar) condition, evaluate C_predicted = J Σ J^T
    under each Σ_eye variant and compare to C_FEM.

    Parameters
    ----------
    J       : [N, 2]  analytic Jacobian from Test 3
    C_FEM   : [N, N]  empirical FEM covariance
    U_pca2  : [N, 2]  top-2 empirical FEM eigenvectors
    Sigma_eye_variants : output of estimate_sigma_eye()
    label   : human-readable identifier

    Returns
    -------
    dict with per-Sigma results plus summary metrics
    """
    results = {'label': label, 'variants': {}}

    # Empirical spectrum (reference)
    w_emp, V_emp = top2_pca(C_FEM)
    results['empirical_top2_eigenvalues'] = w_emp
    results['empirical_effective_rank'] = effective_rank(C_FEM)

    for sigma_name, Sigma_eye in Sigma_eye_variants.items():

        # Predicted covariance
        C_pred = J @ Sigma_eye @ J.T          # [N, N]
        w_pred, V_pred = top2_pca(C_pred)

        # 1. Subspace alignment: does U_jac predict U_pca2?
        U_pred = V_pred                        # [N, 2]
        align = alignment_score(U_pred, U_pca2)

        # 2. Eigenvalue ratio: does the scale match?
        ev_ratio = float(w_pred[0] / (w_emp[0] + 1e-12))

        # 3. Frobenius residual: total unexplained variance fraction
        fro_res = frobenius_residual(C_FEM, C_pred)

        # 4. Residual rank: how many dimensions does the GRU add?
        R = C_FEM - C_pred
        res_rank = effective_rank(R)

        # 5. Capture: what fraction of C_FEM variance lies in span(J)?
        U_J, _ = np.linalg.qr(J)
        cap = capture_fraction(U_J, C_FEM)

        # 6. Optimal scalar: least-squares alpha such that alpha * C_pred ≈ C_FEM.
        #    alpha_opt = tr(C_FEM C_pred) / tr(C_pred²)
        #    If alpha_opt * EV-ratio ≈ 1, the shape is right but the scale is wrong
        #    by a known factor — interpretable as GRU temporal gain attenuation.
        tr_cross = float(np.trace(C_FEM @ C_pred))
        tr_pred2 = float(np.trace(C_pred @ C_pred))
        alpha_opt = tr_cross / (tr_pred2 + 1e-30)

        results['variants'][sigma_name] = {
            'C_predicted':        C_pred,
            'U_predicted':        U_pred,
            'pred_top2_eigenvalues': w_pred,
            'alignment':          align,      # predicted subspace vs empirical
            'eigenvalue_ratio':   ev_ratio,   # pred_lambda1 / emp_lambda1
            'frobenius_residual': fro_res,    # ||C_FEM - C_pred||_F / ||C_FEM||_F
            'residual_rank':      res_rank,   # rank of (C_FEM - C_pred)
            'capture_fraction':   cap,        # fraction of C_FEM in span(J)
            'alpha_opt':          alpha_opt,  # optimal scale factor; ~1/EV-ratio if shape is right
        }

    return results


# ---------------------------------------------------------------------------
# 3.  Main pipeline
# ---------------------------------------------------------------------------

def run_test4(
    J_per_condition: dict,
    C_FEM_per_condition: dict,
    U_pca2_per_condition: dict,
    eye_traces: np.ndarray,
    durations: np.ndarray = None,
    gru_window: int = 8,
    output_dir: str = '.',
):
    """
    Run the full Test 4 pipeline.

    Parameters
    ----------
    J_per_condition      : {(orientation_deg, logmar): np.ndarray [N,2]}
    C_FEM_per_condition  : {(orientation_deg, logmar): np.ndarray [N,N]}
    U_pca2_per_condition : {(orientation_deg, logmar): np.ndarray [N,2]}
    eye_traces           : [M, T_max, 2]  NaN-padded traces
    durations            : [M] int  valid frame counts (if None, uses T_max for all)
    gru_window           : GRU integration window (frames)
    output_dir           : where to save figures and summary

    Returns
    -------
    all_results          : list of per-condition result dicts
    Sigma_eye_variants   : dict of [2,2] covariance matrices
    """
    os.makedirs(output_dir, exist_ok=True)

    # Estimate Σ_eye once (shared across conditions)
    Sigma_eye_variants = estimate_sigma_eye(eye_traces, durations=durations,
                                            gru_window=gru_window)
    print("Σ_eye estimates:")
    for name, S in Sigma_eye_variants.items():
        evals = np.linalg.eigvalsh(S)[::-1]
        print(f"  {name}:  trace={np.trace(S):.4f},  "
              f"eigenvalues=[{evals[0]:.4f}, {evals[1]:.4f}],  "
              f"anisotropy={evals[0]/evals[1]:.2f}")

    all_results = []
    for key in sorted(J_per_condition.keys()):
        ori, logmar = key
        label = f"ori={ori}°  logmar={logmar:.2f}"

        res = evaluate_mechanistic_model(
            J=J_per_condition[key],
            C_FEM=C_FEM_per_condition[key],
            U_pca2=U_pca2_per_condition[key],
            Sigma_eye_variants=Sigma_eye_variants,
            label=label,
        )
        all_results.append((key, res))

    # Print summary table
    _print_summary(all_results, Sigma_eye_variants)

    # Figures
    plot_eigenspectra_with_empirical(all_results, C_FEM_per_condition,
                                     Sigma_eye_variants, output_dir)
    _plot_residual_summary(all_results, Sigma_eye_variants, output_dir)

    return all_results, Sigma_eye_variants


# ---------------------------------------------------------------------------
# 4.  Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_results, Sigma_eye_variants, primary: str = None):
    sigma_names = list(Sigma_eye_variants.keys())
    if primary is None or primary not in sigma_names:
        primary = sigma_names[0]

    print("\n" + "="*100)
    print(f"TEST 4 SUMMARY  (primary Σ_eye = {primary})")
    print("="*100)
    header = f"{'Condition':<30} {'Align':>7} {'EV-ratio':>9} {'alpha_opt':>10} {'Frob-res':>9} "
    header += f"{'Res-rank':>9} {'Cap-frac':>9}"
    print(header)
    print("-" * 100)

    for key, res in all_results:
        v = res['variants'][primary]
        print(
            f"{res['label']:<30} "
            f"{v['alignment']:>7.3f} "
            f"{v['eigenvalue_ratio']:>9.3f} "
            f"{v['alpha_opt']:>10.4f} "
            f"{v['frobenius_residual']:>9.3f} "
            f"{v['residual_rank']:>9d} "
            f"{v['capture_fraction']:>9.3f}"
        )

    print("-" * 100)
    print("\nSensitivity to Σ_eye choice (alignment, Frobenius residual):")
    print(f"{'Condition':<30}", end='')
    for name in sigma_names:
        print(f"  {name[:10]:>10}", end='')
    print()

    for key, res in all_results:
        print(f"{res['label']:<30}", end='')
        for name in sigma_names:
            v = res['variants'][name]
            print(f"  {v['alignment']:.3f}/{v['frobenius_residual']:.3f}", end='')
        print()


# ---------------------------------------------------------------------------
# 5.  Figures
# ---------------------------------------------------------------------------


def plot_eigenspectra_with_empirical(
    all_results,
    C_FEM_per_condition: dict,
    Sigma_eye_variants: dict,
    output_dir: str,
    primary: str = 'per_frame',
):
    """
    Proper version that takes C_FEM_per_condition directly.
    Call this from the main script after run_test4().
    """
    conditions = all_results
    n_cond = len(conditions)
    ncols = (n_cond + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7))
    axes = np.array(axes).flatten()

    for ax, (key, res) in zip(axes, conditions):
        C_FEM = C_FEM_per_condition[key]
        C_pred = res['variants'][primary]['C_predicted']

        w_emp  = np.sort(np.linalg.eigvalsh(C_FEM))[::-1]
        w_pred = np.sort(np.linalg.eigvalsh(C_pred))[::-1]
        w_res  = np.sort(np.linalg.eigvalsh(C_FEM - C_pred))[::-1]

        ranks = np.arange(1, len(w_emp) + 1)
        ax.semilogy(ranks, w_emp,  'k-',  lw=1.5, label='Empirical C_FEM')
        ax.semilogy(ranks, w_pred, 'b--', lw=1.5, label='Predicted J Σ Jᵀ')
        ax.semilogy(ranks, np.abs(w_res), 'r:', lw=1.0, label='Residual', alpha=0.7)
        ax.axvline(2, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(res['label'], fontsize=9)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Eigenvalue')
        ax.legend(fontsize=6)

        v = res['variants'][primary]
        ax.text(0.97, 0.97,
                f"align={v['alignment']:.2f}\nFrob={v['frobenius_residual']:.2f}\n"
                f"EV-r={v['eigenvalue_ratio']:.2f}\nres-rank={v['residual_rank']}",
                transform=ax.transAxes, va='top', ha='right', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    for ax in axes[n_cond:]:
        ax.set_visible(False)

    fig.suptitle('Test 4: Empirical vs Predicted Eigenspectra  '
                 f'(Σ_eye = {primary})', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test4_eigenspectra_full.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/test4_eigenspectra_full.png")


def _plot_residual_summary(all_results, Sigma_eye_variants, output_dir):
    """
    Bar chart: Frobenius residual and alignment across conditions,
    for each Σ_eye variant. Reveals whether the choice of Σ_eye matters.
    """
    sigma_names = list(Sigma_eye_variants.keys())
    labels = [res['label'] for _, res in all_results]
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i, name in enumerate(sigma_names):
        aligns = [res['variants'][name]['alignment']
                  for _, res in all_results]
        frobs  = [res['variants'][name]['frobenius_residual']
                  for _, res in all_results]
        ax1.bar(x + i * width, aligns, width, label=name, alpha=0.8)
        ax2.bar(x + i * width, frobs,  width, label=name, alpha=0.8)

    ax1.axhline(0.7, color='k', ls='--', lw=0.8, label='threshold (0.70)')
    ax1.set_title('Subspace alignment  (predicted vs empirical)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('cos² alignment')
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)

    ax2.set_title('Frobenius residual  ||C_FEM − C_pred||_F / ||C_FEM||_F')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Residual fraction')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test4_residual_summary.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/test4_residual_summary.png")


# ---------------------------------------------------------------------------
# 6.  Residual decomposition — the key View B diagnostic
# ---------------------------------------------------------------------------

def decompose_residual(
    J: np.ndarray,
    C_FEM: np.ndarray,
    Sigma_eye: np.ndarray,
    n_components: int = 10,
) -> dict:
    """
    Decompose the residual R = C_FEM - C_predicted into its principal modes.

    The residual's eigenspectrum tells you:
    - How many dimensions the GRU adds beyond the Jacobian model (View B rank)
    - Whether the residual is structured (View B) or noise-like (flat spectrum)
    - Whether the residual directions correlate with J's column space
      (meaning J is right direction but wrong magnitude)
      or are orthogonal (genuinely new GRU-induced modes)

    Parameters
    ----------
    J         : [N, 2]  analytic Jacobian
    C_FEM     : [N, N]  empirical covariance
    Sigma_eye : [2, 2]  effective eye covariance
    n_components : number of residual eigenvectors to return

    Returns
    -------
    dict with:
        'residual_eigenvalues'  : top n_components eigenvalues of R
        'residual_eigenvectors' : corresponding eigenvectors [N, n_components]
        'overlap_with_J'        : capture_fraction(U_J, R) — View B vs overlap error
        'residual_is_structured': bool (True if top eigenvalue >> mean)
    """
    C_pred = J @ Sigma_eye @ J.T
    R = C_FEM - C_pred

    w_R, V_R = eigh(R)
    idx = np.argsort(w_R)[::-1]
    w_R = w_R[idx]
    V_R = V_R[:, idx]

    # SNR and structure test use only positive eigenvalues of R to avoid sign artifacts
    w_R_pos = np.maximum(w_R, 0.0)
    top_ev  = w_R_pos[0]
    mean_ev = w_R_pos.mean() + 1e-12
    is_structured = bool(top_ev > 5 * mean_ev)

    # PSD projection of residual (zero out negative eigenvalues) for overlap computation
    R_psd = V_R @ np.diag(w_R_pos) @ V_R.T
    U_J, _ = np.linalg.qr(J)
    overlap = capture_fraction(U_J, R_psd)

    # Where do the top residual directions point relative to J?
    cos_sq_with_J = []
    for k in range(min(n_components, V_R.shape[1])):
        v = V_R[:, k:k+1]
        c = float((v.T @ U_J @ U_J.T @ v).squeeze())
        cos_sq_with_J.append(c)

    return {
        'residual_eigenvalues':   w_R[:n_components],
        'residual_eigenvectors':  V_R[:, :n_components],
        'overlap_with_J':         overlap,
        'residual_is_structured': is_structured,
        'top_ev_snr':             float(top_ev / (mean_ev + 1e-12)),
        'cos_sq_resid_dirs_with_J': cos_sq_with_J,
    }


def print_residual_interpretation(decomp: dict, label: str = ''):
    """Human-readable interpretation of decompose_residual output."""
    print(f"\nResidual decomposition — {label}")
    print(f"  Top eigenvalue SNR (structured?): {decomp['top_ev_snr']:.1f}  "
          f"({'structured' if decomp['residual_is_structured'] else 'flat/noise-like'})")
    print(f"  Residual overlap with span(J):    {decomp['overlap_with_J']:.3f}")
    print(f"  Top residual eigenvalues: "
          f"{np.array2string(decomp['residual_eigenvalues'][:5], precision=4)}")
    print(f"  cos² of top-5 residual dirs with J: "
          f"{np.array2string(np.array(decomp['cos_sq_resid_dirs_with_J'][:5]), precision=3)}")

    # Interpretation
    if decomp['residual_is_structured']:
        if decomp['overlap_with_J'] > 0.5:
            msg = ("Residual is structured and overlaps J → J predicts the right "
                   "subspace but wrong magnitude; scale mismatch, not View B.")
        else:
            msg = ("Residual is structured and orthogonal to J → genuine View B "
                   "modes from GRU temporal mixing, not explained by static Jacobian.")
    else:
        msg = ("Residual is approximately flat → C_FEM ≈ C_predicted, "
               "Frobenius error is noise not structure.")
    print(f"  → {msg}")


# ---------------------------------------------------------------------------
# 7.  Combined two-component model (Test 4c)
# ---------------------------------------------------------------------------

def fit_combined_model(
    J_static: np.ndarray,
    J_eff: np.ndarray,
    C_FEM: np.ndarray,
    U_pca2: np.ndarray,
    Sigma_trial: np.ndarray,
    Sigma_within: np.ndarray,
    label: str = '',
) -> dict:
    """
    Fit scalar α in the two-component model:

        C_pred(α) = J_eff Σ_trial J_effᵀ  +  α · J_static Σ_within J_staticᵀ

    where:
      Σ_trial  = covariance of per-trace mean positions (between-trial drift)
      Σ_within = per_frame = covariance of deviations from each trace's centroid
                 (within-trial component; passed in as Sigma_eye_variants['per_frame'])

    α is the GRU's effective attenuation factor for within-trial fluctuations.
    Solved analytically (not a line search):
        A = J_eff Σ_trial J_effᵀ
        B = J_static Σ_within J_staticᵀ
        α = tr((C_FEM − A) B) / tr(B²)   [least-squares minimiser of ||C_FEM - A - αB||_F]

    α ≈ 1   → GRU barely attenuates within-trial fluctuations
    α ≈ 0   → GRU completely suppresses them
    Expected range: 0.01–0.1 (given EV-ratio story)
    """
    A = J_eff   @ Sigma_trial  @ J_eff.T    # between-trial term
    B = J_static @ Sigma_within @ J_static.T  # within-trial term (pre-attenuation)

    # Analytic least-squares: minimises ||C_FEM - A - αB||_F²
    C_deficit = C_FEM - A
    tr_cross  = float(np.trace(C_deficit @ B))
    tr_BB     = float(np.trace(B @ B))
    alpha_raw = tr_cross / (tr_BB + 1e-30)
    if alpha_raw < 0:
        print(f"  WARNING [{label}]: alpha_raw={alpha_raw:.4f} < 0; "
              f"J_eff Σ_trial J_effᵀ overcorrects in the J direction. Clamping to 0.")
    alpha = max(alpha_raw, 0.0)

    C_pred = A + alpha * B

    # Metrics
    fro_res  = frobenius_residual(C_FEM, C_pred)
    w_pred, V_pred = top2_pca(C_pred)
    w_emp,  _      = top2_pca(C_FEM)
    ev_ratio = float(w_pred[0] / (w_emp[0] + 1e-12))
    U_J_eff, _ = np.linalg.qr(J_eff)
    align = alignment_score(U_J_eff, U_pca2)

    # How much of C_FEM variance is in each component?
    frac_between = float(np.trace(A)) / (float(np.trace(C_pred)) + 1e-12)
    frac_within  = 1.0 - frac_between

    return {
        'label':              label,
        'alpha':              alpha,
        'C_predicted':        C_pred,
        'frobenius_residual': fro_res,
        'eigenvalue_ratio':   ev_ratio,
        'alignment':          align,
        'pred_top2_eigenvalues': w_pred,
        'frac_between_trial': frac_between,
        'frac_within_trial':  frac_within,
    }


def run_combined_model(
    J_per_condition: dict,
    J_eff_per_condition: dict,
    C_FEM_per_condition: dict,
    U_pca2_per_condition: dict,
    Sigma_trial: np.ndarray,
    Sigma_within: np.ndarray,
    output_dir: str = '.',
) -> list:
    """
    Run the two-component combined model for every condition and print results.

    Parameters
    ----------
    J_per_condition     : {(ori, lm): [N,2]}  static (fwAD) Jacobian
    J_eff_per_condition : {(ori, lm): [N,2]}  effective (FD-averaged) Jacobian
    Sigma_trial         : [2,2]  between-trial position covariance
    Sigma_within        : [2,2]  within-trial position covariance (= Sigma_per_frame)
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for key in sorted(J_per_condition.keys()):
        ori, logmar = key
        label = f"ori={ori}°  logmar={logmar:.2f}"
        res = fit_combined_model(
            J_static  = J_per_condition[key],
            J_eff     = J_eff_per_condition[key],
            C_FEM     = C_FEM_per_condition[key],
            U_pca2    = U_pca2_per_condition[key],
            Sigma_trial  = Sigma_trial,
            Sigma_within = Sigma_within,
            label = label,
        )
        all_results.append((key, res))

    # Summary table
    print("\n" + "="*100)
    print("TEST 4c SUMMARY  (two-component model: J_eff Σ_trial J_effᵀ + α J Σ_within Jᵀ)")
    print("="*100)
    print(f"{'Condition':<30} {'alpha':>8} {'Align':>7} {'EV-ratio':>9} "
          f"{'Frob-res':>9} {'f_between':>10} {'f_within':>9}")
    print("-" * 100)
    for key, res in all_results:
        print(
            f"{res['label']:<30} "
            f"{res['alpha']:>8.4f} "
            f"{res['alignment']:>7.3f} "
            f"{res['eigenvalue_ratio']:>9.3f} "
            f"{res['frobenius_residual']:>9.3f} "
            f"{res['frac_between_trial']:>10.3f} "
            f"{res['frac_within_trial']:>9.3f}"
        )
    print("-" * 100)

    # Eigenspectrum plot
    n_cond = len(all_results)
    ncols  = (n_cond + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7))
    axes = np.array(axes).flatten()
    for ax, (key, res) in zip(axes, all_results):
        C_FEM  = C_FEM_per_condition[key]
        C_pred = res['C_predicted']
        w_emp  = np.sort(np.linalg.eigvalsh(C_FEM))[::-1]
        w_pred = np.sort(np.linalg.eigvalsh(C_pred))[::-1]
        ranks  = np.arange(1, len(w_emp) + 1)
        ax.semilogy(ranks, w_emp,  'k-',  lw=1.5, label='Empirical C_FEM')
        ax.semilogy(ranks, w_pred, 'm--', lw=1.5, label='Combined (4c)')
        ax.axvline(2, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(res['label'], fontsize=9)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Eigenvalue')
        ax.legend(fontsize=6)
        ax.text(0.97, 0.97,
                f"α={res['alpha']:.3f}\nalign={res['alignment']:.2f}\n"
                f"Frob={res['frobenius_residual']:.3f}\nEV-r={res['eigenvalue_ratio']:.2f}",
                transform=ax.transAxes, va='top', ha='right', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    for ax in axes[n_cond:]:
        ax.set_visible(False)
    fig.suptitle('Test 4c: Two-component model eigenspectra', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'test4c_eigenspectra.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    return all_results


def run_test4b(
    J_eff_per_condition: dict,
    C_FEM_per_condition: dict,
    U_pca2_per_condition: dict,
    Sigma_trial: np.ndarray,
    output_dir: str = '.',
) -> list:
    """
    Test 4b: C_FEM ≈ J_eff Σ_trial J_effᵀ

    J_eff is the expected local sensitivity averaged over the FEM trace
    distribution (computed by finite differences in jacobian_test3.py with
    --run_eff_jacobian).  Sigma_trial is the covariance of per-trace mean
    positions.  Together they give the between-trial pushforward prediction
    that accounts for the GRU's temporal integration.

    Parameters
    ----------
    J_eff_per_condition   : {(orientation_deg, logmar): np.ndarray [N,2]}
    C_FEM_per_condition   : {same keys: np.ndarray [N,N]}
    U_pca2_per_condition  : {same keys: np.ndarray [N,2]}
    Sigma_trial           : [2,2]  covariance of per-trace mean positions

    Returns
    -------
    all_results : list of per-condition result dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    Sigma_eye_variants = {'trial': Sigma_trial}

    all_results = []
    for key in sorted(J_eff_per_condition.keys()):
        ori, logmar = key
        label = f"ori={ori}°  logmar={logmar:.2f} [J_eff]"
        res = evaluate_mechanistic_model(
            J=J_eff_per_condition[key],
            C_FEM=C_FEM_per_condition[key],
            U_pca2=U_pca2_per_condition[key],
            Sigma_eye_variants=Sigma_eye_variants,
            label=label,
        )
        all_results.append((key, res))

    _print_summary(all_results, Sigma_eye_variants, primary='trial')
    plot_eigenspectra_with_empirical(
        all_results, C_FEM_per_condition, Sigma_eye_variants,
        output_dir, primary='trial'
    )
    _plot_residual_summary(all_results, Sigma_eye_variants, output_dir)

    # Residual decompositions
    primary = 'trial'
    print("\n" + "="*70)
    print("TEST 4b RESIDUAL DECOMPOSITION  (J_eff, Σ_trial)")
    print("="*70)
    for key, res in all_results:
        decomp = decompose_residual(
            J=J_eff_per_condition[key],
            C_FEM=C_FEM_per_condition[key],
            Sigma_eye=Sigma_trial,
        )
        print_residual_interpretation(decomp, label=res['label'])

    return all_results


# ---------------------------------------------------------------------------
# 8.  Example / usage stub
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Test 3 saves one .npz per logmar: test3_lm{lm:.2f}.npz
    # Keys inside: J_ori{ori}, C_FEM_ori{ori}, U_pca2_ori{ori} for ori in [0,90,180,270]
    parser.add_argument('--test3_dir', default='declan/jacobian_results',
                        help='Directory with Test 3 per-logmar .npz outputs')
    parser.add_argument('--eye_traces', default='scripts/temporal_decoding/data/eye_traces.npz')
    parser.add_argument('--output_dir', default='declan/jacobian_results/test4')
    parser.add_argument('--gru_window', type=int, default=8)
    args = parser.parse_args()

    orientations = [0, 90, 180, 270]

    # Discover available logmar files rather than hard-coding
    import glob as _glob
    lm_files = [f for f in sorted(_glob.glob(os.path.join(args.test3_dir, 'test3_lm*.npz')))
                if '_grid' not in os.path.basename(f)]
    if not lm_files:
        raise FileNotFoundError(f"No test3_lm*.npz files found in {args.test3_dir}")

    J_per_condition = {}
    C_FEM_per_condition = {}
    U_pca2_per_condition = {}
    J_eff_per_condition = {}
    J_int_per_condition = {}
    sigma_trial_list = []

    for fname in lm_files:
        # Parse logmar from filename, e.g. test3_lm-0.20.npz → -0.20
        base = os.path.basename(fname)          # test3_lm-0.20.npz
        try:
            lm = float(base.replace('test3_lm', '').replace('.npz', ''))
        except ValueError:
            raise ValueError(f"Could not parse logmar from filename: {fname!r}; "
                             f"expected format test3_lm{{logmar}}.npz")
        d = np.load(fname)
        if 'sigma_trial' in d:
            sigma_trial_list.append(d['sigma_trial'])
        for ori in orientations:
            j_key    = f'J_ori{ori}'
            c_key    = f'C_FEM_ori{ori}'
            u_key    = f'U_pca2_ori{ori}'
            jeff_key = f'J_eff_ori{ori}'
            if j_key not in d:
                print(f"  WARNING: {fname} missing key {j_key}, skipping")
                continue
            key = (ori, lm)
            J_per_condition[key]      = d[j_key]   # [N, 2]
            C_FEM_per_condition[key]  = d[c_key]   # [N, N]
            U_pca2_per_condition[key] = d[u_key]   # [N, 2]
            if jeff_key in d:
                J_eff_per_condition[key] = d[jeff_key]  # [N, 2]
            jint_key = f'J_int_ori{ori}'
            if jint_key in d:
                J_int_per_condition[key] = d[jint_key]  # [N, 2]

    print(f"Loaded {len(J_per_condition)} conditions from {len(lm_files)} logmar file(s)")
    if J_eff_per_condition:
        print(f"  J_eff available for {len(J_eff_per_condition)} conditions (Test 4b/4c ready)")
    else:
        print("  J_eff not found — re-run jacobian_test3.py with --run_eff_jacobian to enable Test 4b/4c")
    if J_int_per_condition:
        print(f"  J_int available for {len(J_int_per_condition)} conditions (Test 4d ready)")
    else:
        print("  J_int not found — re-run jacobian_test3.py with --run_int_jacobian to enable Test 4d")

    # --- Load eye traces ---
    et = np.load(args.eye_traces)
    eye_traces = et['traces']                        # [M, T_max, 2]
    durations  = et['durations'].astype(int) if 'durations' in et else None

    # --- Run Test 4 ---
    all_results, Sigma_eye_variants = run_test4(
        J_per_condition=J_per_condition,
        C_FEM_per_condition=C_FEM_per_condition,
        U_pca2_per_condition=U_pca2_per_condition,
        eye_traces=eye_traces,
        durations=durations,
        gru_window=args.gru_window,
        output_dir=args.output_dir,
    )

    # --- Residual decomposition for each condition ---
    primary = 'per_frame'
    print("\n" + "="*70)
    print(f"RESIDUAL DECOMPOSITION  (Σ_eye = {primary})")
    print("="*70)
    for key, res in all_results:
        decomp = decompose_residual(
            J=J_per_condition[key],
            C_FEM=C_FEM_per_condition[key],
            Sigma_eye=Sigma_eye_variants[primary],
        )
        print_residual_interpretation(decomp, label=res['label'])

    # --- Test 4b: effective Jacobian (only if J_eff was saved by test3) ---
    all_results_4b = None
    if J_eff_per_condition:
        # Sigma_trial: use value saved in .npz if present, else compute inline
        if sigma_trial_list:
            Sigma_trial = np.mean(sigma_trial_list, axis=0)
        else:
            _T_max = eye_traces.shape[1]
            _durs  = durations if durations is not None else np.full(len(eye_traces), _T_max, dtype=int)
            _means = np.stack([eye_traces[i, :int(_durs[i])].mean(0) for i in range(len(_durs))])
            Sigma_trial = np.cov(_means.T)
        print(f"\nΣ_trial: trace={np.trace(Sigma_trial):.4f}  "
              f"eigenvalues={np.linalg.eigvalsh(Sigma_trial)[::-1].round(5)}")
        all_results_4b = run_test4b(
            J_eff_per_condition=J_eff_per_condition,
            C_FEM_per_condition=C_FEM_per_condition,
            U_pca2_per_condition=U_pca2_per_condition,
            Sigma_trial=Sigma_trial,
            output_dir=args.output_dir,
        )

        # --- Test 4c: two-component combined model ---
        # Sigma_within = Sigma_per_frame (within-trial deviations from centroid).
        # By law of total variance: Sigma_total = Sigma_trial + Sigma_within.
        # These are independent additive terms — NOT Sigma_frame - Sigma_trial.
        Sigma_within = Sigma_eye_variants['per_frame']
        print(f"\nΣ_within (= per_frame): trace={np.trace(Sigma_within):.4f}  "
              f"eigenvalues={np.linalg.eigvalsh(Sigma_within)[::-1].round(5)}")
        all_results_4c = run_combined_model(
            J_per_condition     = J_per_condition,
            J_eff_per_condition = J_eff_per_condition,
            C_FEM_per_condition = C_FEM_per_condition,
            U_pca2_per_condition= U_pca2_per_condition,
            Sigma_trial  = Sigma_trial,
            Sigma_within = Sigma_within,
            output_dir   = args.output_dir,
        )
    else:
        all_results_4c = None

    # --- Test 4d: J_int × {Σ_trial, Σ_total} ---
    # J_int was built by integrating over the full FEM position distribution P_FEM,
    # so the self-consistent Σ pairing is Σ_total = Σ_trial + Σ_within (the full
    # marginal position variance of that same distribution).  Σ_trial alone discards
    # the within-trial component and was wrong for J_int.
    # Kept alongside for comparison; primary column is 'total'.
    all_results_4d = None
    if J_int_per_condition:
        if sigma_trial_list:
            _Sigma_trial = np.mean(sigma_trial_list, axis=0)
        else:
            _T_max = eye_traces.shape[1]
            _durs  = durations if durations is not None else np.full(len(eye_traces), _T_max, dtype=int)
            _means = np.stack([eye_traces[i, :int(_durs[i])].mean(0) for i in range(len(_durs))])
            _Sigma_trial = np.cov(_means.T)

        Sigma_within = Sigma_eye_variants['per_frame']
        Sigma_total  = _Sigma_trial + Sigma_within
        print(f"\nΣ_total (= Σ_trial + Σ_within):  trace={np.trace(Sigma_total):.4f}  "
              f"eigenvalues={np.linalg.eigvalsh(Sigma_total)[::-1].round(5)}")

        Sigma_eye_int_variants = {
            'trial': _Sigma_trial,  # between-trial only (underpairs J_int — kept for comparison)
            'total': Sigma_total,   # full marginal variance — self-consistent with J_int
        }

        all_results_4d = []
        for key in sorted(J_int_per_condition.keys()):
            ori, logmar = key
            label = f"ori={ori}°  logmar={logmar:.2f} [J_int]"
            res = evaluate_mechanistic_model(
                J=J_int_per_condition[key],
                C_FEM=C_FEM_per_condition[key],
                U_pca2=U_pca2_per_condition[key],
                Sigma_eye_variants=Sigma_eye_int_variants,
                label=label,
            )
            all_results_4d.append((key, res))

        _print_summary(all_results_4d, Sigma_eye_int_variants, primary='total')
        plot_eigenspectra_with_empirical(
            all_results_4d, C_FEM_per_condition,
            Sigma_eye_int_variants, args.output_dir, primary='total'
        )

    # --- Save all results ---
    import pickle
    out_path = os.path.join(args.output_dir, 'test4_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({
            'all_results':    all_results,
            'all_results_4b': all_results_4b,
            'all_results_4c': all_results_4c,
            'all_results_4d': all_results_4d,
            'Sigma_eye_variants': Sigma_eye_variants,
        }, f)
    print(f"\nSaved full results to {out_path}")
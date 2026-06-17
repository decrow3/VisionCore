"""Pilot hidden-pose Vernier observer using local translation geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


THETA_PLUS = "plus"
THETA_MINUS = "minus"
THETA_LABELS = (THETA_PLUS, THETA_MINUS)
SUPPORTED_JOINT_CONTROLS = ("correct_chart", "wrong_chart", "random_basis", "gain_only")
SUPPORTED_JOINT_OBSERVERS = ("enumerated", "map")
SUPPORTED_COMPACT_COVARIANCE = ("full", "diagonal")
SUPPORTED_LIKELIHOOD_NORMALIZATION = ("residual", "full")


def likelihood_score_family(normalization: str) -> str:
    """Human-readable score family for observer outputs."""
    if str(normalization) == "full":
        return "gaussian_log_likelihood"
    if str(normalization) == "residual":
        return "mahalanobis_residual_score"
    raise ValueError(
        f"Unsupported likelihood normalization {normalization!r}; "
        f"expected {SUPPORTED_LIKELIHOOD_NORMALIZATION}"
    )


def build_compact_translation_basis(
    jacobian_counts_per_arcmin: np.ndarray,
    *,
    compact_k: int,
    control: str,
    seed: int,
) -> np.ndarray:
    """Return a unit-space basis for the requested joint-observer control."""
    jac = np.asarray(jacobian_counts_per_arcmin, dtype=np.float64)
    if jac.ndim != 4 or jac.shape[-1] != 2:
        raise ValueError(f"jacobian must be (theta, time, units, 2), got {jac.shape}")
    n_units = int(jac.shape[2])
    control = str(control)
    if control not in SUPPORTED_JOINT_CONTROLS:
        raise ValueError(f"Unsupported joint control {control!r}; expected one of {SUPPORTED_JOINT_CONTROLS}")
    if control == "gain_only":
        u = np.ones((n_units, 1), dtype=np.float64)
        return u / max(float(np.linalg.norm(u)), 1e-12)
    k = min(max(int(compact_k), 1), n_units)
    if control == "random_basis":
        rng = np.random.default_rng(int(seed) + 7919 * int(k))
        q, _ = np.linalg.qr(rng.standard_normal((n_units, k)))
        return q[:, :k]
    mat = np.moveaxis(jac, 2, -1).reshape(-1, n_units)
    mat = mat[np.isfinite(mat).all(axis=1)]
    if mat.shape[0] == 0:
        return np.eye(n_units, k, dtype=np.float64)
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    _left, _s, vh = np.linalg.svd(mat, full_matrices=False)
    if vh.shape[0] < k:
        pad = np.eye(n_units, k - vh.shape[0], dtype=np.float64)
        basis = np.concatenate([vh.T, pad], axis=1)
        q, _ = np.linalg.qr(basis)
        return q[:, :k]
    return vh[:k].T


def candidate_chart(
    jacobian_counts_per_arcmin: np.ndarray,
    u_trans: np.ndarray,
    *,
    candidate_index: int,
    control: str,
) -> np.ndarray:
    """Project a candidate translation Jacobian into compact coordinates."""
    if str(control) not in SUPPORTED_JOINT_CONTROLS:
        raise ValueError(f"Unsupported joint control {control!r}; expected one of {SUPPORTED_JOINT_CONTROLS}")
    jac = np.asarray(jacobian_counts_per_arcmin, dtype=np.float64)
    source_index = 1 - int(candidate_index) if str(control) == "wrong_chart" else int(candidate_index)
    return np.einsum("uk,tud->tkd", np.asarray(u_trans, dtype=np.float64), jac[source_index])


def solve_pose_map(
    z: np.ndarray,
    a_chart: np.ndarray,
    sigma_z: np.ndarray,
    *,
    amplitude_lambda: float,
    smoothness_lambda: float,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Solve the dense pilot MAP problem for a hidden 2D pose trajectory."""
    z = np.asarray(z, dtype=np.float64)
    a = np.asarray(a_chart, dtype=np.float64)
    sigma = np.asarray(sigma_z, dtype=np.float64)
    if z.ndim != 2 or a.ndim != 3 or a.shape[:2] != z.shape or a.shape[2] != 2:
        raise ValueError(f"Expected z=(T,k), A=(T,k,2); got z={z.shape}, A={a.shape}")
    if sigma.ndim == 2:
        sigma = np.asarray([np.diag(row) for row in sigma], dtype=np.float64)
    if sigma.shape != (z.shape[0], z.shape[1], z.shape[1]):
        raise ValueError(f"Expected compact noise covariance (T,k,k), got {sigma.shape}")
    t = int(z.shape[0])
    k = int(z.shape[1])
    h = np.zeros((2 * t, 2 * t), dtype=np.float64)
    b = np.zeros(2 * t, dtype=np.float64)
    precision_blocks: list[np.ndarray] = []
    for ti in range(t):
        keep = np.isfinite(z[ti]) & np.isfinite(sigma[ti]).all(axis=0) & np.isfinite(sigma[ti]).all(axis=1)
        if int(np.sum(keep)) == 0:
            precision_blocks.append(np.zeros((k, k), dtype=np.float64))
            continue
        cov = sigma[ti][np.ix_(keep, keep)]
        cov = 0.5 * (cov + cov.T) + float(epsilon) * np.eye(cov.shape[0], dtype=np.float64)
        try:
            precision = np.linalg.solve(cov, np.eye(cov.shape[0], dtype=np.float64))
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)
        precision_full = np.zeros((k, k), dtype=np.float64)
        precision_full[np.ix_(keep, keep)] = precision
        precision_blocks.append(precision_full)
        at = a[ti, keep, :]
        zt = z[ti, keep]
        sl = slice(2 * ti, 2 * ti + 2)
        h[sl, sl] += at.T @ precision @ at
        b[sl] += at.T @ precision @ zt
    amp = max(float(amplitude_lambda), 0.0)
    smooth = max(float(smoothness_lambda), 0.0)
    prior = np.zeros_like(h)
    if amp > 0.0:
        prior += amp * np.eye(2 * t, dtype=np.float64)
    if smooth > 0.0 and t > 1:
        for ti in range(t):
            sl = slice(2 * ti, 2 * ti + 2)
            prior[sl, sl] += smooth * (1.0 if ti in {0, t - 1} else 2.0) * np.eye(2)
            if ti > 0:
                prev = slice(2 * (ti - 1), 2 * (ti - 1) + 2)
                prior[sl, prev] -= smooth * np.eye(2)
                prior[prev, sl] -= smooth * np.eye(2)
    h += prior
    h += float(epsilon) * np.eye(2 * t, dtype=np.float64)
    try:
        tau = np.linalg.solve(h, b).reshape(t, 2)
    except np.linalg.LinAlgError:
        tau = (np.linalg.pinv(h) @ b).reshape(t, 2)
    pred = np.einsum("tkd,td->tk", a, tau)
    neural = 0.0
    for ti in range(t):
        resid = z[ti] - pred[ti]
        neural += float(resid @ precision_blocks[ti] @ resid)
    amp_loss = float(amp * np.sum(tau * tau))
    diffs = np.diff(tau, axis=0)
    smooth_loss = float(smooth * np.sum(diffs * diffs))
    sign, logdet_h = np.linalg.slogdet(h)
    logdet_h = float(logdet_h) if sign > 0 else float("nan")
    total = neural + amp_loss + smooth_loss
    return {
        "tau": tau,
        "neural_loss": neural,
        "amplitude_loss": amp_loss,
        "smoothness_loss": smooth_loss,
        "total_loss": total,
        "logdet_hessian": logdet_h,
        "approx_marginal_loss": total + logdet_h if np.isfinite(logdet_h) else float("nan"),
    }


def compact_candidate_loss(
    z: np.ndarray,
    pred_z: np.ndarray,
    sigma_z: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Compact-channel Gaussian loss for comparable zero/known/joint scores."""
    z = np.asarray(z, dtype=np.float64)
    pred = np.asarray(pred_z, dtype=np.float64)
    sigma = np.asarray(sigma_z, dtype=np.float64)
    if pred.shape != z.shape:
        pred = np.broadcast_to(pred, z.shape)
    if sigma.ndim == 2:
        sigma = np.asarray([np.diag(row) for row in sigma], dtype=np.float64)
    if sigma.shape != (z.shape[0], z.shape[1], z.shape[1]):
        raise ValueError(f"Expected compact noise covariance (T,k,k), got {sigma.shape}")
    loss = 0.0
    for ti in range(z.shape[0]):
        resid = z[ti] - pred[ti]
        keep = np.isfinite(resid) & np.isfinite(sigma[ti]).all(axis=0) & np.isfinite(sigma[ti]).all(axis=1)
        if int(np.sum(keep)) == 0:
            continue
        cov = sigma[ti][np.ix_(keep, keep)]
        cov = 0.5 * (cov + cov.T) + float(epsilon) * np.eye(cov.shape[0], dtype=np.float64)
        try:
            sol = np.linalg.solve(cov, resid[keep])
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(cov) @ resid[keep]
        loss += float(resid[keep] @ sol)
    return loss


def logsumexp(values: np.ndarray) -> float:
    """Small local logsumexp to keep the observer dependency-light."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return float("-inf")
    vmax = float(np.max(vals))
    if not np.isfinite(vmax):
        return vmax
    return float(vmax + np.log(np.sum(np.exp(vals - vmax))))


def merge_duplicate_states(states: np.ndarray, log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate identical lattice states with logsumexp before beam pruning."""
    states = np.asarray(states, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if states.shape[0] != log_weights.shape[0]:
        raise ValueError("states/log_weights length mismatch")
    if states.shape[0] == 0:
        return states, log_weights
    keys = np.round(states, decimals=12)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    merged = np.full(unique.shape[0], -np.inf, dtype=np.float64)
    for idx in range(unique.shape[0]):
        merged[idx] = logsumexp(log_weights[inverse == idx])
    return unique.astype(np.float64, copy=False), merged


def build_discrete_gaussian_step_prior(
    *,
    max_step_arcmin: float,
    sigma_arcmin: float,
    step_arcmin: float | None = None,
) -> dict[str, np.ndarray]:
    """Build a deterministic 2D Gaussian step prior on an arcmin lattice."""
    max_step = float(max_step_arcmin)
    sigma = float(sigma_arcmin)
    if max_step < 0.0:
        raise ValueError("max_step_arcmin must be nonnegative")
    if sigma <= 0.0:
        raise ValueError("sigma_arcmin must be positive")
    spacing = float(step_arcmin) if step_arcmin is not None else (1.0 if max_step >= 1.0 else max(max_step, 1e-6))
    if spacing <= 0.0:
        raise ValueError("step_arcmin must be positive")
    axis = np.arange(-max_step, max_step + 0.5 * spacing, spacing, dtype=np.float64)
    if axis.size == 0:
        axis = np.asarray([0.0], dtype=np.float64)
    if not np.any(np.isclose(axis, 0.0)):
        axis = np.sort(np.concatenate([axis, np.asarray([0.0], dtype=np.float64)]))
    steps = np.asarray([(dx, dy) for dx in axis for dy in axis], dtype=np.float64)
    logp = -0.5 * np.sum(steps * steps, axis=1) / max(sigma * sigma, 1e-12)
    logp = logp - logsumexp(logp)
    return {"steps": steps, "log_probs": logp, "axis": axis}


def _compact_precision_and_logdet(cov: np.ndarray, *, epsilon: float) -> tuple[np.ndarray, float]:
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    scale = float(np.nanmean(np.diag(cov))) if cov.size else 1.0
    scale = max(scale, 1.0)
    eye = np.eye(cov.shape[0], dtype=np.float64)
    jitter = float(epsilon) * scale
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(cov + jitter * eye)
            precision = np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
            logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
            return 0.5 * (precision + precision.T), logdet
        except np.linalg.LinAlgError:
            jitter *= 10.0
    stabilized = cov + jitter * eye
    sign, logdet = np.linalg.slogdet(stabilized)
    precision = np.linalg.pinv(stabilized)
    return 0.5 * (precision + precision.T), float(logdet) if sign > 0 else float("nan")


def compact_log_likelihood(
    z_obs_t: np.ndarray,
    z_pred_t: np.ndarray,
    cov_t: np.ndarray,
    *,
    likelihood_scale: float = 1.0,
    normalization: str = "residual",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Gaussian compact log likelihood for one time bin and many predictions."""
    if str(normalization) not in SUPPORTED_LIKELIHOOD_NORMALIZATION:
        raise ValueError(
            f"Unsupported likelihood normalization {normalization!r}; "
            f"expected {SUPPORTED_LIKELIHOOD_NORMALIZATION}"
        )
    obs = np.asarray(z_obs_t, dtype=np.float64)
    pred = np.asarray(z_pred_t, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
    precision, logdet = _compact_precision_and_logdet(cov_t, epsilon=epsilon)
    resid = pred - obs[None, :]
    quad = np.einsum("nk,kl,nl->n", resid, precision, resid)
    logp = -0.5 * quad
    if str(normalization) == "full":
        logp = logp - 0.5 * logdet
    return float(likelihood_scale) * logp


def score_fixed_eye_log_evidence(
    z_obs: np.ndarray,
    chart: np.ndarray,
    cov_z: np.ndarray,
    trajectory_arcmin: np.ndarray,
    *,
    likelihood_scale: float,
    likelihood_normalization: str,
    epsilon: float,
) -> float:
    """Score a fixed eye trajectory as compact log evidence."""
    z = np.asarray(z_obs, dtype=np.float64)
    a = np.asarray(chart, dtype=np.float64)
    traj = np.asarray(trajectory_arcmin, dtype=np.float64)
    cov = np.asarray(cov_z, dtype=np.float64)
    t = min(z.shape[0], a.shape[0], traj.shape[0], cov.shape[0])
    total = 0.0
    for ti in range(t):
        pred = a[ti] @ traj[ti]
        total += float(
            compact_log_likelihood(
                z[ti],
                pred,
                cov[ti],
                likelihood_scale=likelihood_scale,
                normalization=likelihood_normalization,
                epsilon=epsilon,
            )[0]
        )
    return float(total)


def score_joint_eye_evidence_enumerated(
    z_obs: np.ndarray,
    chart: np.ndarray,
    cov_z: np.ndarray,
    step_prior: dict[str, np.ndarray],
    *,
    max_particles: int,
    likelihood_scale: float,
    likelihood_normalization: str = "residual",
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Deterministic beam-filter approximation to trajectory-marginal evidence."""
    z = np.asarray(z_obs, dtype=np.float64)
    a = np.asarray(chart, dtype=np.float64)
    cov = np.asarray(cov_z, dtype=np.float64)
    steps = np.asarray(step_prior["steps"], dtype=np.float64)
    log_step = np.asarray(step_prior["log_probs"], dtype=np.float64)
    if steps.ndim != 2 or steps.shape[1] != 2:
        raise ValueError(f"steps must be (n, 2), got {steps.shape}")
    if steps.shape[0] != log_step.shape[0]:
        raise ValueError("step prior steps/log_probs length mismatch")
    t = min(z.shape[0], a.shape[0], cov.shape[0])
    max_particles = max(int(max_particles), 1)
    states = np.zeros((1, 2), dtype=np.float64)
    logw = np.zeros(1, dtype=np.float64)
    log_evidence = 0.0
    retained_mass_by_t: list[float] = []
    n_states_by_t: list[int] = []
    posterior_mean = np.zeros((t, 2), dtype=np.float64)
    posterior_var = np.zeros((t, 2), dtype=np.float64)
    entropy_by_t: list[float] = []
    for ti in range(t):
        prev_n = states.shape[0]
        cand_states = states[:, None, :] + steps[None, :, :]
        cand_states = cand_states.reshape(prev_n * steps.shape[0], 2)
        cand_logw = np.repeat(logw, steps.shape[0]) + np.tile(log_step, prev_n)
        pred = np.einsum("kd,nd->nk", a[ti], cand_states)
        cand_logw = cand_logw + compact_log_likelihood(
            z[ti],
            pred,
            cov[ti],
            likelihood_scale=likelihood_scale,
            normalization=likelihood_normalization,
            epsilon=epsilon,
        )
        logz_inc = logsumexp(cand_logw)
        if not np.isfinite(logz_inc):
            logw = np.full(1, 0.0, dtype=np.float64)
            states = np.zeros((1, 2), dtype=np.float64)
            retained_mass_by_t.append(float("nan"))
            n_states_by_t.append(1)
            entropy_by_t.append(float("nan"))
            continue
        log_evidence += logz_inc
        cand_logw = cand_logw - logz_inc
        cand_states, cand_logw = merge_duplicate_states(cand_states, cand_logw)
        if cand_logw.size > max_particles:
            top = np.argpartition(cand_logw, -max_particles)[-max_particles:]
            top = top[np.argsort(cand_logw[top])[::-1]]
            retained_mass = float(np.sum(np.exp(cand_logw[top])))
            log_evidence += float(np.log(max(retained_mass, 1e-300)))
            states = cand_states[top]
            logw = cand_logw[top] - np.log(max(retained_mass, 1e-300))
        else:
            retained_mass = 1.0
            states = cand_states
            logw = cand_logw
        weights = np.exp(logw)
        weights = weights / max(float(np.sum(weights)), 1e-300)
        mean = weights @ states
        var = weights @ ((states - mean[None, :]) ** 2)
        posterior_mean[ti] = mean
        posterior_var[ti] = var
        retained_mass_by_t.append(retained_mass)
        n_states_by_t.append(int(states.shape[0]))
        entropy_by_t.append(float(-np.sum(weights * np.log(np.maximum(weights, 1e-300)))))
    return {
        "log_evidence": float(log_evidence),
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "retained_mass_by_t": np.asarray(retained_mass_by_t, dtype=np.float64),
        "n_states_by_t": np.asarray(n_states_by_t, dtype=np.int32),
        "entropy_by_t": np.asarray(entropy_by_t, dtype=np.float64),
        "final_states": states,
        "final_log_weights": logw,
    }


def compact_noise_covariance(
    unit_noise_diag: np.ndarray,
    u_trans: np.ndarray,
    *,
    epsilon: float = 1e-8,
    mode: str = "full",
) -> np.ndarray:
    """Project diagonal unit noise into the full compact covariance."""
    if str(mode) not in SUPPORTED_COMPACT_COVARIANCE:
        raise ValueError(f"Unsupported compact covariance mode {mode!r}; expected {SUPPORTED_COMPACT_COVARIANCE}")
    diag = np.asarray(unit_noise_diag, dtype=np.float64)
    u = np.asarray(u_trans, dtype=np.float64)
    cov = np.einsum("uk,tu,ul->tkl", u, diag, u)
    if str(mode) == "diagonal":
        cov = np.asarray([np.diag(np.diag(c)) for c in cov], dtype=np.float64)
    eye = np.eye(u.shape[1], dtype=np.float64)
    return 0.5 * (cov + np.swapaxes(cov, 1, 2)) + float(epsilon) * eye[None, :, :]


def joint_geometry_vernier_observer_trial(
    observed_counts: np.ndarray,
    true_label: str,
    mu0_counts: np.ndarray,
    jacobian_counts_per_arcmin: np.ndarray,
    u_trans: np.ndarray,
    *,
    control: str,
    amplitude_lambda: float,
    smoothness_lambda: float,
    phi: float,
    true_pose_arcmin: np.ndarray | None = None,
    known_u_trans: np.ndarray | None = None,
    known_candidate_counts: np.ndarray | None = None,
    observer_mode: str = "enumerated",
    step_prior: dict[str, np.ndarray] | None = None,
    max_particles: int = 2000,
    likelihood_scale: float = 1.0,
    likelihood_normalization: str = "residual",
    covariance_mode: str = "full",
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Classify one observed Vernier trial with a geometry-aware hidden-pose observer."""
    if str(observer_mode) not in SUPPORTED_JOINT_OBSERVERS:
        raise ValueError(f"Unsupported joint observer {observer_mode!r}; expected {SUPPORTED_JOINT_OBSERVERS}")
    y = np.asarray(observed_counts, dtype=np.float64)
    mu0 = np.asarray(mu0_counts, dtype=np.float64)
    jac = np.asarray(jacobian_counts_per_arcmin, dtype=np.float64)
    u = np.asarray(u_trans, dtype=np.float64)
    t = min(y.shape[0], mu0.shape[1], jac.shape[1])
    y = y[:t]
    mu0 = mu0[:, :t]
    jac = jac[:, :t]
    pose = None if true_pose_arcmin is None else np.asarray(true_pose_arcmin, dtype=np.float64)[:t]
    known_counts = None
    if known_candidate_counts is not None:
        known_counts = np.asarray(known_candidate_counts, dtype=np.float64)
        if known_counts.ndim != 3 or known_counts.shape[0] != len(THETA_LABELS):
            raise ValueError(
                f"known_candidate_counts must be (theta, time, units), got {known_counts.shape}"
            )
        if known_counts.shape[1] < t:
            raise ValueError(
                f"known_candidate_counts has {known_counts.shape[1]} time bins, "
                f"but at least {t} are required"
            )
        if known_counts.shape[2] != y.shape[1]:
            raise ValueError(
                f"known_candidate_counts has {known_counts.shape[2]} units, "
                f"but observed_counts has {y.shape[1]}"
            )
        known_counts = known_counts[:, :t]

    map_results: dict[str, dict[str, Any]] = {}
    zero_log_evidence: dict[str, float] = {}
    known_log_evidence_control: dict[str, float] = {}
    known_log_evidence_correct: dict[str, float] = {}
    joint_log_evidence: dict[str, float] = {}
    known_upper_available = pose is not None and (known_u_trans is not None or str(control) == "correct_chart")
    known_u = u if known_u_trans is None else np.asarray(known_u_trans, dtype=np.float64)
    for ci, label in enumerate(THETA_LABELS):
        residual = y - mu0[ci]
        z = residual @ u
        noise_diag = np.maximum(float(phi) * np.maximum(mu0[ci], 0.0), float(epsilon))
        sigma_z = compact_noise_covariance(noise_diag, u, epsilon=epsilon, mode=covariance_mode)
        a_chart = candidate_chart(jac, u, candidate_index=ci, control=control)
        zero_log_evidence[label] = score_fixed_eye_log_evidence(
            z,
            a_chart,
            sigma_z,
            np.zeros((t, 2), dtype=np.float64),
            likelihood_scale=float(likelihood_scale),
            likelihood_normalization=str(likelihood_normalization),
            epsilon=epsilon,
        )
        if pose is None:
            known_log_evidence_control[label] = float("nan")
            known_log_evidence_correct[label] = float("nan")
        else:
            known_log_evidence_control[label] = score_fixed_eye_log_evidence(
                z,
                a_chart,
                sigma_z,
                pose,
                likelihood_scale=float(likelihood_scale),
                likelihood_normalization=str(likelihood_normalization),
                epsilon=epsilon,
            )

            if known_upper_available:
                z_known = residual @ known_u
                sigma_known = compact_noise_covariance(noise_diag, known_u, epsilon=epsilon, mode=covariance_mode)
                if known_counts is None:
                    correct_chart = candidate_chart(jac, known_u, candidate_index=ci, control="correct_chart")
                    known_log_evidence_correct[label] = score_fixed_eye_log_evidence(
                        z_known,
                        correct_chart,
                        sigma_known,
                        pose,
                        likelihood_scale=float(likelihood_scale),
                        likelihood_normalization=str(likelihood_normalization),
                        epsilon=epsilon,
                    )
                else:
                    pred_known = (known_counts[ci] - mu0[ci]) @ known_u
                    known_noise_diag = np.maximum(float(phi) * np.maximum(known_counts[ci], 0.0), float(epsilon))
                    sigma_known_exact = compact_noise_covariance(
                        known_noise_diag,
                        known_u,
                        epsilon=epsilon,
                        mode=covariance_mode,
                    )
                    known_log_evidence_correct[label] = 0.0
                    for ti in range(t):
                        known_log_evidence_correct[label] += float(
                            compact_log_likelihood(
                                z_known[ti],
                                pred_known[ti],
                                sigma_known_exact[ti],
                                likelihood_scale=float(likelihood_scale),
                                normalization=str(likelihood_normalization),
                                epsilon=epsilon,
                            )[0]
                        )
            else:
                known_log_evidence_correct[label] = float("nan")

        if str(observer_mode) == "map":
            map_results[label] = solve_pose_map(
                z,
                a_chart,
                sigma_z,
                amplitude_lambda=amplitude_lambda,
                smoothness_lambda=smoothness_lambda,
                epsilon=epsilon,
            )
            joint_log_evidence[label] = -float(map_results[label]["approx_marginal_loss"])
        else:
            if step_prior is None:
                raise ValueError("step_prior is required for enumerated joint observer")
            map_results[label] = score_joint_eye_evidence_enumerated(
                z,
                a_chart,
                sigma_z,
                step_prior,
                max_particles=int(max_particles),
                likelihood_scale=float(likelihood_scale),
                likelihood_normalization=str(likelihood_normalization),
                epsilon=epsilon,
            )
            joint_log_evidence[label] = float(map_results[label]["log_evidence"])

    chosen = THETA_PLUS if joint_log_evidence[THETA_PLUS] >= joint_log_evidence[THETA_MINUS] else THETA_MINUS
    pred_zero = THETA_PLUS if zero_log_evidence[THETA_PLUS] >= zero_log_evidence[THETA_MINUS] else THETA_MINUS
    known_pair = np.asarray([known_log_evidence_correct[THETA_PLUS], known_log_evidence_correct[THETA_MINUS]])
    pred_known = (
        THETA_PLUS
        if np.isfinite(known_pair).all() and known_log_evidence_correct[THETA_PLUS] >= known_log_evidence_correct[THETA_MINUS]
        else (THETA_MINUS if np.isfinite(known_pair).all() else "")
    )
    true = str(true_label)
    true_idx = THETA_LABELS.index(true)
    other = THETA_MINUS if true == THETA_PLUS else THETA_PLUS
    if str(observer_mode) == "map":
        tau_hat = np.asarray(map_results[true]["tau"], dtype=np.float64)
    else:
        tau_hat = np.asarray(map_results[true]["posterior_mean"], dtype=np.float64)
    pose_rmse = (
        float(np.sqrt(np.mean((tau_hat - pose) ** 2)))
        if pose is not None and pose.shape == tau_hat.shape
        else float("nan")
    )
    joint_true = float(joint_log_evidence[true])
    zero_true = float(zero_log_evidence[true])
    known_true = float(known_log_evidence_correct[true])
    joint_margin = float(joint_log_evidence[true] - joint_log_evidence[other])
    zero_margin = float(zero_log_evidence[true] - zero_log_evidence[other])
    known_margin = float(known_log_evidence_correct[true] - known_log_evidence_correct[other])
    score_family = likelihood_score_family(str(likelihood_normalization))
    gap_denom = known_true - zero_true
    gap_closure = (
        (joint_true - zero_true) / gap_denom
        if (
            str(control) == "correct_chart"
            and score_family == "gaussian_log_likelihood"
            and np.isfinite(gap_denom)
            and abs(gap_denom) > 1e-12
        )
        else float("nan")
    )
    margin_gap_denom = known_margin - zero_margin
    margin_gap_closure = (
        (joint_margin - zero_margin) / margin_gap_denom
        if str(control) == "correct_chart" and np.isfinite(margin_gap_denom) and abs(margin_gap_denom) > 1e-12
        else float("nan")
    )
    map_loss_plus = -float(joint_log_evidence[THETA_PLUS])
    map_loss_minus = -float(joint_log_evidence[THETA_MINUS])
    retained = map_results[true].get("retained_mass_by_t", np.asarray([], dtype=np.float64))
    n_states = map_results[true].get("n_states_by_t", np.asarray([], dtype=np.int32))
    entropy = map_results[true].get("entropy_by_t", np.asarray([], dtype=np.float64))
    known_eye_reference = "exact_candidate_counts" if known_counts is not None else "local_linear_chart"
    decision_rule = (
        "joint_log_evidence"
        if str(observer_mode) == "enumerated" and score_family == "gaussian_log_likelihood"
        else (
            "joint_mahalanobis_residual_score"
            if str(observer_mode) == "enumerated"
            else "map_approx_marginal_loss"
        )
    )
    return {
        "map_loss_plus": map_loss_plus,
        "map_loss_minus": map_loss_minus,
        "approx_marginal_loss_plus": map_loss_plus,
        "approx_marginal_loss_minus": map_loss_minus,
        "chosen_theta": chosen,
        "correct": bool(chosen == true),
        "pred_zero": pred_zero,
        "pred_known": pred_known,
        "zero_correct": bool(pred_zero == true),
        "known_correct": bool(pred_known == true) if pred_known else float("nan"),
        "decision_rule": decision_rule,
        "joint_observer": str(observer_mode),
        "joint_geometry_mode": "instantaneous_local_chart",
        "joint_likelihood_normalization": str(likelihood_normalization),
        "joint_score_family": score_family,
        "joint_evidence_is_normalized_log_probability": bool(str(likelihood_normalization) == "full"),
        "known_eye_reference": known_eye_reference,
        "known_eye_covariance_reference": "candidate_known_counts" if known_counts is not None else "candidate_baseline_mu0",
        "compact_covariance_mode": str(covariance_mode),
        "confidence_gap": float(joint_log_evidence[THETA_PLUS] - joint_log_evidence[THETA_MINUS]),
        "true_label": true,
        "true_theta_index": int(true_idx),
        "zero_log_evidence_plus": float(zero_log_evidence[THETA_PLUS]),
        "zero_log_evidence_minus": float(zero_log_evidence[THETA_MINUS]),
        "known_log_evidence_plus": float(known_log_evidence_correct[THETA_PLUS]),
        "known_log_evidence_minus": float(known_log_evidence_correct[THETA_MINUS]),
        "known_log_evidence_plus_control_chart": float(known_log_evidence_control[THETA_PLUS]),
        "known_log_evidence_minus_control_chart": float(known_log_evidence_control[THETA_MINUS]),
        "joint_log_evidence_plus": float(joint_log_evidence[THETA_PLUS]),
        "joint_log_evidence_minus": float(joint_log_evidence[THETA_MINUS]),
        "zero_log_evidence_true": zero_true,
        "known_log_evidence_true": known_true,
        "joint_log_evidence_true": joint_true,
        "joint_score": joint_margin,
        "zero_eye_score": zero_margin,
        "known_eye_score": known_margin,
        "known_eye_score_control_chart": float(
            known_log_evidence_control[true] - known_log_evidence_control[other]
        ),
        "gap_closure_vs_zero_known": float(gap_closure),
        "margin_gap_closure_vs_zero_known": float(margin_gap_closure),
        "neural_only_gap_closure_vs_zero_known": float("nan"),
        "pose_rmse_arcmin": pose_rmse,
        "inferred_tau_shape": f"{tau_hat.shape[0]}x{tau_hat.shape[1]}",
        "joint_tau_hat": tau_hat,
        "zero_eye_loss_plus": -float(zero_log_evidence[THETA_PLUS]),
        "zero_eye_loss_minus": -float(zero_log_evidence[THETA_MINUS]),
        "known_eye_loss_plus": -float(known_log_evidence_correct[THETA_PLUS]),
        "known_eye_loss_minus": -float(known_log_evidence_correct[THETA_MINUS]),
        "known_eye_loss_plus_control_chart": -float(known_log_evidence_control[THETA_PLUS]),
        "known_eye_loss_minus_control_chart": -float(known_log_evidence_control[THETA_MINUS]),
        "joint_retained_mass_min": float(np.nanmin(retained)) if retained.size else float("nan"),
        "joint_retained_mass_mean": float(np.nanmean(retained)) if retained.size else float("nan"),
        "joint_n_states_final": int(n_states[-1]) if n_states.size else 0,
        "joint_n_states_max": int(np.max(n_states)) if n_states.size else 0,
        "joint_entropy_final": float(entropy[-1]) if entropy.size else float("nan"),
    }


def summarize_joint_geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate joint-observer trial rows by condition/control/k."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("readout", ""),
            row.get("condition", ""),
            row.get("fd_step_arcmin", ""),
            row.get("inference_mode", ""),
            row.get("joint_control", ""),
            row.get("joint_observer", ""),
            row.get("joint_geometry_mode", "instantaneous_local_chart"),
            row.get("joint_likelihood_normalization", "residual"),
            row.get("joint_score_family", likelihood_score_family(row.get("joint_likelihood_normalization", "residual"))),
            row.get("known_eye_reference", "local_linear_chart"),
            row.get("known_eye_covariance_reference", ""),
            row.get("compact_covariance_mode", ""),
            row.get("compact_k", ""),
            row.get("translation_eps_arcmin", ""),
            row.get("pose_smoothness_lambda", ""),
            row.get("pose_amplitude_lambda", ""),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        (
            readout,
            condition,
            fd_step,
            inference_mode,
            control,
            observer,
            geometry_mode,
            likelihood_normalization,
            score_family,
            known_eye_reference,
            known_eye_covariance_reference,
            cov_mode,
            compact_k,
            eps,
            smooth,
            amp,
        ) = key
        grp = groups[key]
        correct = _bool_values(row.get("correct") for row in grp)
        zero_correct = _bool_values(row.get("zero_correct") for row in grp)
        known_correct = _bool_values(row.get("known_correct") for row in grp)
        joint = _finite_values(row.get("joint_score") for row in grp)
        zero = _finite_values(row.get("zero_eye_score") for row in grp)
        known = _finite_values(row.get("known_eye_score") for row in grp)
        known_control = _finite_values(row.get("known_eye_score_control_chart") for row in grp)
        gap = _finite_values(row.get("gap_closure_vs_zero_known") for row in grp)
        margin_gap = _finite_values(row.get("margin_gap_closure_vs_zero_known") for row in grp)
        pose = _finite_values(row.get("pose_rmse_arcmin") for row in grp)
        retained = _finite_values(row.get("joint_retained_mass_min") for row in grp)
        n_states = _finite_values(row.get("joint_n_states_final") for row in grp)
        out.append(
            {
                "readout": readout,
                "condition": condition,
                "fd_step_arcmin": fd_step,
                "inference_mode": inference_mode,
                "joint_control": control,
                "joint_observer": observer,
                "joint_geometry_mode": geometry_mode,
                "joint_likelihood_normalization": likelihood_normalization,
                "joint_score_family": score_family,
                "known_eye_reference": known_eye_reference,
                "known_eye_covariance_reference": known_eye_covariance_reference,
                "compact_covariance_mode": cov_mode,
                "compact_k": compact_k,
                "translation_eps_arcmin": eps,
                "pose_smoothness_lambda": smooth,
                "pose_amplitude_lambda": amp,
                "n": len(grp),
                "accuracy": float(np.mean(correct)) if correct.size else float("nan"),
                "zero_accuracy": float(np.mean(zero_correct)) if zero_correct.size else float("nan"),
                "known_accuracy": float(np.mean(known_correct)) if known_correct.size else float("nan"),
                "mean_joint_score": _mean_or_nan(joint),
                "mean_zero_eye_score": _mean_or_nan(zero),
                "mean_known_eye_score": _mean_or_nan(known),
                "mean_known_eye_score_control_chart": _mean_or_nan(known_control),
                "mean_gap_closure_vs_zero_known": _mean_or_nan(gap),
                "median_gap_closure_vs_zero_known": float(np.median(gap)) if gap.size else float("nan"),
                "mean_margin_gap_closure_vs_zero_known": _mean_or_nan(margin_gap),
                "median_margin_gap_closure_vs_zero_known": float(np.median(margin_gap)) if margin_gap.size else float("nan"),
                "mean_pose_rmse_arcmin": _mean_or_nan(pose),
                "mean_joint_retained_mass_min": _mean_or_nan(retained),
                "mean_joint_n_states_final": _mean_or_nan(n_states),
            }
        )
    return out


def write_joint_geometry_gap_figure(out_dir: Path, summary_rows: list[dict[str, Any]]) -> Path | None:
    """Write a compact gap-closure plot for the joint-geometry pilot."""
    if not summary_rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    controls = sorted({str(row.get("joint_control", "")) for row in summary_rows})
    for control in controls:
        rows = [row for row in summary_rows if str(row.get("joint_control", "")) == control]
        rows = sorted(rows, key=lambda r: (float(r.get("compact_k", 0) or 0), str(r.get("condition", ""))))
        x = np.asarray([float(row.get("compact_k", 0) or 0) for row in rows], dtype=float)
        y = np.asarray(
            [
                float(
                    row.get(
                        "mean_margin_gap_closure_vs_zero_known",
                        row.get("mean_gap_closure_vs_zero_known", np.nan),
                    )
                )
                for row in rows
            ],
            dtype=float,
        )
        keep = np.isfinite(x) & np.isfinite(y)
        if np.any(keep):
            ax.plot(x[keep], y[keep], marker="o", linewidth=1.8, label=control)
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle=":")
    ax.set_xlabel("compact k")
    ax.set_ylabel("margin gap closure vs zero/known")
    ax.set_title("Joint geometry-aware Vernier observer")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = fig_dir / "joint_geometry_gap_closure.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _finite_values(values: Any) -> np.ndarray:
    converted: list[float] = []
    for v in values:
        try:
            converted.append(float(v))
        except (TypeError, ValueError):
            converted.append(float("nan"))
    arr = np.asarray(converted, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _bool_values(values: Any) -> np.ndarray:
    out: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            out.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            out.append(True)
        elif text in {"false", "0"}:
            out.append(False)
    return np.asarray(out, dtype=bool)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")

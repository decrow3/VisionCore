from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np


ORIENTATIONS = (0, 90, 180, 270)
EPS = 1e-12


def format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def orthonormal_basis(J: np.ndarray, svd_eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    U, S, _ = np.linalg.svd(np.asarray(J, dtype=np.float64), full_matrices=False)
    if S.size == 0:
        return np.zeros((J.shape[0], 0), dtype=np.float64), S
    keep = S > (svd_eps * max(S[0], EPS))
    return U[:, keep], S


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    if U.size == 0 or V.size == 0:
        return np.array([], dtype=np.float64)
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    singular_values = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.arccos(singular_values)


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    angles = principal_angles(U, V)
    if angles.size == 0:
        return float("nan")
    return float(np.mean(np.cos(angles) ** 2))


def project_onto_subspace(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    if U.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    return U @ (U.T @ x)


def compute_signal_covariance(class_means: np.ndarray) -> np.ndarray:
    centered = np.asarray(class_means, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    denom = max(centered.shape[0] - 1, 1)
    cov = (centered.T @ centered) / denom
    return 0.5 * (cov + cov.T)


def compute_alpha(U: np.ndarray, C_signal: np.ndarray) -> float:
    denom = float(np.trace(C_signal)) + EPS
    if U.size == 0:
        return 0.0
    return float(np.trace(U.T @ C_signal @ U) / denom)


def ellipse_from_covariance(C2d: np.ndarray, n_points: int = 200) -> np.ndarray:
    evals, evecs = np.linalg.eigh(np.asarray(C2d, dtype=np.float64))
    evals = np.clip(evals, 0.0, None)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    return (evecs @ np.diag(np.sqrt(evals)) @ circle).T


def _rate_cache_path(rates_dir: Path, logmar: float, orientation: int, condition: str) -> Path:
    lm = format_logmar(logmar)
    hires = rates_dir / f"rates_hires_lm{lm}_ori{orientation}_{condition}.npz"
    lores = rates_dir / f"rates_lm{lm}_ori{orientation}_{condition}.npz"
    if hires.exists():
        return hires
    if lores.exists():
        return lores
    raise FileNotFoundError(f"Missing cached rates for lm={lm} ori={orientation} cond={condition} in {rates_dir}")


def load_eoptotype_trial_means(
    logmar: float,
    condition: str,
    rates_dir: str | Path,
    orientations: tuple[int, ...] = ORIENTATIONS,
) -> dict[int, np.ndarray]:
    rates_dir = Path(rates_dir)
    out: dict[int, np.ndarray] = {}
    for orientation in orientations:
        path = _rate_cache_path(rates_dir, logmar, orientation, condition)
        data = np.load(path, allow_pickle=True)
        rates = np.asarray(data["rates"], dtype=np.float64)
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        trial_means = np.stack([rates[i, : lengths[i]].mean(axis=0) for i in range(rates.shape[0])], axis=0)
        out[int(orientation)] = trial_means
    return out


def find_jacobian_bundle(jacobian_dir: str | Path, logmar: float) -> Path:
    jacobian_dir = Path(jacobian_dir)
    lm = format_logmar(logmar)
    candidates = [
        jacobian_dir / f"test3_lm{lm}.npz",
        jacobian_dir / f"test3_lm{lm}_grid7.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing Jacobian bundle for lm={lm} in {jacobian_dir}")


def load_eoptotype_jacobian(
    logmar: float,
    jacobian_dir: str | Path,
    jacobian_kind: str = "int",
    orientations: tuple[int, ...] = ORIENTATIONS,
) -> tuple[dict[int, np.ndarray], Path]:
    path = find_jacobian_bundle(jacobian_dir, logmar)
    data = np.load(path, allow_pickle=True)
    prefix_map = {
        "int": "J_int_ori",
        "eff": "J_eff_ori",
        "point": "J_ori",
    }
    try:
        prefix = prefix_map[jacobian_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported jacobian_kind={jacobian_kind!r}") from exc

    out: dict[int, np.ndarray] = {}
    for orientation in orientations:
        key = f"{prefix}{orientation}"
        if key not in data.files:
            raise KeyError(f"Key {key} not present in {path.name}")
        out[int(orientation)] = np.asarray(data[key], dtype=np.float64)
    return out, path


def zscore_trial_means_and_jacobians(
    trial_means_by_ori: dict[int, np.ndarray],
    jacobians_by_ori: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    stacked = np.concatenate([trial_means_by_ori[ori] for ori in sorted(trial_means_by_ori)], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    std = np.where(std > 1e-9, std, 1.0)

    trial_means_z = {
        ori: (arr - mean[None, :]) / std[None, :]
        for ori, arr in trial_means_by_ori.items()
    }
    jacobians_z = {
        ori: jac / std[:, None]
        for ori, jac in jacobians_by_ori.items()
    }
    return trial_means_z, jacobians_z, mean, std


def compute_translation_mimicry(
    mu_a: np.ndarray,
    mu_b: np.ndarray,
    J_a: np.ndarray,
    ridge_scale: float = 1e-6,
    arcmin_limits: tuple[float, ...] = (0.5, 1.0, 2.0),
    svd_eps: float = 1e-9,
    n_constrained_angles: int = 720,
) -> dict[str, float]:
    d = np.asarray(mu_b - mu_a, dtype=np.float64)
    J = np.asarray(J_a, dtype=np.float64)
    norm_d2 = float(d @ d) + EPS

    U, S = orthonormal_basis(J, svd_eps=svd_eps)
    proj = project_onto_subspace(d, U)
    mimicry_proj = float(proj @ proj) / norm_d2
    residual_proj = float((d - proj) @ (d - proj)) / norm_d2

    JTJ = J.T @ J
    ridge = float(ridge_scale * np.trace(JTJ) / max(JTJ.shape[0], 1))
    A = JTJ + ridge * np.eye(JTJ.shape[0], dtype=np.float64)
    delta = np.linalg.solve(A, J.T @ d)
    pred_ls = J @ delta
    mimicry_ls = float(pred_ls @ pred_ls) / norm_d2
    residual_ls = float((d - pred_ls) @ (d - pred_ls)) / norm_d2
    singular_max = float(S[0]) if S.size else 0.0
    singular_min = float(S[-1]) if S.size else 0.0
    jacobian_rank = int(U.shape[1])
    jacobian_cond = float(np.linalg.cond(A)) if np.all(np.isfinite(A)) else float("inf")

    row = {
        "mimicry_unconstrained": mimicry_proj,
        "residual_unconstrained": residual_proj,
        "mimicry_unconstrained_ls": mimicry_ls,
        "residual_unconstrained_ls": residual_ls,
        "cosine_alignment": float(np.sqrt(max(mimicry_proj, 0.0))),
        "dx_star_deg": float(delta[0]),
        "dy_star_deg": float(delta[1]),
        "translation_mag_deg": float(np.linalg.norm(delta)),
        "translation_mag_arcmin": float(np.linalg.norm(delta) * 60.0),
        "translation_angle_rad": float(np.arctan2(delta[1], delta[0])),
        "identity_norm": float(np.sqrt(norm_d2)),
        "jacobian_norm_x": float(np.linalg.norm(J[:, 0])),
        "jacobian_norm_y": float(np.linalg.norm(J[:, 1])),
        "jacobian_condition_number": jacobian_cond,
        "jacobian_rank": jacobian_rank,
        "ridge_lambda": ridge,
        "jacobian_singular_max": singular_max,
        "jacobian_singular_min": singular_min,
        "jacobian_near_zero": bool(singular_max < 1e-10),
        "projection_ls_gap": float(abs(mimicry_proj - mimicry_ls)),
    }

    if delta.shape[0] != 2:
        raise ValueError("Constrained mimicry assumes a 2D translation parameterization")

    angles = np.linspace(0.0, 2.0 * np.pi, n_constrained_angles, endpoint=False)
    candidates_unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for lim in arcmin_limits:
        lim_deg = float(lim) / 60.0
        if row["translation_mag_deg"] <= lim_deg:
            delta_c = delta
        else:
            candidates = lim_deg * candidates_unit
            preds = J @ candidates.T
            residuals = d[:, None] - preds
            scores = -np.einsum("ni,ni->i", residuals, residuals)
            delta_c = candidates[int(np.argmax(scores))]
        pred_c = J @ delta_c
        key = str(lim).replace(".", "p")
        row[f"mimicry_constrained_{key}_arcmin"] = float(pred_c @ pred_c) / norm_d2
    return row


def ordered_matrix_from_rows(rows: list[dict], key: str, orientations: tuple[int, ...]) -> np.ndarray:
    mat = np.full((len(orientations), len(orientations)), np.nan, dtype=np.float64)
    idx = {ori: i for i, ori in enumerate(orientations)}
    for row in rows:
        i = idx[int(row["orientation_a"])]
        j = idx[int(row["orientation_b"])]
        mat[i, j] = float(row[key])
    return mat


def symmetrize_ordered_matrix(mat: np.ndarray, reducer: str = "mean") -> np.ndarray:
    out = np.array(mat, copy=True, dtype=np.float64)
    n = out.shape[0]
    for i in range(n):
        out[i, i] = np.nan
        for j in range(i + 1, n):
            a = out[i, j]
            b = out[j, i]
            if reducer == "mean":
                value = np.nanmean([a, b])
            elif reducer == "max":
                value = np.nanmax([a, b])
            elif reducer == "min":
                value = np.nanmin([a, b])
            else:
                raise ValueError(f"Unsupported reducer={reducer!r}")
            out[i, j] = value
            out[j, i] = value
    return out


def mean_offdiag(mat: np.ndarray) -> float:
    mask = ~np.eye(mat.shape[0], dtype=bool)
    values = mat[mask]
    return float(np.nanmean(values))


def maybe_git_commit(repo_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def dump_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
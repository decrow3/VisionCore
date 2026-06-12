"""Covariance-aware FEM optimality analysis helpers.

This module keeps the numerically delicate pieces of the analysis independent
from the model-heavy runner.  The runner can render movies and collect rates,
while these functions own the condition parsing, covariance estimation,
covariance-aware Fisher calculation, and compact diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from VisionCore.covariance import project_to_psd
from VisionCore.subspace import directional_variance_capture, participation_ratio

from .common import DT
from .information import expected_counts_from_rates, finite_difference_derivatives, fisher_scalars
from .lagcube_information import _shift_key
from .pipeline import _trajectory_for_condition


SCALED_FAMILIES = (
    "scaled_real",
    "random_amp_scaled",
    "random_amp_cloud_matched_scaled",
    "trajectory_order_shuffle_scaled",
)
DEFAULT_SCALES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_RATE_GAINS = (0.5, 1.0, 2.0)
DEFAULT_NOISE_FLOOR_MULTIPLIERS = (0.5, 1.0, 2.0)
DEFAULT_K_LIST = (2, 5, 10)
_SCALED_RE = re.compile(r"^(?P<family>scaled_real|random_amp_scaled|random_amp_cloud_matched_scaled|trajectory_order_shuffle_scaled)_D(?P<label>[0-9]+(?:p[0-9]+)?)$")


@dataclass(frozen=True)
class ScaledCondition:
    """Parsed scaled-trajectory condition."""

    family: str
    scale: float
    condition: str


@dataclass(frozen=True)
class CovarianceEstimate:
    """A movement-induced covariance estimate and its diagnostics."""

    family: str
    scale: float
    kind: str
    estimator: str
    covariance: np.ndarray
    n_samples: int
    n_pairs: int


def scale_label(scale: float) -> str:
    """Return a stable filesystem/condition label for a movement scale."""
    return f"{float(scale):.6g}".replace(".", "p").replace("-", "m")


def scaled_condition_name(family: str, scale: float) -> str:
    """Return the canonical condition name for a scaled family and scale."""
    if family not in SCALED_FAMILIES:
        raise ValueError(f"Unknown scaled family {family!r}. Use one of {SCALED_FAMILIES}.")
    return f"{family}_D{scale_label(scale)}"


def parse_scaled_condition(condition: str) -> ScaledCondition:
    """Parse names such as ``scaled_real_D0p50``."""
    match = _SCALED_RE.match(str(condition))
    if not match:
        raise ValueError(
            f"Unsupported scaled condition {condition!r}; expected '<family>_D<scale>' "
            f"with family in {SCALED_FAMILIES}."
        )
    family = match.group("family")
    scale = float(match.group("label").replace("p", ".").replace("m", "-"))
    return ScaledCondition(family=family, scale=scale, condition=str(condition))


def scale_trace(trace: np.ndarray, scale: float, t_max: int | None = None) -> np.ndarray:
    """Mean-center an eye trace and multiply displacement by ``scale``."""
    tr = np.asarray(trace if t_max is None else trace[: int(t_max)], dtype=np.float32)
    if tr.ndim != 2 or tr.shape[1] != 2:
        raise ValueError(f"Expected trace with shape (T, 2), got {tr.shape}")
    if tr.shape[0] == 0:
        raise ValueError("Cannot scale an empty trace.")
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    return (center + float(scale) * (tr - center)).astype(np.float32)


def trajectory_for_scaled_family(
    trace: np.ndarray,
    family: str,
    scale: float,
    *,
    t_max: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Build a scaled real or control trajectory.

    Random controls are generated at empirical scale first, then mean-centered
    and scaled around the measured trace center.  This preserves the existing
    deterministic control construction while exposing movement amplitude.
    """
    if family not in SCALED_FAMILIES:
        raise ValueError(f"Unknown scaled family {family!r}. Use one of {SCALED_FAMILIES}.")
    tr = np.asarray(trace[: int(t_max)], dtype=np.float32)
    if family == "scaled_real":
        return scale_trace(tr, scale), f"measured_trace_scaled_D={float(scale):.6g}"

    base_condition = family.removesuffix("_scaled")
    control, desc = _trajectory_for_condition(tr, base_condition, t_max=int(t_max), seed=int(seed))
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    centered = np.asarray(control, dtype=np.float32) - np.mean(control, axis=0, keepdims=True).astype(np.float32)
    return (center + float(scale) * centered).astype(np.float32), f"{desc}_scaled_D={float(scale):.6g}"


def trajectories_for_scaled_family(
    trace: np.ndarray,
    family: str,
    scales: Iterable[float],
    *,
    t_max: int,
    seed: int,
) -> dict[float, tuple[np.ndarray, str]]:
    """Build all scaled trajectories for one pair/family.

    Random-control families can be expensive, especially cloud matching.  Build
    the empirical-scale control once, then reuse its centered path for all
    requested scales.
    """
    if family not in SCALED_FAMILIES:
        raise ValueError(f"Unknown scaled family {family!r}. Use one of {SCALED_FAMILIES}.")
    scale_values = [float(scale) for scale in scales]
    tr = np.asarray(trace[: int(t_max)], dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    if family == "scaled_real":
        return {
            scale: (scale_trace(tr, scale), f"measured_trace_scaled_D={scale:.6g}")
            for scale in scale_values
        }

    base_condition = family.removesuffix("_scaled")
    control, desc = _trajectory_for_condition(tr, base_condition, t_max=int(t_max), seed=int(seed))
    centered = np.asarray(control, dtype=np.float32) - np.mean(control, axis=0, keepdims=True).astype(np.float32)
    return {
        scale: (
            (center + scale * centered).astype(np.float32),
            f"{desc}_scaled_D={scale:.6g}",
        )
        for scale in scale_values
    }


def counts_and_derivatives_from_shifted_rates(
    rates_by_shift: dict[tuple[float, float], np.ndarray],
    *,
    fisher_step_arcmin: float,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return center expected counts ``mu`` and finite-difference ``J``."""
    h_deg = float(fisher_step_arcmin) / 60.0
    mu0 = expected_counts_from_rates(rates_by_shift[_shift_key(0.0, 0.0)], dt)
    dmu = finite_difference_derivatives(
        expected_counts_from_rates(rates_by_shift[_shift_key(h_deg, 0.0)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(-h_deg, 0.0)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(0.0, h_deg)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(0.0, -h_deg)], dt),
        h_deg,
    )
    return mu0.astype(np.float32), dmu.astype(np.float32)


def independent_fisher_by_time(mu_tn: np.ndarray, j_tnd: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Poisson independent Fisher matrix for each time bin."""
    mu = np.asarray(mu_tn, dtype=np.float64)
    jac = np.asarray(j_tnd, dtype=np.float64)
    if mu.ndim != 2 or jac.shape != mu.shape + (2,):
        raise ValueError(f"Expected mu (T, N) and J (T, N, 2), got {mu.shape} and {jac.shape}")
    out = np.zeros((mu.shape[0], 2, 2), dtype=np.float64)
    for t in range(mu.shape[0]):
        safe = np.clip(mu[t], eps, None)
        out[t] = jac[t].T @ (jac[t] / safe[:, None])
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def covariance_fisher_by_time(
    mu_tn: np.ndarray,
    j_tnd: np.ndarray,
    sigma_extra: np.ndarray | None = None,
    *,
    rate_gain: float = 1.0,
    noise_floor_multiplier: float = 1.0,
    ridge_frac: float = 1e-4,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute block-diagonal population-covariance Fisher by time.

    ``sigma_extra`` is an ``N x N`` population covariance shared across time,
    usually the movement-induced covariance for the relevant family/scale.
    """
    mu = float(rate_gain) * np.asarray(mu_tn, dtype=np.float64)
    jac = float(rate_gain) * np.asarray(j_tnd, dtype=np.float64)
    if mu.ndim != 2 or jac.shape != mu.shape + (2,):
        raise ValueError(f"Expected mu (T, N) and J (T, N, 2), got {mu.shape} and {jac.shape}")
    n_units = mu.shape[1]
    if sigma_extra is None:
        extra = np.zeros((n_units, n_units), dtype=np.float64)
    else:
        extra = np.asarray(sigma_extra, dtype=np.float64)
        if extra.shape != (n_units, n_units):
            raise ValueError(f"sigma_extra shape {extra.shape} incompatible with N={n_units}")
        extra = 0.5 * (extra + extra.T)

    out = np.zeros((mu.shape[0], 2, 2), dtype=np.float64)
    eye = np.eye(n_units, dtype=np.float64)
    for t in range(mu.shape[0]):
        diag = np.clip(float(noise_floor_multiplier) * mu[t], eps, None)
        sigma_t = np.diag(diag) + extra
        sigma_t = 0.5 * (sigma_t + sigma_t.T)
        ridge_base = float(np.median(np.diag(sigma_t))) if n_units else 0.0
        ridge = max(float(ridge_frac) * ridge_base, eps)
        inv = np.linalg.pinv(sigma_t + ridge * eye, hermitian=True)
        out[t] = jac[t].T @ inv @ jac[t]
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def cumulative_trace_and_efficiency(f_by_time: np.ndarray, expected_spikes_t: np.ndarray) -> dict[str, np.ndarray]:
    """Return cumulative Fisher trace and trace per expected spike."""
    mats = np.asarray(f_by_time, dtype=np.float64)
    if mats.ndim != 3 or mats.shape[1:] != (2, 2):
        raise ValueError(f"Expected Fisher by time with shape (T, 2, 2), got {mats.shape}")
    expected = np.asarray(expected_spikes_t, dtype=np.float64)
    if expected.shape != (mats.shape[0],):
        raise ValueError(f"Expected spikes shape {expected.shape} incompatible with T={mats.shape[0]}")
    cum = np.cumsum(mats, axis=0)
    trace = np.trace(cum, axis1=1, axis2=2)
    cumulative_spikes = np.cumsum(expected)
    return {
        "cumulative_fisher_trace": trace.astype(np.float32),
        "cumulative_fisher_trace_per_spike": (trace / np.maximum(cumulative_spikes, 1e-12)).astype(np.float32),
        "cumulative_expected_spikes": cumulative_spikes.astype(np.float32),
    }


def _cov_from_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, int]:
    if not rows:
        raise ValueError("Cannot estimate covariance from zero rows.")
    x = np.concatenate([np.asarray(row, dtype=np.float64) for row in rows], axis=0)
    if x.ndim != 2:
        raise ValueError(f"Expected concatenated response rows with shape (samples, units), got {x.shape}")
    if x.shape[0] <= 1:
        cov = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    else:
        x = x - np.mean(x, axis=0, keepdims=True)
        cov = (x.T @ x) / float(x.shape[0] - 1)
    return project_to_psd(cov, eps=0.0), int(x.shape[0])


def movement_covariance_pooled_residual(mu_rows: Iterable[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Pool time samples after subtracting each pair's own mean response."""
    centered: list[np.ndarray] = []
    n_pairs = 0
    for mu in mu_rows:
        arr = np.asarray(mu, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected each mu row with shape (T, N), got {arr.shape}")
        centered.append(arr - np.mean(arr, axis=0, keepdims=True))
        n_pairs += 1
    cov, n_samples = _cov_from_rows(centered)
    return cov, n_samples, n_pairs


def movement_covariance_within_pair(mu_rows: Iterable[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Average PSD within-pair time covariances."""
    covs: list[np.ndarray] = []
    n_samples = 0
    for mu in mu_rows:
        arr = np.asarray(mu, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected each mu row with shape (T, N), got {arr.shape}")
        n_samples += int(arr.shape[0])
        if arr.shape[0] <= 1:
            cov = np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float64)
        else:
            centered = arr - np.mean(arr, axis=0, keepdims=True)
            cov = (centered.T @ centered) / float(arr.shape[0] - 1)
        covs.append(project_to_psd(cov, eps=0.0))
    if not covs:
        raise ValueError("Cannot estimate covariance from zero rows.")
    return project_to_psd(np.mean(covs, axis=0), eps=0.0), n_samples, len(covs)


def covariance_spectrum_row(estimate: CovarianceEstimate, *, reference_trace: float | None = None) -> dict[str, Any]:
    """Return compact covariance-spectrum diagnostics for CSV output."""
    cov = project_to_psd(estimate.covariance, eps=0.0)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    trace = float(np.sum(np.maximum(evals, 0.0)))
    row: dict[str, Any] = {
        "family": estimate.family,
        "scale_D": float(estimate.scale),
        "kind": estimate.kind,
        "estimator": estimate.estimator,
        "n_samples": int(estimate.n_samples),
        "n_pairs": int(estimate.n_pairs),
        "n_units": int(cov.shape[0]),
        "trace": trace,
        "participation_ratio": participation_ratio(cov),
    }
    if reference_trace is not None and np.isfinite(reference_trace) and float(reference_trace) > 0:
        row["trace_over_D1"] = trace / float(reference_trace)
        row["trace_over_expected_D2"] = trace / (float(reference_trace) * float(estimate.scale) ** 2) if estimate.scale else np.nan
    for k in DEFAULT_K_LIST:
        kk = min(int(k), evals.size)
        row[f"top{k}_variance_fraction"] = float(np.sum(np.maximum(evals[:kk], 0.0)) / max(trace, 1e-12)) if kk else np.nan
        row[f"eig{k}"] = float(evals[kk - 1]) if kk else np.nan
    return row


def top_eigenvectors(mat: np.ndarray, k: int) -> np.ndarray:
    """Return top ``k`` orthonormal eigenvectors of a symmetric matrix."""
    arr = np.asarray(mat, dtype=np.float64)
    arr = 0.5 * (arr + arr.T)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got {arr.shape}")
    if arr.shape[0] == 0 or int(k) <= 0:
        return np.zeros((arr.shape[0], 0), dtype=np.float64)
    vals, vecs = np.linalg.eigh(arr)
    order = np.argsort(vals)[::-1]
    kk = min(int(k), arr.shape[0])
    return vecs[:, order[:kk]]


def coding_covariance_from_j(j_rows: Iterable[np.ndarray]) -> np.ndarray:
    """Neuron-space coding covariance ``sum J_t J_t.T``."""
    mats: list[np.ndarray] = []
    for jac in j_rows:
        arr = np.asarray(jac, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(f"Expected J row with shape (T, N, 2), got {arr.shape}")
        mats.append(np.einsum("tnd,tmd->nm", arr, arr))
    if not mats:
        raise ValueError("Cannot build coding covariance from zero J rows.")
    return project_to_psd(np.sum(mats, axis=0), eps=0.0)


def signal_covariance_from_pair_means(mu_rows: Iterable[np.ndarray]) -> np.ndarray:
    """Covariance of pair-mean responses across image/crop/trace identities."""
    means = [np.mean(np.asarray(mu, dtype=np.float64), axis=0) for mu in mu_rows]
    if not means:
        raise ValueError("Cannot build signal covariance from zero rows.")
    x = np.stack(means, axis=0)
    if x.shape[0] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    return project_to_psd((x.T @ x) / float(x.shape[0] - 1), eps=0.0)


def alignment_rows(
    *,
    family: str,
    scale: float,
    kind: str,
    sigma_fem: np.ndarray,
    coding_cov: np.ndarray,
    signal_cov: np.ndarray | None,
    k_list: Iterable[int] = DEFAULT_K_LIST,
) -> list[dict[str, Any]]:
    """Compute displacement/coding and stimulus-signal alignment diagnostics."""
    sigma = project_to_psd(sigma_fem, eps=0.0)
    coding = project_to_psd(coding_cov, eps=0.0)
    signal = None if signal_cov is None else project_to_psd(signal_cov, eps=0.0)
    rows: list[dict[str, Any]] = []
    for k in k_list:
        kk = int(k)
        u_coding = top_eigenvectors(coding, kk)
        u_fem = top_eigenvectors(sigma, kk)
        row: dict[str, Any] = {
            "family": family,
            "scale_D": float(scale),
            "kind": kind,
            "k": kk,
            "sigma_trace": float(np.trace(sigma)),
            "coding_trace": float(np.trace(coding)),
            "coding_variance_fem": directional_variance_capture(sigma, u_coding) if u_coding.size else np.nan,
            "fem_variance_coding": directional_variance_capture(coding, u_fem) if u_fem.size else np.nan,
        }
        if signal is not None:
            u_signal = top_eigenvectors(signal, kk)
            row.update({
                "signal_trace": float(np.trace(signal)),
                "signal_variance_fem": directional_variance_capture(sigma, u_signal) if u_signal.size else np.nan,
                "fem_variance_signal": directional_variance_capture(signal, u_fem) if u_fem.size else np.nan,
            })
        rows.append(row)
    return rows


def fisher_metric_row(
    *,
    row_id: int,
    record: dict[str, Any],
    family: str,
    scale: float,
    regime: str,
    f_by_time: np.ndarray,
    expected_spikes_t: np.ndarray,
    rate_gain: float = 1.0,
    noise_floor_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Summarize one row/regime Fisher trace."""
    cumulative = cumulative_trace_and_efficiency(f_by_time, expected_spikes_t)
    f_final = np.sum(np.asarray(f_by_time, dtype=np.float64), axis=0)
    scalars = fisher_scalars(f_final)
    expected_total = float(cumulative["cumulative_expected_spikes"][-1])
    out: dict[str, Any] = {
        "row_id": int(row_id),
        "example_id": record.get("example_id", ""),
        "kind": record.get("kind", ""),
        "image_index": int(record.get("image_index", -1)),
        "crop_rank": int(record.get("crop_rank", 0)),
        "family": family,
        "condition": scaled_condition_name(family, scale),
        "scale_D": float(scale),
        "regime": regime,
        "rate_gain": float(rate_gain),
        "noise_floor_multiplier": float(noise_floor_multiplier),
        "final_fisher_trace": float(scalars["trace"]),
        "final_fisher_trace_per_spike": float(scalars["trace"] / max(expected_total, 1e-12)),
        "final_fisher_logdet": float(scalars["logdet"]),
        "final_fisher_det": float(scalars["det"]),
        "final_expected_spikes": expected_total,
        "final_cumulative_fisher_trace": float(cumulative["cumulative_fisher_trace"][-1]),
        "final_cumulative_fisher_trace_per_spike": float(cumulative["cumulative_fisher_trace_per_spike"][-1]),
    }
    return out


def sensitivity_metric_rows(
    *,
    row_id: int,
    record: dict[str, Any],
    family: str,
    scale: float,
    mu_tn: np.ndarray,
    j_tnd: np.ndarray,
    sigma_extra: np.ndarray,
    rate_gains: Iterable[float] = DEFAULT_RATE_GAINS,
    noise_floor_multipliers: Iterable[float] = DEFAULT_NOISE_FLOOR_MULTIPLIERS,
    ridge_frac: float = 1e-4,
) -> list[dict[str, Any]]:
    """Compute pose-blind Fisher rows over gain/noise sensitivity settings."""
    expected = np.sum(np.asarray(mu_tn, dtype=np.float64), axis=1)
    rows: list[dict[str, Any]] = []
    for gain in rate_gains:
        for noise in noise_floor_multipliers:
            f = covariance_fisher_by_time(
                mu_tn,
                j_tnd,
                np.asarray(sigma_extra, dtype=np.float64) * (float(gain) ** 2),
                rate_gain=float(gain),
                noise_floor_multiplier=float(noise),
                ridge_frac=float(ridge_frac),
            )
            rows.append(
                fisher_metric_row(
                    row_id=row_id,
                    record=record,
                    family=family,
                    scale=scale,
                    regime="cov_pose_blind_sensitivity",
                    f_by_time=f,
                    expected_spikes_t=expected * float(gain),
                    rate_gain=float(gain),
                    noise_floor_multiplier=float(noise),
                )
            )
    return rows


def parse_csv_list(value: str | Iterable[str] | None) -> list[str]:
    """Parse comma-separated CLI values."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    out: list[str] = []
    for item in value:
        out.extend(parse_csv_list(str(item)))
    return out


def parse_float_list(value: str | Iterable[str] | None) -> list[float]:
    """Parse comma-separated floats."""
    return [float(part) for part in parse_csv_list(value)]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries, returning an empty list for missing files."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

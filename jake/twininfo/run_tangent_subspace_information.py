"""Figure 4E tangent-subspace information analysis.

This runner connects the existing ``jake.twininfo`` natural-image information
pipeline to the compact translation-tangent basis produced by
``declan.twin_feature_tangent_structure``.  The main metric is a derivative-
projection Poisson Fisher analysis: rates stay in the original non-negative
model units, while the local spatial derivatives are decomposed into a compact
reafferent tangent subspace, its orthogonal complement, and dimension-matched
null bases.

Conceptual target
-----------------
The analysis asks whether the spatial information gained from real fixational
retinal motion is concentrated in the same compact tangent subspace that
explains FEM-linked covariance.  It should be interpreted as a model-observer
analysis of retinal-pose information, not as a behavioral acuity claim.

Typical command
---------------

    python -m jake.twininfo.run_tangent_subspace_information \
      --run-name panelE_production_k10_delta025 \
      --tfts-run outputs/twin_feature_tangent_structure_prod_limited_synth \
      --twininfo-run outputs/twininfo/production_all_images \
      --basis-k 10 \
      --basis-delta-arcmin 0.25 \
      --basis-source image_disjoint \
      --conditions real stabilized \
      --n-null-repeats 100 \
      --recompute

Notes
-----
The stop rule is strict: the TFTS basis dimension must match the unit axis of
the rate maps being analyzed.  For the intended first pass, use the canonical
TFTS shared readout, which should produce rate maps with ``N == 756`` units.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_jsonable(row))


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# -----------------------------------------------------------------------------
# Linear algebra and projected Fisher helpers
# -----------------------------------------------------------------------------


def _shift_key(dx: float, dy: float) -> tuple[float, float]:
    return (round(float(dx), 8), round(float(dy), 8))


def _nearest_value(values: Iterable[float], target: float) -> float:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("Cannot choose nearest value from an empty list")
    return float(min(vals, key=lambda v: abs(v - float(target))))


def _eigh_desc(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(cov, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def _orthonormal_basis_from_columns(mat: np.ndarray, k: int | None = None, eps: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for the column span of ``mat``.

    Uses a thin SVD on ``mat`` (N × ncols) directly — O(N · ncols²) rather than
    O(N³) from forming the N×N Gram matrix — which is a large win when ncols << N
    (e.g. 126 tangent vectors, 756 units).

    Parameters
    ----------
    mat:
        Matrix with shape ``(n_units, n_vectors)``.
    k:
        Optional maximum number of basis dimensions.
    eps:
        Squared-singular-value threshold (matches the original eigenvalue threshold
        since eval_i = s_i²).
    """
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {x.shape}")
    keep_cols = np.all(np.isfinite(x), axis=0) & (np.linalg.norm(x, axis=0) > eps)
    x = x[:, keep_cols]
    if x.size == 0:
        raise ValueError("No finite nonzero tangent columns available for basis")
    U, s, _ = np.linalg.svd(x, full_matrices=False)   # thin SVD: U (N, r), s (r,)
    rank = int(np.sum(s ** 2 > float(eps)))
    if rank <= 0:
        raise ValueError("Tangent covariance has numerical rank zero")
    kk = rank if k is None else min(int(k), rank)
    U_k = U[:, :kk].astype(np.float64, copy=False)
    gram_err = float(np.max(np.abs(U_k.T @ U_k - np.eye(kk))))
    if gram_err > 1e-5:
        raise ValueError(
            f"STOP: basis is not orthonormal after construction (max gram error={gram_err:.2e}). "
            "Likely a numerical failure in the SVD."
        )
    return U_k


def _variance_capture(U: np.ndarray, mat: np.ndarray, eps: float = 1e-12) -> float:
    """Fraction of Frobenius energy in ``mat`` captured by span(U)."""
    u = np.asarray(U, dtype=np.float64)
    x = np.asarray(mat, dtype=np.float64)
    denom = float(np.sum(x * x))
    if denom <= eps:
        return float("nan")
    coeff = u.T @ x
    return float(np.sum(coeff * coeff) / denom)


def _project_unit_axis(arr: np.ndarray, U: np.ndarray, *, unit_axis: int = 1) -> np.ndarray:
    """Project an array along its unit axis into ``span(U)``.

    ``arr`` may be a derivative array with shape ``(T, N, ..., D)``.  The unit
    axis is projected as ``U U.T arr`` while all other axes are preserved.
    """
    x = np.asarray(arr, dtype=np.float64)
    u = np.asarray(U, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"Expected basis U with shape (N, k), got {u.shape}")
    if x.shape[unit_axis] != u.shape[0]:
        raise ValueError(
            f"Unit-axis mismatch: arr axis {unit_axis} has {x.shape[unit_axis]} units, "
            f"basis has {u.shape[0]} rows"
        )
    moved = np.moveaxis(x, unit_axis, -2)  # (..., N, D) for derivative arrays.
    coeff = np.einsum("...nd,nk->...kd", moved, u, optimize=True)
    projected = np.einsum("nk,...kd->...nd", u, coeff, optimize=True)
    return np.moveaxis(projected, -2, unit_axis).astype(np.float64, copy=False)


def _orthogonal_component_unit_axis(arr: np.ndarray, U: np.ndarray, *, unit_axis: int = 1) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) - _project_unit_axis(arr, U, unit_axis=unit_axis)


def _fisher_project_unit_axis(
    dmu: np.ndarray,
    mu0: np.ndarray,
    U: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project dmu into span(U) using the Poisson Fisher metric W = diag(1/mu0).

    The unit axis is always axis 1.  Supports arbitrary extra spatial dimensions:
    ``dmu`` shape ``(T, N, *spatial, D)`` and ``mu0`` shape ``(T, N, *spatial)``.
    All (T × prod(spatial)) positions are projected independently using their
    own per-position Fisher weight W = diag(1/mu0).

    Unlike the Euclidean projection (U U.T dmu), this guarantees an exact
    Fisher-information partition:

        Fisher(dmu_proj, mu0) + Fisher(dmu - dmu_proj, mu0) == Fisher(dmu, mu0)

    so tangent and orthogonal-complement Fisher values always sum to the full
    value and neither can exceed it.
    """
    dmu_ = np.asarray(dmu, dtype=np.float64)
    mu_ = np.asarray(mu0, dtype=np.float64)
    U_ = np.asarray(U, dtype=np.float64)

    if dmu_.ndim < 3 or mu_.ndim < 2:
        raise ValueError(
            f"_fisher_project_unit_axis expects dmu (T,N,...,D) and mu0 (T,N,...); "
            f"got dmu {dmu_.shape}, mu0 {mu_.shape}"
        )

    T, N = dmu_.shape[0], dmu_.shape[1]
    D = dmu_.shape[-1]
    spatial = dmu_.shape[2:-1]          # e.g. (H, W) or () for plain (T, N, D)
    if mu_.shape != (T, N) + spatial:
        raise ValueError(
            f"Shape mismatch: dmu {dmu_.shape} implies mu0 should be {(T, N) + spatial}, "
            f"got {mu_.shape}"
        )
    if N != U_.shape[0]:
        raise ValueError(f"Unit axis mismatch: dmu axis 1 = {N}, U rows = {U_.shape[0]}")

    # Batch all non-N dims: (T, N, *spatial, D) → (B, N, D) and (T, N, *spatial) → (B, N)
    # Strategy: moveaxis N from 1 to -2, then flatten leading dims into B.
    dmu_moved = np.moveaxis(dmu_, 1, -2)           # (T, *spatial, N, D)
    mu_moved  = np.moveaxis(mu_,  1, -1)           # (T, *spatial, N)
    B = int(np.prod(dmu_moved.shape[:-2]))         # T * prod(spatial)
    dmu_2d = dmu_moved.reshape(B, N, D)            # (B, N, D)
    mu_2d  = mu_moved.reshape(B, N)               # (B, N)

    w     = 1.0 / np.maximum(mu_2d, eps)                          # (B, N)
    WU    = w[:, :, np.newaxis] * U_[np.newaxis, :, :]            # (B, N, k)
    UTWU  = np.einsum("nk,Bnj->Bkj", U_, WU)                     # (B, k, k)
    WJ    = w[:, :, np.newaxis] * dmu_2d                          # (B, N, D)
    UtWJ  = np.einsum("nk,Bnd->Bkd", U_, WJ)                     # (B, k, D)
    alpha = np.linalg.solve(UTWU, UtWJ)                           # (B, k, D)
    proj_2d = np.einsum("nk,Bkd->Bnd", U_, alpha)                # (B, N, D)

    # Reshape back: (B, N, D) → (T, *spatial, N, D) → (T, N, *spatial, D)
    proj_moved = proj_2d.reshape(dmu_moved.shape)                  # (T, *spatial, N, D)
    return np.moveaxis(proj_moved, -2, 1).astype(np.float64, copy=False)  # (T, N, *spatial, D)


def _finite_difference_dmu_from_rates(
    rates_by_shift: dict[tuple[float, float], np.ndarray],
    *,
    fisher_step_arcmin: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``mu0`` and local spatial derivative ``dmu`` in expected counts.

    Uses the same central finite-difference convention as ``twininfo``.
    """
    from .information import expected_counts_from_rates, finite_difference_derivatives

    h_deg = float(fisher_step_arcmin) / 60.0
    mu0 = expected_counts_from_rates(rates_by_shift[_shift_key(0.0, 0.0)], dt)
    dmu = finite_difference_derivatives(
        expected_counts_from_rates(rates_by_shift[_shift_key(h_deg, 0.0)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(-h_deg, 0.0)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(0.0, h_deg)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(0.0, -h_deg)], dt),
        h_deg,
    )
    return mu0.astype(np.float64, copy=False), dmu.astype(np.float64, copy=False)


def cumulative_projected_fisher(
    rates_by_shift: dict[tuple[float, float], np.ndarray],
    *,
    fisher_step_arcmin: float,
    dt: float,
    U: np.ndarray | None = None,
    component: str = "full",
    projection_mode: str = "derivative_fisher",
) -> dict[str, np.ndarray]:
    """Compute cumulative Fisher traces after derivative projection.

    Parameters
    ----------
    rates_by_shift:
        Mapping from ``(dx_deg, dy_deg)`` to model rates.  Arrays may have shape
        ``(T, N)`` or ``(T, N, H, W)``.
    fisher_step_arcmin:
        Central finite-difference step in arcminutes.
    dt:
        Seconds per sample.
    U:
        Basis over units, shape ``(N, k)``. Required for all components except
        ``full``.
    component:
        One of ``full``, ``basis``, or ``orthogonal``.
    projection_mode:
        ``'derivative_fisher'`` (default): project in the Poisson Fisher metric
        W = diag(1/mu0), guaranteeing Fisher(basis) + Fisher(orthogonal) == Fisher(full).
        ``'derivative_euclidean'``: Euclidean U U.T projection — does NOT satisfy
        the partition and can produce basis Fisher > full Fisher for shuffled bases.
    """
    from .information import fisher_by_time

    mu0, dmu_full = _finite_difference_dmu_from_rates(
        rates_by_shift,
        fisher_step_arcmin=fisher_step_arcmin,
        dt=dt,
    )
    if component == "full":
        dmu = dmu_full
    elif component == "basis":
        if U is None:
            raise ValueError("U is required for component='basis'")
        if projection_mode == "derivative_fisher":
            dmu = _fisher_project_unit_axis(dmu_full, mu0, U)
        else:
            dmu = _project_unit_axis(dmu_full, U, unit_axis=1)
    elif component == "orthogonal":
        if U is None:
            raise ValueError("U is required for component='orthogonal'")
        if projection_mode == "derivative_fisher":
            dmu = dmu_full - _fisher_project_unit_axis(dmu_full, mu0, U)
        else:
            dmu = _orthogonal_component_unit_axis(dmu_full, U, unit_axis=1)
    else:
        raise ValueError("component must be one of 'full', 'basis', 'orthogonal'")

    total_by_t, total_cum, pattern_by_t, pattern_cum = fisher_by_time(mu0, dmu)
    total_trace = np.trace(total_cum, axis1=1, axis2=2)
    pattern_trace = np.trace(pattern_cum, axis1=1, axis2=2)
    expected_spikes = np.cumsum(np.sum(mu0, axis=tuple(range(1, mu0.ndim))))
    return {
        "cumulative_fisher_total": total_trace.astype(np.float32),
        "cumulative_fisher_pattern": pattern_trace.astype(np.float32),
        "cumulative_fisher_pattern_per_spike": (pattern_trace / np.maximum(expected_spikes, 1e-12)).astype(np.float32),
        "cumulative_expected_spikes": expected_spikes.astype(np.float32),
        "fisher_total_by_time": total_by_t.astype(np.float32),
        "fisher_pattern_by_time": pattern_by_t.astype(np.float32),
        "mu0": mu0.astype(np.float32),
        "dmu_full": dmu_full.astype(np.float32),
        "dmu_component": dmu.astype(np.float32),
    }


# -----------------------------------------------------------------------------
# TFTS basis loading
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BasisResult:
    U: np.ndarray
    basis_type: str
    basis_source: str
    basis_k: int
    basis_delta_arcmin: float
    n_units: int
    n_objects_used: int
    n_tangent_vectors: int
    excluded_image_id: int | None
    variance_capture_self: float


def _load_tfts_payload(tfts_run: Path, delta_arcmin: float) -> tuple[float, dict[str, dict[str, Any]]]:
    path = Path(tfts_run) / "tangent_maps" / "twin_tangent_maps.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing TFTS tangent payload: {path}")
    with path.open("rb") as handle:
        cached = pickle.load(handle)
    available = [float(v) for v in cached["delta_arcmins"]]
    delta = _nearest_value(available, float(delta_arcmin))
    payload_by_delta = cached["object_payload"]
    try:
        payload = payload_by_delta[delta]
    except KeyError:
        payload = payload_by_delta[float(delta)]
    return float(delta), {str(k): v for k, v in payload.items()}


def _tangent_matrix_from_payload(
    payload: dict[str, dict[str, Any]],
    *,
    exclude_image_id: int | None = None,
    exclude_object_ids: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    object_ids: list[str] = []
    exclude_object_ids = set() if exclude_object_ids is None else set(exclude_object_ids)
    for object_id in sorted(payload):
        if object_id in exclude_object_ids:
            continue
        meta = payload[object_id]
        if exclude_image_id is not None and int(meta.get("image_id", -999999)) == int(exclude_image_id):
            continue
        bx = np.asarray(meta["bx"], dtype=np.float64).ravel()
        by = np.asarray(meta["by"], dtype=np.float64).ravel()
        if bx.shape != by.shape:
            continue
        if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))):
            continue
        if np.linalg.norm(bx) <= 1e-12 and np.linalg.norm(by) <= 1e-12:
            continue
        cols.extend([bx, by])
        object_ids.append(object_id)
    if not cols:
        raise ValueError("No valid TFTS tangent vectors after filtering")
    return np.stack(cols, axis=1), object_ids


def load_tfts_basis(
    tfts_run: Path,
    *,
    delta_arcmin: float,
    k: int,
    basis_source: str,
    image_id: int | None = None,
) -> BasisResult:
    """Load/recompute a compact TFTS basis.

    ``basis_source='all_objects'`` uses every valid object.  ``'image_disjoint'``
    excludes objects with ``image_id`` matching the current movie before
    learning the basis.  This implements the intended no-image-leakage version
    for Panel E.
    """
    delta, payload = _load_tfts_payload(Path(tfts_run), delta_arcmin)
    exclude_image_id = None
    if basis_source in {"image_disjoint", "leave_image_out", "image_disjoint_leave_one_out"}:
        if image_id is None:
            raise ValueError(f"basis_source={basis_source!r} requires image_id")
        exclude_image_id = int(image_id)
    elif basis_source != "all_objects":
        raise ValueError("basis_source must be 'all_objects' or 'image_disjoint'")

    mat, object_ids = _tangent_matrix_from_payload(payload, exclude_image_id=exclude_image_id)
    U = _orthonormal_basis_from_columns(mat, k=int(k))
    return BasisResult(
        U=U,
        basis_type="tangent",
        basis_source=basis_source,
        basis_k=int(U.shape[1]),
        basis_delta_arcmin=float(delta),
        n_units=int(U.shape[0]),
        n_objects_used=int(len(object_ids)),
        n_tangent_vectors=int(mat.shape[1]),
        excluded_image_id=exclude_image_id,
        variance_capture_self=_variance_capture(U, mat),
    )


def unit_shuffled_tfts_basis(
    tfts_run: Path,
    *,
    delta_arcmin: float,
    k: int,
    basis_source: str,
    image_id: int | None,
    seed: int,
) -> BasisResult:
    """Dimensionality-matched unit-shuffled tangent basis."""
    delta, payload = _load_tfts_payload(Path(tfts_run), delta_arcmin)
    exclude_image_id = int(image_id) if basis_source != "all_objects" and image_id is not None else None
    mat, object_ids = _tangent_matrix_from_payload(payload, exclude_image_id=exclude_image_id)
    rng = np.random.default_rng(int(seed))
    shuf = np.stack([col[rng.permutation(col.shape[0])] for col in mat.T], axis=1)
    U = _orthonormal_basis_from_columns(shuf, k=int(k))
    return BasisResult(
        U=U,
        basis_type="unit_shuffle",
        basis_source=basis_source,
        basis_k=int(U.shape[1]),
        basis_delta_arcmin=float(delta),
        n_units=int(U.shape[0]),
        n_objects_used=int(len(object_ids)),
        n_tangent_vectors=int(mat.shape[1]),
        excluded_image_id=exclude_image_id,
        variance_capture_self=_variance_capture(U, shuf),
    )


def random_orthogonal_basis(n_units: int, k: int, *, seed: int) -> BasisResult:
    rng = np.random.default_rng(int(seed))
    q, _ = np.linalg.qr(rng.normal(size=(int(n_units), int(k))))
    U = q[:, : int(k)]
    return BasisResult(
        U=U,
        basis_type="random_orthogonal",
        basis_source="random",
        basis_k=int(U.shape[1]),
        basis_delta_arcmin=float("nan"),
        n_units=int(U.shape[0]),
        n_objects_used=0,
        n_tangent_vectors=0,
        excluded_image_id=None,
        variance_capture_self=float("nan"),
    )


# -----------------------------------------------------------------------------
# Main production analysis
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDirs:
    run: Path
    cache: Path
    figures: Path
    metadata: Path
    results: Path


def _ensure_run_dirs(run_dir: Path) -> RunDirs:
    dirs = RunDirs(
        run=Path(run_dir),
        cache=Path(run_dir) / "cache",
        figures=Path(run_dir) / "figures",
        metadata=Path(run_dir) / "metadata",
        results=Path(run_dir) / "results",
    )
    for path in (dirs.run, dirs.cache, dirs.figures, dirs.metadata, dirs.results):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _pipeline_config_from_run(twininfo_run: Path | None, overrides: argparse.Namespace) -> Any:
    """Return a ``PipelineConfig`` using an existing run_config when available."""
    from .pipeline import PipelineConfig

    if twininfo_run is None:
        return PipelineConfig(
            run_name=overrides.source_run_name,
            seed=int(overrides.seed),
            image_indices=overrides.image_indices,
            n_crops_per_image=int(overrides.n_crops_per_image),
            n_examples_per_kind=int(overrides.n_examples_per_kind),
            selected_trace_example_ids=tuple(overrides.selected_trace_example_ids),
            t_max=int(overrides.t_max),
            stride=int(overrides.stride),
            population_size=int(overrides.population_size),
            population_selection="top_performance",
            performance_metric="ccnorm",
            population_grid_position_mode="center",
            deduplicate_units=False,
            batch_size=int(overrides.batch_size),
            fisher_step_arcmin=float(overrides.fisher_step_arcmin),
            shift_grid_mode="cross",
            recompute=True,
        )

    cfg_path = Path(twininfo_run) / "metadata" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot reconstruct twininfo config; missing {cfg_path}")
    raw = _read_json(cfg_path)
    valid_fields = {f.name for f in fields(PipelineConfig)}
    kwargs = {key: raw[key] for key in valid_fields if key in raw}
    kwargs["run_name"] = overrides.source_run_name or str(raw.get("run_name") or Path(twininfo_run).name)
    kwargs["make_stimulus_movies"] = False
    kwargs["make_activation_movies"] = False
    kwargs["recompute"] = True
    kwargs["fisher_step_arcmin"] = float(overrides.fisher_step_arcmin)
    kwargs["batch_size"] = int(overrides.batch_size)
    return PipelineConfig(**kwargs)


class _CanonicalPopulation:
    """Minimal object accepted by ``run_shifted_lag_cube_rate_maps``."""

    def __init__(self, readout: Any):
        self.readout = readout


def _load_canonical_context(model_device: str) -> Any:
    """Load the canonical TFTS shared readout context.

    This reuses the audited TFTS loader so the information analysis and tangent
    basis live on the same unit axis.
    """
    from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import _load_twin_context

    return _load_twin_context(model_device=model_device)


def _condition_fisher_rows(
    *,
    rates_by_shift: dict[tuple[float, float], np.ndarray],
    basis: BasisResult,
    null_bases: list[tuple[str, BasisResult]],
    fisher_step_arcmin: float,
    dt: float,
    projection_mode: str = "derivative_fisher",
    metric: str = "pattern_fisher_trace",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Compute full, tangent, orthogonal, and null projected Fisher traces.

    ``mu0`` and ``dmu_full`` are computed once and shared across all projections.
    ``basis`` and ``null_bases`` must be pre-built by the caller (once per
    image/example) so basis construction is not repeated across conditions.
    """
    if projection_mode not in {"derivative_euclidean", "derivative_fisher"}:
        raise NotImplementedError(
            f"projection_mode={projection_mode!r} is not yet implemented. "
            "Supported: 'derivative_fisher' (default), 'derivative_euclidean' (legacy, no partition guarantee)."
        )
    if metric != "pattern_fisher_trace":
        raise NotImplementedError(
            f"metric={metric!r} is not yet implemented. "
            "Only 'pattern_fisher_trace' is supported."
        )

    n_rate_units = int(rates_by_shift[_shift_key(0.0, 0.0)].shape[1])
    if n_rate_units != basis.n_units:
        raise ValueError(
            "STOP: rate-map unit axis does not match TFTS basis. "
            f"rate units={n_rate_units}, basis rows={basis.n_units}. "
            "Use the canonical TFTS readout or add an explicit manifest-alignment step."
        )

    from .information import fisher_by_time

    # Compute mu0 and dmu_full ONCE — reused for every projection below.
    mu0, dmu_full = _finite_difference_dmu_from_rates(
        rates_by_shift, fisher_step_arcmin=fisher_step_arcmin, dt=dt
    )
    expected_spikes = np.cumsum(np.sum(mu0, axis=tuple(range(1, mu0.ndim))))

    def _traces(dmu: np.ndarray) -> dict[str, np.ndarray]:
        total_by_t, total_cum, pattern_by_t, pattern_cum = fisher_by_time(mu0, dmu)
        total_trace   = np.trace(total_cum,   axis1=1, axis2=2)
        pattern_trace = np.trace(pattern_cum, axis1=1, axis2=2)
        return {
            "cumulative_fisher_total":             total_trace.astype(np.float32),
            "cumulative_fisher_pattern":           pattern_trace.astype(np.float32),
            "cumulative_fisher_pattern_per_spike": (
                pattern_trace / np.maximum(expected_spikes, 1e-12)
            ).astype(np.float32),
            "cumulative_expected_spikes":          expected_spikes.astype(np.float32),
        }

    def _project(U: np.ndarray, component: str) -> np.ndarray:
        if projection_mode == "derivative_fisher":
            proj = _fisher_project_unit_axis(dmu_full, mu0, U)
        else:
            proj = _project_unit_axis(dmu_full, U, unit_axis=1)
        return proj if component == "basis" else dmu_full - proj

    # (label, basis_result_or_None, component)
    trace_specs: list[tuple[str, BasisResult | None, str]] = [
        ("full",                  None,  "full"),
        ("tangent",               basis, "basis"),
        ("orthogonal_complement", basis, "orthogonal"),
    ]
    for label, null_br in null_bases:
        trace_specs.append((label, null_br, "basis"))

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for label, basis_result, component in trace_specs:
        dmu = dmu_full if component == "full" else _project(basis_result.U, component)
        fisher = _traces(dmu)

        arrays[f"{label}/cumulative_fisher_pattern"]           = fisher["cumulative_fisher_pattern"]
        arrays[f"{label}/cumulative_fisher_total"]             = fisher["cumulative_fisher_total"]
        arrays[f"{label}/cumulative_fisher_pattern_per_spike"] = fisher["cumulative_fisher_pattern_per_spike"]
        rows.append({
            "basis_label":               label,
            "projection_mode":           projection_mode,
            "metric":                    metric,
            "basis_type":                "full" if basis_result is None else basis_result.basis_type,
            "basis_source":              "full" if basis_result is None else basis_result.basis_source,
            "basis_k":                   int(n_rate_units) if basis_result is None else int(basis_result.basis_k),
            "basis_delta_arcmin":        float("nan") if basis_result is None else float(basis_result.basis_delta_arcmin),
            "basis_n_units":             int(n_rate_units) if basis_result is None else int(basis_result.n_units),
            "basis_n_objects_used":      0 if basis_result is None else int(basis_result.n_objects_used),
            "basis_n_tangent_vectors":   0 if basis_result is None else int(basis_result.n_tangent_vectors),
            "basis_excluded_image_id":   (
                "" if basis_result is None or basis_result.excluded_image_id is None
                else int(basis_result.excluded_image_id)
            ),
            "basis_self_variance_capture": (
                float("nan") if basis_result is None else float(basis_result.variance_capture_self)
            ),
            "final_cumulative_fisher_pattern":           float(fisher["cumulative_fisher_pattern"][-1]),
            "final_cumulative_fisher_total":             float(fisher["cumulative_fisher_total"][-1]),
            "final_cumulative_fisher_pattern_per_spike": float(fisher["cumulative_fisher_pattern_per_spike"][-1]),
            "final_cumulative_expected_spikes":          float(fisher["cumulative_expected_spikes"][-1]),
        })

    # Partition check: tangent + orthogonal total Fisher must equal full.
    if projection_mode == "derivative_fisher":
        full_t = arrays.get("full/cumulative_fisher_total")
        tang_t = arrays.get("tangent/cumulative_fisher_total")
        orth_t = arrays.get("orthogonal_complement/cumulative_fisher_total")
        if full_t is not None and tang_t is not None and orth_t is not None:
            max_err = float(np.max(np.abs((tang_t + orth_t - full_t) / np.maximum(np.abs(full_t), 1.0))))
            print(f"    partition residual: {max_err:.2e}", flush=True)
            if max_err > 1e-3:
                raise ValueError(
                    f"STOP: Fisher partition violated — tangent + orthogonal != full "
                    f"(max relative error = {max_err:.3e}). "
                    "Check _fisher_project_unit_axis implementation."
                )

    return rows, arrays


def _summarize_gain(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize basis-specific real-minus-stabilized gains."""
    # example_id is the unique trace identifier here; it encodes trace_id implicitly.
    # If multiple trace_ids per example_id are ever introduced, add "trace_id" to key_cols.
    key_cols = ("example_id", "image_index", "crop_rank")
    lookup: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in final_rows:
        key = tuple(row[c] for c in key_cols) + (row["basis_label"],)
        lookup.setdefault(key, {})[str(row["condition"])] = row

    basis_labels = sorted({str(row["basis_label"]) for row in final_rows})
    out: list[dict[str, Any]] = []
    movie_keys = sorted({tuple(row[c] for c in key_cols) for row in final_rows})
    for movie_key in movie_keys:
        full_key = movie_key + ("full",)
        full_real = lookup.get(full_key, {}).get("real")
        full_stab = lookup.get(full_key, {}).get("stabilized")
        if full_real is None or full_stab is None:
            continue
        i_real_full  = float(full_real["final_cumulative_fisher_pattern"])
        i_stab_full  = float(full_stab["final_cumulative_fisher_pattern"])
        full_gain    = i_real_full - i_stab_full
        for basis_label in basis_labels:
            conds = lookup.get(movie_key + (basis_label,), {})
            real = conds.get("real")
            stab = conds.get("stabilized")
            if real is None or stab is None:
                continue
            i_real_basis = float(real["final_cumulative_fisher_pattern"])
            i_stab_basis = float(stab["final_cumulative_fisher_pattern"])

            # Basis-specific gain: both real and stabilized use the same projection.
            # Captures how much information the basis adds over its own stabilized baseline.
            gain_basis_baseline = i_real_basis - i_stab_basis

            # Full-population-stabilized gain (handoff preferred formula):
            # numerator uses I_real_basis but the baseline is the full-population stabilized.
            # Directly answers "how much of the full FEM information gain sits in the basis subspace?"
            gain_full_baseline = i_real_basis - i_stab_full

            # Fraction is only interpretable when full_gain > 0; flag rather than mask so
            # callers can decide what to do with negative-gain windows.
            full_gain_positive = full_gain > 0
            out.append({
                "example_id": movie_key[0],
                "image_index": int(movie_key[1]),
                "crop_rank": int(movie_key[2]),
                "kind": str(real.get("kind", "")),
                "full_gain_positive": full_gain_positive,
                "basis_label": basis_label,
                "basis_type": real["basis_type"],
                "basis_k": real["basis_k"],
                "real_final_fisher_pattern": i_real_basis,
                "stabilized_basis_final_fisher_pattern": i_stab_basis,
                "stabilized_full_final_fisher_pattern": i_stab_full,
                "gain_over_stabilized": gain_basis_baseline,
                "gain_vs_full_stab": gain_full_baseline,
                "full_gain_over_stabilized": full_gain,
                # fraction using basis-specific stabilized baseline
                # NaN when full_gain <= 0 — exclude from summary statistics
                "fraction_full_fem_gain_captured": (
                    gain_basis_baseline / full_gain
                    if full_gain_positive and abs(full_gain) > 1e-12 else float("nan")
                ),
                # fraction using full-population stabilized baseline (handoff preferred)
                "fraction_full_fem_gain_captured_full_baseline": (
                    gain_full_baseline / full_gain
                    if full_gain_positive and abs(full_gain) > 1e-12 else float("nan")
                ),
                "fraction_full_real_information": (
                    i_real_basis / i_real_full if abs(i_real_full) > 1e-12 else float("nan")
                ),
            })
    return out


def _plot_panel_summary(
    gain_rows: list[dict[str, Any]],
    series_records: list[dict[str, Any]],
    series_arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Write a 3-panel Figure 4E diagnostic plot.

    Panel 0 — cumulative Fisher trace for a representative positive-gain real movie.
    Panel 1 — absolute FEM gain by basis, split by window kind (all windows).
    Panel 2 — fraction of FEM gain captured by basis, split by kind,
               restricted to positive-gain windows where the fraction is meaningful.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not gain_rows or not series_records:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No rows to plot", ha="center", va="center")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    summary_labels = ["full", "tangent", "orthogonal_complement", "unit_shuffle", "random_orthogonal"]
    kinds = ["microsaccade", "fixation"]
    kind_color = {"microsaccade": "#f58518", "fixation": "#4c78a8"}

    # Positive-gain windows: fraction is interpretable only when full_gain > 0.
    pos_rows = [r for r in gain_rows if float(r.get("full_gain_over_stabilized", 0)) > 0]
    movie_keys_all = {(r["example_id"], r["image_index"], r["crop_rank"])
                      for r in gain_rows if r["basis_label"] == "full"}
    movie_keys_pos = {(r["example_id"], r["image_index"], r["crop_rank"])
                      for r in pos_rows if r["basis_label"] == "full"}

    def _vals(rows: list[dict[str, Any]], col: str, label: str) -> np.ndarray:
        raw = [
            float(r[col]) for r in rows
            if (str(r["basis_label"]) == label or str(r["basis_label"]).startswith(label + "_"))
            and np.isfinite(float(r[col]))
        ]
        return np.asarray(raw, dtype=np.float64)

    def _bar_stats(rows: list[dict[str, Any]], col: str) -> tuple[list[float], list[float]]:
        means, sems = [], []
        for label in summary_labels:
            v = _vals(rows, col, label)
            means.append(float(np.mean(v)) if v.size else float("nan"))
            sems.append(float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)
        return means, sems

    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # --- Panel 0: cumulative trace for first positive-gain real movie ---
    pos_example_ids = {r["example_id"] for r in pos_rows}
    first = next(
        (r for r in series_records if r["condition"] == "real" and r["example_id"] in pos_example_ids),
        next((r for r in series_records if r["condition"] == "real"), None),
    )
    if first is not None:
        time_s = series_arrays.get("time_s", np.array([]))
        trace_labels = ["full", "tangent", "orthogonal_complement", "unit_shuffle_mean", "random_orthogonal_mean"]
        for lbl in trace_labels:
            arr_key = f"{first['series_id']}/{lbl}/cumulative_fisher_pattern"
            if arr_key in series_arrays:
                axs[0].plot(time_s, series_arrays[arr_key], lw=1.5, label=lbl.replace("_", " "))
        gain_sign = "pos" if first["example_id"] in pos_example_ids else "neg"
        axs[0].set_title(f"Cumulative Fisher trace\n({first['example_id']}, {first.get('kind','')}, gain={gain_sign})")
        axs[0].set_xlabel("time (s)")
        axs[0].set_ylabel("cumulative pattern Fisher")
        axs[0].legend(frameon=False, fontsize=7)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    # --- Panel 1: absolute FEM gain by basis × kind (all windows) ---
    x = np.arange(len(summary_labels), dtype=float)
    width = 0.38
    for ki, kind in enumerate(kinds):
        kind_rows = [r for r in gain_rows if r.get("kind") == kind]
        means, sems = _bar_stats(kind_rows, "gain_over_stabilized")
        offset = (ki - 0.5) * width
        axs[1].bar(x + offset, means, width, yerr=sems, capsize=3, alpha=0.85,
                   color=kind_color[kind], label=kind)
    axs[1].axhline(0.0, color="0.25", lw=1)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([s.replace("_", " ") for s in summary_labels], rotation=30, ha="right", fontsize=8)
    axs[1].set_ylabel("absolute FEM gain\n(real − stabilized Fisher)")
    axs[1].set_title(f"Absolute gain by basis × kind\n(all {len(movie_keys_all)} windows)")
    axs[1].legend(frameon=False, fontsize=8)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    # --- Panel 2: fraction of FEM gain, positive-gain windows only ---
    for ki, kind in enumerate(kinds):
        kind_pos = [r for r in pos_rows if r.get("kind") == kind]
        n_kind_pos = len({(r["example_id"], r["image_index"], r["crop_rank"])
                          for r in kind_pos if r["basis_label"] == "full"})
        n_kind_all = len({(r["example_id"], r["image_index"], r["crop_rank"])
                          for r in gain_rows if r["basis_label"] == "full" and r.get("kind") == kind})
        means, sems = _bar_stats(kind_pos, "fraction_full_fem_gain_captured")
        offset = (ki - 0.5) * width
        axs[2].bar(x + offset, means, width, yerr=sems, capsize=3, alpha=0.85,
                   color=kind_color[kind],
                   label=f"{kind} ({n_kind_pos}/{n_kind_all} pos-gain)")
    axs[2].axhline(1.0, color="0.25", lw=1, ls="--", label="full gain")
    axs[2].axhline(0.0, color="0.25", lw=1)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([s.replace("_", " ") for s in summary_labels], rotation=30, ha="right", fontsize=8)
    axs[2].set_ylabel("fraction full FEM gain captured")
    axs[2].set_title(
        f"Fraction of FEM gain by basis × kind\n"
        f"({len(movie_keys_pos)}/{len(movie_keys_all)} positive-gain windows)"
    )
    axs[2].legend(frameon=False, fontsize=7)
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)

    fig.suptitle("Figure 4E diagnostic: tangent-subspace FEM information", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_tangent_subspace_information(args: argparse.Namespace) -> dict[str, Any]:
    from .common import DT, N_LAGS
    from .image_selection import crop_rows, select_image_crops
    from .lagcube_information import finite_difference_shift_set, run_shifted_lag_cube_rate_maps
    from .pipeline import _condition_blocks, _example_seed, _filter_trace_examples
    from .retinal_examples import model_lag_cubes_from_image_trace
    from .trace_selection import run_trace_selection_step

    output_root = Path(args.output_root)
    run_name = args.run_name or "panelE_tangent_subspace_information"
    dirs = _ensure_run_dirs(output_root / run_name)
    summary_path = dirs.metadata / "run_summary.json"
    if summary_path.exists() and not args.recompute:
        return _read_json(summary_path)

    config = _pipeline_config_from_run(Path(args.twininfo_run) if args.twininfo_run else None, args)
    _write_json(dirs.metadata / "source_twininfo_config.json", config.__dict__)
    _write_json(dirs.metadata / "panelE_config.json", vars(args))

    ctx = _load_canonical_context(model_device=str(args.model_device))
    population = _CanonicalPopulation(ctx.readout)
    model = ctx.model
    device = getattr(model, "device", args.model_device)

    examples = run_trace_selection_step(
        figure_dir=dirs.figures,
        metadata_dir=dirs.metadata,
        seed=int(config.seed),
        n_examples_per_kind=int(config.n_examples_per_kind),
        t_max=int(config.t_max),
        stride=int(config.stride),
        model=model,
    )
    examples = _filter_trace_examples(config, examples)
    image_by_index, image_crops, image_figures = select_image_crops(
        image_indices=config.image_indices,
        n_crops_per_image=int(config.n_crops_per_image),
        figure_dir=dirs.figures,
        metadata_path=dirs.metadata / "02_image_crop_hotspots.csv",
        seed=int(config.seed),
        t_max=int(config.t_max),
        trace_arrays=[example.trace for example in examples],
    )
    crops = crop_rows(image_crops)
    shifts = finite_difference_shift_set(float(args.fisher_step_arcmin))

    final_rows: list[dict[str, Any]] = []
    series_records: list[dict[str, Any]] = []
    series_arrays: dict[str, np.ndarray] = {
        "time_s": (np.arange(int(config.t_max), dtype=np.float32) * float(DT)).astype(np.float32)
    }
    movie_manifest: list[dict[str, Any]] = []

    conditions = tuple(str(c) for c in args.conditions)
    pair_count = len(crops) * len(examples)
    pair_i = 0
    for crop in crops:
        image_index = int(crop["image_index"])
        crop_rank = int(crop["crop_rank"])
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        image = image_by_index[image_index]
        for example in examples:
            pair_i += 1
            print(f"PanelE pair {pair_i}/{pair_count}: {example.example_id} image={image_index} crop={crop_rank}", flush=True)
            seed = _example_seed(int(config.seed), example.example_id, image_index, crop_rank)
            # Build TFTS basis and null bases once per (image, example) — reused
            # across conditions (real/stabilized) to avoid redundant construction.
            _basis = load_tfts_basis(
                Path(args.tfts_run),
                delta_arcmin=float(args.basis_delta_arcmin),
                k=int(args.basis_k),
                basis_source=str(args.basis_source),
                image_id=image_index,
            )
            _null_bases: list[tuple[str, BasisResult]] = []
            for _rep in range(int(args.n_null_repeats)):
                _null_bases.append((f"unit_shuffle_{_rep:03d}", unit_shuffled_tfts_basis(
                    Path(args.tfts_run),
                    delta_arcmin=float(args.basis_delta_arcmin),
                    k=int(args.basis_k),
                    basis_source=str(args.basis_source),
                    image_id=image_index,
                    seed=int(seed) + 100000 + _rep,
                )))
                _null_bases.append((f"random_orthogonal_{_rep:03d}", random_orthogonal_basis(
                    _basis.n_units,
                    _basis.basis_k,
                    seed=int(seed) + 200000 + _rep,
                )))
            real_cubes = model_lag_cubes_from_image_trace(
                image,
                example.trace,
                t_max=int(config.t_max),
                crop_center_offset_px=crop_offset,
            )
            series_id_base = f"{example.example_id}_image{image_index:03d}_crop{crop_rank:02d}"
            movie_manifest.append({
                "series_id_base": series_id_base,
                "example_id": example.example_id,
                "kind": example.kind,
                "image_index": image_index,
                "crop_rank": crop_rank,
                "crop_center_offset_x_px": float(crop_offset[0]),
                "crop_center_offset_y_px": float(crop_offset[1]),
            })
            for condition in conditions:
                if condition not in {"real", "stabilized"}:
                    raise ValueError("First-pass Panel E supports only real and stabilized conditions")
                print(f"  condition={condition}", flush=True)
                cubes = _condition_blocks(
                    condition=condition,
                    image=image,
                    trace=example.trace,
                    t_max=int(config.t_max),
                    crop_center_offset_px=crop_offset,
                    real_cubes=real_cubes,
                    control_images={},
                )
                rates_by_shift = run_shifted_lag_cube_rate_maps(
                    model,
                    population,
                    device,
                    cubes,
                    shifts,
                    batch_size=int(args.batch_size),
                )
                rows, arrays = _condition_fisher_rows(
                    rates_by_shift=rates_by_shift,
                    basis=_basis,
                    null_bases=_null_bases,
                    fisher_step_arcmin=float(args.fisher_step_arcmin),
                    dt=float(DT),
                    projection_mode=str(args.projection_mode),
                    metric=str(args.metric),
                )
                # Collapse repeated null traces for the plotting cache, but keep final rows per repeat.
                null_groups: dict[str, list[np.ndarray]] = {"unit_shuffle": [], "random_orthogonal": []}
                for row in rows:
                    row.update({
                        "series_id": f"{series_id_base}_{condition}",
                        "example_id": example.example_id,
                        "kind": example.kind,
                        "image_index": image_index,
                        "crop_rank": crop_rank,
                        "condition": condition,
                        "fisher_step_arcmin": float(args.fisher_step_arcmin),
                    })
                    final_rows.append(row)
                    label = str(row["basis_label"])
                    key = f"{label}/cumulative_fisher_pattern"
                    if label.startswith("unit_shuffle"):
                        null_groups["unit_shuffle"].append(arrays[key])
                    elif label.startswith("random_orthogonal"):
                        null_groups["random_orthogonal"].append(arrays[key])
                    elif label in {"full", "tangent", "orthogonal_complement"}:
                        series_arrays[f"{series_id_base}_{condition}/{label}/cumulative_fisher_pattern"] = arrays[key]
                for group, vals in null_groups.items():
                    if vals:
                        series_arrays[f"{series_id_base}_{condition}/{group}_mean/cumulative_fisher_pattern"] = np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32)
                series_records.append({
                    "series_id": f"{series_id_base}_{condition}",
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                })

    gain_rows = _summarize_gain(final_rows)
    _write_csv(dirs.metadata / "panelE_movie_manifest.csv", movie_manifest)
    _write_csv(dirs.results / "panelE_final_information_by_basis.csv", final_rows)
    _write_csv(dirs.results / "panelE_subspace_capture_summary.csv", gain_rows)
    _write_csv(dirs.metadata / "panelE_information_series_records.csv", series_records)
    np.savez_compressed(dirs.cache / "panelE_cumulative_information_series.npz", **series_arrays)
    _plot_panel_summary(gain_rows, series_records, series_arrays, dirs.figures / "panelE_tangent_subspace_information.pdf")
    _plot_panel_summary(gain_rows, series_records, series_arrays, dirs.figures / "panelE_tangent_subspace_information.png")

    # Null summary: grouped by basis_label prefix × kind.
    # Fraction stats are restricted to positive-gain windows (full_gain > 0) since the
    # fraction is not meaningful when full_gain <= 0. Absolute gain stats use all windows.
    # Grouping by basis_label (not basis_type) avoids conflating orthogonal_complement
    # (basis_type=="tangent") with the tangent basis itself.
    _summary_groups = ["full", "tangent", "orthogonal_complement", "unit_shuffle", "random_orthogonal"]
    pos_gain_rows = [r for r in gain_rows if float(r.get("full_gain_over_stabilized", 0)) > 0]
    null_summary: list[dict[str, Any]] = []
    for kind_filter in ("all", "fixation", "microsaccade"):
        if kind_filter == "all":
            kind_rows = gain_rows
            kind_pos_rows = pos_gain_rows
        else:
            kind_rows = [r for r in gain_rows if r.get("kind") == kind_filter]
            kind_pos_rows = [r for r in pos_gain_rows if r.get("kind") == kind_filter]
        for group in _summary_groups:
            abs_vals = np.asarray([
                float(r["gain_over_stabilized"]) for r in kind_rows
                if (str(r["basis_label"]) == group or str(r["basis_label"]).startswith(group + "_"))
                and np.isfinite(float(r["gain_over_stabilized"]))
            ], dtype=np.float64)
            frac_vals = np.asarray([
                float(r["fraction_full_fem_gain_captured"]) for r in kind_pos_rows
                if (str(r["basis_label"]) == group or str(r["basis_label"]).startswith(group + "_"))
                and np.isfinite(float(r["fraction_full_fem_gain_captured"]))
            ], dtype=np.float64)
            null_summary.append({
                "kind": kind_filter,
                "basis_label_group": group,
                "n_windows_all": int(abs_vals.size),
                "n_windows_positive_gain": int(frac_vals.size),
                "mean_absolute_gain": float(np.mean(abs_vals)) if abs_vals.size else float("nan"),
                "median_absolute_gain": float(np.median(abs_vals)) if abs_vals.size else float("nan"),
                "mean_fraction_full_fem_gain_captured": float(np.mean(frac_vals)) if frac_vals.size else float("nan"),
                "median_fraction_full_fem_gain_captured": float(np.median(frac_vals)) if frac_vals.size else float("nan"),
                "ci_low_fraction": float(np.percentile(frac_vals, 2.5)) if frac_vals.size else float("nan"),
                "ci_high_fraction": float(np.percentile(frac_vals, 97.5)) if frac_vals.size else float("nan"),
            })
    _write_csv(dirs.results / "panelE_basis_null_summary.csv", null_summary)

    summary = {
        "run_dir": str(dirs.run),
        "n_examples": len(examples),
        "n_crops": len(crops),
        "n_conditions": len(conditions),
        "n_final_rows": len(final_rows),
        "n_gain_rows": len(gain_rows),
        "basis_k": int(args.basis_k),
        "basis_delta_arcmin": float(args.basis_delta_arcmin),
        "basis_source": str(args.basis_source),
        "tfts_run": str(args.tfts_run),
        "figure_pdf": str(dirs.figures / "panelE_tangent_subspace_information.pdf"),
        "final_rows_csv": str(dirs.results / "panelE_final_information_by_basis.csv"),
        "gain_summary_csv": str(dirs.results / "panelE_subspace_capture_summary.csv"),
        "image_selection_figures": image_figures,
    }
    _write_json(summary_path, summary)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="panelE_tangent_subspace_information")
    parser.add_argument("--output-root", default="outputs/tangent_subspace_information")
    parser.add_argument("--twininfo-run", default=None,
                        help="Optional existing outputs/twininfo/<run> directory whose run_config is reused.")
    parser.add_argument("--source-run-name", default=None,
                        help="Optional readable source run label when no --twininfo-run is supplied.")
    parser.add_argument("--tfts-run", required=True,
                        help="TFTS output directory containing tangent_maps/twin_tangent_maps.pkl")
    parser.add_argument("--basis-k", type=int, default=10)
    parser.add_argument("--basis-delta-arcmin", type=float, default=0.25)
    parser.add_argument("--basis-source", choices=("all_objects", "image_disjoint"), default="image_disjoint")
    parser.add_argument("--conditions", nargs="+", default=("real", "stabilized"))
    parser.add_argument("--n-null-repeats", type=int, default=100)
    parser.add_argument("--fisher-step-arcmin", type=float, default=0.5)
    parser.add_argument("--projection-mode", default="derivative_fisher",
                        choices=("derivative_fisher", "derivative_euclidean"),
                        help=(
                            "How derivatives are projected into the basis subspace. "
                            "'derivative_fisher' (default): Fisher/Poisson-metric projection, "
                            "guarantees tangent + orthogonal == full Fisher. "
                            "'derivative_euclidean': legacy Euclidean U U.T projection, "
                            "does NOT satisfy the partition and can produce basis > full."
                        ))
    parser.add_argument("--metric", default="pattern_fisher_trace",
                        choices=("pattern_fisher_trace",),
                        help=(
                            "Information metric to compute. "
                            "'pattern_fisher_trace' = trace of the spatial-pattern Poisson Fisher matrix "
                            "(pattern component only; excludes single-spike baseline term). "
                            "Corresponds to 'cumulative_fisher_pattern' in the twininfo pipeline."
                        ))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--recompute", action="store_true")

    # Minimal config path when --twininfo-run is omitted.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-indices", nargs="+", type=int, default=None)
    parser.add_argument("--n-crops-per-image", type=int, default=1)
    parser.add_argument("--n-examples-per-kind", type=int, default=2)
    parser.add_argument("--selected-trace-example-ids", nargs="+", default=())
    parser.add_argument("--t-max", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=756,
                        help="Only used for metadata when --twininfo-run is omitted; canonical readout determines actual N.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_tangent_subspace_information(args)
    print("Tangent-subspace information analysis complete")
    print(f"  run: {summary['run_dir']}")
    print(f"  final rows: {summary['final_rows_csv']}")
    print(f"  gain summary: {summary['gain_summary_csv']}")
    print(f"  figure: {summary['figure_pdf']}")


if __name__ == "__main__":
    main()

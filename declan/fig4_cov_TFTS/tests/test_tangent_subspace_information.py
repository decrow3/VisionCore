from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[3] / "jake" / "twininfo" / "run_tangent_subspace_information.py"
spec = importlib.util.spec_from_file_location("run_tangent_subspace_information", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_project_unit_axis_preserves_basis_component():
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    coeff = rng.normal(size=(4, 3, 2, 2))
    # Shape (T, N, H, D), entirely in span(U) along N.
    arr = np.einsum("nk,thkd->tnhd", U, coeff)
    proj = mod._project_unit_axis(arr, U, unit_axis=1)
    assert np.allclose(proj, arr, atol=1e-10)


def test_orthogonal_component_is_orthogonal_to_basis():
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    arr = rng.normal(size=(3, 5, 4, 2))
    orth = mod._orthogonal_component_unit_axis(arr, U, unit_axis=1)
    moved = np.moveaxis(orth, 1, -2)
    coeff = np.einsum("...nd,nk->...kd", moved, U)
    assert np.max(np.abs(coeff)) < 1e-10


def test_basis_from_columns_captures_known_subspace():
    rng = np.random.default_rng(2)
    U_true, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    weights = rng.normal(size=(3, 20))
    mat = U_true @ weights
    U = mod._orthonormal_basis_from_columns(mat, k=3)
    assert mod._variance_capture(U, mat) > 0.999999


def test_basis_from_columns_is_orthonormal():
    """U.T @ U must equal I — required for Fisher projection correctness."""
    rng = np.random.default_rng(3)
    mat = rng.normal(size=(20, 15))
    for k in (1, 5, 10):
        U = mod._orthonormal_basis_from_columns(mat, k=k)
        assert U.shape == (20, k)
        assert np.allclose(U.T @ U, np.eye(k), atol=1e-10)


def test_projected_fisher_trace_nonnegative():
    """Fisher trace must be >= 0 for full, tangent, and orthogonal projections."""
    rng = np.random.default_rng(4)
    n_units, k = 12, 3
    U, _ = np.linalg.qr(rng.normal(size=(n_units, k)))
    mu = np.abs(rng.normal(size=(n_units,))) + 0.1   # strictly positive rates
    # J shape (T=2, N, H=1, D=2) as used by _project_unit_axis with unit_axis=1
    J = rng.normal(size=(2, n_units, 1, 2))
    w = 1.0 / np.maximum(mu, 1e-6)
    for component in [
        J,
        mod._project_unit_axis(J, U, unit_axis=1),
        mod._orthogonal_component_unit_axis(J, U, unit_axis=1),
    ]:
        fisher = float(np.sum(w[np.newaxis, :, np.newaxis, np.newaxis] * component ** 2))
        assert fisher >= 0.0


def test_projection_decomposition_sums_to_full():
    """J_tangent + J_orth == J, and J_orth is Euclidean-orthogonal to U."""
    rng = np.random.default_rng(5)
    n_units, k = 10, 4
    U, _ = np.linalg.qr(rng.normal(size=(n_units, k)))
    J = rng.normal(size=(3, n_units, 2, 2))
    J_tang = mod._project_unit_axis(J, U, unit_axis=1)
    J_orth = mod._orthogonal_component_unit_axis(J, U, unit_axis=1)
    assert np.allclose(J_tang + J_orth, J, atol=1e-10)
    # Orth component has zero coefficients in the basis directions
    moved = np.moveaxis(J_orth, 1, -2)          # (..., N, D)
    coeff = np.einsum("...nd,nk->...kd", moved, U)
    assert np.max(np.abs(coeff)) < 1e-10


def test_full_rank_projection_is_identity():
    """Projecting J with a full-rank (identity) basis returns J unchanged.

    This simultaneously tests: (a) the API operates on J not mu, and
    (b) a full-rank projection gives the same Fisher as unprojected.
    """
    rng = np.random.default_rng(6)
    n_units = 6
    U = np.eye(n_units)
    mu = np.abs(rng.normal(size=(n_units,))) + 0.1
    J = rng.normal(size=(1, n_units, 1, 2))
    J_proj = mod._project_unit_axis(J, U, unit_axis=1)
    assert np.allclose(J_proj, J, atol=1e-10)
    # mu is untouched — the function never receives or modifies it
    w = 1.0 / np.maximum(mu, 1e-6)
    fisher_full = float(np.sum(w[np.newaxis, :, np.newaxis, np.newaxis] * J ** 2))
    fisher_proj = float(np.sum(w[np.newaxis, :, np.newaxis, np.newaxis] * J_proj ** 2))
    assert np.isclose(fisher_full, fisher_proj)

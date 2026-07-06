#!/usr/bin/env python3
"""Decode signed local image-feature derivatives from compact response channels.

This runner implements the first-pass analysis described in
``local_derivative_channel_analysis_plan.md``.  It uses the cached
``twin_tangent_maps.pkl`` objects as a manifest-verifiable fresh cache: the
response derivatives and shifted response endpoints are read from that cache,
while signed image-feature derivative targets are recomputed in this run from
the cached stimulus-history tensors using the same subpixel shift convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from declan.compact_retinal_translation_geometry.run_compact_retinal_translation_geometry import (
    ACCENT,
    BRIDGE,
    DEFAULT_CLOSURE_ROOT,
    DEFAULT_TFTS_ROOT,
    GREEN,
    MODEL,
    NULL,
    TEXT,
    VISIONCORE_ROOT,
    _clean_axes,
    _save_fig,
)


DEFAULT_OUTPUT_ROOT = (
    VISIONCORE_ROOT
    / "outputs"
    / "compact_retinal_translation_geometry"
    / "local_derivative_channel_v1"
)

PRIMARY_TARGETS = ("phase_vector", "gabor_even_odd")
EPS = 1e-12


@dataclass(frozen=True)
class DerivativeObject:
    object_id: str
    group_id: str
    image_id: str
    trial_index: str
    time_index: str
    delta_arcmin: float
    delta_model_px: float
    r0: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    history: np.ndarray


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    ux: float
    uy: float


@dataclass(frozen=True)
class BasisSpec:
    basis_type: str
    basis_role: str
    basis_training_source: str
    basis_draw: str
    k_requested: int
    matrix: np.ndarray | None

    @property
    def k_effective(self) -> int:
        if self.matrix is None:
            return 0
        return int(self.matrix.shape[1])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not keys:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _parse_int_list(value: str) -> list[int]:
    out = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if not out or any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return out


def _parse_float_list(value: str, *, positive: bool = True) -> list[float]:
    out = sorted({float(part.strip()) for part in str(value).split(",") if part.strip()})
    if positive:
        bad = any((not np.isfinite(v)) or v <= 0.0 for v in out)
        msg = "expected a comma-separated list of positive finite values"
    else:
        bad = any((not np.isfinite(v)) or v < 0.0 for v in out)
        msg = "expected a comma-separated list of nonnegative finite values"
    if not out or bad:
        raise argparse.ArgumentTypeError(msg)
    return out


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _repo_commit() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=VISIONCORE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"],
                cwd=VISIONCORE_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception as exc:
        return {"git_commit": None, "git_dirty": None, "git_error": f"{type(exc).__name__}: {exc}"}


def _orth(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D")
    if mat.size == 0 or mat.shape[1] == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(mat, mode="reduced")
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.asarray([], dtype=np.float64)
    scale = float(np.max(diag)) if diag.size else 0.0
    keep = diag > max(float(eps), scale * float(eps))
    return q[:, keep]


def _pca_basis(rows: np.ndarray, max_components: int) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("PCA input must be 2D")
    if x.shape[0] == 0:
        return np.zeros((x.shape[1], 0), dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] == 0:
        return np.zeros((rows.shape[1], 0), dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(x, full_matrices=False)
    if svals.size == 0:
        return np.zeros((x.shape[1], 0), dtype=np.float64)
    keep = svals > max(float(svals[0]), 1.0) * 1e-10
    n_keep = int(min(np.sum(keep), int(max_components), vt.shape[0]))
    return _orth(vt[:n_keep].T)


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((int(n_units), max(int(k), 1))), mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs[None, :])[:, : int(k)]


def _orth_residual_basis(primary: np.ndarray, nuisance: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if primary.ndim != 2 or nuisance.ndim != 2:
        raise ValueError("primary and nuisance bases must be 2D")
    if primary.shape[0] != nuisance.shape[0]:
        raise ValueError("primary and nuisance bases must have matching row counts")
    if primary.shape[1] == 0:
        return primary[:, :0]
    if nuisance.shape[1] > 0:
        qn = _orth(nuisance)
        residual = primary - qn @ (qn.T @ primary)
    else:
        residual = primary.copy()
    return _orth(residual, eps=tol)


def _principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = _orth(a)
    qb = _orth(b)
    k = int(min(qa.shape[1], qb.shape[1]))
    if k <= 0:
        return np.asarray([], dtype=np.float64)
    return np.clip(np.linalg.svd(qa[:, :k].T @ qb[:, :k], compute_uv=False), 0.0, 1.0)


def _as_vector(meta: dict[str, Any], key: str, object_id: str) -> np.ndarray:
    arr = np.asarray(meta[key], dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} for {object_id} is not a finite vector")
    return arr


def _group_label(meta: dict[str, Any], object_id: str, group_by: str) -> str:
    if group_by == "image_id":
        return str(meta.get("image_id", object_id))
    if group_by == "trial_index":
        return f"{meta.get('image_id', 'unknown')}/{meta.get('trial_index', object_id)}"
    if group_by == "object_id":
        return str(object_id)
    raise ValueError(f"Unsupported group_by={group_by!r}")


def _load_tangent_payload(tfts_root: Path) -> tuple[list[float], dict[float, dict[str, dict[str, Any]]], Path]:
    path = Path(tfts_root) / "tangent_maps" / "twin_tangent_maps.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing tangent-map cache: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    deltas = [float(v) for v in payload["delta_arcmins"]]
    by_delta = {
        float(delta): {str(oid): dict(meta) for oid, meta in objects.items()}
        for delta, objects in payload["object_payload"].items()
    }
    return deltas, by_delta, path


def _nearest_available(requested: float, available: list[float], tolerance: float = 1e-9) -> float | None:
    if not available:
        return None
    best = float(min(available, key=lambda val: abs(float(val) - float(requested))))
    if abs(best - float(requested)) <= tolerance:
        return best
    return None


def _collect_objects(
    *,
    tfts_root: Path,
    requested_epsilons: list[float],
    group_by: str,
) -> tuple[dict[float, list[DerivativeObject]], list[dict[str, Any]], dict[str, Any]]:
    available, by_delta, source_path = _load_tangent_payload(tfts_root)
    objects_by_delta: dict[float, list[DerivativeObject]] = {}
    skipped: list[dict[str, Any]] = []
    used: list[float] = []
    n_units: int | None = None
    for req in requested_epsilons:
        delta = _nearest_available(req, available)
        if delta is None:
            skipped.append(
                {
                    "delta_arcmin_requested": float(req),
                    "object_id": "",
                    "skip_reason": f"requested_delta_unavailable; available={','.join(map(str, available))}",
                }
            )
            continue
        used.append(float(delta))
        rows: list[DerivativeObject] = []
        for object_id, meta in sorted(by_delta[delta].items()):
            try:
                r0 = _as_vector(meta, "r0", object_id)
                bx = _as_vector(meta, "bx", object_id)
                by = _as_vector(meta, "by", object_id)
                history = np.asarray(meta["history"], dtype=np.float64)
                if history.ndim != 3 or not np.all(np.isfinite(history)):
                    raise ValueError("history is not a finite (T,H,W) tensor")
                delta_model_px = float(meta.get("delta_model_px", float("nan")))
                if not np.isfinite(delta_model_px) or delta_model_px <= 0.0:
                    raise ValueError("delta_model_px missing or non-positive")
                if not (r0.shape == bx.shape == by.shape):
                    raise ValueError(f"mismatched response shapes r0={r0.shape} bx={bx.shape} by={by.shape}")
                if n_units is None:
                    n_units = int(r0.size)
                elif int(r0.size) != n_units:
                    raise ValueError(f"unit count {r0.size} does not match expected {n_units}")
            except Exception as exc:
                skipped.append(
                    {
                        "delta_arcmin_requested": float(req),
                        "delta_arcmin_used": float(delta),
                        "object_id": str(object_id),
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            rows.append(
                DerivativeObject(
                    object_id=str(object_id),
                    group_id=_group_label(meta, str(object_id), group_by),
                    image_id=str(meta.get("image_id", "")),
                    trial_index=str(meta.get("trial_index", "")),
                    time_index=str(meta.get("time_index", "")),
                    delta_arcmin=float(delta),
                    delta_model_px=delta_model_px,
                    r0=r0,
                    bx=bx,
                    by=by,
                    history=history,
                )
            )
        if rows:
            objects_by_delta[float(delta)] = rows
    if not objects_by_delta:
        raise ValueError("No usable derivative objects found for requested epsilon values")
    meta = {
        "tangent_source": str(source_path.resolve()),
        "requested_epsilons_arcmin": requested_epsilons,
        "available_epsilons_arcmin": available,
        "used_epsilons_arcmin": sorted(objects_by_delta),
        "n_units": int(n_units or 0),
    }
    return objects_by_delta, skipped, meta


def _assign_group_folds(group_ids: list[str], *, n_folds: int, seed: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    groups = sorted(set(str(v) for v in group_ids))
    if int(n_folds) < 2:
        raise ValueError("--n-folds must be at least 2")
    if len(groups) < int(n_folds):
        raise ValueError(f"Requested {n_folds} folds, but only found {len(groups)} groups")
    counts = {group: sum(str(v) == group for v in group_ids) for group in groups}
    rng = np.random.default_rng(int(seed))
    tie_break = {group: float(rng.random()) for group in groups}
    ordered = sorted(groups, key=lambda group: (-counts[group], tie_break[group], group))
    fold_loads = [0 for _ in range(int(n_folds))]
    fold_group_counts = [0 for _ in range(int(n_folds))]
    group_to_fold: dict[str, int] = {}
    for group in ordered:
        fold_id = min(range(int(n_folds)), key=lambda idx: (fold_loads[idx], fold_group_counts[idx], idx))
        group_to_fold[group] = int(fold_id)
        fold_loads[fold_id] += int(counts[group])
        fold_group_counts[fold_id] += 1
    rows: list[dict[str, Any]] = []
    for fold_id in range(int(n_folds)):
        fold_groups = sorted(group for group, fid in group_to_fold.items() if fid == fold_id)
        rows.append(
            {
                "fold_id": int(fold_id),
                "n_groups": int(len(fold_groups)),
                "n_objects": int(sum(counts[group] for group in fold_groups)),
                "group_ids": ",".join(fold_groups),
            }
        )
    return group_to_fold, rows


def parse_directions(value: str) -> list[DirectionSpec]:
    aliases: dict[str, tuple[float, float]] = {
        "+x": (1.0, 0.0),
        "x": (1.0, 0.0),
        "-x": (-1.0, 0.0),
        "+y": (0.0, 1.0),
        "y": (0.0, 1.0),
        "-y": (0.0, -1.0),
        "+x+y": (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        "x+y": (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        "+x-y": (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        "x-y": (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        "-x+y": (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        "-x-y": (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
    }
    specs: list[DirectionSpec] = []
    for raw in _parse_str_list(value):
        key = raw.strip().lower()
        if key == "cardinal":
            specs.extend(parse_directions("+x,-x,+y,-y"))
            continue
        if key == "diagonal":
            specs.extend(parse_directions("+x+y,+x-y,-x+y,-x-y"))
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"Unsupported direction {raw!r}")
        ux, uy = aliases[key]
        name = raw if raw.startswith(("+", "-")) else f"+{raw}"
        specs.append(DirectionSpec(name=name, ux=float(ux), uy=float(uy)))
    dedup: dict[str, DirectionSpec] = {}
    for spec in specs:
        dedup[spec.name] = spec
    return list(dedup.values())


def _frame_from_history(history: np.ndarray, mode: str) -> np.ndarray:
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError(f"history must have shape (T,H,W), got {h.shape}")
    if mode == "current":
        frame = h[0]
    elif mode == "center":
        frame = h[h.shape[0] // 2]
    elif mode == "mean":
        frame = np.mean(h, axis=0)
    else:
        raise ValueError(f"Unsupported feature frame mode: {mode}")
    return np.asarray(frame, dtype=np.float64)


def _shift_stack_subpixel(stack: np.ndarray, *, dx_px: float, dy_px: float) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 2:
        arr3 = arr[None, :, :]
        squeeze = True
    elif arr.ndim == 3:
        arr3 = arr
        squeeze = False
    else:
        raise ValueError(f"Expected 2D frame or 3D stack, got {arr.shape}")
    out = ndimage.shift(
        arr3,
        shift=(0.0, float(dy_px), float(dx_px)),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return out[0] if squeeze else out


def _gabor_kernel(wavelength: float, theta: float, phase: float) -> np.ndarray:
    sigma_x = 0.56 * float(wavelength)
    sigma_y = 0.84 * float(wavelength)
    radius = int(max(3, math.ceil(3.0 * max(sigma_x, sigma_y))))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float64)
    ct = math.cos(float(theta))
    st = math.sin(float(theta))
    x_theta = xx * ct + yy * st
    y_theta = -xx * st + yy * ct
    envelope = np.exp(-0.5 * ((x_theta / sigma_x) ** 2 + (y_theta / sigma_y) ** 2))
    kernel = envelope * np.cos((2.0 * math.pi * x_theta / float(wavelength)) + float(phase))
    kernel = kernel - np.mean(kernel)
    norm = float(np.sqrt(np.sum(kernel * kernel)))
    if norm > 0.0:
        kernel = kernel / norm
    return kernel


def _region_weights(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sigma = max(min(h, w) / 5.0, 1.0)
    center = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    masks = {
        "center": center,
        "left": (xx <= cx).astype(np.float64),
        "right": (xx >= cx).astype(np.float64),
        "upper": (yy <= cy).astype(np.float64),
        "lower": (yy >= cy).astype(np.float64),
    }
    out: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        den = float(np.sum(mask))
        out[name] = mask / den if den > 0.0 else mask
    return out


def _pool_regions(arr: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([float(np.sum(np.asarray(arr, dtype=np.float64) * weights[name])) for name in weights], dtype=np.float64)


def _grid_pool(arr: np.ndarray, grid: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    rows = np.array_split(np.arange(a.shape[0]), int(grid))
    cols = np.array_split(np.arange(a.shape[1]), int(grid))
    vals: list[float] = []
    for rr in rows:
        for cc in cols:
            vals.append(float(np.mean(a[np.ix_(rr, cc)])))
    return np.asarray(vals, dtype=np.float64)


class FeatureExtractor:
    def __init__(
        self,
        *,
        target_families: list[str],
        gabor_wavelengths: list[float],
        gabor_orientations_deg: list[float],
        grid_size: int,
    ) -> None:
        self.target_families = list(target_families)
        self.gabor_wavelengths = list(gabor_wavelengths)
        self.gabor_orientations_deg = list(gabor_orientations_deg)
        self.grid_size = int(grid_size)
        self._kernel_specs: list[tuple[str, float, float]] = []
        for wavelength in self.gabor_wavelengths:
            for orientation in self.gabor_orientations_deg:
                label = f"wl{float(wavelength):g}_ori{float(orientation):g}"
                self._kernel_specs.append((label, float(wavelength), float(orientation)))
        self._weights_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._gabor_filter_cache: dict[tuple[int, int], list[tuple[str, str, np.ndarray, np.ndarray]]] = {}

    def _weights(self, shape: tuple[int, int]) -> dict[str, np.ndarray]:
        key = (int(shape[0]), int(shape[1]))
        if key not in self._weights_cache:
            self._weights_cache[key] = _region_weights(key)
        return self._weights_cache[key]

    def _gabor_filters(self, shape: tuple[int, int]) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
        key = (int(shape[0]), int(shape[1]))
        if key in self._gabor_filter_cache:
            return self._gabor_filter_cache[key]
        h, w = key
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        weights = self._weights(key)
        filters: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for label, wavelength, orientation in self._kernel_specs:
            theta = math.radians(float(orientation))
            ct = math.cos(theta)
            st = math.sin(theta)
            sigma_x = 0.56 * float(wavelength)
            sigma_y = 0.84 * float(wavelength)
            for region_name, region_weight in weights.items():
                den = float(np.sum(region_weight))
                cy = float(np.sum(yy * region_weight) / den) if den > 0.0 else (h - 1) / 2.0
                cx = float(np.sum(xx * region_weight) / den) if den > 0.0 else (w - 1) / 2.0
                xr = xx - cx
                yr = yy - cy
                x_theta = xr * ct + yr * st
                y_theta = -xr * st + yr * ct
                envelope = np.exp(-0.5 * ((x_theta / sigma_x) ** 2 + (y_theta / sigma_y) ** 2))
                even = region_weight * envelope * np.cos(2.0 * math.pi * x_theta / float(wavelength))
                odd = region_weight * envelope * np.sin(2.0 * math.pi * x_theta / float(wavelength))
                even = even - np.mean(even)
                odd = odd - np.mean(odd)
                even_norm = float(np.sqrt(np.sum(even * even)))
                odd_norm = float(np.sqrt(np.sum(odd * odd)))
                if even_norm > 0.0:
                    even = even / even_norm
                if odd_norm > 0.0:
                    odd = odd / odd_norm
                filters.append((label, region_name, even, odd))
        self._gabor_filter_cache[key] = filters
        return filters

    def describe(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        f = np.asarray(frame, dtype=np.float64)
        if f.ndim != 2:
            raise ValueError(f"frame must be 2D, got {f.shape}")
        f0 = f - float(np.mean(f))
        out: dict[str, np.ndarray] = {}
        need_gabor = any(
            family in self.target_families
            for family in ("gabor_even_odd", "gabor_energy", "phase_vector")
        )
        if need_gabor:
            even_odd_vals: list[np.ndarray] = []
            energy_vals: list[np.ndarray] = []
            phase_vals: list[np.ndarray] = []
            for _, _, even_filter, odd_filter in self._gabor_filters((f0.shape[0], f0.shape[1])):
                even_pool = np.asarray([float(np.sum(f0 * even_filter))], dtype=np.float64)
                odd_pool = np.asarray([float(np.sum(f0 * odd_filter))], dtype=np.float64)
                amp_pool = np.sqrt(even_pool * even_pool + odd_pool * odd_pool)
                if "gabor_even_odd" in self.target_families:
                    even_odd_vals.extend([even_pool, odd_pool])
                if "gabor_energy" in self.target_families:
                    energy_vals.append(amp_pool)
                if "phase_vector" in self.target_families:
                    phase_vals.extend([even_pool / (amp_pool + 1e-8), odd_pool / (amp_pool + 1e-8)])
            if "gabor_even_odd" in self.target_families:
                out["gabor_even_odd"] = np.concatenate(even_odd_vals) if even_odd_vals else np.zeros(0)
            if "gabor_energy" in self.target_families:
                out["gabor_energy"] = np.concatenate(energy_vals) if energy_vals else np.zeros(0)
            if "phase_vector" in self.target_families:
                out["phase_vector"] = np.concatenate(phase_vals) if phase_vals else np.zeros(0)
        if "bandpass_signed" in self.target_families:
            vals = []
            for sigma1, sigma2 in ((0.6, 1.2), (1.2, 2.4), (2.4, 4.8)):
                dog = ndimage.gaussian_filter(f0, sigma=sigma1, mode="nearest") - ndimage.gaussian_filter(
                    f0,
                    sigma=sigma2,
                    mode="nearest",
                )
                vals.append(_grid_pool(dog, self.grid_size))
            out["bandpass_signed"] = np.concatenate(vals)
        if "raw_pixel_grid" in self.target_families:
            out["raw_pixel_grid"] = _grid_pool(f0, self.grid_size)
        if "image_gradient" in self.target_families:
            gy, gx = np.gradient(f0)
            out["image_gradient"] = np.concatenate([_grid_pool(gx, self.grid_size), _grid_pool(gy, self.grid_size)])
        return out

    def inventory_rows(self, frame_shape: tuple[int, int]) -> list[dict[str, Any]]:
        dummy = np.zeros(frame_shape, dtype=np.float64)
        desc = self.describe(dummy)
        rows: list[dict[str, Any]] = []
        for family in self.target_families:
            rows.append(
                {
                    "target_family": family,
                    "target_dim": int(desc.get(family, np.zeros(0)).size),
                    "target_role": "primary" if family in PRIMARY_TARGETS else "secondary_or_control",
                    "feature_source": "cached_history_frame_shifted_in_run",
                    "gabor_wavelengths_px": ",".join(f"{v:g}" for v in self.gabor_wavelengths),
                    "gabor_orientations_deg": ",".join(f"{v:g}" for v in self.gabor_orientations_deg),
                    "grid_size": int(self.grid_size),
                }
            )
        return rows


def _edge_axis(frame: np.ndarray) -> tuple[str, float, float]:
    f = np.asarray(frame, dtype=np.float64)
    gy, gx = np.gradient(f - float(np.mean(f)))
    ex = float(np.mean(gx * gx))
    ey = float(np.mean(gy * gy))
    return ("x" if ex >= ey else "y"), ex, ey


def _build_feature_samples(
    *,
    objects_by_delta: dict[float, list[DerivativeObject]],
    directions: list[DirectionSpec],
    feature_frame_mode: str,
    extractor: FeatureExtractor,
    target_families: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    dr_rows: list[np.ndarray] = []
    r0_rows: list[np.ndarray] = []
    derivative_targets: dict[str, list[np.ndarray]] = {family: [] for family in target_families}
    shifted_targets: dict[str, list[np.ndarray]] = {family: [] for family in target_families}
    object_feature_rows: list[dict[str, Any]] = []
    sample_idx = 0
    for delta in sorted(objects_by_delta):
        for obj in objects_by_delta[delta]:
            frame = _frame_from_history(obj.history, feature_frame_mode)
            edge_name, grad_x_energy, grad_y_energy = _edge_axis(frame)
            base_desc = extractor.describe(frame)
            for family in target_families:
                object_feature_rows.append(
                    {
                        "epsilon_arcmin": float(delta),
                        "object_id": obj.object_id,
                        "target_family": family,
                        "base_feature_l2_norm": float(np.linalg.norm(base_desc[family])),
                        "target_dim": int(base_desc[family].size),
                    }
                )
            for direction in directions:
                dx = float(obj.delta_model_px) * float(direction.ux)
                dy = float(obj.delta_model_px) * float(direction.uy)
                plus_desc = extractor.describe(_shift_stack_subpixel(frame, dx_px=dx, dy_px=dy))
                minus_desc = extractor.describe(_shift_stack_subpixel(frame, dx_px=-dx, dy_px=-dy))
                for family in target_families:
                    derivative_targets[family].append((plus_desc[family] - minus_desc[family]) / (2.0 * obj.delta_model_px))
                    shifted_targets[family].append(plus_desc[family])
                sample_rows.append(
                    {
                        "sample_index": int(sample_idx),
                        "epsilon_arcmin": float(delta),
                        "delta_model_px": float(obj.delta_model_px),
                        "object_id": obj.object_id,
                        "group_id": obj.group_id,
                        "image_id": obj.image_id,
                        "trial_index": obj.trial_index,
                        "time_index": obj.time_index,
                        "direction": direction.name,
                        "direction_x": float(direction.ux),
                        "direction_y": float(direction.uy),
                        "edge_normal_axis": edge_name,
                        "gradient_x_energy": grad_x_energy,
                        "gradient_y_energy": grad_y_energy,
                    }
                )
                dr_rows.append(direction.ux * obj.bx + direction.uy * obj.by)
                r0_rows.append(obj.r0)
                sample_idx += 1
    sample_df = pd.DataFrame(sample_rows)
    dr = np.stack(dr_rows, axis=0).astype(np.float64)
    r0 = np.stack(r0_rows, axis=0).astype(np.float64)
    derivative_arrays = {family: np.stack(vals, axis=0).astype(np.float64) for family, vals in derivative_targets.items()}
    shifted_arrays = {family: np.stack(vals, axis=0).astype(np.float64) for family, vals in shifted_targets.items()}
    return sample_df, dr, r0, derivative_arrays, shifted_arrays, object_feature_rows


def _fit_fold_bases(
    *,
    objects: list[DerivativeObject],
    train_groups: set[str],
    max_k: int,
) -> dict[str, np.ndarray]:
    train = [obj for obj in objects if obj.group_id in train_groups]
    if not train:
        raise ValueError("Cannot fit bases with no training objects")
    n_units = int(train[0].r0.size)
    max_compact = min(max(2 * int(max_k), int(max_k) + 1), 2 * len(train), n_units)
    compact = _pca_basis(np.stack([v for obj in train for v in (obj.bx, obj.by)], axis=0), max_compact)
    static = _pca_basis(np.stack([obj.r0 for obj in train], axis=0), min(int(max_k), len(train), n_units))
    global_rate = np.ones((n_units, 1), dtype=np.float64) / math.sqrt(float(n_units))
    target_pc1 = compact[:, :1] if compact.shape[1] else np.zeros((n_units, 0), dtype=np.float64)
    return {
        "compact_full": compact,
        "static_full": static,
        "global_rate": global_rate,
        "target_pc1": target_pc1,
    }


def _basis_specs_for_k(
    *,
    bases: dict[str, np.ndarray],
    k: int,
    n_units: int,
    rng: np.random.Generator,
    n_random: int,
    n_unit_shuffle: int,
) -> list[BasisSpec]:
    compact = bases["compact_full"]
    static = bases["static_full"]
    specs: list[BasisSpec] = [
        BasisSpec("compact", "primary_basis", "fold_train_bx_by_pca", "observed", int(k), compact[:, : min(int(k), compact.shape[1])]),
        BasisSpec("static_pc", "primary_control", "fold_train_r0_pca", "observed", int(k), static[:, : min(int(k), static.shape[1])]),
        BasisSpec(
            "compact_resid_static",
            "residual_control",
            "compact_after_projecting_out_static_pc",
            "observed",
            int(k),
            _orth_residual_basis(compact[:, : min(int(k), compact.shape[1])], static[:, : min(int(k), static.shape[1])]),
        ),
        BasisSpec(
            "static_resid_compact",
            "residual_control",
            "static_pc_after_projecting_out_compact",
            "observed",
            int(k),
            _orth_residual_basis(static[:, : min(int(k), static.shape[1])], compact[:, : min(int(k), compact.shape[1])]),
        ),
        BasisSpec(
            "noncompact_complement",
            "specificity_control",
            "next_compact_tangent_pcs_after_primary_k",
            "observed",
            int(k),
            compact[:, min(int(k), compact.shape[1]) : min(2 * int(k), compact.shape[1])],
        ),
        BasisSpec("global_rate", "gain_control", "analytic_global_rate_axis", "observed", int(k), bases["global_rate"]),
        BasisSpec("target_pc1", "dominant_tangent_pc_control", "fold_train_bx_by_pc1", "observed", int(k), bases["target_pc1"]),
    ]
    for draw in range(int(n_random)):
        specs.append(
            BasisSpec(
                "random",
                "basis_control",
                "isotropic_random_orthonormal",
                f"random_{draw:03d}",
                int(k),
                _random_basis(n_units, int(k), rng),
            )
        )
    for draw in range(int(n_unit_shuffle)):
        u = compact[:, : min(int(k), compact.shape[1])]
        if u.shape[1] > 0:
            u = u[rng.permutation(n_units), :]
        specs.append(
            BasisSpec(
                "unit_shuffle_compact",
                "basis_control",
                "fold_train_compact_unit_permutation",
                f"shuffle_{draw:03d}",
                int(k),
                u,
            )
        )
    return specs


def _ridge_lambda_grid(x_train: np.ndarray) -> np.ndarray:
    x = np.asarray(x_train, dtype=np.float64)
    n_features = max(int(x.shape[1]), 1)
    scale = float(np.trace(x.T @ x) / n_features) if x.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.logspace(-4, 4, 17, dtype=np.float64) * scale


def _standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, keepdims=True)
    std = np.std(train, axis=0, keepdims=True)
    std = np.where(std > 1e-10, std, 1.0)
    return (train - mean) / std, (test - mean) / std, mean, std


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    if x_train.shape[1] > x_train.shape[0]:
        xxt = x_train @ x_train.T
        alpha = np.linalg.solve(
            xxt + float(lam) * np.eye(xxt.shape[0], dtype=np.float64),
            y_train,
        )
        return x_train.T @ alpha
    xtx = x_train.T @ x_train
    return np.linalg.solve(xtx + float(lam) * np.eye(xtx.shape[0], dtype=np.float64), x_train.T @ y_train)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.size == 0 or p.size == 0:
        return {
            "R2_mean": float("nan"),
            "R2_median": float("nan"),
            "R2_pooled": float("nan"),
            "feature_cosine_mean": float("nan"),
            "neg_mse_standardized": float("nan"),
            "vector_correlation": float("nan"),
        }
    ss_res = np.sum((y - p) ** 2, axis=0)
    ss_tot = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    ss_res_pool = float(np.sum((y - p) ** 2))
    ss_tot_pool = float(np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2))
    y_norm = np.linalg.norm(y, axis=1)
    p_norm = np.linalg.norm(p, axis=1)
    den = y_norm * p_norm
    ok = den > 1e-12
    cosine = np.full(y.shape[0], np.nan, dtype=np.float64)
    cosine[ok] = np.sum(y[ok] * p[ok], axis=1) / den[ok]
    yy = y.ravel()
    pp = p.ravel()
    return {
        "R2_mean": float(np.nanmean(r2)),
        "R2_median": float(np.nanmedian(r2)),
        "R2_pooled": float(1.0 - ss_res_pool / ss_tot_pool) if ss_tot_pool > 1e-12 else float("nan"),
        "feature_cosine_mean": float(np.nanmean(cosine)),
        "neg_mse_standardized": float(-np.mean((y - p) ** 2)),
        "vector_correlation": float(np.corrcoef(yy, pp)[0, 1]) if np.std(yy) > 1e-12 and np.std(pp) > 1e-12 else float("nan"),
    }


def _condition_folds(groups: np.ndarray, seed: int, n_folds: int = 3) -> list[np.ndarray]:
    unique = np.unique(groups.astype(str))
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    return [fold for fold in np.array_split(shuffled, min(int(n_folds), unique.size)) if fold.size]


def _inner_select_lambda(x_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, seed: int) -> float:
    lambdas = _ridge_lambda_grid(x_train)
    unique_groups = np.unique(groups_train.astype(str))
    if unique_groups.size < 3 or x_train.shape[0] < 10:
        return float(lambdas[len(lambdas) // 2])
    scores = np.full(lambdas.shape, 0.0, dtype=np.float64)
    counts = np.zeros(lambdas.shape, dtype=np.int64)
    for held_groups in _condition_folds(groups_train, seed=int(seed), n_folds=3):
        test = np.isin(groups_train.astype(str), held_groups.astype(str))
        train = ~test
        if np.sum(train) < 5 or np.sum(test) < 3:
            continue
        xtr, xte, _, _ = _standardize_train_test(x_train[train], x_train[test])
        ytr = y_train[train]
        yte = y_train[test]
        for li, lam in enumerate(lambdas):
            try:
                w = _fit_ridge(xtr, ytr, float(lam))
            except np.linalg.LinAlgError:
                continue
            scores[li] += _metric_dict(yte, xte @ w)["R2_mean"]
            counts[li] += 1
    valid = counts > 0
    if not np.any(valid):
        return float(lambdas[len(lambdas) // 2])
    mean_scores = np.full(lambdas.shape, -np.inf, dtype=np.float64)
    mean_scores[valid] = scores[valid] / counts[valid]
    return float(lambdas[int(np.argmax(mean_scores))])


def _decode_train_test(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
    ridge_selection: str = "inner_cv",
) -> tuple[np.ndarray, np.ndarray, float, int]:
    x_train = np.asarray(x[train_mask], dtype=np.float64)
    x_test = np.asarray(x[test_mask], dtype=np.float64)
    y_train = np.asarray(y[train_mask], dtype=np.float64)
    y_test = np.asarray(y[test_mask], dtype=np.float64)
    keep_x = np.isfinite(x_train).all(axis=0) & np.isfinite(x_test).all(axis=0)
    keep_y = np.isfinite(y_train).all(axis=0) & np.isfinite(y_test).all(axis=0)
    if np.sum(keep_x) == 0 or np.sum(keep_y) == 0 or x_train.shape[0] < 5 or x_test.shape[0] < 3:
        return np.zeros((np.sum(test_mask), np.sum(keep_y)), dtype=np.float64), y_test[:, keep_y], float("nan"), int(np.sum(keep_x))
    x_train = x_train[:, keep_x]
    x_test = x_test[:, keep_x]
    y_train = y_train[:, keep_y]
    y_test = y_test[:, keep_y]
    x_train_s, x_test_s, _, _ = _standardize_train_test(x_train, x_test)
    y_train_s, y_test_s, _, _ = _standardize_train_test(y_train, y_test)
    if ridge_selection == "middle":
        lambdas = _ridge_lambda_grid(x_train_s)
        lam = float(lambdas[len(lambdas) // 2])
    elif ridge_selection == "inner_cv":
        lam = _inner_select_lambda(x_train_s, y_train_s, groups[train_mask], int(seed))
    else:
        raise ValueError(f"Unsupported ridge_selection={ridge_selection!r}")
    try:
        w = _fit_ridge(x_train_s, y_train_s, lam)
        pred = x_test_s @ w
    except np.linalg.LinAlgError:
        pred = np.full_like(y_test_s, np.nan)
    return pred, y_test_s, float(lam), int(np.sum(keep_x))


def _prediction_key(
    *,
    epsilon: float,
    target_family: str,
    basis_type: str,
    k: int,
    basis_draw: str,
    target_mode: str,
) -> tuple[float, str, str, int, str, str]:
    return (float(epsilon), str(target_family), str(basis_type), int(k), str(basis_draw), str(target_mode))


def _add_prediction(
    store: dict[tuple[float, str, str, int, str, str], dict[str, list[Any]]],
    key: tuple[float, str, str, int, str, str],
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    bucket = store.setdefault(key, {"sample_indices": [], "y_true": [], "y_pred": []})
    bucket["sample_indices"].append(np.asarray(sample_indices, dtype=np.int64))
    bucket["y_true"].append(np.asarray(y_true, dtype=np.float64))
    bucket["y_pred"].append(np.asarray(y_pred, dtype=np.float64))


def _packed_prediction(bucket: dict[str, list[Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.concatenate(bucket["sample_indices"], axis=0)
    y_true = np.concatenate(bucket["y_true"], axis=0)
    y_pred = np.concatenate(bucket["y_pred"], axis=0)
    order = np.argsort(idx)
    return idx[order], y_true[order], y_pred[order]


def _derivative_predictive_basis(x_train_full: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    x_s, _, _, _ = _standardize_train_test(x_train_full, x_train_full)
    y_s, _, _, _ = _standardize_train_test(y_train, y_train)
    cross = x_s.T @ y_s
    if cross.ndim != 2 or not np.all(np.isfinite(cross)):
        return np.zeros((x_train_full.shape[1], 0), dtype=np.float64)
    u, _, _ = np.linalg.svd(cross, full_matrices=False)
    return _orth(u[:, : min(int(k), u.shape[1])])


def _score_decode_models(
    *,
    objects_by_delta: dict[float, list[DerivativeObject]],
    sample_df: pd.DataFrame,
    dr: np.ndarray,
    r0: np.ndarray,
    derivative_targets: dict[str, np.ndarray],
    shifted_targets: dict[str, np.ndarray],
    group_to_fold: dict[str, int],
    target_families: list[str],
    k_list: list[int],
    primary_k: int,
    n_random: int,
    n_unit_shuffle: int,
    seed: int,
    run_feature_recovery: bool,
    ridge_selection: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[float, str, str, int, str, str], dict[str, list[Any]]],
]:
    metric_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    prediction_store: dict[tuple[float, str, str, int, str, str], dict[str, list[Any]]] = {}
    groups = sample_df["group_id"].astype(str).to_numpy()
    directions = sample_df["direction"].astype(str).to_numpy()
    sample_indices_all = sample_df["sample_index"].to_numpy(dtype=np.int64)
    n_units = int(dr.shape[1])
    max_k = int(max(k_list))
    for fold_id in sorted(set(group_to_fold.values())):
        test_groups = {group for group, fid in group_to_fold.items() if int(fid) == int(fold_id)}
        train_groups = set(group_to_fold) - test_groups
        leakage_rows.append(
            {
                "fold_id": int(fold_id),
                "n_train_groups": int(len(train_groups)),
                "n_test_groups": int(len(test_groups)),
                "n_shared_groups": int(len(train_groups & test_groups)),
                "status": "pass" if not (train_groups & test_groups) else "fail",
            }
        )
        for epsilon, objects in sorted(objects_by_delta.items()):
            eps_mask = np.isclose(sample_df["epsilon_arcmin"].to_numpy(dtype=np.float64), float(epsilon))
            train_mask = eps_mask & np.isin(groups, list(train_groups))
            test_mask = eps_mask & np.isin(groups, list(test_groups))
            if np.sum(train_mask) < 5 or np.sum(test_mask) < 3:
                continue
            bases = _fit_fold_bases(objects=objects, train_groups=train_groups, max_k=max_k)
            compact = bases["compact_full"]
            static = bases["static_full"]
            for k in k_list:
                rng = np.random.default_rng(int(seed) + int(fold_id) * 10007 + int(round(float(epsilon) * 10000)) + int(k) * 917)
                specs = _basis_specs_for_k(
                    bases=bases,
                    k=int(k),
                    n_units=n_units,
                    rng=rng,
                    n_random=n_random,
                    n_unit_shuffle=n_unit_shuffle,
                )
                if int(k) == int(primary_k):
                    specs.append(BasisSpec("full_response", "sanity_check", "all_units_response_derivative", "observed", 0, None))
                basis_rows.extend(
                    [
                        {
                            "epsilon_arcmin": float(epsilon),
                            "fold_id": int(fold_id),
                            "basis_type": spec.basis_type,
                            "basis_role": spec.basis_role,
                            "basis_training_source": spec.basis_training_source,
                            "basis_draw": spec.basis_draw,
                            "k_requested": int(spec.k_requested),
                            "k_effective": int(n_units if spec.matrix is None else spec.k_effective),
                            "fit_scope": "fold_image_disjoint_train_groups",
                            "n_train_groups": int(len(train_groups)),
                            "n_test_groups": int(len(test_groups)),
                        }
                        for spec in specs
                    ]
                )
                cos_compact_static = _principal_cosines(compact[:, : min(int(k), compact.shape[1])], static[:, : min(int(k), static.shape[1])])
                bridge_rows.append(
                    {
                        "epsilon_arcmin": float(epsilon),
                        "fold_id": int(fold_id),
                        "target_family": "basis_only",
                        "k": int(k),
                        "subspace_a": "compact",
                        "subspace_b": "static_pc",
                        "mean_principal_cosine": float(np.mean(cos_compact_static)) if cos_compact_static.size else float("nan"),
                        "mean_squared_principal_cosine": float(np.mean(cos_compact_static**2)) if cos_compact_static.size else float("nan"),
                        "min_principal_cosine": float(np.min(cos_compact_static)) if cos_compact_static.size else float("nan"),
                        "principal_cosines": ";".join(f"{float(v):.8g}" for v in cos_compact_static),
                        "bridge_status": "available",
                    }
                )
                for target_family in target_families:
                    y_deriv = derivative_targets[target_family]
                    derivative_basis = _derivative_predictive_basis(dr[train_mask], y_deriv[train_mask], int(k))
                    for basis_name, basis_matrix in (
                        ("compact", compact[:, : min(int(k), compact.shape[1])]),
                        ("static_pc", static[:, : min(int(k), static.shape[1])]),
                    ):
                        cosines = _principal_cosines(derivative_basis, basis_matrix)
                        bridge_rows.append(
                            {
                                "epsilon_arcmin": float(epsilon),
                                "fold_id": int(fold_id),
                                "target_family": target_family,
                                "k": int(k),
                                "subspace_a": "full_response_derivative_predictive_svd",
                                "subspace_b": basis_name,
                                "mean_principal_cosine": float(np.mean(cosines)) if cosines.size else float("nan"),
                                "mean_squared_principal_cosine": float(np.mean(cosines**2)) if cosines.size else float("nan"),
                                "min_principal_cosine": float(np.min(cosines)) if cosines.size else float("nan"),
                                "principal_cosines": ";".join(f"{float(v):.8g}" for v in cosines),
                                "bridge_status": "available",
                            }
                        )
                    for spec in specs:
                        if spec.basis_type == "full_response":
                            x = dr
                            k_out = 0
                        else:
                            if spec.matrix is None or spec.matrix.shape[1] == 0:
                                continue
                            x = dr @ spec.matrix
                            k_out = int(k)
                        pred, true, lam, k_eff_x = _decode_train_test(
                            x,
                            y_deriv,
                            groups,
                            train_mask,
                            test_mask,
                            seed=int(seed) + int(fold_id) * 99 + int(k) * 11 + len(metric_rows),
                            ridge_selection=ridge_selection,
                        )
                        test_indices = sample_indices_all[test_mask]
                        _add_prediction(
                            prediction_store,
                            _prediction_key(
                                epsilon=epsilon,
                                target_family=target_family,
                                basis_type=spec.basis_type,
                                k=k_out,
                                basis_draw=spec.basis_draw,
                                target_mode="derivative",
                            ),
                            test_indices,
                            true,
                            pred,
                        )
                        for direction_name in ["all", *sorted(set(directions[test_mask]))]:
                            if direction_name == "all":
                                local = np.ones(test_indices.shape, dtype=bool)
                            else:
                                local = directions[test_mask] == direction_name
                            if np.sum(local) < 2:
                                continue
                            metrics = _metric_dict(true[local], pred[local])
                            metric_rows.append(
                                {
                                    "epsilon_arcmin": float(epsilon),
                                    "fold_id": int(fold_id),
                                    "target_family": target_family,
                                    "target_mode": "derivative",
                                    "basis_type": spec.basis_type,
                                    "basis_draw": spec.basis_draw,
                                    "basis_role": spec.basis_role,
                                    "k": int(k_out),
                                    "k_effective": int(n_units if spec.matrix is None else spec.k_effective),
                                    "direction": direction_name,
                                    "ridge_lambda": lam,
                                    "n_train_samples": int(np.sum(train_mask)),
                                    "n_test_samples": int(np.sum(local)),
                                    "n_train_groups": int(len(train_groups)),
                                    "n_test_groups": int(len(test_groups)),
                                    "target_dim": int(true.shape[1]),
                                    **metrics,
                                }
                            )
                    if run_feature_recovery and int(k) == int(primary_k):
                        static_only_pred, static_only_true, static_only_lam, _ = _decode_train_test(
                            r0,
                            shifted_targets[target_family],
                            groups,
                            train_mask,
                            test_mask,
                            seed=int(seed) + int(fold_id) * 771 + len(utility_rows),
                            ridge_selection=ridge_selection,
                        )
                        static_metrics = _metric_dict(static_only_true, static_only_pred)
                        utility_rows.append(
                            {
                                "epsilon_arcmin": float(epsilon),
                                "fold_id": int(fold_id),
                                "target_family": target_family,
                                "readout_variant": "static_only",
                                "basis_type": "none",
                                "basis_draw": "observed",
                                "k": 0,
                                "ridge_lambda": static_only_lam,
                                "delta_R2_mean_vs_static_only": 0.0,
                                "delta_cosine_vs_static_only": 0.0,
                                **static_metrics,
                            }
                        )
                        utility_specs = [s for s in specs if s.basis_type in {"compact", "static_pc", "compact_resid_static", "random", "unit_shuffle_compact", "noncompact_complement"}]
                        for spec in utility_specs:
                            if spec.matrix is None or spec.matrix.shape[1] == 0:
                                continue
                            x = np.concatenate([r0, dr @ spec.matrix], axis=1)
                            pred, true, lam, _ = _decode_train_test(
                                x,
                                shifted_targets[target_family],
                                groups,
                                train_mask,
                                test_mask,
                                seed=int(seed) + int(fold_id) * 991 + len(utility_rows),
                                ridge_selection=ridge_selection,
                            )
                            metrics = _metric_dict(true, pred)
                            utility_rows.append(
                                {
                                    "epsilon_arcmin": float(epsilon),
                                    "fold_id": int(fold_id),
                                    "target_family": target_family,
                                    "readout_variant": f"static_plus_{spec.basis_type}",
                                    "basis_type": spec.basis_type,
                                    "basis_draw": spec.basis_draw,
                                    "k": int(spec.k_requested),
                                    "k_effective": int(spec.k_effective),
                                    "ridge_lambda": lam,
                                    "delta_R2_mean_vs_static_only": float(metrics["R2_mean"] - static_metrics["R2_mean"]),
                                    "delta_cosine_vs_static_only": float(metrics["feature_cosine_mean"] - static_metrics["feature_cosine_mean"]),
                                    **metrics,
                                }
                            )
    return metric_rows, basis_rows, bridge_rows, utility_rows, leakage_rows, prediction_store


def _bootstrap_metric(
    *,
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_df: pd.DataFrame,
    metric_name: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    meta = sample_df.set_index("sample_index").loc[np.asarray(sample_indices, dtype=np.int64)]
    groups = meta["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    observed = _metric_dict(y_true, y_pred)[metric_name]
    if int(n_bootstrap) <= 0 or unique_groups.size <= 1:
        return observed, float("nan"), float("nan"), int(unique_groups.size)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for bi in range(int(n_bootstrap)):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        mask = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        boot[bi] = _metric_dict(y_true[mask], y_pred[mask])[metric_name]
    return observed, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)), int(unique_groups.size)


def _bootstrap_rows(
    *,
    prediction_store: dict[tuple[float, str, str, int, str, str], dict[str, list[Any]]],
    sample_df: pd.DataFrame,
    primary_k: int,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bootstrap_bases = {"compact", "static_pc", "compact_resid_static", "static_resid_compact", "full_response", "global_rate", "target_pc1"}
    for key, bucket in sorted(prediction_store.items(), key=lambda kv: str(kv[0])):
        epsilon, target_family, basis_type, k, basis_draw, target_mode = key
        if target_mode != "derivative":
            continue
        if basis_type not in bootstrap_bases:
            continue
        if basis_type != "full_response" and int(k) != int(primary_k):
            continue
        if basis_draw != "observed":
            continue
        sample_indices, y_true, y_pred = _packed_prediction(bucket)
        for metric_name in ("R2_mean", "R2_pooled", "feature_cosine_mean", "neg_mse_standardized"):
            mean, lo, hi, n_groups = _bootstrap_metric(
                sample_indices=sample_indices,
                y_true=y_true,
                y_pred=y_pred,
                sample_df=sample_df,
                metric_name=metric_name,
                n_bootstrap=n_bootstrap,
                seed=int(seed) + len(rows) * 37,
            )
            rows.append(
                {
                    "epsilon_arcmin": float(epsilon),
                    "target_family": target_family,
                    "target_mode": target_mode,
                    "basis_type": basis_type,
                    "basis_draw": basis_draw,
                    "k": int(k),
                    "metric": metric_name,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_samples": int(sample_indices.size),
                    "n_groups": n_groups,
                    "n_bootstrap": int(n_bootstrap),
                }
            )
    return rows


def _paired_bootstrap_difference(
    *,
    lhs_bucket: dict[str, list[Any]],
    rhs_bucket: dict[str, list[Any]],
    sample_df: pd.DataFrame,
    metric_name: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, float, int]:
    lhs_idx, lhs_true, lhs_pred = _packed_prediction(lhs_bucket)
    rhs_idx, rhs_true, rhs_pred = _packed_prediction(rhs_bucket)
    common = np.intersect1d(lhs_idx, rhs_idx)
    if common.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    lhs_pos = np.searchsorted(lhs_idx, common)
    rhs_pos = np.searchsorted(rhs_idx, common)
    y_true = lhs_true[lhs_pos]
    lhs_pred_c = lhs_pred[lhs_pos]
    rhs_pred_c = rhs_pred[rhs_pos]
    observed = _metric_dict(y_true, lhs_pred_c)[metric_name] - _metric_dict(rhs_true[rhs_pos], rhs_pred_c)[metric_name]
    meta = sample_df.set_index("sample_index").loc[common]
    groups = meta["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if int(n_bootstrap) <= 0 or unique_groups.size <= 1:
        return observed, float("nan"), float("nan"), float("nan"), int(unique_groups.size)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for bi in range(int(n_bootstrap)):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        mask = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        boot[bi] = _metric_dict(y_true[mask], lhs_pred_c[mask])[metric_name] - _metric_dict(y_true[mask], rhs_pred_c[mask])[metric_name]
    p_two = float(min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))))
    return observed, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)), p_two, int(unique_groups.size)


def _compact_minus_static_rows(
    *,
    prediction_store: dict[tuple[float, str, str, int, str, str], dict[str, list[Any]]],
    sample_df: pd.DataFrame,
    primary_k: int,
    primary_targets: list[str],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    epsilons = sorted({key[0] for key in prediction_store})
    for epsilon in epsilons:
        for target_family in primary_targets:
            lhs_key = _prediction_key(
                epsilon=epsilon,
                target_family=target_family,
                basis_type="compact",
                k=primary_k,
                basis_draw="observed",
                target_mode="derivative",
            )
            rhs_key = _prediction_key(
                epsilon=epsilon,
                target_family=target_family,
                basis_type="static_pc",
                k=primary_k,
                basis_draw="observed",
                target_mode="derivative",
            )
            if lhs_key not in prediction_store or rhs_key not in prediction_store:
                continue
            for metric_name in ("R2_mean", "R2_pooled", "feature_cosine_mean", "neg_mse_standardized"):
                mean, lo, hi, p_two, n_groups = _paired_bootstrap_difference(
                    lhs_bucket=prediction_store[lhs_key],
                    rhs_bucket=prediction_store[rhs_key],
                    sample_df=sample_df,
                    metric_name=metric_name,
                    n_bootstrap=n_bootstrap,
                    seed=int(seed) + len(rows) * 131,
                )
                rows.append(
                    {
                        "epsilon_arcmin": float(epsilon),
                        "target_family": target_family,
                        "lhs_basis": "compact",
                        "rhs_basis": "static_pc",
                        "k": int(primary_k),
                        "metric": metric_name,
                        "mean_lhs_minus_rhs": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "bootstrap_p_two_sided_about_zero": p_two,
                        "n_groups": n_groups,
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
    return rows


def _norm_residual(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    den = 0.5 * (na + nb)
    return float(np.linalg.norm(a + b) / den) if den > 1e-12 else float("nan")


def _linearity_rows_for_values(
    *,
    sample_df: pd.DataFrame,
    sample_indices: np.ndarray,
    values: np.ndarray,
    row_prefix: dict[str, Any],
) -> list[dict[str, Any]]:
    meta = sample_df.set_index("sample_index").loc[np.asarray(sample_indices, dtype=np.int64)].reset_index()
    lookup = {
        (str(row.object_id), float(row.epsilon_arcmin), str(row.direction)): idx
        for idx, row in meta.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for axis, pos_name, neg_name in (("x", "+x", "-x"), ("y", "+y", "-y")):
        vals = []
        for object_id in sorted(set(meta["object_id"].astype(str))):
            for epsilon in sorted(set(meta["epsilon_arcmin"].astype(float))):
                kp = (object_id, float(epsilon), pos_name)
                kn = (object_id, float(epsilon), neg_name)
                if kp in lookup and kn in lookup:
                    vals.append(_norm_residual(values[lookup[kp]], values[lookup[kn]]))
        finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        rows.append(
            {
                **row_prefix,
                "linearity_test": "antisymmetry",
                "axis": axis,
                "epsilon_arcmin": "all",
                "mean_residual": float(np.mean(finite)) if finite.size else float("nan"),
                "median_residual": float(np.median(finite)) if finite.size else float("nan"),
                "n_objects": int(finite.size),
            }
        )
    epsilons = sorted(float(v) for v in set(meta["epsilon_arcmin"].astype(float)))
    for e0, e1 in zip(epsilons[:-1], epsilons[1:], strict=False):
        vals = []
        if not np.isclose(e1 / max(e0, EPS), 2.0, rtol=0.15):
            continue
        for object_id in sorted(set(meta["object_id"].astype(str))):
            for direction in sorted(set(meta["direction"].astype(str))):
                k0 = (object_id, float(e0), direction)
                k1 = (object_id, float(e1), direction)
                if k0 in lookup and k1 in lookup:
                    a = values[lookup[k0]]
                    b = values[lookup[k1]]
                    den = 0.5 * (float(np.linalg.norm(a)) + float(np.linalg.norm(b)))
                    vals.append(float(np.linalg.norm(b - a) / den) if den > 1e-12 else float("nan"))
        finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        rows.append(
            {
                **row_prefix,
                "linearity_test": "scaling_derivative_consistency",
                "axis": "all",
                "epsilon_arcmin": f"{e0:g}_to_{e1:g}",
                "mean_residual": float(np.mean(finite)) if finite.size else float("nan"),
                "median_residual": float(np.median(finite)) if finite.size else float("nan"),
                "n_objects": int(finite.size),
            }
        )
    if {"+x+y", "+x", "+y"}.issubset(set(meta["direction"].astype(str))):
        vals = []
        for object_id in sorted(set(meta["object_id"].astype(str))):
            for epsilon in epsilons:
                kd = (object_id, float(epsilon), "+x+y")
                kx = (object_id, float(epsilon), "+x")
                ky = (object_id, float(epsilon), "+y")
                if kd in lookup and kx in lookup and ky in lookup:
                    lhs = math.sqrt(2.0) * values[lookup[kd]]
                    rhs = values[lookup[kx]] + values[lookup[ky]]
                    den = 0.5 * (float(np.linalg.norm(lhs)) + float(np.linalg.norm(rhs)))
                    vals.append(float(np.linalg.norm(lhs - rhs) / den) if den > 1e-12 else float("nan"))
        finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        rows.append(
            {
                **row_prefix,
                "linearity_test": "additivity_xy_diagonal",
                "axis": "x_plus_y",
                "epsilon_arcmin": "all",
                "mean_residual": float(np.mean(finite)) if finite.size else float("nan"),
                "median_residual": float(np.median(finite)) if finite.size else float("nan"),
                "n_objects": int(finite.size),
            }
        )
    normal_minus_tangent = []
    for object_id in sorted(set(meta["object_id"].astype(str))):
        block = meta[meta["object_id"].astype(str) == object_id]
        normal_axis = str(block["edge_normal_axis"].iloc[0])
        normal_dirs = ["+x", "-x"] if normal_axis == "x" else ["+y", "-y"]
        tangent_dirs = ["+y", "-y"] if normal_axis == "x" else ["+x", "-x"]
        for epsilon in epsilons:
            normal_norms = [
                float(np.linalg.norm(values[lookup[(object_id, float(epsilon), direction)]]))
                for direction in normal_dirs
                if (object_id, float(epsilon), direction) in lookup
            ]
            tangent_norms = [
                float(np.linalg.norm(values[lookup[(object_id, float(epsilon), direction)]]))
                for direction in tangent_dirs
                if (object_id, float(epsilon), direction) in lookup
            ]
            if normal_norms and tangent_norms:
                normal_minus_tangent.append(float(np.mean(normal_norms) - np.mean(tangent_norms)))
    finite_nt = np.asarray([v for v in normal_minus_tangent if np.isfinite(v)], dtype=np.float64)
    rows.append(
        {
            **row_prefix,
            "linearity_test": "edge_normal_minus_tangent_norm",
            "axis": "image_gradient_dominant_axis",
            "epsilon_arcmin": "all",
            "mean_residual": float(np.mean(finite_nt)) if finite_nt.size else float("nan"),
            "median_residual": float(np.median(finite_nt)) if finite_nt.size else float("nan"),
            "n_objects": int(finite_nt.size),
        }
    )
    return rows


def _linearity_rows(
    *,
    prediction_store: dict[tuple[float, str, str, int, str, str], dict[str, list[Any]]],
    sample_df: pd.DataFrame,
    primary_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_true: set[tuple[str, str]] = set()
    for key, bucket in sorted(prediction_store.items(), key=lambda kv: str(kv[0])):
        epsilon, target_family, basis_type, k, basis_draw, target_mode = key
        if target_mode != "derivative":
            continue
        if basis_type != "full_response" and int(k) != int(primary_k):
            continue
        if basis_draw != "observed":
            continue
        sample_indices, y_true, y_pred = _packed_prediction(bucket)
        true_key = (target_family, target_mode)
        if true_key not in seen_true:
            seen_true.add(true_key)
            rows.extend(
                _linearity_rows_for_values(
                    sample_df=sample_df,
                    sample_indices=sample_indices,
                    values=y_true,
                    row_prefix={
                        "target_family": target_family,
                        "target_mode": target_mode,
                        "basis_type": "true_target",
                        "basis_draw": "none",
                        "k": 0,
                        "source": "true",
                    },
                )
            )
        rows.extend(
            _linearity_rows_for_values(
                sample_df=sample_df,
                sample_indices=sample_indices,
                values=y_pred,
                row_prefix={
                    "target_family": target_family,
                    "target_mode": target_mode,
                    "basis_type": basis_type,
                    "basis_draw": basis_draw,
                    "k": int(k),
                    "source": "decoded",
                },
            )
        )
    return rows


def _raw_pixel_jacobian_descriptor(frame: np.ndarray, direction: DirectionSpec, grid_size: int) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float64)
    gy, gx = np.gradient(f - float(np.mean(f)))
    # Positive dx in the cached grid_sample convention samples f(x - dx), so
    # the direct derivative of shifted image content is the negative gradient.
    directional = -(float(direction.ux) * gx + float(direction.uy) * gy)
    return _grid_pool(directional, int(grid_size))


def _provenance_audit(
    *,
    tfts_root: Path,
    tangent_meta: dict[str, Any],
    sample_df: pd.DataFrame,
    objects_by_delta: dict[float, list[DerivativeObject]],
    feature_frame_mode: str,
    grid_size: int,
) -> dict[str, Any]:
    summary_path = Path(tfts_root) / "twin_feature_tangent_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    manifest_status = "pass" if summary.get("status") in {"production_completed", "completed"} else "warn"
    cache_has_endpoints = True
    try:
        _, by_delta, _ = _load_tangent_payload(Path(tfts_root))
        for delta in tangent_meta["used_epsilons_arcmin"]:
            for meta in by_delta[float(delta)].values():
                for key in ("rx_p", "rx_m", "ry_p", "ry_m", "history"):
                    if key not in meta:
                        cache_has_endpoints = False
                        break
    except Exception:
        cache_has_endpoints = False
    first_delta = sorted(objects_by_delta)[0]
    first_obj = objects_by_delta[first_delta][0]
    frame = _frame_from_history(first_obj.history, feature_frame_mode)
    direction = DirectionSpec("+x", 1.0, 0.0)
    plus = _shift_stack_subpixel(frame, dx_px=first_obj.delta_model_px, dy_px=0.0)
    minus = _shift_stack_subpixel(frame, dx_px=-first_obj.delta_model_px, dy_px=0.0)
    direct = (_grid_pool(plus - float(np.mean(plus)), grid_size) - _grid_pool(minus - float(np.mean(minus)), grid_size)) / (
        2.0 * first_obj.delta_model_px
    )
    jac = _raw_pixel_jacobian_descriptor(frame, direction, grid_size)
    den = float(np.linalg.norm(direct) * np.linalg.norm(jac))
    jac_cos = float(np.dot(direct, jac) / den) if den > 1e-12 else float("nan")
    jac_rel = float(np.linalg.norm(direct - jac) / max(np.linalg.norm(direct), EPS))
    ramp = np.tile(np.linspace(-1.0, 1.0, frame.shape[1]), (frame.shape[0], 1))
    ramp_direct = (_shift_stack_subpixel(ramp, dx_px=0.25, dy_px=0.0) - _shift_stack_subpixel(ramp, dx_px=-0.25, dy_px=0.0)) / 0.5
    ramp_mean = float(np.mean(ramp_direct[:, 2:-2])) if ramp.shape[1] > 4 else float(np.mean(ramp_direct))
    shared_groups = 0
    for fold_id in sorted(sample_df.get("fold_id", pd.Series(dtype=int)).dropna().unique()):
        train_groups = set(sample_df.loc[sample_df["fold_id"] != fold_id, "group_id"].astype(str))
        test_groups = set(sample_df.loc[sample_df["fold_id"] == fold_id, "group_id"].astype(str))
        shared_groups += len(train_groups & test_groups)
    return {
        "checks": [
            {
                "name": "fresh_or_manifest_verified_cache",
                "status": manifest_status,
                "details": "using cached shifted responses and histories with production summary"
                if manifest_status == "pass"
                else "summary manifest missing or not marked production_completed",
            },
            {
                "name": "shifted_response_endpoint_cache",
                "status": "pass" if cache_has_endpoints else "fail",
                "details": "rx_p/rx_m/ry_p/ry_m/history present in tangent cache",
            },
            {
                "name": "units_and_shift_convention_recorded",
                "status": "pass",
                "details": "feature derivative and response derivative both use cached delta_model_px; positive dx follows grid_sample_border_align_corners_true",
            },
            {
                "name": "df_direct_vs_raw_pixel_jacobian",
                "status": "pass" if np.isfinite(jac_cos) and jac_cos > 0.95 else "warn",
                "cosine": jac_cos,
                "relative_error": jac_rel,
                "details": "raw-pixel-grid target compared to pooled image-gradient independent path",
            },
            {
                "name": "synthetic_ramp_shift_sign",
                "status": "pass" if np.isfinite(ramp_mean) and ramp_mean < 0.0 else "warn",
                "mean_derivative_for_positive_dx_on_x_ramp": ramp_mean,
                "details": "positive dx moves image content right under grid_sample, yielding a negative derivative on an increasing x ramp",
            },
            {
                "name": "fold_group_leakage",
                "status": "pass" if shared_groups == 0 else "fail",
                "n_shared_groups": int(shared_groups),
            },
        ],
        "source_summary_path": str(summary_path.resolve()),
        "source_summary_status": summary.get("status"),
        "source_model_ppd": summary.get("model_ppd"),
        "source_history_length_frames": summary.get("history_length_frames"),
        "source_prediction_validation_max_abs_diff_max": summary.get("prediction_validation_max_abs_diff_max"),
        "feature_frame_mode": feature_frame_mode,
        "shift_interpolation": "scipy_ndimage_shift_order1_nearest_equivalent_to_cached_grid_sample_sign",
        "derivative_units": "per_model_pixel",
    }


def _summarize_scale_sweep(
    bootstrap_rows: list[dict[str, Any]],
    linearity_rows: list[dict[str, Any]],
    *,
    primary_k: int,
    primary_targets: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boot = pd.DataFrame(bootstrap_rows)
    lin = pd.DataFrame(linearity_rows)
    if boot.empty:
        return rows
    for epsilon, block in boot.groupby("epsilon_arcmin", sort=True):
        for target_family in primary_targets:
            rec: dict[str, Any] = {
                "epsilon_arcmin": float(epsilon),
                "target_family": target_family,
                "k": int(primary_k),
            }
            for basis in ("compact", "static_pc", "compact_resid_static", "full_response"):
                row = block[
                    (block["target_family"].astype(str) == target_family)
                    & (block["basis_type"].astype(str) == basis)
                    & (block["metric"].astype(str) == "R2_mean")
                ]
                rec[f"{basis}_R2_mean"] = float(row["mean"].iloc[0]) if not row.empty else float("nan")
            if not lin.empty:
                lin_block = lin[
                    (lin["target_family"].astype(str) == target_family)
                    & (lin["basis_type"].astype(str) == "compact")
                    & (lin["linearity_test"].astype(str) == "scaling_derivative_consistency")
                ]
                rec["compact_scaling_residual_median"] = float(lin_block["median_residual"].median()) if not lin_block.empty else float("nan")
            rows.append(rec)
    return rows


def _write_figures(
    *,
    out: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    linearity: pd.DataFrame,
    bridge: pd.DataFrame,
    utility: pd.DataFrame,
    primary_k: int,
) -> None:
    fig_dir = out / "figures"
    if not bootstrap.empty:
        block = bootstrap[
            (bootstrap["target_family"].astype(str) == "phase_vector")
            & (bootstrap["metric"].astype(str) == "R2_mean")
            & (bootstrap["basis_type"].astype(str).isin(["compact", "static_pc"]))
        ].copy()
        if not block.empty:
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            for basis, color in (("compact", MODEL), ("static_pc", ACCENT)):
                b = block[block["basis_type"].astype(str) == basis].sort_values("epsilon_arcmin")
                ax.plot(b["epsilon_arcmin"], b["mean"], marker="o", color=color, lw=2.0, label=basis)
                if np.all(np.isfinite(b["ci_low"])) and np.all(np.isfinite(b["ci_high"])):
                    ax.fill_between(b["epsilon_arcmin"], b["ci_low"], b["ci_high"], color=color, alpha=0.14, linewidth=0)
            ax.set_xlabel("epsilon (arcmin)")
            ax.set_ylabel("held-out R2")
            ax.set_title("Signed phase derivative readout", loc="left", fontweight="bold")
            _clean_axes(ax, grid=True)
            ax.legend(frameon=False)
            fig.tight_layout()
            _save_fig(fig, fig_dir / "signed_phase_compact_vs_static")
    if not metrics.empty:
        block = metrics[
            (metrics["target_family"].astype(str) == "phase_vector")
            & (metrics["direction"].astype(str) == "all")
            & (metrics["basis_draw"].astype(str) == "observed")
            & (metrics["basis_type"].astype(str).isin(["compact", "static_pc", "compact_resid_static"]))
            & (metrics["k"].astype(int) > 0)
        ].copy()
        if not block.empty:
            summary = (
                block.groupby(["basis_type", "k"], as_index=False)
                .agg(R2_mean=("R2_mean", "mean"))
                .sort_values(["basis_type", "k"])
            )
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            colors = {"compact": MODEL, "static_pc": ACCENT, "compact_resid_static": GREEN}
            for basis in ("compact", "static_pc", "compact_resid_static"):
                b = summary[summary["basis_type"].astype(str) == basis]
                if b.empty:
                    continue
                ax.plot(b["k"], b["R2_mean"], marker="o", lw=1.8, color=colors[basis], label=basis)
            ax.set_xlabel("basis dimension k")
            ax.set_ylabel("fold-mean held-out R2")
            ax.set_title("k sweep derivative prediction", loc="left", fontweight="bold")
            _clean_axes(ax, grid=True)
            ax.legend(frameon=False)
            fig.tight_layout()
            _save_fig(fig, fig_dir / "k_sweep_derivative_prediction")
    if not linearity.empty:
        block = linearity[
            (linearity["basis_type"].astype(str).isin(["true_target", "compact", "static_pc"]))
            & (linearity["linearity_test"].astype(str).isin(["antisymmetry", "scaling_derivative_consistency"]))
            & (linearity["target_family"].astype(str) == "phase_vector")
        ].copy()
        if not block.empty:
            summary = (
                block.groupby(["basis_type", "linearity_test"], as_index=False)
                .agg(median_residual=("median_residual", "median"))
            )
            labels = [f"{r.basis_type}\n{r.linearity_test.replace('_', ' ')}" for r in summary.itertuples()]
            fig, ax = plt.subplots(figsize=(6.2, 3.2))
            ax.bar(np.arange(len(summary)), summary["median_residual"], color=[MODEL if "compact" in x else ACCENT if "static" in x else NULL for x in summary["basis_type"]])
            ax.set_xticks(np.arange(len(summary)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("median normalized residual")
            ax.set_title("Linearity residuals by scale", loc="left", fontweight="bold")
            _clean_axes(ax, grid=True)
            fig.tight_layout()
            _save_fig(fig, fig_dir / "linearity_residuals_by_scale")
    if not bridge.empty:
        block = bridge[
            (bridge["target_family"].astype(str).isin(["basis_only", "phase_vector"]))
            & (bridge["k"].astype(int) == int(primary_k))
        ].copy()
        if not block.empty:
            labels = [f"{r.subspace_a}\nvs {r.subspace_b}" for r in block.itertuples()]
            fig, ax = plt.subplots(figsize=(max(5.2, 0.45 * len(labels)), 3.2))
            ax.bar(np.arange(len(block)), block["mean_principal_cosine"], color=BRIDGE)
            ax.set_xticks(np.arange(len(block)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("mean principal cosine")
            ax.set_title("Subspace bridge principal angles", loc="left", fontweight="bold")
            ax.set_ylim(0.0, 1.0)
            _clean_axes(ax, grid=True)
            fig.tight_layout()
            _save_fig(fig, fig_dir / "subspace_bridge_principal_angles")
    if not utility.empty:
        block = utility[
            (utility["target_family"].astype(str) == "phase_vector")
            & (utility["basis_draw"].astype(str) == "observed")
            & (utility["readout_variant"].astype(str) != "static_only")
        ].copy()
        if not block.empty:
            summary = (
                block.groupby("readout_variant", as_index=False)
                .agg(delta_R2_mean_vs_static_only=("delta_R2_mean_vs_static_only", "mean"))
                .sort_values("delta_R2_mean_vs_static_only")
            )
            fig, ax = plt.subplots(figsize=(5.4, 3.2))
            ax.barh(summary["readout_variant"], summary["delta_R2_mean_vs_static_only"], color=GREEN)
            ax.axvline(0.0, color=TEXT, lw=0.9, alpha=0.55)
            ax.set_xlabel("Delta R2 vs static only")
            ax.set_title("Feature recovery utility", loc="left", fontweight="bold")
            _clean_axes(ax, grid=True)
            fig.tight_layout()
            _save_fig(fig, fig_dir / "feature_recovery_utility")


def _decision_table(
    *,
    compact_minus_static: pd.DataFrame,
    bootstrap: pd.DataFrame,
    linearity: pd.DataFrame,
    primary_k: int,
    primary_targets: list[str],
) -> dict[str, Any]:
    primary = compact_minus_static[
        (compact_minus_static["metric"].astype(str) == "R2_mean")
        & (compact_minus_static["target_family"].astype(str).isin(primary_targets))
        & (compact_minus_static["k"].astype(int) == int(primary_k))
    ] if not compact_minus_static.empty else pd.DataFrame()
    compact_positive = (not primary.empty) and float(primary["mean_lhs_minus_rhs"].mean()) > 0.0
    compact_ci_positive = (not primary.empty) and np.all(primary["ci_low"].to_numpy(dtype=np.float64) > 0.0)
    decoded_lin_block = linearity[
        (linearity["basis_type"].astype(str) == "compact")
        & (linearity["target_family"].astype(str).isin(primary_targets))
        & (linearity["linearity_test"].astype(str).isin(["antisymmetry", "scaling_derivative_consistency"]))
    ] if not linearity.empty else pd.DataFrame()
    target_lin_block = linearity[
        (linearity["basis_type"].astype(str) == "true_target")
        & (linearity["target_family"].astype(str).isin(primary_targets))
        & (linearity["linearity_test"].astype(str).isin(["antisymmetry", "scaling_derivative_consistency"]))
    ] if not linearity.empty else pd.DataFrame()
    target_linearity_pass = (not target_lin_block.empty) and float(target_lin_block["median_residual"].median()) < 0.75
    full_block = bootstrap[
        (bootstrap["basis_type"].astype(str) == "full_response")
        & (bootstrap["metric"].astype(str) == "R2_mean")
        & (bootstrap["target_family"].astype(str).isin(primary_targets))
    ] if not bootstrap.empty else pd.DataFrame()
    compact_block = bootstrap[
        (bootstrap["basis_type"].astype(str) == "compact")
        & (bootstrap["metric"].astype(str) == "R2_mean")
        & (bootstrap["target_family"].astype(str).isin(primary_targets))
    ] if not bootstrap.empty else pd.DataFrame()

    full_r2_mean = float(full_block["mean"].mean()) if not full_block.empty else None
    full_r2_ci_low_min = float(full_block["ci_low"].min()) if not full_block.empty else None
    compact_r2_mean = float(compact_block["mean"].mean()) if not compact_block.empty else None
    compact_r2_ci_low_min = float(compact_block["ci_low"].min()) if not compact_block.empty else None
    full_response_reliably_positive = (
        full_r2_mean is not None
        and full_r2_ci_low_min is not None
        and full_r2_mean > 0.0
        and full_r2_ci_low_min > 0.0
    )
    full_response_above_compact = (
        full_r2_mean is not None
        and compact_r2_mean is not None
        and full_r2_mean > compact_r2_mean
    )

    if compact_ci_positive and target_linearity_pass:
        outcome = "strong_compact_specific_local_derivative_result"
        interpretation = (
            "Compact beats fold/image-disjoint static PCs on primary signed derivative targets, "
            "and the true target derivatives pass the configured signed-derivative consistency screen."
        )
    elif compact_positive:
        outcome = "compact_advantage_but_linearity_or_ci_incomplete"
        interpretation = (
            "Compact is above static PCs on average, but the first-pass confidence or target-consistency "
            "criteria are not strong enough for the primary strong-positive language."
        )
    elif full_response_above_compact and full_response_reliably_positive:
        outcome = "full_response_contains_more_derivative_information_than_compact"
        interpretation = (
            "Full response-space prediction is stronger than the compact channel; local derivative "
            "information may exist without being concentrated in the compact object."
        )
    elif full_response_above_compact and not full_response_reliably_positive:
        outcome = "decoder_sanity_failed_no_reliable_heldout_derivative_readout"
        interpretation = (
            "Full response-space prediction is numerically above compact, but the all-unit held-out "
            "primary R2 does not clear zero. Treat the derivative readout as inconclusive rather than "
            "evidence for diffuse derivative information."
        )
    else:
        outcome = "not_compact_specific_on_first_pass"
        interpretation = (
            "The current first pass does not isolate a compact-specific signed local derivative "
            "component beyond static-PC controls."
        )
    return {
        "primary_k": int(primary_k),
        "primary_targets": primary_targets,
        "outcome": outcome,
        "interpretation": interpretation,
        "compact_minus_static_R2_mean_mean": float(primary["mean_lhs_minus_rhs"].mean()) if not primary.empty else None,
        "compact_minus_static_R2_ci_low_min": float(primary["ci_low"].min()) if not primary.empty else None,
        "compact_R2_mean_mean": compact_r2_mean,
        "compact_R2_ci_low_min": compact_r2_ci_low_min,
        "full_response_R2_mean_mean": full_r2_mean,
        "full_response_R2_ci_low_min": full_r2_ci_low_min,
        "full_response_reliably_positive": bool(full_response_reliably_positive),
        "true_target_linearity_median_residual": (
            float(target_lin_block["median_residual"].median()) if not target_lin_block.empty else None
        ),
        "compact_decoded_linearity_diagnostic_median_residual": (
            float(decoded_lin_block["median_residual"].median()) if not decoded_lin_block.empty else None
        ),
        "compact_linearity_median_residual": (
            float(decoded_lin_block["median_residual"].median()) if not decoded_lin_block.empty else None
        ),
        "rules": [
            "Strong positive requires compact-minus-static primary R2 CI above zero and true-target signed-derivative consistency median residual below 0.75.",
            "Decoded compact linearity is reported as a diagnostic only; antisymmetry can be induced by the signed sample construction plus a linear decoder.",
            "Full response supports diffuse-derivative language only when its primary held-out R2 mean and CI low are both above zero.",
            "Covariance-closure bridge remains unavailable unless external basis vectors are serialized.",
        ],
    }


def _write_readme(
    *,
    out: Path,
    config: dict[str, Any],
    decision: dict[str, Any],
    compact_minus_static: pd.DataFrame,
) -> None:
    def _fmt(value: Any) -> str:
        return "unavailable" if value is None else f"{float(value):.6g}"

    primary_rows = compact_minus_static[
        (compact_minus_static["metric"].astype(str) == "R2_mean")
        & (compact_minus_static["target_family"].astype(str).isin(config["primary_targets"]))
    ] if not compact_minus_static.empty else pd.DataFrame()
    lines = [
        "# Local Derivative Channel Analysis",
        "",
        "This directory contains the first-pass signed local image-feature derivative readout from the compact retinal-translation channel.",
        "",
        "## Configuration",
        "",
        f"- tangent cache: `{config['tangent_source']}`",
        f"- epsilon values: `{', '.join(str(v) for v in config['epsilon_arcmin'])}` arcmin",
        f"- derivative units: `{config['derivative_units']}`",
        f"- feature frame mode: `{config['feature_frame_mode']}`",
        f"- split group: `{config['group_by']}`",
        f"- folds: `{config['n_folds']}`",
        f"- primary k: `{config['primary_k']}`",
        "",
        "## Decision",
        "",
        f"- outcome: `{decision['outcome']}`",
        f"- interpretation: {decision['interpretation']}",
        "",
    ]
    if (
        decision.get("compact_R2_mean_mean") is not None
        or decision.get("full_response_R2_mean_mean") is not None
    ):
        lines.extend(
            [
                "## Decoder Sanity",
                "",
                f"- compact primary held-out R2 mean: `{_fmt(decision.get('compact_R2_mean_mean'))}`",
                f"- compact primary held-out R2 CI-low minimum: `{_fmt(decision.get('compact_R2_ci_low_min'))}`",
                f"- full-response primary held-out R2 mean: `{_fmt(decision.get('full_response_R2_mean_mean'))}`",
                f"- full-response primary held-out R2 CI-low minimum: `{_fmt(decision.get('full_response_R2_ci_low_min'))}`",
                f"- full-response reliably positive: `{bool(decision.get('full_response_reliably_positive'))}`",
                f"- true-target linearity median residual: `{_fmt(decision.get('true_target_linearity_median_residual'))}`",
                f"- compact decoded linearity diagnostic median residual: `{_fmt(decision.get('compact_decoded_linearity_diagnostic_median_residual'))}`",
                "",
            ]
        )
    if not primary_rows.empty:
        lines.extend(["## Primary Compact Minus Static-PC R2", ""])
        for row in primary_rows.sort_values(["epsilon_arcmin", "target_family"]).itertuples():
            lines.append(
                f"- epsilon `{float(row.epsilon_arcmin):g}`, `{row.target_family}`: "
                f"{float(row.mean_lhs_minus_rhs):.4f} "
                f"[{float(row.ci_low):.4f}, {float(row.ci_high):.4f}]"
            )
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `run_manifest.json`: configuration, source paths, git commit, folds, targets, and seeds.",
            "- `provenance_audit.json`: provenance gate checks and shift/sign conventions.",
            "- `image_fold_assignments.csv`: image/group-disjoint folds.",
            "- `feature_target_inventory.csv`: feature-family definitions and dimensions.",
            "- `basis_inventory.csv`: fold-trained bases and controls.",
            "- `derivative_prediction_metrics.csv`: held-out fold metrics.",
            "- `derivative_prediction_bootstrap.csv`: clustered bootstrap summaries over held-out groups.",
            "- `compact_minus_static_primary.csv`: pre-committed compact-minus-static-PC contrast.",
            "- `linearity_consistency_metrics.csv`: antisymmetry, scaling, additivity, and edge-axis checks.",
            "- `scale_sweep_summary.csv`: primary scale sweep summary.",
            "- `subspace_bridge_principal_angles.csv`: reconstructable derivative/static/compact subspace overlaps.",
            "- `covariance_bridge_metrics.csv`: covariance bridge status.",
            "- `feature_recovery_utility.csv`: shifted-feature recovery utility readouts.",
            "- `decision_table.json`: rule-based interpretation summary.",
            "",
            "The covariance-closure bridge is marked unavailable unless a source run provides serialized basis vectors. Metrics from closure summaries alone are not sufficient to compute principal angles or derivative-basis capture.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    requested_eps = _parse_float_list(args.epsilon_arcmin)
    k_list = _parse_int_list(args.k_list)
    primary_targets = _parse_str_list(args.primary_targets)
    target_families = _parse_str_list(args.target_families)
    directions = parse_directions(args.directions)
    objects_by_delta, skipped_rows, tangent_meta = _collect_objects(
        tfts_root=Path(args.tfts_root),
        requested_epsilons=requested_eps,
        group_by=str(args.group_by),
    )
    write_csv(out / "skipped_derivative_objects.csv", skipped_rows)

    all_base_groups: list[str] = []
    image_rows: list[dict[str, Any]] = []
    for epsilon, objects in sorted(objects_by_delta.items()):
        for obj in objects:
            all_base_groups.append(obj.group_id)
            image_rows.append(
                {
                    "epsilon_arcmin": float(epsilon),
                    "object_id": obj.object_id,
                    "group_id": obj.group_id,
                    "image_id": obj.image_id,
                    "trial_index": obj.trial_index,
                    "time_index": obj.time_index,
                }
            )
    group_to_fold, fold_rows = _assign_group_folds(all_base_groups, n_folds=int(args.n_folds), seed=int(args.seed))
    for row in image_rows:
        row["fold_id"] = int(group_to_fold[str(row["group_id"])])
    write_csv(out / "image_fold_assignments.csv", image_rows)
    write_csv(out / "fold_inventory.csv", fold_rows)

    extractor = FeatureExtractor(
        target_families=target_families,
        gabor_wavelengths=_parse_float_list(args.gabor_wavelengths),
        gabor_orientations_deg=_parse_float_list(args.gabor_orientations_deg, positive=False),
        grid_size=int(args.grid_size),
    )
    frame_shape = tuple(_frame_from_history(next(iter(next(iter(objects_by_delta.values())))).history, str(args.feature_frame_mode)).shape)
    feature_inventory = extractor.inventory_rows(frame_shape)
    write_csv(out / "feature_target_inventory.csv", feature_inventory)
    sample_df, dr, r0, derivative_targets, shifted_targets, object_feature_rows = _build_feature_samples(
        objects_by_delta=objects_by_delta,
        directions=directions,
        feature_frame_mode=str(args.feature_frame_mode),
        extractor=extractor,
        target_families=target_families,
    )
    sample_df["fold_id"] = sample_df["group_id"].astype(str).map(group_to_fold).astype(int)
    write_csv(out / "feature_sample_inventory.csv", sample_df.to_dict(orient="records"))
    write_csv(out / "feature_object_inventory.csv", object_feature_rows)

    provenance = _provenance_audit(
        tfts_root=Path(args.tfts_root),
        tangent_meta=tangent_meta,
        sample_df=sample_df,
        objects_by_delta=objects_by_delta,
        feature_frame_mode=str(args.feature_frame_mode),
        grid_size=int(args.grid_size),
    )
    write_json(out / "provenance_audit.json", provenance)

    metric_rows, basis_rows, bridge_rows, utility_rows, leakage_rows, prediction_store = _score_decode_models(
        objects_by_delta=objects_by_delta,
        sample_df=sample_df,
        dr=dr,
        r0=r0,
        derivative_targets=derivative_targets,
        shifted_targets=shifted_targets,
        group_to_fold=group_to_fold,
        target_families=target_families,
        k_list=k_list,
        primary_k=int(args.primary_k),
        n_random=int(args.n_random),
        n_unit_shuffle=int(args.n_unit_shuffle),
        seed=int(args.seed),
        run_feature_recovery=bool(args.run_feature_recovery_utility),
        ridge_selection=str(args.ridge_selection),
    )
    basis_rows.append(
        {
            "epsilon_arcmin": "",
            "fold_id": "",
            "basis_type": "rf_readout_permuted_compact",
            "basis_role": "basis_control",
            "basis_training_source": "rf_readout_metadata",
            "basis_draw": "unavailable",
            "k_requested": int(args.primary_k),
            "k_effective": 0,
            "fit_scope": "not_run",
            "status": "unavailable_no_rf_readout_metadata_in_tangent_cache",
        }
    )
    write_csv(out / "basis_inventory.csv", basis_rows)
    write_csv(out / "fold_leakage_audit.csv", leakage_rows)
    write_csv(out / "derivative_prediction_metrics.csv", metric_rows)
    write_csv(out / "subspace_bridge_principal_angles.csv", bridge_rows)
    write_csv(out / "feature_recovery_utility.csv", utility_rows)

    bootstrap_rows = _bootstrap_rows(
        prediction_store=prediction_store,
        sample_df=sample_df,
        primary_k=int(args.primary_k),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 40000,
    )
    compact_minus_static_rows = _compact_minus_static_rows(
        prediction_store=prediction_store,
        sample_df=sample_df,
        primary_k=int(args.primary_k),
        primary_targets=primary_targets,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 50000,
    )
    linearity_rows = _linearity_rows(
        prediction_store=prediction_store,
        sample_df=sample_df,
        primary_k=int(args.primary_k),
    )
    scale_rows = _summarize_scale_sweep(
        bootstrap_rows,
        linearity_rows,
        primary_k=int(args.primary_k),
        primary_targets=primary_targets,
    )
    write_csv(out / "derivative_prediction_bootstrap.csv", bootstrap_rows)
    write_csv(out / "compact_minus_static_primary.csv", compact_minus_static_rows)
    write_csv(out / "linearity_consistency_metrics.csv", linearity_rows)
    write_csv(out / "scale_sweep_summary.csv", scale_rows)
    write_csv(
        out / "covariance_bridge_metrics.csv",
        [
            {
                "bridge_metric": "derivative_basis_to_covariance_closure_basis",
                "status": "unavailable",
                "reason": "closure artifacts provide metrics/manifests but no serialized basis vectors",
                "closure_root": str(Path(args.closure_root).resolve()),
            }
        ],
    )

    decision = _decision_table(
        compact_minus_static=pd.DataFrame(compact_minus_static_rows),
        bootstrap=pd.DataFrame(bootstrap_rows),
        linearity=pd.DataFrame(linearity_rows),
        primary_k=int(args.primary_k),
        primary_targets=primary_targets,
    )
    write_json(out / "decision_table.json", decision)

    config = {
        "analysis_name": "local_derivative_channel_analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tangent_source": tangent_meta["tangent_source"],
        "closure_root": str(Path(args.closure_root).resolve()),
        "out_root": str(out.resolve()),
        "epsilon_arcmin": sorted(objects_by_delta),
        "requested_epsilon_arcmin": requested_eps,
        "directions": [spec.name for spec in directions],
        "direction_vectors": {spec.name: [spec.ux, spec.uy] for spec in directions},
        "target_families": target_families,
        "primary_targets": primary_targets,
        "feature_frame_mode": str(args.feature_frame_mode),
        "feature_config": {
            "gabor_wavelengths_px": _parse_float_list(args.gabor_wavelengths),
            "gabor_orientations_deg": _parse_float_list(args.gabor_orientations_deg, positive=False),
            "grid_size": int(args.grid_size),
        },
        "group_by": str(args.group_by),
        "n_folds": int(args.n_folds),
        "k_list": k_list,
        "primary_k": int(args.primary_k),
        "n_random": int(args.n_random),
        "n_unit_shuffle": int(args.n_unit_shuffle),
        "n_bootstrap": int(args.n_bootstrap),
        "ridge_selection": str(args.ridge_selection),
        "seed": int(args.seed),
        "n_units": int(tangent_meta["n_units"]),
        "n_samples": int(sample_df.shape[0]),
        "n_groups": int(len(group_to_fold)),
        "derivative_units": "per_model_pixel",
        "shift_convention": "grid_sample_border_align_corners_true_positive_dx_samples_x_minus_dx",
        "interpolation_mode": "bilinear_border",
        "provenance_status": {
            row["name"]: row["status"] for row in provenance["checks"]
        },
        **_repo_commit(),
    }
    write_json(out / "run_manifest.json", config)

    _write_figures(
        out=out,
        metrics=pd.DataFrame(metric_rows),
        bootstrap=pd.DataFrame(bootstrap_rows),
        linearity=pd.DataFrame(linearity_rows),
        bridge=pd.DataFrame(bridge_rows),
        utility=pd.DataFrame(utility_rows),
        primary_k=int(args.primary_k),
    )
    _write_readme(
        out=out,
        config=config,
        decision=decision,
        compact_minus_static=pd.DataFrame(compact_minus_static_rows),
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--closure-root", type=Path, default=DEFAULT_CLOSURE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epsilon-arcmin", type=str, default="0.125,0.25,0.5")
    parser.add_argument("--directions", type=str, default="+x,-x,+y,-y")
    parser.add_argument("--target-families", type=str, default="gabor_even_odd,phase_vector,bandpass_signed,gabor_energy,raw_pixel_grid")
    parser.add_argument("--primary-targets", type=str, default="phase_vector,gabor_even_odd")
    parser.add_argument("--feature-frame-mode", choices=["current", "center", "mean"], default="current")
    parser.add_argument("--gabor-wavelengths", type=str, default="4,8,16")
    parser.add_argument("--gabor-orientations-deg", type=str, default="0,45,90,135")
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--k-list", type=str, default="2,5,10,20,30")
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--group-by", choices=["image_id", "trial_index", "object_id"], default="image_id")
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--n-unit-shuffle", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--ridge-selection",
        choices=["inner_cv", "middle"],
        default="inner_cv",
        help="Use inner image-disjoint CV for production, or the middle grid value for fast smoke/debug runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-feature-recovery-utility", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = analyze(args)
    print(f"Wrote local derivative channel analysis to {config['out_root']}")
    print(
        f"Samples: {config['n_samples']} across {config['n_groups']} {config['group_by']} groups; "
        f"primary k={config['primary_k']}"
    )


if __name__ == "__main__":
    main()

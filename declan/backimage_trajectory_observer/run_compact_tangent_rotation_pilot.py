from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import DEFAULT_INPUT

BX_COLOR = "#0072B2"
BY_COLOR = "#D55E00"
PRED_COLOR = "#CC79A7"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def _parse_ints(value: str) -> list[int]:
    vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    return sorted({v for v in vals if v > 0})


def _parse_float_pair(value: str) -> tuple[float, float]:
    bits = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    if len(bits) != 2:
        raise ValueError(f"Expected two comma-separated floats, got {value!r}")
    norm = float(np.hypot(bits[0], bits[1]))
    if norm <= 0:
        raise ValueError("Direction vector must be nonzero")
    return bits[0] / norm, bits[1] / norm


def _fit_pca(x: np.ndarray, n_components: int, *, center: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    mean = np.mean(arr, axis=0) if center else np.zeros(arr.shape[1], dtype=np.float64)
    xc = arr - mean[None, :]
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    evals = (s * s) / max(arr.shape[0] - 1, 1)
    k = min(int(n_components), vt.shape[0])
    return mean, vt[:k].T, evals, xc @ vt[:k].T


def _project(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)[None, :]) @ np.asarray(basis, dtype=np.float64)


def _project_vector(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(basis, dtype=np.float64)


def _capture(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)[None, :]
    den = float(np.sum(arr * arr))
    if den <= 0 or basis.size == 0:
        return float("nan")
    coeff = arr @ basis
    return float(np.sum(coeff * coeff) / den)


def _vector_capture(x: np.ndarray, basis: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    den = float(np.sum(arr * arr))
    if den <= 0 or basis.size == 0:
        return float("nan")
    coeff = arr @ basis
    return float(np.sum(coeff * coeff) / den)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    centered = y - np.mean(y, axis=0, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _safe_stat(vals: np.ndarray, fn: str) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if fn == "median":
        return float(np.median(arr))
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "q25":
        return float(np.percentile(arr, 25))
    if fn == "q75":
        return float(np.percentile(arr, 75))
    raise ValueError(fn)


@dataclass(frozen=True)
class TangentCache:
    r0: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    object_ids: np.ndarray
    source_rows: np.ndarray
    step_deg: float
    response_reduction: str


@dataclass(frozen=True)
class BasisPack:
    subset: str
    dim: int
    tangent_mean: np.ndarray
    tangent_basis: np.ndarray
    tangent_evals: np.ndarray
    r0_mean: np.ndarray
    train_mask: np.ndarray
    subset_mask: np.ndarray


@dataclass(frozen=True)
class GeneratorPack:
    mode: str
    direction_name: str
    direction: tuple[float, float]
    gx: np.ndarray
    gy: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    r2_x_train: float
    r2_y_train: float
    r2_x_test: float
    r2_y_test: float


def _load_npz_cache(path: Path) -> TangentCache:
    z = np.load(path, allow_pickle=True)
    for key in ("r0", "bx", "by"):
        if key not in z.files:
            raise ValueError(f"{path} is missing required array {key!r}")
    r0 = np.asarray(z["r0"], dtype=np.float64)
    bx = np.asarray(z["bx"], dtype=np.float64)
    by = np.asarray(z["by"], dtype=np.float64)
    if r0.shape != bx.shape or r0.shape != by.shape or r0.ndim != 2:
        raise ValueError(f"Expected r0/bx/by arrays with the same 2D shape, got {r0.shape}, {bx.shape}, {by.shape}")
    object_ids = np.asarray(z["object_ids"] if "object_ids" in z.files else [f"obj_{i}" for i in range(r0.shape[0])])
    source_rows = np.asarray(z["source_rows"] if "source_rows" in z.files else np.arange(r0.shape[0]), dtype=np.int64)
    step_deg = float(np.ravel(z["step_deg"])[0]) if "step_deg" in z.files else float("nan")
    response_reduction = str(np.ravel(z["response_reduction"])[0]) if "response_reduction" in z.files else "unknown"
    finite = np.isfinite(r0).all(axis=1) & np.isfinite(bx).all(axis=1) & np.isfinite(by).all(axis=1)
    finite &= (np.linalg.norm(bx, axis=1) > 1e-12) | (np.linalg.norm(by, axis=1) > 1e-12)
    return TangentCache(
        r0=r0[finite],
        bx=bx[finite],
        by=by[finite],
        object_ids=object_ids[finite],
        source_rows=source_rows[finite],
        step_deg=step_deg,
        response_reduction=response_reduction,
    )


def _metadata_from_csv(path: Path, source_rows: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source_row" not in df.columns:
        df = df.copy()
        df["source_row"] = np.arange(df.shape[0], dtype=int)
    lookup = df.drop_duplicates("source_row").set_index("source_row", drop=False)
    rows = []
    for src in np.asarray(source_rows, dtype=np.int64):
        if int(src) in lookup.index:
            rows.append(lookup.loc[int(src)].to_dict())
        else:
            rows.append({"source_row": int(src)})
    return pd.DataFrame(rows)


def _dominant_sf(row: pd.Series) -> str:
    bands = [
        ("0-2cpd", "image_power_0_2_cpd_fraction"),
        ("2-4cpd", "image_power_2_4_cpd_fraction"),
        ("4-8cpd", "image_power_4_8_cpd_fraction"),
        ("8plus_cpd", "image_power_8plus_cpd_fraction"),
    ]
    vals = []
    for name, col in bands:
        vals.append((name, float(pd.to_numeric(row.get(col), errors="coerce")) if col in row else np.nan))
    finite = [(name, val) for name, val in vals if np.isfinite(val)]
    if not finite:
        return "unknown"
    return max(finite, key=lambda kv: kv[1])[0]


def _add_metadata_columns(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.copy()
    if "image_orientation_coherence" in df.columns:
        df["structure_score"] = pd.to_numeric(df["image_orientation_coherence"], errors="coerce")
    elif "anisotropy" in df.columns:
        df["structure_score"] = pd.to_numeric(df["anisotropy"], errors="coerce")
    else:
        df["structure_score"] = np.nan
    df["dominant_sf_band"] = [_dominant_sf(row) for _, row in df.iterrows()]
    axis_col = "image_dominant_orientation_deg" if "image_dominant_orientation_deg" in df.columns else "image_edge_axis_deg"
    if axis_col in df.columns:
        angle = pd.to_numeric(df[axis_col], errors="coerce").to_numpy(dtype=np.float64)
        # Four broad unsigned orientation bins.
        wrapped = np.mod(angle, 180.0)
        bins = np.floor((wrapped + 22.5) / 45.0).astype(float)
        labels = []
        for val in bins:
            if not np.isfinite(val):
                labels.append("unknown")
            else:
                labels.append(["ori_0", "ori_45", "ori_90", "ori_135"][int(val) % 4])
        df["orientation_bin"] = labels
    else:
        df["orientation_bin"] = "unknown"
    return df


def _subset_masks(meta: pd.DataFrame, *, min_subset: int) -> dict[str, np.ndarray]:
    n = int(meta.shape[0])
    masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}
    if "structure_score" in meta.columns:
        vals = pd.to_numeric(meta["structure_score"], errors="coerce").to_numpy(dtype=np.float64)
        if np.sum(np.isfinite(vals)) >= int(min_subset):
            q = float(np.nanpercentile(vals, 67))
            masks["high_structure"] = np.isfinite(vals) & (vals >= q)
            masks["high_orientation_coherence"] = masks["high_structure"].copy()
    if "dominant_sf_band" in meta.columns:
        for band in ("4-8cpd", "8plus_cpd"):
            masks[f"dominant_{band}"] = meta["dominant_sf_band"].to_numpy() == band
    if "orientation_bin" in meta.columns:
        for label in sorted(set(str(v) for v in meta["orientation_bin"].to_numpy() if str(v) != "unknown")):
            masks[f"orientation_{label}"] = meta["orientation_bin"].to_numpy() == label
    return {name: mask for name, mask in masks.items() if int(np.sum(mask)) >= int(min_subset)}


def _train_test_mask(mask: np.ndarray, source_rows: np.ndarray, test_fraction: float, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(int(seed))
    sources = np.unique(np.asarray(source_rows, dtype=np.int64)[idx])
    shuffled_sources = sources.copy()
    rng.shuffle(shuffled_sources)
    n_test = max(1, int(round(float(test_fraction) * len(sources)))) if len(sources) >= 4 else 0
    train = mask.copy()
    if n_test:
        test_sources = set(int(v) for v in shuffled_sources[:n_test])
        train &= ~np.asarray([int(src) in test_sources for src in source_rows], dtype=bool)
    if int(np.sum(train)) < 2:
        train = mask.copy()
    return train


def _fit_basis(cache: TangentCache, subset: str, subset_mask: np.ndarray, dim: int, *, test_fraction: float, seed: int) -> BasisPack:
    train_mask = _train_test_mask(subset_mask, cache.source_rows, test_fraction, seed)
    train_tangents = np.concatenate([cache.bx[train_mask], cache.by[train_mask]], axis=0)
    tangent_mean, basis, evals, _ = _fit_pca(train_tangents, int(dim), center=True)
    r0_mean = np.mean(cache.r0[train_mask], axis=0)
    return BasisPack(
        subset=subset,
        dim=int(dim),
        tangent_mean=tangent_mean,
        tangent_basis=basis,
        tangent_evals=evals,
        r0_mean=r0_mean,
        train_mask=train_mask,
        subset_mask=subset_mask,
    )


def _basis_energy_rows(cache: TangentCache, pack: BasisPack) -> list[dict[str, object]]:
    rows = []
    test_mask = pack.subset_mask & ~pack.train_mask
    split_masks = {"train": pack.train_mask, "test": test_mask, "all_subset": pack.subset_mask}
    total = float(np.sum(pack.tangent_evals)) + 1e-12
    for split, mask in split_masks.items():
        if int(np.sum(mask)) < 1:
            continue
        tang = np.concatenate([cache.bx[mask], cache.by[mask]], axis=0)
        static = cache.r0[mask]
        raw_tangent_energy = float(np.sum(cache.bx[mask] * cache.bx[mask]) + np.sum(cache.by[mask] * cache.by[mask]))
        mean_energy = float(tang.shape[0] * np.sum(pack.tangent_mean * pack.tangent_mean))
        rows.append(
            {
                "subset": pack.subset,
                "dim": int(pack.dim),
                "split": split,
                "n_images": int(np.sum(mask)),
                "n_tangent_vectors": int(tang.shape[0]),
                "tangent_centered_variance_capture": _capture(tang, pack.tangent_mean, pack.tangent_basis),
                "tangent_vector_energy_capture": _vector_capture(tang, pack.tangent_basis),
                "static_response_centered_capture_in_tangent_basis": _capture(static, pack.r0_mean, pack.tangent_basis),
                "static_response_vector_capture_in_tangent_basis": _vector_capture(static, pack.tangent_basis),
                "tangent_mean_norm_fraction": float(mean_energy / (raw_tangent_energy + 1e-12)),
                "train_eigen_capture": float(np.sum(pack.tangent_evals[: pack.dim]) / total),
                "train_participation_ratio": float((np.sum(pack.tangent_evals) ** 2) / (np.sum(pack.tangent_evals**2) + 1e-12)),
            }
        )
    return rows


def _project_pack(cache: TangentCache, pack: BasisPack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx = _project_vector(cache.bx, pack.tangent_basis)
    cy = _project_vector(cache.by, pack.tangent_basis)
    a = _project(cache.r0, pack.r0_mean, pack.tangent_basis)
    return cx, cy, a


def _fit_unconstrained(a: np.ndarray, y: np.ndarray, train_mask: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a[train_mask], dtype=np.float64)
    yy = np.asarray(y[train_mask], dtype=np.float64)
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    coeff = np.linalg.solve(design.T @ design + reg, design.T @ yy)
    return coeff[:-1].T, coeff[-1]


def _fit_skew_constrained(a: np.ndarray, y: np.ndarray, train_mask: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a[train_mask], dtype=np.float64)
    yy = np.asarray(y[train_mask], dtype=np.float64)
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(yy, axis=0, keepdims=True)
    xc = x - x_mean
    yc = yy - y_mean
    n, k = xc.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if not pairs:
        return np.zeros((k, k), dtype=np.float64), np.ravel(y_mean)
    design = np.zeros((n * k, len(pairs)), dtype=np.float64)
    target = yc.reshape(-1)
    for col, (p, q) in enumerate(pairs):
        design[np.arange(n) * k + p, col] = xc[:, q]
        design[np.arange(n) * k + q, col] = -xc[:, p]
    reg = np.eye(len(pairs), dtype=np.float64) * float(ridge)
    weights = np.linalg.solve(design.T @ design + reg, design.T @ target)
    g = np.zeros((k, k), dtype=np.float64)
    for w, (p, q) in zip(weights, pairs, strict=False):
        g[p, q] = float(w)
        g[q, p] = -float(w)
    intercept = np.ravel(y_mean - x_mean @ g.T)
    return g, intercept


def _truncate_skew_rank(g: np.ndarray, rank: int) -> np.ndarray:
    skew = 0.5 * (np.asarray(g, dtype=np.float64) - np.asarray(g, dtype=np.float64).T)
    u, s, vt = np.linalg.svd(skew, full_matrices=False)
    r = min(int(rank), len(s))
    r -= r % 2
    if r <= 0:
        return np.zeros_like(skew)
    out = (u[:, :r] * s[:r]) @ vt[:r, :]
    return 0.5 * (out - out.T)


def _predict(a: np.ndarray, g: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) @ np.asarray(g, dtype=np.float64).T + np.asarray(intercept, dtype=np.float64)[None, :]


def _generator_fits(
    cx: np.ndarray,
    cy: np.ndarray,
    a: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    ridge: float,
) -> list[GeneratorPack]:
    fits: list[GeneratorPack] = []
    gx, ix = _fit_unconstrained(a, cx, train_mask, ridge)
    gy, iy = _fit_unconstrained(a, cy, train_mask, ridge)
    fits.append(_make_generator_pack("unconstrained_ridge", gx, gy, ix, iy, a, cx, cy, train_mask, eval_mask))
    gx_s, ix_s = _fit_skew_constrained(a, cx, train_mask, ridge)
    gy_s, iy_s = _fit_skew_constrained(a, cy, train_mask, ridge)
    fits.append(_make_generator_pack("skew_constrained_ridge", gx_s, gy_s, ix_s, iy_s, a, cx, cy, train_mask, eval_mask))
    gx_l = _truncate_skew_rank(gx_s, rank=min(4, gx_s.shape[0]))
    gy_l = _truncate_skew_rank(gy_s, rank=min(4, gy_s.shape[0]))
    ix_l = np.mean(cx[train_mask] - a[train_mask] @ gx_l.T, axis=0)
    iy_l = np.mean(cy[train_mask] - a[train_mask] @ gy_l.T, axis=0)
    fits.append(_make_generator_pack("low_rank_skew_projected", gx_l, gy_l, ix_l, iy_l, a, cx, cy, train_mask, eval_mask))
    return fits


def _make_generator_pack(
    mode: str,
    gx: np.ndarray,
    gy: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    a: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> GeneratorPack:
    test_mask = np.asarray(eval_mask, dtype=bool) & ~np.asarray(train_mask, dtype=bool)
    px = _predict(a, gx, ix)
    py = _predict(a, gy, iy)
    return GeneratorPack(
        mode=mode,
        direction_name="x/y",
        direction=(float("nan"), float("nan")),
        gx=gx,
        gy=gy,
        ix=ix,
        iy=iy,
        r2_x_train=_r2(cx[train_mask], px[train_mask]),
        r2_y_train=_r2(cy[train_mask], py[train_mask]),
        r2_x_test=_r2(cx[test_mask], px[test_mask]) if np.sum(test_mask) >= 2 else float("nan"),
        r2_y_test=_r2(cy[test_mask], py[test_mask]) if np.sum(test_mask) >= 2 else float("nan"),
    )


def _cosine_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _unit_rows(y_true)
    yp = _unit_rows(y_pred)
    vals = np.sum(yt * yp, axis=1)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _generator_metric_rows(
    subset: str,
    dim: int,
    fit: GeneratorPack,
    a: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    directions: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    gx = fit.gx
    gy = fit.gy
    comm = gx @ gy - gy @ gx
    den = float(np.linalg.norm(gx) * np.linalg.norm(gy) + 1e-12)
    test_mask = np.asarray(eval_mask, dtype=bool) & ~np.asarray(train_mask, dtype=bool)
    base = {
        "subset": subset,
        "dim": int(dim),
        "fit_mode": fit.mode,
        "r2_x_train": fit.r2_x_train,
        "r2_y_train": fit.r2_y_train,
        "r2_x_test": fit.r2_x_test,
        "r2_y_test": fit.r2_y_test,
        "skew_fraction_gx": _skew_fraction(gx),
        "skew_fraction_gy": _skew_fraction(gy),
        "commutator_relative_norm": float(np.linalg.norm(comm) / den),
        "norm_gx": float(np.linalg.norm(gx)),
        "norm_gy": float(np.linalg.norm(gy)),
    }
    rows = []
    for direction_name, direction in directions.items():
        vx, vy = float(direction[0]), float(direction[1])
        gv = vx * gx + vy * gy
        iv = vx * fit.ix + vy * fit.iy
        cv = vx * cx + vy * cy
        pred = _predict(a, gv, iv)
        row = dict(base)
        row.update(
            {
                "direction_name": direction_name,
                "vx": vx,
                "vy": vy,
                "r2_v_train": _r2(cv[train_mask], pred[train_mask]),
                "r2_v_test": _r2(cv[test_mask], pred[test_mask]) if np.sum(test_mask) >= 2 else float("nan"),
                "cosine_v_train_mean": _cosine_mean(cv[train_mask], pred[train_mask]),
                "cosine_v_test_mean": _cosine_mean(cv[test_mask], pred[test_mask]) if np.sum(test_mask) >= 2 else float("nan"),
                "skew_fraction_gv": _skew_fraction(gv),
                "norm_gv": float(np.linalg.norm(gv)),
            }
        )
        rows.append(row)
    return rows


def _skew_fraction(g: np.ndarray) -> float:
    mat = np.asarray(g, dtype=np.float64)
    total = float(np.sum(mat * mat))
    if total <= 0:
        return float("nan")
    skew = 0.5 * (mat - mat.T)
    return float(np.sum(skew * skew) / total)


def _eigenspectrum_rows(subset: str, dim: int, fit: GeneratorPack, directions: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mats = {"Gx": fit.gx, "Gy": fit.gy}
    for name, v in directions.items():
        mats[f"Gv_{name}"] = float(v[0]) * fit.gx + float(v[1]) * fit.gy
    for mat_name, mat in mats.items():
        vals = np.linalg.eigvals(mat)
        order = np.argsort(np.abs(vals))[::-1]
        for rank, idx in enumerate(order):
            ev = vals[idx]
            rows.append(
                {
                    "subset": subset,
                    "dim": int(dim),
                    "fit_mode": fit.mode,
                    "matrix": mat_name,
                    "eigen_rank": int(rank),
                    "real": float(np.real(ev)),
                    "imag": float(np.imag(ev)),
                    "abs": float(abs(ev)),
                    "frequency_like_abs_imag": float(abs(np.imag(ev))),
                }
            )
    return rows


def _leading_complex_plane(g: np.ndarray) -> tuple[np.ndarray, str]:
    vals, vecs = np.linalg.eig(np.asarray(g, dtype=np.float64))
    candidates = np.flatnonzero(np.abs(np.imag(vals)) > 1e-8)
    if candidates.size:
        idx = int(candidates[np.argmax(np.abs(np.imag(vals[candidates])))])
        q = np.stack([np.real(vecs[:, idx]), np.imag(vecs[:, idx])], axis=1)
        q, _ = np.linalg.qr(q)
        return q[:, :2], "complex_eigenmode"
    skew = 0.5 * (g - g.T)
    vals_s, vecs_s = np.linalg.eig(skew)
    idx = int(np.argmax(np.abs(np.imag(vals_s))))
    q = np.stack([np.real(vecs_s[:, idx]), np.imag(vecs_s[:, idx])], axis=1)
    q, _ = np.linalg.qr(q)
    return q[:, :2], "skew_eigenmode"


def _third_axis_from_residual(a: np.ndarray, plane: np.ndarray) -> np.ndarray:
    proj = a @ plane @ plane.T
    resid = a - proj
    _, basis, _, _ = _fit_pca(resid, 1, center=True)
    return basis[:, 0]


def _color_values(meta: pd.DataFrame, color_by: str) -> tuple[np.ndarray, str, str]:
    if color_by == "dominant_sf_band" and "dominant_sf_band" in meta.columns:
        labels = sorted(set(str(v) for v in meta["dominant_sf_band"]))
        lookup = {label: i for i, label in enumerate(labels)}
        vals = np.asarray([lookup[str(v)] for v in meta["dominant_sf_band"]], dtype=float)
        return vals, "tab10", "dominant SF"
    if color_by == "orientation_bin" and "orientation_bin" in meta.columns:
        labels = sorted(set(str(v) for v in meta["orientation_bin"]))
        lookup = {label: i for i, label in enumerate(labels)}
        vals = np.asarray([lookup[str(v)] for v in meta["orientation_bin"]], dtype=float)
        return vals, "tab10", "orientation bin"
    if color_by in meta.columns:
        vals = pd.to_numeric(meta[color_by], errors="coerce").to_numpy(dtype=np.float64)
        return vals, "viridis", color_by
    return np.arange(meta.shape[0], dtype=float), "viridis", "image index"


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        return
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    half = 0.55 * span
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_zlim(float(center[2] - half), float(center[2] + half))


def _plot_tangent_vectors_3d(
    path: Path,
    cx: np.ndarray,
    cy: np.ndarray,
    meta: pd.DataFrame,
    mask: np.ndarray,
    *,
    direction: tuple[float, float],
    color_by: str,
    title: str,
) -> None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    cv = direction[0] * cx + direction[1] * cy
    all_points = np.concatenate([cx[idx, :3], cy[idx, :3], cv[idx, :3]], axis=0)
    vals, cmap, label = _color_values(meta.iloc[idx].reset_index(drop=True), color_by)
    fig = plt.figure(figsize=(13.0, 9.5), constrained_layout=True)
    views = [(20, -55), (20, 35), (75, -90), (5, -5)]
    for i, view in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(cx[idx, 0], cx[idx, 1], cx[idx, 2], s=12, color=BX_COLOR, alpha=0.48, label="cx")
        ax.scatter(cy[idx, 0], cy[idx, 1], cy[idx, 2], s=12, color=BY_COLOR, alpha=0.48, label="cy")
        sc = ax.scatter(cv[idx, 0], cv[idx, 1], cv[idx, 2], s=18, c=vals, cmap=cmap, alpha=0.86, label="cv")
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("tPC1")
        ax.set_ylabel("tPC2")
        ax.set_zlabel("tPC3")
        _set_equal_3d(ax, all_points)
        if i == 1:
            ax.legend(frameon=False, loc="upper left")
            fig.colorbar(sc, ax=ax, shrink=0.62, label=label)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_ellipses_3d(
    path: Path,
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    *,
    max_ellipses: int,
    seed: int,
    title: str,
) -> None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    rng = np.random.default_rng(int(seed))
    if idx.size > int(max_ellipses):
        norms = np.linalg.norm(cx[idx, :3], axis=1) + np.linalg.norm(cy[idx, :3], axis=1)
        # Prefer large, visible ellipses but keep deterministic diversity.
        top = idx[np.argsort(norms)[-min(idx.size, int(max_ellipses) * 3) :]]
        idx = np.sort(rng.choice(top, size=int(max_ellipses), replace=False))
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    ellipse_points = [
        np.outer(np.cos(theta), cx[j, :3]) + np.outer(np.sin(theta), cy[j, :3])
        for j in idx
    ]
    all_points = np.concatenate(ellipse_points, axis=0)
    fig = plt.figure(figsize=(13.0, 9.5), constrained_layout=True)
    views = [(20, -55), (20, 35), (75, -90), (5, -5)]
    for i, view in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        for pts in ellipse_points:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#255C99", alpha=0.45, lw=0.9)
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("tPC1")
        ax.set_ylabel("tPC2")
        ax.set_zlabel("tPC3")
        _set_equal_3d(ax, all_points)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_generator_flow_3d(
    path: Path,
    a: np.ndarray,
    cv: np.ndarray,
    pred: np.ndarray,
    g: np.ndarray,
    mask: np.ndarray,
    *,
    max_arrows: int,
    seed: int,
    title: str,
) -> None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    plane, source = _leading_complex_plane(g)
    third = _third_axis_from_residual(a[mask], plane)
    axes = np.column_stack([plane[:, 0], plane[:, 1], third])
    xyz = a @ axes
    obs = cv @ axes
    prd = pred @ axes
    if idx.size > int(max_arrows):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_arrows), replace=False))
    spread = float(np.nanstd(xyz[idx])) * 0.16
    obs_u = _unit_rows(obs[idx])
    prd_u = _unit_rows(prd[idx])
    all_points = np.concatenate([xyz[idx], xyz[idx] + obs_u * spread, xyz[idx] + prd_u * spread], axis=0)
    fig = plt.figure(figsize=(13.0, 9.5), constrained_layout=True)
    views = [(20, -55), (20, 35), (75, -90), (5, -5)]
    for i, view in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], s=11, color="0.45", alpha=0.5)
        ax.quiver(
            xyz[idx, 0],
            xyz[idx, 1],
            xyz[idx, 2],
            obs_u[:, 0] * spread,
            obs_u[:, 1] * spread,
            obs_u[:, 2] * spread,
            color=BX_COLOR,
            alpha=0.50,
            linewidth=0.7,
            label="observed",
        )
        ax.quiver(
            xyz[idx, 0],
            xyz[idx, 1],
            xyz[idx, 2],
            prd_u[:, 0] * spread,
            prd_u[:, 1] * spread,
            prd_u[:, 2] * spread,
            color=PRED_COLOR,
            alpha=0.50,
            linewidth=0.7,
            label="predicted",
        )
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("eigenplane real")
        ax.set_ylabel("eigenplane imag")
        ax.set_zlabel("content PC")
        _set_equal_3d(ax, all_points)
        if i == 1:
            ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"{title}; plane={source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _subset_summary_rows(cache: TangentCache, meta: pd.DataFrame, subsets: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, mask in subsets.items():
        bx = cache.bx[mask]
        by = cache.by[mask]
        cos = np.sum(bx * by, axis=1) / (np.linalg.norm(bx, axis=1) * np.linalg.norm(by, axis=1) + 1e-12)
        rows.append(
            {
                "subset": name,
                "n_images": int(np.sum(mask)),
                "norm_bx_median": _safe_stat(np.linalg.norm(bx, axis=1), "median"),
                "norm_by_median": _safe_stat(np.linalg.norm(by, axis=1), "median"),
                "angle_bx_by_deg_median": _safe_stat(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), "median"),
                "structure_score_median": _safe_stat(pd.to_numeric(meta.loc[mask, "structure_score"], errors="coerce"), "median")
                if "structure_score" in meta.columns
                else float("nan"),
            }
        )
    return rows


def _readme(out_dir: Path) -> None:
    text = """# Compact Tangent Rotation Pilot

This directory contains tangent-space plots and metrics, not raw response
trajectory plots. The input is a finite-difference cache with `r0`, `bx`, and
`by`, where:

```text
bx(I) = [r(I + eps x) - r(I - eps x)] / (2 eps)
by(I) = [r(I + eps y) - r(I - eps y)] / (2 eps)
```

The compact basis is fit only on train-image tangents. PCA centering is used to
learn the basis axes, but derivative vectors are then projected without
subtracting the tangent mean:

```text
cx(I) = bx(I) U
cy(I) = by(I) U
```

This matters because `bx` and `by` are vectors, not response states. Static
coordinates `a(I)` are still centered response coordinates,
`(r0(I) - mean_train_r0) U`. Generator fits ask whether
`cx(I) ~= Gx a(I)` and `cy(I) ~= Gy a(I)` inside the compact tangent
coordinates.

Figure interpretation:

- `tangent_vector_3d.png`: tangent vectors in compact tangent PC space.
- `tangent_ellipses_3d.png`: each curve is the local unit-displacement ellipse
  `cx cos(theta) + cy sin(theta)` for one image.
- `generator_eigenplane_flow_3d.png`: static points projected onto the leading
  complex/skew plane of a selected generator direction; arrows compare observed
  and predicted tangent flow.

Use this as a phase-plane/rotation diagnostic. It does not claim that the full
neural response trajectory is a torus or that these are behavioral variables.
Report both centered tangent variance capture and uncentered tangent vector
energy capture; use the latter when quoting compact tangent subspace capture.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _synthetic_control(name: str, n: int, units: int, seed: int) -> TangentCache:
    rng = np.random.default_rng(int(seed))
    if name == "single_grating":
        z = rng.uniform(0, 2 * np.pi, size=n)
        r0_lat = np.column_stack([np.cos(z), np.sin(z)])
        bx_lat = np.column_stack([-np.sin(z), np.cos(z)])
        by_lat = np.zeros_like(bx_lat)
    elif name == "two_gratings":
        z1 = rng.uniform(0, 2 * np.pi, size=n)
        z2 = rng.uniform(0, 2 * np.pi, size=n)
        r0_lat = np.column_stack([np.cos(z1), np.sin(z1), np.cos(z2), np.sin(z2)])
        bx_lat = np.column_stack([-np.sin(z1), np.cos(z1), -0.35 * np.sin(z2), 0.35 * np.cos(z2)])
        by_lat = np.column_stack([0.15 * -np.sin(z1), 0.15 * np.cos(z1), -np.sin(z2), np.cos(z2)])
    elif name == "mixture":
        phases = rng.uniform(0, 2 * np.pi, size=(n, 5))
        amps = rng.uniform(0.2, 1.0, size=(n, 5))
        r0_lat = np.concatenate([amps * np.cos(phases), amps * np.sin(phases)], axis=1)
        bx_lat = np.concatenate([-amps * np.sin(phases), amps * np.cos(phases)], axis=1)
        by_lat = rng.normal(scale=0.35, size=bx_lat.shape)
    else:
        raise ValueError(name)
    q, _ = np.linalg.qr(rng.normal(size=(units, r0_lat.shape[1])))
    r0 = r0_lat @ q.T + 0.02 * rng.normal(size=(n, units))
    bx = bx_lat @ q.T + 0.02 * rng.normal(size=(n, units))
    by = by_lat @ q.T + 0.02 * rng.normal(size=(n, units))
    return TangentCache(
        r0=r0,
        bx=bx,
        by=by,
        object_ids=np.asarray([f"{name}_{i}" for i in range(n)]),
        source_rows=np.arange(n, dtype=np.int64),
        step_deg=float("nan"),
        response_reduction=f"synthetic_{name}",
    )


def _analyze_cache(
    cache: TangentCache,
    meta: pd.DataFrame,
    out_dir: Path,
    *,
    dims: list[int],
    directions: dict[str, tuple[float, float]],
    min_subset: int,
    test_fraction: float,
    ridge: float,
    seed: int,
    color_by: str,
    max_ellipses: int,
    max_arrows: int,
    plot_dim: int,
    plot_first_n: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _add_metadata_columns(meta)
    subsets = _subset_masks(meta, min_subset=min_subset)
    energy_rows: list[dict[str, object]] = []
    generator_rows: list[dict[str, object]] = []
    eigen_rows: list[dict[str, object]] = []
    subset_rows = _subset_summary_rows(cache, meta, subsets)
    saved_npz: dict[str, np.ndarray] = {}
    representative_plotted = False
    restricted_plotted: set[str] = set()

    for subset_name, subset_mask in subsets.items():
        for dim in dims:
            if int(np.sum(subset_mask)) < max(4, dim + 2):
                continue
            pack = _fit_basis(cache, subset_name, subset_mask, dim, test_fraction=test_fraction, seed=seed + 17 * dim + len(subset_name))
            cx, cy, a = _project_pack(cache, pack)
            energy_rows.extend(_basis_energy_rows(cache, pack))
            saved_npz[f"{subset_name}_k{dim}_basis"] = pack.tangent_basis.astype(np.float32)
            saved_npz[f"{subset_name}_k{dim}_tangent_mean"] = pack.tangent_mean.astype(np.float32)
            saved_npz[f"{subset_name}_k{dim}_r0_mean"] = pack.r0_mean.astype(np.float32)
            fits = _generator_fits(cx, cy, a, pack.train_mask, pack.subset_mask, ridge=ridge)
            for fit in fits:
                generator_rows.extend(
                    _generator_metric_rows(subset_name, dim, fit, a, cx, cy, pack.train_mask, pack.subset_mask, directions)
                )
                eigen_rows.extend(_eigenspectrum_rows(subset_name, dim, fit, directions))
            if subset_name == "all" and dim == int(plot_dim):
                saved_npz["all_dim_selected_cx"] = cx.astype(np.float32)
                saved_npz["all_dim_selected_cy"] = cy.astype(np.float32)
                saved_npz["all_dim_selected_a"] = a.astype(np.float32)

            if (not representative_plotted) and subset_name == "all" and dim == int(plot_dim) and dim >= 3:
                direction = directions[next(iter(directions))]
                plot_cols = min(3, cx.shape[1])
                _plot_tangent_vectors_3d(
                    out_dir / "tangent_vector_3d.png",
                    cx[:, :plot_cols],
                    cy[:, :plot_cols],
                    meta,
                    subset_mask,
                    direction=direction,
                    color_by=color_by,
                    title=f"Compact tangent vectors, subset={subset_name}, k={dim}, first {plot_cols}",
                )
                _plot_ellipses_3d(
                    out_dir / "tangent_ellipses_3d.png",
                    cx[:, :plot_cols],
                    cy[:, :plot_cols],
                    subset_mask,
                    max_ellipses=max_ellipses,
                    seed=seed + 5,
                    title=f"Image-specific tangent ellipses, subset={subset_name}, k={dim}, first {plot_cols}",
                )
                fit = fits[0]
                g = direction[0] * fit.gx + direction[1] * fit.gy
                cv = direction[0] * cx + direction[1] * cy
                pred = _predict(a, g, direction[0] * fit.ix + direction[1] * fit.iy)
                _plot_generator_flow_3d(
                    out_dir / "generator_eigenplane_flow_3d.png",
                    a,
                    cv,
                    pred,
                    g,
                    subset_mask,
                    max_arrows=max_arrows,
                    seed=seed + 6,
                    title=f"Generator eigenplane flow, subset={subset_name}, k={dim}",
                )
                representative_plotted = True
            if dim == int(plot_dim) and dim >= 3 and subset_name not in restricted_plotted:
                direction = directions[next(iter(directions))]
                restricted_dir = out_dir / "restricted_plots"
                plot_cols = min(3, cx.shape[1])
                _plot_tangent_vectors_3d(
                    restricted_dir / f"{subset_name}_tangent_vector_3d.png",
                    cx[:, :plot_cols],
                    cy[:, :plot_cols],
                    meta,
                    subset_mask,
                    direction=direction,
                    color_by=color_by,
                    title=f"Compact tangent vectors, subset={subset_name}, k={dim}, first {plot_cols}",
                )
                _plot_ellipses_3d(
                    restricted_dir / f"{subset_name}_tangent_ellipses_3d.png",
                    cx[:, :plot_cols],
                    cy[:, :plot_cols],
                    subset_mask,
                    max_ellipses=max_ellipses,
                    seed=seed + 500 + len(restricted_plotted),
                    title=f"Image-specific tangent ellipses, subset={subset_name}, k={dim}, first {plot_cols}",
                )
                fit = fits[0]
                g = direction[0] * fit.gx + direction[1] * fit.gy
                cv = direction[0] * cx + direction[1] * cy
                pred = _predict(a, g, direction[0] * fit.ix + direction[1] * fit.iy)
                _plot_generator_flow_3d(
                    restricted_dir / f"{subset_name}_generator_eigenplane_flow_3d.png",
                    a,
                    cv,
                    pred,
                    g,
                    subset_mask,
                    max_arrows=max_arrows,
                    seed=seed + 700 + len(restricted_plotted),
                    title=f"Generator eigenplane flow, subset={subset_name}, k={dim}",
                )
                restricted_plotted.add(subset_name)

    _write_csv(out_dir / "tangent_energy_capture.csv", energy_rows)
    _write_csv(out_dir / "generator_fit_metrics.csv", generator_rows)
    _write_csv(out_dir / "generator_eigenspectrum.csv", eigen_rows)
    _write_csv(out_dir / "restricted_subset_summary.csv", subset_rows)
    if saved_npz:
        np.savez_compressed(out_dir / "compact_tangent_basis_and_scores.npz", **saved_npz)
    _readme(out_dir)
    summary = {
        "n_images": int(cache.r0.shape[0]),
        "n_unique_source_rows": int(np.unique(cache.source_rows).size),
        "source_rows_are_unique": bool(np.unique(cache.source_rows).size == cache.source_rows.size),
        "n_units": int(cache.r0.shape[1]),
        "response_reduction": cache.response_reduction,
        "step_deg": cache.step_deg,
        "dims": dims,
        "directions": directions,
        "plot_dim": int(plot_dim),
        "plot_first_n": int(plot_first_n),
        "subsets": {name: int(np.sum(mask)) for name, mask in subsets.items()},
        "outputs": [
            "tangent_energy_capture.csv",
            "generator_fit_metrics.csv",
            "generator_eigenspectrum.csv",
            "restricted_subset_summary.csv",
            "tangent_vector_3d.png",
            "tangent_ellipses_3d.png",
            "generator_eigenplane_flow_3d.png",
            "restricted_plots/",
            "compact_tangent_basis_and_scores.npz",
            "README.md",
        ],
    }
    _write_json(out_dir / "compact_tangent_rotation_pilot_summary.json", summary)
    return summary


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    cache = _load_npz_cache(Path(args.cache))
    meta = _metadata_from_csv(Path(args.metadata_csv), cache.source_rows)
    dims = _parse_ints(args.dims)
    if int(args.plot_dim) not in dims:
        dims = sorted(set(dims + [int(args.plot_dim)]))
    directions = {"v": _parse_float_pair(args.direction)}
    out_dir = Path(args.out_dir)
    summary = _analyze_cache(
        cache,
        meta,
        out_dir,
        dims=dims,
        directions=directions,
        min_subset=int(args.min_subset),
        test_fraction=float(args.test_fraction),
        ridge=float(args.ridge),
        seed=int(args.seed),
        color_by=str(args.color_by),
        max_ellipses=int(args.max_ellipses),
        max_arrows=int(args.max_arrows),
        plot_dim=int(args.plot_dim),
        plot_first_n=int(args.plot_first_n),
    )
    if bool(args.synthetic_controls):
        syn_root = out_dir / "positive_controls"
        control_summaries = {}
        for i, name in enumerate(("single_grating", "two_gratings", "mixture")):
            syn = _synthetic_control(name, n=max(64, int(args.min_subset) * 3), units=cache.r0.shape[1], seed=int(args.seed) + 100 + i)
            syn_meta = pd.DataFrame({"source_row": syn.source_rows, "structure_score": np.ones(syn.r0.shape[0])})
            control_summaries[name] = _analyze_cache(
                syn,
                syn_meta,
                syn_root / name,
                dims=dims,
                directions=directions,
                min_subset=int(args.min_subset),
                test_fraction=float(args.test_fraction),
                ridge=float(args.ridge),
                seed=int(args.seed) + 200 + i,
                color_by="source_row",
                max_ellipses=int(args.max_ellipses),
                max_arrows=int(args.max_arrows),
                plot_dim=int(args.plot_dim),
                plot_first_n=int(args.plot_first_n),
            )
        summary["positive_controls"] = control_summaries
        _write_json(out_dir / "compact_tangent_rotation_pilot_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finite-difference compact tangent rotation/phase-plane pilot.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
            "backimage_cardinal_tangent_charts_n128_step025_mean/backimage_cardinal_tangent_cache.npz"
        ),
        help="NPZ cache with r0, bx, by arrays from finite-difference twin responses.",
    )
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dims", type=str, default="3,6,9,12")
    parser.add_argument("--direction", type=str, default="1,0", help="vx,vy direction used for cv and Gv plots.")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-subset", type=int, default=12)
    parser.add_argument("--color-by", type=str, default="structure_score")
    parser.add_argument("--max-ellipses", type=int, default=36)
    parser.add_argument("--max-arrows", type=int, default=96)
    parser.add_argument("--plot-dim", type=int, default=9, help="Compact basis dimension to use for the main 3D plots.")
    parser.add_argument("--plot-first-n", type=int, default=3, help="Recorded for provenance; 3D plots display the leading 3 axes.")
    parser.add_argument("--synthetic-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

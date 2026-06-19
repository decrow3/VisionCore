from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _finite_vals(x: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _finite_median(x: list[float] | np.ndarray) -> float:
    vals = _finite_vals(x)
    return float(np.median(vals)) if vals.size else float("nan")


def _finite_mean(x: list[float] | np.ndarray) -> float:
    vals = _finite_vals(x)
    return float(np.mean(vals)) if vals.size else float("nan")


def _finite_ci(x: list[float] | np.ndarray, q: float) -> float:
    vals = _finite_vals(x)
    return float(np.percentile(vals, q)) if vals.size else float("nan")


def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def _participation_ratio(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0:
        return float("nan")
    den = float(np.sum(v * v))
    if den <= 0:
        return float("nan")
    num = float(np.sum(v))
    return float((num * num) / den)


def _rank_at(frac: np.ndarray, threshold: float) -> int:
    if frac.size == 0:
        return 0
    cum = np.cumsum(np.asarray(frac, dtype=np.float64))
    return int(np.searchsorted(cum, float(threshold), side="left")) + 1


def _rank_correlation(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 3:
        return float("nan")
    x = x[keep]
    y = y[keep]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))
    den = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.dot(rx, ry) / den) if den > 0 else float("nan")


def _load_tangent_payload(input_root: Path) -> tuple[list[float], dict[float, dict[str, dict[str, Any]]], Path, dict[str, Any]]:
    if input_root.is_file():
        pkl_path = input_root
    else:
        pkl_path = input_root / "tangent_maps" / "twin_tangent_maps.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find tangent map cache at {pkl_path}. "
            "Run run_twin_feature_tangent_structure.py first, or pass the pickle path directly."
        )
    with pkl_path.open("rb") as handle:
        cached = pickle.load(handle)
    delta_arcmins = [float(v) for v in cached["delta_arcmins"]]
    payload = {
        float(d): {str(oid): dict(meta) for oid, meta in block.items()}
        for d, block in cached["object_payload"].items()
    }
    metadata = dict(cached.get("metadata", {}))
    return delta_arcmins, payload, pkl_path, metadata


def _valid_object_ids(payload: dict[str, dict[str, Any]], eps: float = 1e-12) -> list[str]:
    valid: list[str] = []
    for oid, meta in payload.items():
        try:
            r0 = np.asarray(meta["r0"], dtype=np.float64)
            bx = np.asarray(meta["bx"], dtype=np.float64)
            by = np.asarray(meta["by"], dtype=np.float64)
        except Exception:
            continue
        same_shape = r0.shape == bx.shape == by.shape and r0.ndim == 1
        finite = np.isfinite(r0).all() and np.isfinite(bx).all() and np.isfinite(by).all()
        # A one-axis-zero tangent is still informative: it is a flat local
        # ellipse/line. Drop only objects with no translation response at all.
        nonzero = (_safe_norm(bx) > eps) or (_safe_norm(by) > eps)
        if same_shape and finite and nonzero:
            valid.append(str(oid))
    return sorted(valid)


@dataclass(frozen=True)
class BasisResult:
    basis: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    participation_ratio: float
    capture: float
    rank_50: int
    rank_75: int
    rank_90: int
    rank_95: int


def _basis_from_columns(mat: np.ndarray, k: int, *, convention: str = "uncentered") -> BasisResult:
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2 or min(m.shape) == 0:
        empty = np.zeros((m.shape[0] if m.ndim == 2 else 0, 0), dtype=np.float64)
        return BasisResult(empty, np.zeros(0), 0, float("nan"), float("nan"), 0, 0, 0, 0)
    keep_cols = np.all(np.isfinite(m), axis=0) & (np.linalg.norm(m, axis=0) > 1e-12)
    m = m[:, keep_cols]
    if m.size == 0:
        empty = np.zeros((mat.shape[0], 0), dtype=np.float64)
        return BasisResult(empty, np.zeros(0), 0, float("nan"), float("nan"), 0, 0, 0, 0)
    if convention == "centered_across_tangents_per_unit":
        m = m - np.mean(m, axis=1, keepdims=True)
    elif convention != "uncentered":
        raise ValueError(
            "basis_convention must be 'uncentered' or "
            "'centered_across_tangents_per_unit'"
        )
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    evals = np.maximum(s * s, 0.0)
    total = float(np.sum(evals))
    frac = evals / (total + 1e-12)
    rank = int(np.sum(evals > max(1e-12, 1e-8 * float(np.max(evals)))))
    kk = max(1, min(int(k), u.shape[1], rank if rank > 0 else u.shape[1]))
    basis = u[:, :kk]
    capture = float(np.sum(evals[:kk]) / (total + 1e-12)) if total > 0 else float("nan")
    return BasisResult(
        basis=basis,
        eigenvalues=evals,
        rank=rank,
        participation_ratio=_participation_ratio(evals),
        capture=capture,
        rank_50=_rank_at(frac, 0.50),
        rank_75=_rank_at(frac, 0.75),
        rank_90=_rank_at(frac, 0.90),
        rank_95=_rank_at(frac, 0.95),
    )


def _variance_capture(u: np.ndarray, vec: np.ndarray) -> float:
    v = np.asarray(vec, dtype=np.float64)
    den = float(np.sum(v * v))
    if den <= 0 or u.size == 0:
        return float("nan")
    coeff = u.T @ v
    return float(np.sum(coeff * coeff) / den)


def _ellipse_metrics(jx: np.ndarray, jy: np.ndarray, radius_px: float) -> dict[str, float]:
    a = np.stack([np.asarray(jx, dtype=np.float64), np.asarray(jy, dtype=np.float64)], axis=1)
    s = np.linalg.svd(a, compute_uv=False)
    if s.size < 2:
        s = np.pad(s, (0, 2 - s.size))
    major = float(s[0])
    minor = float(s[1])
    denom = major * major + minor * minor
    nx = _safe_norm(jx)
    ny = _safe_norm(jy)
    cos_xy = float(np.dot(jx, jy) / (nx * ny + 1e-12))
    return {
        "ellipse_major_sensitivity": major,
        "ellipse_minor_sensitivity": minor,
        "ellipse_aspect_minor_over_major": float(minor / (major + 1e-12)),
        "ellipse_circularity": float((2.0 * major * minor) / (denom + 1e-12)),
        "ellipse_area_at_ring_radius": float(np.pi * float(radius_px) * float(radius_px) * major * minor),
        "tangent_cos_xy": cos_xy,
        "tangent_abs_cos_xy": abs(cos_xy),
    }


def _pair_slices(k: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(0, int(k), 2):
        pairs.append((i, min(i + 2, int(k))))
    return pairs


def _pair_energy_metrics(jx: np.ndarray, jy: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(jx, dtype=np.float64)
    y = np.asarray(jy, dtype=np.float64)
    pair_energy = []
    for a, b in _pair_slices(x.size):
        pair_energy.append(float(np.sum(x[a:b] * x[a:b]) + np.sum(y[a:b] * y[a:b])))
    e = np.asarray(pair_energy, dtype=np.float64)
    total = float(np.sum(e))
    if total <= 0 or e.size == 0:
        return {
            "n_coordinate_pairs": int(e.size),
            "best_pair_index": -1,
            "best_pair_energy_fraction": float("nan"),
            "top2_pair_energy_fraction": float("nan"),
            "pair_energy_entropy_fraction": float("nan"),
        }
    frac = e / total
    best = int(np.argmax(frac))
    top2 = float(np.sum(np.sort(frac)[-min(2, frac.size) :]))
    entropy = -float(np.sum(frac * np.log(frac + 1e-12)))
    entropy_fraction = entropy / float(np.log(e.size)) if e.size > 1 else 0.0
    return {
        "n_coordinate_pairs": int(e.size),
        "best_pair_index": int(best),
        "best_pair_energy_fraction": float(frac[best]),
        "top2_pair_energy_fraction": top2,
        "pair_energy_entropy_fraction": float(entropy_fraction),
    }


def _frame_from_history(history: Any) -> np.ndarray | None:
    if history is None:
        return None
    h = np.asarray(history, dtype=np.float64)
    if h.size == 0:
        return None
    frame = h[0] if h.ndim >= 3 else h
    frame = np.squeeze(frame)
    if frame.ndim != 2:
        return None
    return frame


def _image_frequency_stats(history: Any) -> dict[str, float]:
    frame = _frame_from_history(history)
    if frame is None:
        return {
            "image_rms_contrast": float("nan"),
            "image_gradient_rms": float("nan"),
            "image_gradient_anisotropy": float("nan"),
            "image_frequency_centroid_cyc_per_px": float("nan"),
            "image_high_frequency_power_fraction": float("nan"),
        }
    f = np.asarray(frame, dtype=np.float64)
    f = f - float(np.mean(f))
    gx = np.diff(f, axis=1, prepend=f[:, :1])
    gy = np.diff(f, axis=0, prepend=f[:1, :])
    gxx = float(np.mean(gx * gx))
    gyy = float(np.mean(gy * gy))
    gxy = float(np.mean(gx * gy))
    anis = float(np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy * gxy) / (gxx + gyy + 1e-12))
    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(f.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(f.shape[1]))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        centroid = float("nan")
        high_frac = float("nan")
    else:
        centroid = float(np.sum(radius * power) / total_power)
        high_frac = float(np.sum(power[radius >= 0.25]) / total_power)
    return {
        "image_rms_contrast": float(np.std(frame)),
        "image_gradient_rms": float(np.sqrt(np.mean(gx * gx + gy * gy))),
        "image_gradient_anisotropy": anis,
        "image_frequency_centroid_cyc_per_px": centroid,
        "image_high_frequency_power_fraction": high_frac,
    }


@dataclass(frozen=True)
class GeneratorFit:
    generator: np.ndarray
    intercept: np.ndarray
    r2: float
    prediction_norm_fraction: float


def _fit_generator(z0: np.ndarray, dz: np.ndarray, ridge: float) -> GeneratorFit:
    z = np.asarray(z0, dtype=np.float64)
    y = np.asarray(dz, dtype=np.float64)
    if z.ndim != 2 or y.ndim != 2 or z.shape != y.shape or z.shape[0] < 2:
        k = int(z.shape[1]) if z.ndim == 2 else 0
        return GeneratorFit(np.zeros((k, k)), np.zeros(k), float("nan"), float("nan"))
    a = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(a.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    coeff = np.linalg.solve(a.T @ a + reg, a.T @ y)
    pred = a @ coeff
    resid = y - pred
    centered = y - np.mean(y, axis=0, keepdims=True)
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum(centered * centered))
    pred_norm = float(np.linalg.norm(pred) / (np.linalg.norm(y) + 1e-12))
    return GeneratorFit(
        generator=coeff[:-1].T,
        intercept=coeff[-1],
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        prediction_norm_fraction=pred_norm,
    )


def _skew_part(g: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(g, dtype=np.float64) - np.asarray(g, dtype=np.float64).T)


def _sym_part(g: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(g, dtype=np.float64) + np.asarray(g, dtype=np.float64).T)


def _generator_shape_metrics(g: np.ndarray) -> dict[str, float]:
    mat = np.asarray(g, dtype=np.float64)
    total = float(np.sum(mat * mat))
    if total <= 0:
        return {
            "generator_fro_norm": 0.0,
            "generator_skew_energy_fraction": float("nan"),
            "generator_symmetric_energy_fraction": float("nan"),
            "generator_diagonal_energy_fraction": float("nan"),
        }
    skew = _skew_part(mat)
    sym = _sym_part(mat)
    diag = np.diag(np.diag(mat))
    return {
        "generator_fro_norm": float(np.sqrt(total)),
        "generator_skew_energy_fraction": float(np.sum(skew * skew) / total),
        "generator_symmetric_energy_fraction": float(np.sum(sym * sym) / total),
        "generator_diagonal_energy_fraction": float(np.sum(diag * diag) / total),
    }


def _schur_pair_basis(skew: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        from scipy.linalg import schur

        _, q = schur(np.asarray(skew, dtype=np.float64), output="real")
        return np.asarray(q, dtype=np.float64), "real_schur_skew"
    except Exception:
        return np.eye(int(skew.shape[0]), dtype=np.float64), "identity_fallback"


def _block_rotation_metrics(g: np.ndarray, pair_basis: np.ndarray) -> dict[str, float]:
    q = np.asarray(pair_basis, dtype=np.float64)
    gb = q.T @ np.asarray(g, dtype=np.float64) @ q
    total = float(np.sum(gb * gb))
    if total <= 0:
        return {
            "pair_block_energy_fraction": float("nan"),
            "pair_offblock_energy_fraction": float("nan"),
            "pair_block_skew_fraction_of_total": float("nan"),
            "pair_block_skew_fraction_within_blocks": float("nan"),
        }
    block_energy = 0.0
    block_skew_energy = 0.0
    for a, b in _pair_slices(gb.shape[0]):
        block = gb[a:b, a:b]
        block_energy += float(np.sum(block * block))
        sk = _skew_part(block)
        block_skew_energy += float(np.sum(sk * sk))
    return {
        "pair_block_energy_fraction": float(block_energy / total),
        "pair_offblock_energy_fraction": float(1.0 - block_energy / total),
        "pair_block_skew_fraction_of_total": float(block_skew_energy / total),
        "pair_block_skew_fraction_within_blocks": float(block_skew_energy / (block_energy + 1e-12)),
    }


def _parse_basis_dims(value: str) -> list[int]:
    dims = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    return sorted({d for d in dims if d > 0})


def _derivative_scale(meta: dict[str, Any], delta_arcmin: float, derivative_units: str) -> float:
    if derivative_units == "model_px":
        return 1.0
    delta_px = float(meta.get("delta_model_px", float("nan")))
    if not np.isfinite(delta_px) or abs(float(delta_arcmin)) <= 1e-12:
        return 1.0
    px_per_arcmin = delta_px / float(delta_arcmin)
    if derivative_units == "arcmin":
        return float(px_per_arcmin)
    if derivative_units == "degree":
        return float(px_per_arcmin * 60.0)
    raise ValueError("derivative_units must be one of 'model_px', 'arcmin', or 'degree'")


def _radius_px_for_payload(meta: dict[str, Any], delta_arcmin: float, ring_radius_arcmin: float | None) -> tuple[float, float]:
    delta_px = float(meta.get("delta_model_px", float("nan")))
    if not np.isfinite(delta_px) or abs(float(delta_arcmin)) <= 1e-12:
        return 1.0, float("nan")
    arcmin_to_px = delta_px / float(delta_arcmin)
    radius_arcmin = float(delta_arcmin) if ring_radius_arcmin is None else float(ring_radius_arcmin)
    return float(radius_arcmin * arcmin_to_px), radius_arcmin


def _random_pair_null_medians(
    jx_rows: np.ndarray,
    jy_rows: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(jx_rows, dtype=np.float64)
    y = np.asarray(jy_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] < 2 or repeats <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(repeats)):
        q, _ = np.linalg.qr(rng.normal(size=(x.shape[1], x.shape[1])))
        fracs = []
        xr = x @ q
        yr = y @ q
        for row_x, row_y in zip(xr, yr, strict=False):
            fracs.append(float(_pair_energy_metrics(row_x, row_y)["best_pair_energy_fraction"]))
        vals.append(_finite_median(fracs))
    return np.asarray(vals, dtype=np.float64)


def _plot_summary(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    ok_rows = [r for r in summary_rows if str(r.get("status", "")) == "ok"]
    if not ok_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    deltas = sorted({float(r["delta"]) for r in ok_rows})
    for k in sorted({int(r["basis_k"]) for r in ok_rows}):
        block = [r for r in ok_rows if int(r["basis_k"]) == k]
        xs = [float(r["delta"]) for r in block]
        order = np.argsort(xs)
        xs = list(np.asarray(xs)[order])
        circ = list(np.asarray([float(r["ellipse_circularity_median"]) for r in block])[order])
        pair = list(np.asarray([float(r["best_pair_energy_fraction_median"]) for r in block])[order])
        skew = list(np.asarray([float(r["generator_skew_energy_fraction_mean"]) for r in block])[order])
        axes[0].plot(xs, circ, marker="o", label=f"k={k}")
        axes[1].plot(xs, pair, marker="o", label=f"k={k}")
        axes[2].plot(xs, skew, marker="o", label=f"k={k}")
    axes[0].set_title("Ring ellipse circularity")
    axes[1].set_title("Best 2D pair energy")
    axes[2].set_title("Generator skew fraction")
    for ax in axes:
        ax.set_xlabel("finite-difference delta (arcmin)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        if len(deltas) <= 6:
            ax.set_xticks(deltas)
    axes[0].set_ylabel("median")
    axes[0].legend(frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_ring_examples(rows: list[dict[str, object]], examples: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    candidates = [
        r
        for r in rows
        if str(r.get("status", "")) == "ok"
        and str(r.get("object_id")) in examples
        and int(r.get("best_pair_index", -1)) >= 0
    ]
    candidates = sorted(candidates, key=lambda r: float(r.get("ellipse_area_at_ring_radius", 0.0)), reverse=True)[:9]
    if not candidates:
        return
    ncols = min(3, len(candidates))
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False, constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    for ax, row in zip(axes.ravel(), candidates, strict=False):
        oid = str(row["object_id"])
        jx, jy = examples[oid]
        pair_idx = int(row["best_pair_index"])
        a = 2 * pair_idx
        b = min(a + 2, jx.size)
        if b - a < 2:
            ax.axis("off")
            continue
        radius = float(row.get("ring_radius_model_px", 1.0))
        xy = radius * (np.outer(np.cos(theta), jx[a:b]) + np.outer(np.sin(theta), jy[a:b]))
        ax.plot(xy[:, 0], xy[:, 1], color="#255c99", lw=1.6)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=14, color="#d95f02")
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.axvline(0.0, color="0.85", lw=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{oid} pair {pair_idx}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(candidates) :]:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    delta_arcmins, payload_by_delta, pkl_path, source_metadata = _load_tangent_payload(Path(args.input_root))
    requested_deltas = (
        {float(v.strip()) for v in str(args.deltas).split(",") if v.strip()}
        if args.deltas
        else set(delta_arcmins)
    )
    basis_dims = _parse_basis_dims(str(args.basis_dims))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    generator_rows: list[dict[str, object]] = []
    basis_rows: list[dict[str, object]] = []
    ring_examples: dict[tuple[float, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for delta in delta_arcmins:
        if float(delta) not in requested_deltas:
            continue
        payload = payload_by_delta.get(float(delta), {})
        object_ids = _valid_object_ids(payload)
        if len(object_ids) < int(args.min_objects):
            summary_rows.append(
                {
                    "delta": float(delta),
                    "basis_k": -1,
                    "n_objects": int(len(object_ids)),
                    "status": "not_run_insufficient_valid_objects",
                }
            )
            continue

        bx_full = np.stack([
            np.asarray(payload[oid]["bx"], dtype=np.float64)
            * _derivative_scale(payload[oid], float(delta), str(args.derivative_units))
            for oid in object_ids
        ], axis=1)
        by_full = np.stack([
            np.asarray(payload[oid]["by"], dtype=np.float64)
            * _derivative_scale(payload[oid], float(delta), str(args.derivative_units))
            for oid in object_ids
        ], axis=1)
        tangent_mat = np.concatenate([bx_full, by_full], axis=1)

        for requested_k in basis_dims:
            basis = _basis_from_columns(tangent_mat, k=int(requested_k), convention=str(args.basis_convention))
            u = basis.basis
            if u.size == 0:
                summary_rows.append(
                    {
                        "delta": float(delta),
                        "basis_k_requested": int(requested_k),
                        "basis_k": 0,
                        "n_objects": int(len(object_ids)),
                        "status": "not_run_empty_basis",
                    }
                )
                continue
            k_eff = int(u.shape[1])
            z_rows = []
            jx_rows = []
            jy_rows = []
            object_block: list[dict[str, object]] = []
            example_block: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for oid in object_ids:
                meta = payload[oid]
                r0 = np.asarray(meta["r0"], dtype=np.float64)
                scale = _derivative_scale(meta, float(delta), str(args.derivative_units))
                bx = np.asarray(meta["bx"], dtype=np.float64) * scale
                by = np.asarray(meta["by"], dtype=np.float64) * scale
                z0 = u.T @ r0
                jx = u.T @ bx
                jy = u.T @ by
                radius_px, radius_arcmin = _radius_px_for_payload(meta, float(delta), args.ring_radius_arcmin)
                row = {
                    "delta": float(delta),
                    "basis_k_requested": int(requested_k),
                    "basis_k": int(k_eff),
                    "object_id": str(oid),
                    "image_id": int(meta.get("image_id", -1)),
                    "trial_index": int(meta.get("trial_index", -1)),
                    "time_index": int(meta.get("time_index", -1)),
                    "ring_radius_arcmin": float(radius_arcmin),
                    "ring_radius_model_px": float(radius_px),
                    "norm_bx_full": _safe_norm(bx),
                    "norm_by_full": _safe_norm(by),
                    "compact_capture_bx": _variance_capture(u, bx),
                    "compact_capture_by": _variance_capture(u, by),
                    **_ellipse_metrics(jx, jy, radius_px=radius_px),
                    **_pair_energy_metrics(jx, jy),
                    **_image_frequency_stats(meta.get("history")),
                    "trajectory_source": "linearized_tangent_ring",
                    "basis_source": "tangent_union_pca",
                    "basis_convention": str(args.basis_convention),
                    "derivative_units": str(args.derivative_units),
                    "status": "ok",
                }
                object_block.append(row)
                z_rows.append(z0)
                jx_rows.append(jx)
                jy_rows.append(jy)
                example_block[str(oid)] = (jx, jy)

            z = np.asarray(z_rows, dtype=np.float64)
            jx_mat = np.asarray(jx_rows, dtype=np.float64)
            jy_mat = np.asarray(jy_rows, dtype=np.float64)
            gx_fit = _fit_generator(z, jx_mat, ridge=float(args.generator_ridge))
            gy_fit = _fit_generator(z, jy_mat, ridge=float(args.generator_ridge))
            gx = gx_fit.generator
            gy = gy_fit.generator
            comm = gx @ gy - gy @ gx
            comm_rel = float(np.linalg.norm(comm) / (np.linalg.norm(gx) * np.linalg.norm(gy) + 1e-12))
            for axis_name, fit in (("x", gx_fit), ("y", gy_fit)):
                skew = _skew_part(fit.generator)
                q, q_source = _schur_pair_basis(skew)
                generator_rows.append(
                    {
                        "delta": float(delta),
                        "basis_k_requested": int(requested_k),
                        "basis_k": int(k_eff),
                        "fit_axis": axis_name,
                        "n_objects": int(len(object_ids)),
                        "generator_fit_r2": float(fit.r2),
                        "generator_prediction_norm_fraction": float(fit.prediction_norm_fraction),
                        "generator_intercept_norm": _safe_norm(fit.intercept),
                        "commutator_relative_norm": comm_rel,
                        "pair_basis_source": q_source,
                        **_generator_shape_metrics(fit.generator),
                        **_block_rotation_metrics(fit.generator, q),
                        "status": "ok",
                    }
                )

            pair_null = _random_pair_null_medians(
                jx_mat,
                jy_mat,
                repeats=int(args.pair_null_repeats),
                seed=int(args.seed) + int(round(float(delta) * 1000.0)) + int(requested_k) * 100,
            )
            object_rows.extend(object_block)
            ring_examples[(float(delta), int(k_eff))] = example_block
            obs_best_pair = [float(r["best_pair_energy_fraction"]) for r in object_block]
            obs_circ = [float(r["ellipse_circularity"]) for r in object_block]
            obs_area = [float(r["ellipse_area_at_ring_radius"]) for r in object_block]
            obs_aspect = [float(r["ellipse_aspect_minor_over_major"]) for r in object_block]
            gradient = [float(r["image_gradient_rms"]) for r in object_block]
            freq_centroid = [float(r["image_frequency_centroid_cyc_per_px"]) for r in object_block]
            anis = [float(r["image_gradient_anisotropy"]) for r in object_block]
            gen_block = [
                r
                for r in generator_rows
                if float(r["delta"]) == float(delta)
                and int(r["basis_k_requested"]) == int(requested_k)
                and str(r["fit_axis"]) in {"x", "y"}
            ]
            summary_rows.append(
                {
                    "delta": float(delta),
                    "basis_k_requested": int(requested_k),
                    "basis_k": int(k_eff),
                    "n_objects": int(len(object_ids)),
                    "n_units": int(tangent_mat.shape[0]),
                    "tangent_union_rank": int(basis.rank),
                    "tangent_union_participation_ratio": float(basis.participation_ratio),
                    "tangent_union_capture_by_basis": float(basis.capture),
                    "tangent_union_rank_50": int(basis.rank_50),
                    "tangent_union_rank_75": int(basis.rank_75),
                    "tangent_union_rank_90": int(basis.rank_90),
                    "tangent_union_rank_95": int(basis.rank_95),
                    "compact_capture_bx_median": _finite_median([float(r["compact_capture_bx"]) for r in object_block]),
                    "compact_capture_by_median": _finite_median([float(r["compact_capture_by"]) for r in object_block]),
                    "ellipse_circularity_median": _finite_median(obs_circ),
                    "ellipse_circularity_ci_low": _finite_ci(obs_circ, 2.5),
                    "ellipse_circularity_ci_high": _finite_ci(obs_circ, 97.5),
                    "ellipse_aspect_median": _finite_median(obs_aspect),
                    "ellipse_area_median": _finite_median(obs_area),
                    "best_pair_energy_fraction_median": _finite_median(obs_best_pair),
                    "best_pair_energy_fraction_ci_low": _finite_ci(obs_best_pair, 2.5),
                    "best_pair_energy_fraction_ci_high": _finite_ci(obs_best_pair, 97.5),
                    "best_pair_random_rotation_null_median": _finite_median(pair_null),
                    "best_pair_random_rotation_null_ci_low": _finite_ci(pair_null, 2.5),
                    "best_pair_random_rotation_null_ci_high": _finite_ci(pair_null, 97.5),
                    "best_pair_effect_over_random_rotation": float(_finite_median(obs_best_pair) - _finite_median(pair_null)),
                    "generator_fit_r2_mean": _finite_mean([float(r["generator_fit_r2"]) for r in gen_block]),
                    "generator_skew_energy_fraction_mean": _finite_mean([float(r["generator_skew_energy_fraction"]) for r in gen_block]),
                    "generator_pair_block_energy_fraction_mean": _finite_mean([float(r["pair_block_energy_fraction"]) for r in gen_block]),
                    "generator_pair_block_skew_fraction_mean": _finite_mean([float(r["pair_block_skew_fraction_of_total"]) for r in gen_block]),
                    "commutator_relative_norm": comm_rel,
                    "spearman_area_vs_gradient_rms": _rank_correlation(obs_area, gradient),
                    "spearman_area_vs_frequency_centroid": _rank_correlation(obs_area, freq_centroid),
                    "spearman_circularity_vs_gradient_anisotropy": _rank_correlation(obs_circ, anis),
                    "spearman_best_pair_vs_frequency_centroid": _rank_correlation(obs_best_pair, freq_centroid),
                    "trajectory_source": "linearized_tangent_ring",
                    "basis_source": "tangent_union_pca",
                    "basis_convention": str(args.basis_convention),
                    "derivative_units": str(args.derivative_units),
                    "status": "ok",
                }
            )
            basis_rows.append(
                {
                    "delta": float(delta),
                    "basis_k_requested": int(requested_k),
                    "basis_k": int(k_eff),
                    "n_objects": int(len(object_ids)),
                    "n_units": int(tangent_mat.shape[0]),
                    "rank": int(basis.rank),
                    "participation_ratio": float(basis.participation_ratio),
                    "capture": float(basis.capture),
                    "rank_50": int(basis.rank_50),
                    "rank_75": int(basis.rank_75),
                    "rank_90": int(basis.rank_90),
                    "rank_95": int(basis.rank_95),
                    "status": "ok",
                    "basis_convention": str(args.basis_convention),
                }
            )

    _write_csv(out_dir / "phase_rotation_object_metrics.csv", object_rows)
    _write_csv(out_dir / "phase_rotation_summary.csv", summary_rows)
    _write_csv(out_dir / "phase_rotation_generator_metrics.csv", generator_rows)
    _write_csv(out_dir / "phase_rotation_basis_metrics.csv", basis_rows)
    _plot_summary(summary_rows, out_dir / "figures" / "phase_rotation_summary.png")

    ok_summary = [r for r in summary_rows if str(r.get("status", "")) == "ok"]
    if ok_summary:
        primary = max(
            ok_summary,
            key=lambda r: (
                int(r.get("basis_k", 0)) == int(args.example_basis_k),
                -abs(float(r.get("delta", 0.0)) - float(args.example_delta if args.example_delta is not None else r.get("delta", 0.0))),
                int(r.get("basis_k", 0)),
            ),
        )
        d0 = float(primary["delta"])
        k0 = int(primary["basis_k"])
        rows0 = [r for r in object_rows if float(r["delta"]) == d0 and int(r["basis_k"]) == k0]
        _plot_ring_examples(rows0, ring_examples.get((d0, k0), {}), out_dir / "figures" / f"ring_trajectory_examples_delta{d0:g}_k{k0}.png")

    manifest = {
        "input_tangent_cache": str(pkl_path),
        "output_dir": str(out_dir),
        "basis_dims": basis_dims,
        "deltas_analyzed": sorted(float(d) for d in requested_deltas),
        "ring_radius_arcmin": args.ring_radius_arcmin,
        "basis_convention": str(args.basis_convention),
        "derivative_units": str(args.derivative_units),
        "source_metadata": source_metadata,
        "pair_null_repeats": int(args.pair_null_repeats),
        "generator_ridge": float(args.generator_ridge),
        "summary_status_counts": {
            str(status): int(sum(1 for r in summary_rows if str(r.get("status", "")) == str(status)))
            for status in sorted({str(r.get("status", "")) for r in summary_rows})
        },
    }
    _save_json(out_dir / "phase_rotation_manifest.json", manifest)
    readme = [
        "# Phase-Rotation Probe",
        "",
        "This downstream analysis probes whether cached TFTS translation tangents have an internal phase-like organization.",
        "",
        "Important scope note: `trajectory_source=linearized_tangent_ring` means the ring loops are local tangent predictions, not newly sampled finite-response orbits. A positive result here is a reason to run a finite ring-translation pass, not a literal torus claim by itself.",
        "",
        f"Basis convention: `{args.basis_convention}`. Derivative units: `{args.derivative_units}`.",
        "",
        "Primary outputs:",
        "- phase_rotation_object_metrics.csv",
        "- phase_rotation_summary.csv",
        "- phase_rotation_generator_metrics.csv",
        "- phase_rotation_basis_metrics.csv",
        "- figures/phase_rotation_summary.png",
        "- figures/ring_trajectory_examples_*.png",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe phase-like rotational structure in cached TFTS tangent maps.")
    p.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="TFTS output root containing tangent_maps/twin_tangent_maps.pkl, or the pickle path itself.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_phase_rotation_probe",
    )
    p.add_argument("--deltas", type=str, default=None, help="Comma-separated finite-difference deltas to analyze. Default: all.")
    p.add_argument("--basis-dims", type=str, default="2,4,6,10,20")
    p.add_argument(
        "--basis-convention",
        choices=("uncentered", "centered_across_tangents_per_unit"),
        default="uncentered",
        help="Tangent-basis convention. The centered option matches the later manifest-basis audit convention.",
    )
    p.add_argument(
        "--derivative-units",
        choices=("model_px", "arcmin", "degree"),
        default="model_px",
        help="Units for bx/by before projection. Cached TFTS maps store response derivatives per model pixel.",
    )
    p.add_argument("--min-objects", type=int, default=8)
    p.add_argument("--ring-radius-arcmin", type=float, default=None)
    p.add_argument("--pair-null-repeats", type=int, default=100)
    p.add_argument("--generator-ridge", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--example-delta", type=float, default=None)
    p.add_argument("--example-basis-k", type=int, default=6)
    return p


def main() -> None:
    manifest = analyze(build_parser().parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

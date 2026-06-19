from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT
from declan.twin_feature_tangent_structure.run_phase_rotation_probe import (
    _basis_from_columns,
    _derivative_scale,
    _finite_ci,
    _finite_mean,
    _finite_median,
    _image_frequency_stats,
    _load_tangent_payload,
    _parse_basis_dims,
    _safe_norm,
    _save_json,
    _valid_object_ids,
    _write_csv,
)


def _unique_shift_rows(rows: list[tuple[float, float]]) -> np.ndarray:
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for dx, dy in rows:
        key = (round(float(dx), 10), round(float(dy), 10))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(dx), float(dy)))
    return np.asarray(out, dtype=np.float64)


def _pca_rank_metrics(dz: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(dz, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0:
        return {
            "pca_rank": 0,
            "pca_participation_ratio": float("nan"),
            "line_fraction": float("nan"),
            "plane_fraction": float("nan"),
            "top3_fraction": float("nan"),
        }
    xc = x - np.mean(x, axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    evals = np.maximum(s * s, 0.0)
    total = float(np.sum(evals))
    if total <= 1e-12:
        return {
            "pca_rank": 0,
            "pca_participation_ratio": float("nan"),
            "line_fraction": float("nan"),
            "plane_fraction": float("nan"),
            "top3_fraction": float("nan"),
        }
    rank = int(np.sum(evals > max(1e-12, 1e-8 * float(np.max(evals)))))
    pr = float((np.sum(evals) ** 2) / (np.sum(evals * evals) + 1e-12))
    return {
        "pca_rank": rank,
        "pca_participation_ratio": pr,
        "line_fraction": float(np.sum(evals[:1]) / total),
        "plane_fraction": float(np.sum(evals[:2]) / total),
        "top3_fraction": float(np.sum(evals[:3]) / total),
    }


def _feature_matrix(shifts: np.ndarray, degree: int) -> np.ndarray:
    s = np.asarray(shifts, dtype=np.float64)
    dx = s[:, 0]
    dy = s[:, 1]
    if int(degree) == 1:
        return np.stack([dx, dy], axis=1)
    if int(degree) == 2:
        return np.stack([dx, dy, dx * dx, dx * dy, dy * dy], axis=1)
    raise ValueError("degree must be 1 or 2")


def _through_origin_fit_r2(shifts: np.ndarray, dz: np.ndarray, degree: int, ridge: float = 1e-9) -> float:
    x = _feature_matrix(np.asarray(shifts, dtype=np.float64), int(degree))
    y = np.asarray(dz, dtype=np.float64)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[keep]
    y = y[keep]
    if x.shape[0] <= x.shape[1] or y.size == 0:
        return float("nan")
    reg = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    coeff = np.linalg.solve(x.T @ x + reg, x.T @ y)
    pred = x @ coeff
    den = float(np.sum(y * y))
    if den <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / den)


def _sheet_fit_metrics(shifts: np.ndarray, dz: np.ndarray) -> dict[str, float]:
    lin = _through_origin_fit_r2(shifts, dz, degree=1)
    quad = _through_origin_fit_r2(shifts, dz, degree=2)
    return {
        "linear_sheet_r2_energy": float(lin),
        "quadratic_sheet_r2_energy": float(quad),
        "quadratic_gain_over_linear": float(quad - lin) if np.isfinite(lin) and np.isfinite(quad) else float("nan"),
    }


def _ring_fourier_metrics(theta: np.ndarray, dz: np.ndarray) -> dict[str, float]:
    th = np.asarray(theta, dtype=np.float64).ravel()
    y = np.asarray(dz, dtype=np.float64)
    keep = np.isfinite(th) & np.isfinite(y).all(axis=1)
    th = th[keep]
    y = y[keep]
    if th.size < 6 or y.ndim != 2 or y.shape[0] != th.size:
        return {
            "ring_planarity_fraction": float("nan"),
            "ring_first_harmonic_fraction": float("nan"),
            "ring_second_harmonic_fraction": float("nan"),
            "ring_radial_cv_in_best_plane": float("nan"),
            "ring_cyclic_step_cv": float("nan"),
            "ellipse_like_score": float("nan"),
            "circle_like_score": float("nan"),
        }
    order = np.argsort(th)
    th = th[order]
    y = y[order]
    yc = y - np.mean(y, axis=0, keepdims=True)
    total = float(np.sum(yc * yc))
    if total <= 1e-12:
        return {
            "ring_planarity_fraction": float("nan"),
            "ring_first_harmonic_fraction": float("nan"),
            "ring_second_harmonic_fraction": float("nan"),
            "ring_radial_cv_in_best_plane": float("nan"),
            "ring_cyclic_step_cv": float("nan"),
            "ellipse_like_score": float("nan"),
            "circle_like_score": float("nan"),
        }
    pca = _pca_rank_metrics(y)
    f1 = np.stack([np.cos(th), np.sin(th)], axis=1)
    f2 = np.stack([np.cos(2.0 * th), np.sin(2.0 * th)], axis=1)

    def _capture(feat: np.ndarray) -> float:
        coeff = np.linalg.lstsq(feat, yc, rcond=None)[0]
        pred = feat @ coeff
        return float(np.sum(pred * pred) / total)

    h1 = _capture(f1)
    h2 = _capture(f2)
    _, _, vt = np.linalg.svd(yc, full_matrices=False)
    xy = yc @ vt[:2].T
    radius = np.linalg.norm(xy, axis=1)
    radial_cv = float(np.std(radius) / (np.mean(radius) + 1e-12))
    steps = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    step_cv = float(np.std(steps) / (np.mean(steps) + 1e-12))
    planarity = float(pca["plane_fraction"])
    ellipse_score = float(planarity * h1)
    circle_score = float(ellipse_score * max(0.0, 1.0 - min(radial_cv, 1.0)))
    return {
        "ring_planarity_fraction": planarity,
        "ring_first_harmonic_fraction": h1,
        "ring_second_harmonic_fraction": h2,
        "ring_radial_cv_in_best_plane": radial_cv,
        "ring_cyclic_step_cv": step_cv,
        "ellipse_like_score": ellipse_score,
        "circle_like_score": circle_score,
    }


def _opposition_symmetry(a: np.ndarray, b: np.ndarray) -> float:
    den = _safe_norm(a) + _safe_norm(b)
    if den <= 1e-12:
        return float("nan")
    return float(_safe_norm(a + b) / den)


def _classify_shape(row: dict[str, object]) -> str:
    line = float(row.get("line_fraction", float("nan")))
    plane = float(row.get("plane_fraction", float("nan")))
    lin = float(row.get("linear_sheet_r2_energy", float("nan")))
    quad = float(row.get("quadratic_sheet_r2_energy", float("nan")))
    gain = float(row.get("quadratic_gain_over_linear", float("nan")))
    ring_plane = float(row.get("ring_planarity_fraction", float("nan")))
    h1 = float(row.get("ring_first_harmonic_fraction", float("nan")))
    radial = float(row.get("ring_radial_cv_in_best_plane", float("nan")))
    if np.isfinite(line) and line >= 0.85:
        return "line_or_ribbon"
    if np.isfinite(ring_plane) and np.isfinite(h1) and ring_plane >= 0.78 and h1 >= 0.75:
        if np.isfinite(radial) and radial <= 0.25:
            return "round_loop"
        return "elliptical_loop"
    if np.isfinite(plane) and plane >= 0.82 and np.isfinite(lin) and lin >= 0.82:
        return "flat_translation_sheet"
    if np.isfinite(plane) and plane >= 0.78 and np.isfinite(quad) and quad >= 0.82 and np.isfinite(gain) and gain >= 0.08:
        return "curved_translation_sheet"
    if np.isfinite(plane) and plane >= 0.72:
        return "planar_but_irregular"
    return "distributed_no_simple_shape"


def _cached_cardinal_orbit(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    required = ["r0", "rx_p", "rx_m", "ry_p", "ry_m"]
    if not all(k in meta for k in required):
        raise KeyError("cached cardinal endpoint fields are missing")
    delta_px = float(meta.get("delta_model_px", float("nan")))
    if not np.isfinite(delta_px) or delta_px <= 0:
        raise ValueError("delta_model_px is missing or invalid")
    points = np.stack(
        [
            np.asarray(meta["r0"], dtype=np.float64),
            np.asarray(meta["rx_p"], dtype=np.float64),
            np.asarray(meta["rx_m"], dtype=np.float64),
            np.asarray(meta["ry_p"], dtype=np.float64),
            np.asarray(meta["ry_m"], dtype=np.float64),
        ],
        axis=0,
    )
    shifts = np.asarray(
        [
            (0.0, 0.0),
            (delta_px, 0.0),
            (-delta_px, 0.0),
            (0.0, delta_px),
            (0.0, -delta_px),
        ],
        dtype=np.float64,
    )
    ring_theta = np.zeros(0, dtype=np.float64)
    return shifts, points, ring_theta, "cached_cardinal_endpoints"


def _regenerated_orbit(
    ctx: Any,
    meta: dict[str, Any],
    model_device: str,
    *,
    ring_radius_arcmin: float,
    ring_points: int,
    grid_radius_arcmin: float,
    grid_step_arcmin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import (
        _predict_rate_from_history,
        _shift_movie_subpixel,
        _movie_to_thw,
    )

    history = np.asarray(meta["history"], dtype=np.float32)
    delta_arcmin = float(meta.get("delta_arcmin", float("nan")))
    delta_px = float(meta.get("delta_model_px", float("nan")))
    if not np.isfinite(delta_arcmin) or abs(delta_arcmin) <= 1e-12 or not np.isfinite(delta_px):
        raise ValueError("delta_arcmin/delta_model_px are required for regenerated orbits")
    px_per_arcmin = delta_px / delta_arcmin
    ring_radius_px = float(ring_radius_arcmin) * px_per_arcmin
    grid_radius_px = float(grid_radius_arcmin) * px_per_arcmin
    grid_step_px = float(grid_step_arcmin) * px_per_arcmin
    vals = np.arange(-grid_radius_px, grid_radius_px + 0.5 * grid_step_px, grid_step_px, dtype=np.float64)
    shift_rows: list[tuple[float, float]] = [(0.0, 0.0)]
    shift_rows.extend((float(dx), float(dy)) for dy in vals for dx in vals)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ring_points), endpoint=False, dtype=np.float64)
    ring_shifts = [(float(ring_radius_px * np.cos(t)), float(ring_radius_px * np.sin(t))) for t in theta]
    shift_rows.extend(ring_shifts)
    shifts = _unique_shift_rows(shift_rows)
    ring_lookup = {(round(dx, 10), round(dy, 10)): float(t) for (dx, dy), t in zip(ring_shifts, theta, strict=False)}
    ring_theta: list[float] = []

    h0 = _movie_to_thw(history).to(str(model_device))
    points: list[np.ndarray] = []
    for dx, dy in shifts:
        if abs(float(dx)) <= 1e-12 and abs(float(dy)) <= 1e-12:
            shifted = history
        else:
            shifted = _shift_movie_subpixel(h0, dx_px=float(dx), dy_px=float(dy)).detach().cpu().numpy()
        points.append(_predict_rate_from_history(ctx, shifted, model_device=model_device))
        ring_theta.append(ring_lookup.get((round(float(dx), 10), round(float(dy), 10)), float("nan")))
    return shifts, np.asarray(points, dtype=np.float64), np.asarray(ring_theta, dtype=np.float64), "regenerated_ring_grid_orbit"


def _random_basis_capture(
    full_deltas_by_object: list[np.ndarray],
    n_units: int,
    k: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    if repeats <= 0 or not full_deltas_by_object or k <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    vals: list[float] = []
    for _ in range(int(repeats)):
        q, _ = np.linalg.qr(rng.normal(size=(int(n_units), int(k))))
        caps = []
        for full in full_deltas_by_object:
            den = float(np.sum(full * full))
            if den <= 1e-12:
                continue
            z = np.asarray(full, dtype=np.float64) @ q
            caps.append(float(np.sum(z * z) / den))
        vals.append(_finite_median(caps))
    return np.asarray(vals, dtype=np.float64)


def _plot_summary(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    ok = [r for r in summary_rows if str(r.get("status", "")) == "ok"]
    if not ok:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    axes = axes.ravel()
    for k in sorted({int(r["basis_k"]) for r in ok}):
        block = [r for r in ok if int(r["basis_k"]) == k]
        order = np.argsort([float(r["delta"]) for r in block])
        xs = np.asarray([float(r["delta"]) for r in block], dtype=np.float64)[order]
        cap = np.asarray([float(r["compact_orbit_energy_fraction_median"]) for r in block], dtype=np.float64)[order]
        null = np.asarray([float(r["random_basis_orbit_energy_fraction_median"]) for r in block], dtype=np.float64)[order]
        plane = np.asarray([float(r["plane_fraction_median"]) for r in block], dtype=np.float64)[order]
        line = np.asarray([float(r["line_fraction_median"]) for r in block], dtype=np.float64)[order]
        lin = np.asarray([float(r["linear_sheet_r2_median"]) for r in block], dtype=np.float64)[order]
        quad = np.asarray([float(r["quadratic_sheet_r2_median"]) for r in block], dtype=np.float64)[order]
        axes[0].plot(xs, cap, marker="o", label=f"compact k={k}")
        axes[0].plot(xs, null, marker="x", ls="--", alpha=0.65, label=f"random k={k}")
        axes[1].plot(xs, plane, marker="o", label=f"plane k={k}")
        axes[1].plot(xs, line, marker="x", ls="--", label=f"line k={k}")
        axes[2].plot(xs, lin, marker="o", label=f"linear k={k}")
        axes[2].plot(xs, quad, marker="x", ls="--", label=f"quadratic k={k}")
    label_counts = Counter()
    for row in ok:
        for part in str(row.get("shape_label_counts", "")).split(";"):
            if not part:
                continue
            label, count = part.rsplit(":", 1)
            label_counts[label] += int(count)
    if label_counts:
        labels = sorted(label_counts, key=lambda x: (-label_counts[x], x))
        axes[3].barh(labels, [label_counts[l] for l in labels], color="#4f7cac")
        axes[3].invert_yaxis()
        axes[3].set_title("Shape labels across summaries")
        axes[3].set_xlabel("count")
    axes[0].set_title("Orbit energy captured")
    axes[1].set_title("PCA rank shape")
    axes[2].set_title("Sheet model fit")
    axes[0].set_ylabel("median fraction")
    axes[1].set_ylabel("median fraction")
    axes[2].set_ylabel("median R2-energy")
    for ax in axes[:3]:
        ax.set_xlabel("finite-difference delta (arcmin)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_examples(examples: list[dict[str, Any]], out_path: Path) -> None:
    if not examples:
        return
    examples = examples[:9]
    ncols = min(3, len(examples))
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), squeeze=False, constrained_layout=True)
    for ax, ex in zip(axes.ravel(), examples, strict=False):
        dz = np.asarray(ex["dz"], dtype=np.float64)
        shifts = np.asarray(ex["shifts"], dtype=np.float64)
        ring_theta = np.asarray(ex["ring_theta"], dtype=np.float64)
        yc = dz - np.mean(dz, axis=0, keepdims=True)
        if yc.shape[0] < 2:
            ax.axis("off")
            continue
        _, _, vt = np.linalg.svd(yc, full_matrices=False)
        xy = yc @ vt[:2].T if vt.shape[0] >= 2 else np.pad(yc @ vt[:1].T, ((0, 0), (0, 1)))
        ring = np.isfinite(ring_theta)
        grid = ~ring
        if np.any(grid):
            ax.scatter(xy[grid, 0], xy[grid, 1], s=24, c=np.linalg.norm(shifts[grid], axis=1), cmap="viridis", edgecolor="0.25", linewidth=0.4)
        if np.any(ring):
            order = np.argsort(ring_theta[ring])
            pts = xy[ring][order]
            ax.plot(pts[:, 0], pts[:, 1], color="#d95f02", lw=1.5)
            ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#d95f02")
        ax.axhline(0.0, color="0.86", lw=0.8)
        ax.axvline(0.0, color="0.86", lw=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"d{float(ex['delta']):g} k{int(ex['basis_k'])} | {ex['object_id']} | {ex['shape_label']}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(examples) :]:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _select_plot_examples(candidates: list[dict[str, Any]], max_examples: int = 9) -> list[dict[str, Any]]:
    if not candidates:
        return []
    by_label: dict[str, list[dict[str, Any]]] = {}
    for ex in candidates:
        by_label.setdefault(str(ex["shape_label"]), []).append(ex)
    selected: list[dict[str, Any]] = []
    priority = [
        "elliptical_loop",
        "round_loop",
        "curved_translation_sheet",
        "flat_translation_sheet",
        "line_or_ribbon",
        "planar_but_irregular",
        "distributed_no_simple_shape",
    ]
    for label in priority:
        block = sorted(
            by_label.get(label, []),
            key=lambda ex: (float(ex["compact_energy"]), float(ex["plane_fraction"])),
            reverse=True,
        )
        selected.extend(block[:2])
        if len(selected) >= int(max_examples):
            return selected[: int(max_examples)]
    selected_keys = {(ex["object_id"], ex["basis_k"], ex["delta"], ex["shape_label"]) for ex in selected}
    remaining = sorted(
        candidates,
        key=lambda ex: (float(ex["compact_energy"]), float(ex["plane_fraction"])),
        reverse=True,
    )
    for ex in remaining:
        key = (ex["object_id"], ex["basis_k"], ex["delta"], ex["shape_label"])
        if key in selected_keys:
            continue
        selected.append(ex)
        if len(selected) >= int(max_examples):
            break
    return selected[: int(max_examples)]


def _label_count_string(labels: list[str]) -> str:
    counts = Counter(labels)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


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

    ctx = None
    if str(args.trajectory_source) == "regenerated_ring_grid":
        from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import _load_twin_context

        ctx = _load_twin_context(str(args.device))

    object_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    basis_rows: list[dict[str, object]] = []
    plot_examples: list[dict[str, Any]] = []

    for delta in delta_arcmins:
        if float(delta) not in requested_deltas:
            continue
        payload = payload_by_delta.get(float(delta), {})
        object_ids = _valid_object_ids(payload)
        if int(args.max_objects) > 0:
            object_ids = object_ids[: int(args.max_objects)]
        if len(object_ids) < int(args.min_objects):
            summary_rows.append(
                {
                    "delta": float(delta),
                    "basis_k": -1,
                    "n_objects": int(len(object_ids)),
                    "trajectory_source": str(args.trajectory_source),
                    "status": "not_run_insufficient_valid_objects",
                }
            )
            continue
        bx_full = np.stack(
            [
                np.asarray(payload[oid]["bx"], dtype=np.float64)
                * _derivative_scale(payload[oid], float(delta), str(args.derivative_units))
                for oid in object_ids
            ],
            axis=1,
        )
        by_full = np.stack(
            [
                np.asarray(payload[oid]["by"], dtype=np.float64)
                * _derivative_scale(payload[oid], float(delta), str(args.derivative_units))
                for oid in object_ids
            ],
            axis=1,
        )
        tangent_mat = np.concatenate([bx_full, by_full], axis=1)

        orbit_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
        for oid in object_ids:
            meta = payload[oid]
            try:
                if str(args.trajectory_source) == "cached_cardinal":
                    orbit_cache[oid] = _cached_cardinal_orbit(meta)
                else:
                    orbit_cache[oid] = _regenerated_orbit(
                        ctx,
                        meta,
                        str(args.device),
                        ring_radius_arcmin=float(args.ring_radius_arcmin or delta),
                        ring_points=int(args.ring_points),
                        grid_radius_arcmin=float(args.grid_radius_arcmin or delta),
                        grid_step_arcmin=float(args.grid_step_arcmin or delta),
                    )
            except Exception:
                continue
        object_ids = [oid for oid in object_ids if oid in orbit_cache]
        if len(object_ids) < int(args.min_objects):
            summary_rows.append(
                {
                    "delta": float(delta),
                    "basis_k": -1,
                    "n_objects": int(len(object_ids)),
                    "trajectory_source": str(args.trajectory_source),
                    "status": "not_run_insufficient_valid_orbits",
                }
            )
            continue

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
                        "trajectory_source": str(args.trajectory_source),
                        "status": "not_run_empty_basis",
                    }
                )
                continue
            k_eff = int(u.shape[1])
            block_rows: list[dict[str, object]] = []
            full_deltas_for_null: list[np.ndarray] = []
            examples_for_block: list[dict[str, Any]] = []
            for oid in object_ids:
                meta = payload[oid]
                shifts, points, ring_theta, actual_source = orbit_cache[oid]
                if points.ndim != 2 or points.shape[0] != shifts.shape[0]:
                    continue
                z = points @ u
                z0 = z[0]
                dz = z - z0[None, :]
                full_dz = points - points[0][None, :]
                full_deltas_for_null.append(full_dz)
                den = float(np.sum(full_dz * full_dz))
                compact_energy = float(np.sum(dz * dz) / den) if den > 1e-12 else float("nan")
                metrics: dict[str, object] = {
                    "delta": float(delta),
                    "basis_k_requested": int(requested_k),
                    "basis_k": int(k_eff),
                    "object_id": str(oid),
                    "image_id": int(meta.get("image_id", -1)),
                    "trial_index": int(meta.get("trial_index", -1)),
                    "time_index": int(meta.get("time_index", -1)),
                    "n_orbit_points": int(points.shape[0]),
                    "compact_orbit_energy_fraction": compact_energy,
                    "orbit_full_energy": float(den),
                    "max_shift_model_px": float(np.max(np.linalg.norm(shifts, axis=1))),
                    "trajectory_source": str(actual_source),
                    "basis_source": "tangent_union_pca",
                    "basis_convention": str(args.basis_convention),
                    "derivative_units_for_basis": str(args.derivative_units),
                    "status": "ok",
                    **_pca_rank_metrics(dz),
                    **_sheet_fit_metrics(shifts, dz),
                    **_image_frequency_stats(meta.get("history")),
                }
                if str(actual_source) == "cached_cardinal_endpoints" and points.shape[0] >= 5:
                    metrics.update(
                        {
                            "x_opposition_symmetry": _opposition_symmetry(dz[1], dz[2]),
                            "y_opposition_symmetry": _opposition_symmetry(dz[3], dz[4]),
                        }
                    )
                ring_keep = np.isfinite(ring_theta)
                if int(np.sum(ring_keep)) >= 6:
                    metrics.update(_ring_fourier_metrics(ring_theta[ring_keep], dz[ring_keep]))
                else:
                    metrics.update(_ring_fourier_metrics(np.zeros(0), np.zeros((0, dz.shape[1]))))
                metrics["shape_label"] = _classify_shape(metrics)
                block_rows.append(metrics)
                examples_for_block.append(
                    {
                        "delta": float(delta),
                        "basis_k": int(k_eff),
                        "object_id": str(oid),
                        "shape_label": str(metrics["shape_label"]),
                        "dz": dz,
                        "shifts": shifts,
                        "ring_theta": ring_theta,
                        "compact_energy": compact_energy,
                        "plane_fraction": float(metrics["plane_fraction"]),
                    }
                )

            random_caps = _random_basis_capture(
                full_deltas_for_null,
                n_units=int(tangent_mat.shape[0]),
                k=int(k_eff),
                repeats=int(args.random_basis_repeats),
                seed=int(args.seed) + int(round(float(delta) * 1000.0)) + int(k_eff) * 1000,
            )
            labels = [str(r["shape_label"]) for r in block_rows]
            object_rows.extend(block_rows)
            summary_rows.append(
                {
                    "delta": float(delta),
                    "basis_k_requested": int(requested_k),
                    "basis_k": int(k_eff),
                    "n_objects": int(len(block_rows)),
                    "n_units": int(tangent_mat.shape[0]),
                    "tangent_union_rank": int(basis.rank),
                    "tangent_union_capture_by_basis": float(basis.capture),
                    "compact_orbit_energy_fraction_median": _finite_median([float(r["compact_orbit_energy_fraction"]) for r in block_rows]),
                    "compact_orbit_energy_fraction_ci_low": _finite_ci([float(r["compact_orbit_energy_fraction"]) for r in block_rows], 2.5),
                    "compact_orbit_energy_fraction_ci_high": _finite_ci([float(r["compact_orbit_energy_fraction"]) for r in block_rows], 97.5),
                    "random_basis_orbit_energy_fraction_median": _finite_median(random_caps),
                    "random_basis_orbit_energy_fraction_ci_low": _finite_ci(random_caps, 2.5),
                    "random_basis_orbit_energy_fraction_ci_high": _finite_ci(random_caps, 97.5),
                    "compact_energy_effect_over_random": float(
                        _finite_median([float(r["compact_orbit_energy_fraction"]) for r in block_rows])
                        - _finite_median(random_caps)
                    ),
                    "line_fraction_median": _finite_median([float(r["line_fraction"]) for r in block_rows]),
                    "plane_fraction_median": _finite_median([float(r["plane_fraction"]) for r in block_rows]),
                    "top3_fraction_median": _finite_median([float(r["top3_fraction"]) for r in block_rows]),
                    "linear_sheet_r2_median": _finite_median([float(r["linear_sheet_r2_energy"]) for r in block_rows]),
                    "quadratic_sheet_r2_median": _finite_median([float(r["quadratic_sheet_r2_energy"]) for r in block_rows]),
                    "quadratic_gain_median": _finite_median([float(r["quadratic_gain_over_linear"]) for r in block_rows]),
                    "ring_first_harmonic_fraction_median": _finite_median([float(r["ring_first_harmonic_fraction"]) for r in block_rows]),
                    "ellipse_like_score_median": _finite_median([float(r["ellipse_like_score"]) for r in block_rows]),
                    "circle_like_score_median": _finite_median([float(r["circle_like_score"]) for r in block_rows]),
                    "x_opposition_symmetry_median": _finite_median([float(r.get("x_opposition_symmetry", float("nan"))) for r in block_rows]),
                    "y_opposition_symmetry_median": _finite_median([float(r.get("y_opposition_symmetry", float("nan"))) for r in block_rows]),
                    "spearman_energy_vs_gradient_rms": _spearman(
                        [float(r["compact_orbit_energy_fraction"]) for r in block_rows],
                        [float(r["image_gradient_rms"]) for r in block_rows],
                    ),
                    "shape_label_counts": _label_count_string(labels),
                    "trajectory_source": str(args.trajectory_source),
                    "basis_source": "tangent_union_pca",
                    "basis_convention": str(args.basis_convention),
                    "status": "ok",
                }
            )
            basis_rows.append(
                {
                    "delta": float(delta),
                    "basis_k_requested": int(requested_k),
                    "basis_k": int(k_eff),
                    "n_objects": int(len(block_rows)),
                    "n_units": int(tangent_mat.shape[0]),
                    "rank": int(basis.rank),
                    "participation_ratio": float(basis.participation_ratio),
                    "capture": float(basis.capture),
                    "rank_50": int(basis.rank_50),
                    "rank_75": int(basis.rank_75),
                    "rank_90": int(basis.rank_90),
                    "rank_95": int(basis.rank_95),
                    "basis_convention": str(args.basis_convention),
                    "status": "ok",
                }
            )
            plot_examples.extend(examples_for_block)

    _write_csv(out_dir / "shape_atlas_object_metrics.csv", object_rows)
    _write_csv(out_dir / "shape_atlas_summary.csv", summary_rows)
    _write_csv(out_dir / "shape_atlas_basis_metrics.csv", basis_rows)
    _plot_summary(summary_rows, out_dir / "figures" / "shape_atlas_summary.png")
    _plot_examples(_select_plot_examples(plot_examples), out_dir / "figures" / "shape_atlas_examples.png")

    manifest = {
        "input_tangent_cache": str(pkl_path),
        "output_dir": str(out_dir),
        "basis_dims": basis_dims,
        "deltas_analyzed": sorted(float(d) for d in requested_deltas),
        "trajectory_source": str(args.trajectory_source),
        "basis_convention": str(args.basis_convention),
        "derivative_units_for_basis": str(args.derivative_units),
        "random_basis_repeats": int(args.random_basis_repeats),
        "source_metadata": source_metadata,
        "summary_status_counts": {
            str(status): int(sum(1 for r in summary_rows if str(r.get("status", "")) == str(status)))
            for status in sorted({str(r.get("status", "")) for r in summary_rows})
        },
    }
    _save_json(out_dir / "shape_atlas_manifest.json", manifest)
    readme = [
        "# Shape Atlas Probe",
        "",
        "This analysis asks a deliberately broad question: when translation responses are projected into the compact tangent basis, do they look like a recognizable geometric object?",
        "",
        "The cached-cardinal mode uses the stored center/x+/x-/y+/y- finite endpoints, so it can score compactness, line-vs-plane structure, local sheet fit, and opposition symmetry without re-running the twin. The regenerated-ring-grid mode reloads the twin from stored histories and adds finite rings/grids for loop and curved-sheet tests.",
        "",
        f"Trajectory source: `{args.trajectory_source}`. Basis convention: `{args.basis_convention}`.",
        "",
        "Primary outputs:",
        "- shape_atlas_object_metrics.csv",
        "- shape_atlas_summary.csv",
        "- shape_atlas_basis_metrics.csv",
        "- figures/shape_atlas_summary.png",
        "- figures/shape_atlas_examples.png",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


def _spearman(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe recognizable shape classes in compact TFTS translation geometry.")
    p.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="TFTS output root containing tangent_maps/twin_tangent_maps.pkl, or the pickle path itself.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_shape_atlas_probe",
    )
    p.add_argument("--deltas", type=str, default=None, help="Comma-separated finite-difference deltas to analyze. Default: all.")
    p.add_argument("--basis-dims", type=str, default="2,4,6,10,20")
    p.add_argument(
        "--basis-convention",
        choices=("uncentered", "centered_across_tangents_per_unit"),
        default="uncentered",
    )
    p.add_argument(
        "--derivative-units",
        choices=("model_px", "arcmin", "degree"),
        default="model_px",
        help="Units used only for constructing the tangent basis. Orbit points remain finite response differences.",
    )
    p.add_argument(
        "--trajectory-source",
        choices=("cached_cardinal", "regenerated_ring_grid"),
        default="cached_cardinal",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--min-objects", type=int, default=8)
    p.add_argument("--max-objects", type=int, default=0, help="0 means all valid objects.")
    p.add_argument("--ring-radius-arcmin", type=float, default=None)
    p.add_argument("--ring-points", type=int, default=16)
    p.add_argument("--grid-radius-arcmin", type=float, default=None)
    p.add_argument("--grid-step-arcmin", type=float, default=None)
    p.add_argument("--random-basis-repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    manifest = analyze(build_parser().parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

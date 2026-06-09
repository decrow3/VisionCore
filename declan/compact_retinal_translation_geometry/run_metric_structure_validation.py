#!/usr/bin/env python3
"""Validate compact-space retinal-translation metric structure.

The current tangent-map cache contains cardinal finite translations
(``+/-x`` and ``+/-y``) at a small step sweep.  This validator therefore runs
the promoted metric tests that are supported by that cache and marks diagonal
composition / true direction-held-out prediction as unavailable until the cache
is extended with diagonal or arbitrary translated responses.
"""
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
import pandas as pd

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_TFTS_ROOT = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
DEFAULT_OUT_ROOT = VISIONCORE_ROOT / "outputs" / "compact_retinal_translation_geometry"


@dataclass(frozen=True)
class BasisPayload:
    name: str
    matrix: np.ndarray | None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_tangent_maps(tfts_root: Path) -> tuple[list[float], dict[float, dict[str, dict[str, Any]]]]:
    path = tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing tangent-map cache: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    deltas = [float(v) for v in payload["delta_arcmins"]]
    objects = {
        float(delta): {str(oid): meta for oid, meta in object_payload.items()}
        for delta, object_payload in payload["object_payload"].items()
    }
    return deltas, objects


def _nearest(values: list[float], target: float) -> float:
    if not values:
        raise ValueError("No values available.")
    return float(min(values, key=lambda v: abs(float(v) - float(target))))


def _valid_object_ids(payload: dict[str, dict[str, Any]], required: tuple[str, ...] = ("r0", "bx", "by")) -> list[str]:
    out: list[str] = []
    for oid, meta in sorted(payload.items()):
        ok = True
        for name in required:
            if name not in meta:
                ok = False
                break
            arr = np.asarray(meta[name], dtype=np.float64)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                ok = False
                break
        if ok:
            bx = np.asarray(meta.get("bx", []), dtype=np.float64)
            by = np.asarray(meta.get("by", []), dtype=np.float64)
            if bx.size and (np.linalg.norm(bx) <= 1e-12 or np.linalg.norm(by) <= 1e-12):
                ok = False
        if ok:
            out.append(str(oid))
    return out


def _orth(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(x, dtype=np.float64))
    keep = np.abs(np.diag(r)) > eps
    return q[:, keep]


def _basis_from_payload(payload: dict[str, dict[str, Any]], k: int) -> np.ndarray:
    object_ids = _valid_object_ids(payload, required=("bx", "by"))
    if not object_ids:
        raise ValueError("No finite tangent objects available for basis construction.")
    bx = np.stack([np.asarray(payload[oid]["bx"], dtype=np.float64) for oid in object_ids], axis=1)
    by = np.stack([np.asarray(payload[oid]["by"], dtype=np.float64) for oid in object_ids], axis=1)
    b = np.concatenate([bx, by], axis=1)
    c = b @ b.T
    vals, vecs = np.linalg.eigh(0.5 * (c + c.T))
    order = np.argsort(vals)[::-1]
    return vecs[:, order[: int(k)]]


def _basis_list(u_compact: np.ndarray, seed: int) -> list[BasisPayload]:
    rng = np.random.default_rng(int(seed))
    q_rand = _orth(rng.normal(size=u_compact.shape))
    perm = rng.permutation(u_compact.shape[0])
    q_shuf = _orth(u_compact[perm, :])
    return [
        BasisPayload("compact_k10", u_compact),
        BasisPayload("random_k10", q_rand[:, : u_compact.shape[1]]),
        BasisPayload("unit_shuffled_compact_k10", q_shuf[:, : u_compact.shape[1]]),
        BasisPayload("full_population", None),
    ]


def _project(basis: BasisPayload, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if basis.matrix is None:
        return arr
    return basis.matrix.T @ arr


def _project_j(basis: BasisPayload, j: np.ndarray) -> np.ndarray:
    if basis.matrix is None:
        return np.asarray(j, dtype=np.float64)
    return basis.matrix.T @ np.asarray(j, dtype=np.float64)


def _metric_stats(g: np.ndarray, *, abs_eps: float, rel_eps: float) -> dict[str, float | int]:
    gg = 0.5 * (np.asarray(g, dtype=np.float64) + np.asarray(g, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(gg)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    max_eval = float(vals[0]) if vals.size else 0.0
    eps = max(float(abs_eps), float(rel_eps) * max_eval)
    rank = int(np.sum(vals > eps))
    tr = float(np.sum(vals))
    det = float(np.prod(vals)) if vals.size == 2 else float("nan")
    cond = float(vals[0] / vals[-1]) if vals.size == 2 and vals[-1] > eps else float("inf")
    anis = float((vals[0] - vals[-1]) / (vals[0] + vals[-1] + 1e-12)) if vals.size == 2 else float("nan")
    angle = float(np.arctan2(vecs[1, 0], vecs[0, 0])) if vals.size == 2 else float("nan")
    return {
        "lambda_1": float(vals[0]) if vals.size else float("nan"),
        "lambda_2": float(vals[1]) if vals.size > 1 else float("nan"),
        "rank": rank,
        "trace": tr,
        "determinant": det,
        "condition_number": cond,
        "anisotropy": anis,
        "principal_axis_angle_rad": angle,
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def _regression_stats(pred: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p = p[ok]
    y = y[ok]
    if p.size < 3 or float(np.var(p)) <= 1e-18 or float(np.var(y)) <= 1e-18:
        return {
            "n": int(p.size),
            "pearson_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "median_relative_abs_error": float("nan"),
        }
    slope, intercept = np.polyfit(p, y, deg=1)
    fit = slope * p + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    rel = np.abs(y - p) / np.maximum(np.abs(y), 1e-12)
    return {
        "n": int(p.size),
        "pearson_r": float(np.corrcoef(p, y)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot),
        "median_relative_abs_error": float(np.median(rel)),
    }


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(yp)
    y = y[ok]
    yp = yp[ok]
    if y.size < 2 or float(np.var(y)) <= 1e-18:
        return float("nan")
    return float(1.0 - np.sum((y - yp) ** 2) / np.sum((y - np.mean(y)) ** 2))


def _angular_error(delta_true: np.ndarray, delta_hat: np.ndarray) -> float:
    dt = np.asarray(delta_true, dtype=np.float64)
    dh = np.asarray(delta_hat, dtype=np.float64)
    den = float(np.linalg.norm(dt) * np.linalg.norm(dh))
    if den <= 1e-12:
        return float("nan")
    c = float(np.clip(np.dot(dt, dh) / den, -1.0, 1.0))
    return float(np.arccos(c))


def _shift_specs(meta: dict[str, Any]) -> list[dict[str, Any]]:
    dpx = float(meta["delta_model_px"])
    darc = float(meta["delta_arcmin"])
    return [
        {"shift_label": "+x", "response_key": "rx_p", "axis": "x", "direction": 1, "delta_px_x": dpx, "delta_px_y": 0.0, "delta_arcmin_x": darc, "delta_arcmin_y": 0.0},
        {"shift_label": "-x", "response_key": "rx_m", "axis": "x", "direction": -1, "delta_px_x": -dpx, "delta_px_y": 0.0, "delta_arcmin_x": -darc, "delta_arcmin_y": 0.0},
        {"shift_label": "+y", "response_key": "ry_p", "axis": "y", "direction": 1, "delta_px_x": 0.0, "delta_px_y": dpx, "delta_arcmin_x": 0.0, "delta_arcmin_y": darc},
        {"shift_label": "-y", "response_key": "ry_m", "axis": "y", "direction": -1, "delta_px_x": 0.0, "delta_px_y": -dpx, "delta_arcmin_x": 0.0, "delta_arcmin_y": -darc},
    ]


def _local_metrics(
    primary_payload: dict[str, dict[str, Any]],
    bases: list[BasisPayload],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metric_by_basis_object: dict[tuple[str, str], dict[str, Any]] = {}
    for oid in _valid_object_ids(primary_payload, required=("r0", "bx", "by")):
        meta = primary_payload[oid]
        j_full = np.stack(
            [np.asarray(meta["bx"], dtype=np.float64), np.asarray(meta["by"], dtype=np.float64)],
            axis=1,
        )
        for basis in bases:
            jk = _project_j(basis, j_full)
            g = jk.T @ jk
            stats = _metric_stats(g, abs_eps=float(args.metric_rank_abs_eps), rel_eps=float(args.metric_rank_rel_eps))
            row = {
                "object_id": oid,
                "image_id": int(meta.get("image_id", -1)),
                "trial_index": int(meta.get("trial_index", -1)),
                "time_index": int(meta.get("time_index", -1)),
                "basis_type": basis.name,
                "basis_k": 0 if basis.matrix is None else int(basis.matrix.shape[1]),
                "estimation_step_arcmin": float(meta["delta_arcmin"]),
                "estimation_step_model_px": float(meta["delta_model_px"]),
                "g_xx": float(g[0, 0]),
                "g_xy": float(g[0, 1]),
                "g_yy": float(g[1, 1]),
                **stats,
            }
            rows.append(row)
            metric_by_basis_object[(basis.name, oid)] = {"g": g, "jk": jk, "row": row}
    return rows, metric_by_basis_object


def _quadratic_prediction(
    payload_by_delta: dict[float, dict[str, dict[str, Any]]],
    metric_by_basis_object: dict[tuple[str, str], dict[str, Any]],
    bases: list[BasisPayload],
    primary_delta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_objects = set.intersection(*(set(payload.keys()) for payload in payload_by_delta.values()))
    for delta, payload in sorted(payload_by_delta.items()):
        for oid in sorted(common_objects):
            meta = payload[oid]
            if not all(key in meta and np.all(np.isfinite(meta[key])) for key in ("r0", "rx_p", "rx_m", "ry_p", "ry_m")):
                continue
            r0 = np.asarray(meta["r0"], dtype=np.float64)
            for basis in bases:
                metric_payload = metric_by_basis_object.get((basis.name, oid))
                if metric_payload is None:
                    continue
                g = np.asarray(metric_payload["g"], dtype=np.float64)
                for spec in _shift_specs(meta):
                    dr = np.asarray(meta[spec["response_key"]], dtype=np.float64) - r0
                    z = _project(basis, dr)
                    disp = np.array([spec["delta_px_x"], spec["delta_px_y"]], dtype=np.float64)
                    pred_sq = float(disp.T @ g @ disp)
                    actual_sq = float(np.dot(z, z))
                    eval_type = "cardinal_calibration" if abs(float(delta) - float(primary_delta)) < 1e-9 else "cardinal_step_sweep"
                    rows.append(
                        {
                            "object_id": oid,
                            "image_id": int(meta.get("image_id", -1)),
                            "basis_type": basis.name,
                            "estimation_step_arcmin": float(primary_delta),
                            "test_step_arcmin": float(delta),
                            "test_step_model_px": float(meta["delta_model_px"]),
                            "evaluation_type": eval_type,
                            "shift_label": spec["shift_label"],
                            "axis": spec["axis"],
                            "delta_px_x": float(spec["delta_px_x"]),
                            "delta_px_y": float(spec["delta_px_y"]),
                            "actual_squared_distance": actual_sq,
                            "metric_predicted_squared_distance": pred_sq,
                            "relative_abs_error": float(abs(actual_sq - pred_sq) / max(abs(actual_sq), 1e-12)),
                        }
                    )
    return rows


def _opposition(
    primary_payload: dict[str, dict[str, Any]],
    bases: list[BasisPayload],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for oid in _valid_object_ids(primary_payload, required=("r0", "rx_p", "rx_m", "ry_p", "ry_m")):
        meta = primary_payload[oid]
        r0 = np.asarray(meta["r0"], dtype=np.float64)
        for basis in bases:
            for axis, pos_key, neg_key in [("x", "rx_p", "rx_m"), ("y", "ry_p", "ry_m")]:
                z_pos = _project(basis, np.asarray(meta[pos_key], dtype=np.float64) - r0)
                z_neg = _project(basis, np.asarray(meta[neg_key], dtype=np.float64) - r0)
                rows.append(
                    {
                        "object_id": oid,
                        "image_id": int(meta.get("image_id", -1)),
                        "basis_type": basis.name,
                        "finite_difference_step_arcmin": float(meta["delta_arcmin"]),
                        "axis": axis,
                        "opposition_cosine": _cos(z_pos, -z_neg),
                        "norm_positive": float(np.linalg.norm(z_pos)),
                        "norm_negative": float(np.linalg.norm(z_neg)),
                    }
                )
    return rows


def _scaling(
    payload_by_delta: dict[float, dict[str, dict[str, Any]]],
    metric_by_basis_object: dict[tuple[str, str], dict[str, Any]],
    bases: list[BasisPayload],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_objects = sorted(set.intersection(*(set(payload.keys()) for payload in payload_by_delta.values())))
    for oid in common_objects:
        for basis in bases:
            metric_payload = metric_by_basis_object.get((basis.name, oid))
            if metric_payload is None:
                continue
            g = np.asarray(metric_payload["g"], dtype=np.float64)
            for axis, direction, response_key in [
                ("x", 1, "rx_p"),
                ("x", -1, "rx_m"),
                ("y", 1, "ry_p"),
                ("y", -1, "ry_m"),
            ]:
                steps: list[float] = []
                norms: list[float] = []
                pred_sq: list[float] = []
                actual_sq: list[float] = []
                for delta, payload in sorted(payload_by_delta.items()):
                    meta = payload[oid]
                    if not all(key in meta and np.all(np.isfinite(meta[key])) for key in ("r0", response_key)):
                        continue
                    r0 = np.asarray(meta["r0"], dtype=np.float64)
                    z = _project(basis, np.asarray(meta[response_key], dtype=np.float64) - r0)
                    dpx = float(meta["delta_model_px"]) * float(direction)
                    disp = np.array([dpx, 0.0], dtype=np.float64) if axis == "x" else np.array([0.0, dpx], dtype=np.float64)
                    steps.append(float(delta))
                    norms.append(float(np.linalg.norm(z)))
                    actual_sq.append(float(np.dot(z, z)))
                    pred_sq.append(float(disp.T @ g @ disp))
                if len(steps) < 3:
                    continue
                x = np.asarray(steps, dtype=np.float64)
                y = np.asarray(norms, dtype=np.float64)
                slope, intercept = np.polyfit(x, y, deg=1)
                fit = slope * x + intercept
                ss_res = float(np.sum((y - fit) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                metric_stats = _regression_stats(np.asarray(pred_sq), np.asarray(actual_sq))
                rows.append(
                    {
                        "object_id": oid,
                        "basis_type": basis.name,
                        "axis": axis,
                        "direction": int(direction),
                        "n_steps": len(steps),
                        "steps_arcmin": ";".join(f"{v:g}" for v in steps),
                        "norms": ";".join(f"{v:.6g}" for v in norms),
                        "norm_slope": float(slope),
                        "norm_intercept": float(intercept),
                        "norm_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
                        "norm_monotonic_increasing": bool(np.all(np.diff(y) >= -1e-12)),
                        "metric_squared_distance_r2": float(metric_stats["r2"]),
                        "metric_squared_distance_slope": float(metric_stats["slope"]),
                        "metric_squared_distance_median_relative_abs_error": float(metric_stats["median_relative_abs_error"]),
                    }
                )
    return rows


def _coordinate_recovery(
    payload_by_delta: dict[float, dict[str, dict[str, Any]]],
    metric_by_basis_object: dict[tuple[str, str], dict[str, Any]],
    bases: list[BasisPayload],
    primary_delta: float,
    ridge_fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_objects = sorted(set.intersection(*(set(payload.keys()) for payload in payload_by_delta.values())))
    for delta, payload in sorted(payload_by_delta.items()):
        for oid in common_objects:
            meta = payload[oid]
            if not all(key in meta and np.all(np.isfinite(meta[key])) for key in ("r0", "rx_p", "rx_m", "ry_p", "ry_m")):
                continue
            r0 = np.asarray(meta["r0"], dtype=np.float64)
            for basis in bases:
                metric_payload = metric_by_basis_object.get((basis.name, oid))
                if metric_payload is None:
                    continue
                g = np.asarray(metric_payload["g"], dtype=np.float64)
                jk = np.asarray(metric_payload["jk"], dtype=np.float64)
                lam = float(ridge_fraction) * float(np.trace(g) / 2.0 + 1e-12)
                inv = np.linalg.pinv(g + lam * np.eye(2))
                for spec in _shift_specs(meta):
                    dr = np.asarray(meta[spec["response_key"]], dtype=np.float64) - r0
                    z = _project(basis, dr)
                    rhs = jk.T @ z
                    delta_hat = inv @ rhs
                    delta_true = np.array([spec["delta_px_x"], spec["delta_px_y"]], dtype=np.float64)
                    eval_type = "cardinal_calibration" if abs(float(delta) - float(primary_delta)) < 1e-9 else "cardinal_step_sweep"
                    rows.append(
                        {
                            "object_id": oid,
                            "image_id": int(meta.get("image_id", -1)),
                            "basis_type": basis.name,
                            "evaluation_type": eval_type,
                            "test_step_arcmin": float(delta),
                            "shift_label": spec["shift_label"],
                            "delta_true_px_x": float(delta_true[0]),
                            "delta_true_px_y": float(delta_true[1]),
                            "delta_hat_px_x": float(delta_hat[0]),
                            "delta_hat_px_y": float(delta_hat[1]),
                            "error_px": float(np.linalg.norm(delta_hat - delta_true)),
                            "angular_error_rad": _angular_error(delta_true, delta_hat),
                            "magnitude_true_px": float(np.linalg.norm(delta_true)),
                            "magnitude_hat_px": float(np.linalg.norm(delta_hat)),
                            "magnitude_error_px": float(abs(np.linalg.norm(delta_hat) - np.linalg.norm(delta_true))),
                            "ridge_lambda": lam,
                        }
                    )
    return rows


def _composition(primary_payload: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    diagonal_keys = {"rxy_p", "rxy_m", "rdiag_p", "rdiag_m", "rxp_yp", "rxp_ym", "rxm_yp", "rxm_ym"}
    has_diagonal = any(any(str(k).lower() in diagonal_keys for k in meta.keys()) for meta in primary_payload.values())
    return [
        {
            "status": "not_run" if not has_diagonal else "ready",
            "reason": "diagonal translated responses are not present in current tangent-map cache" if not has_diagonal else "",
            "required_cache_extension": "Add finite responses for dx+dy, dx-dy, and/or arbitrary held-out displacement samples.",
        }
    ]


def _cross_image_regularities(local_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(local_metric_rows)
    if df.empty:
        return []
    compact = df[df["basis_type"] == "compact_k10"].copy()
    rows: list[dict[str, Any]] = []
    for image_id, g in compact.groupby("image_id"):
        rows.append(
            {
                "image_id": int(image_id),
                "n_contexts": int(len(g)),
                "rank2_fraction": float(np.mean(g["rank"].astype(int) >= 2)),
                "trace_median": float(np.nanmedian(g["trace"].to_numpy(dtype=np.float64))),
                "anisotropy_median": float(np.nanmedian(g["anisotropy"].to_numpy(dtype=np.float64))),
                "orientation_median_rad": float(np.nanmedian(g["principal_axis_angle_rad"].to_numpy(dtype=np.float64))),
            }
        )
    rows.append(
        {
            "image_id": "pooled",
            "n_contexts": int(len(compact)),
            "rank2_fraction": float(np.mean(compact["rank"].astype(int) >= 2)),
            "trace_median": float(np.nanmedian(compact["trace"].to_numpy(dtype=np.float64))),
            "anisotropy_median": float(np.nanmedian(compact["anisotropy"].to_numpy(dtype=np.float64))),
            "orientation_median_rad": float(np.nanmedian(compact["principal_axis_angle_rad"].to_numpy(dtype=np.float64))),
        }
    )
    return rows


def _summary(
    *,
    local_metric_rows: list[dict[str, Any]],
    quadratic_rows: list[dict[str, Any]],
    opposition_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    composition_rows: list[dict[str, Any]],
    coordinate_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local = pd.DataFrame(local_metric_rows)
    compact_metric = local[local["basis_type"] == "compact_k10"] if not local.empty else pd.DataFrame()
    if not compact_metric.empty:
        rank2_frac = float(np.mean(compact_metric["rank"].astype(int) >= 2))
        cond = compact_metric["condition_number"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        cond_med = float(np.nanmedian(cond))
        rows.append(
            {
                "test": "local_metric_rank2_fraction",
                "status": "pass" if rank2_frac >= float(args.rank2_fraction_min) else "fail",
                "metric_value": rank2_frac,
                "threshold": f">={float(args.rank2_fraction_min):.2f}",
                "details": "fraction of contexts with two compact metric eigenvalues above threshold",
            }
        )
        rows.append(
            {
                "test": "local_metric_conditioning",
                "status": "pass" if np.isfinite(cond_med) and cond_med <= float(args.condition_number_warn) else "warn",
                "metric_value": cond_med,
                "threshold": f"median condition <= {float(args.condition_number_warn):g}",
                "details": "condition number is computed only for rank-2 contexts",
            }
        )
    else:
        rows.append({"test": "local_metric_rank2_fraction", "status": "fail", "metric_value": float("nan"), "threshold": "compact rows present"})

    qdf = pd.DataFrame(quadratic_rows)
    for eval_type, label, min_r2 in [
        ("cardinal_calibration", "quadratic_cardinal_calibration_r2", args.quadratic_cardinal_r2_min),
        ("cardinal_step_sweep", "quadratic_step_sweep_prediction_r2", args.quadratic_step_r2_min),
    ]:
        block = qdf[(qdf["basis_type"] == "compact_k10") & (qdf["evaluation_type"] == eval_type)] if not qdf.empty else pd.DataFrame()
        stats = _regression_stats(
            block["metric_predicted_squared_distance"].to_numpy(dtype=np.float64) if not block.empty else np.array([]),
            block["actual_squared_distance"].to_numpy(dtype=np.float64) if not block.empty else np.array([]),
        )
        rows.append(
            {
                "test": label,
                "status": "pass" if np.isfinite(float(stats["r2"])) and float(stats["r2"]) >= float(min_r2) else "warn",
                "metric_value": float(stats["r2"]),
                "threshold": f"R2 >= {float(min_r2):.2f}",
                "details": f"n={stats['n']}; slope={stats['slope']}; pearson_r={stats['pearson_r']}",
            }
        )
    has_heldout_direction = False
    rows.append(
        {
            "test": "quadratic_direction_heldout_prediction",
            "status": "not_run" if not has_heldout_direction else "ready",
            "metric_value": "",
            "threshold": "requires diagonal/arbitrary translated responses",
            "details": "Current cache has cardinal +/-x,+/-y translations only.",
        }
    )

    odf = pd.DataFrame(opposition_rows)
    if not odf.empty:
        med = odf.groupby("basis_type")["opposition_cosine"].median().to_dict()
        compact = float(med.get("compact_k10", float("nan")))
        random = float(med.get("random_k10", float("nan")))
        shuf = float(med.get("unit_shuffled_compact_k10", float("nan")))
        rows.append(
            {
                "test": "opposition_vs_null",
                "status": "pass" if compact > random and compact > shuf else "warn",
                "metric_value": compact,
                "threshold": "compact median > random and unit-shuffled compact medians",
                "details": f"compact={compact:.4f}; random={random:.4f}; unit_shuffle={shuf:.4f}",
            }
        )

    sdf = pd.DataFrame(scaling_rows)
    if not sdf.empty:
        block = sdf[sdf["basis_type"] == "compact_k10"]
        norm_r2 = float(np.nanmedian(block["norm_r2"].to_numpy(dtype=np.float64)))
        mono = float(np.mean(block["norm_monotonic_increasing"].astype(bool)))
        metric_r2 = float(np.nanmedian(block["metric_squared_distance_r2"].to_numpy(dtype=np.float64)))
        rows.append(
            {
                "test": "scaling_norm_r2",
                "status": "pass" if norm_r2 >= float(args.scaling_r2_min) and mono >= 0.5 else "warn",
                "metric_value": norm_r2,
                "threshold": f"median norm R2 >= {float(args.scaling_r2_min):.2f}; monotonic fraction >= 0.5",
                "details": f"monotonic_fraction={mono:.4f}",
            }
        )
        rows.append(
            {
                "test": "scaling_metric_squared_distance_r2",
                "status": "pass" if metric_r2 >= float(args.quadratic_step_r2_min) else "warn",
                "metric_value": metric_r2,
                "threshold": f"median per-object metric squared-distance R2 >= {float(args.quadratic_step_r2_min):.2f}",
                "details": "Uses primary-step metric to predict squared distances across step sweep.",
            }
        )

    cstatus = str(composition_rows[0]["status"]) if composition_rows else "not_run"
    rows.append(
        {
            "test": "local_composition",
            "status": cstatus,
            "metric_value": "",
            "threshold": "requires diagonal translated responses",
            "details": "" if not composition_rows else str(composition_rows[0].get("reason", "")),
        }
    )

    rdf = pd.DataFrame(coordinate_rows)
    if not rdf.empty:
        for eval_type, label, min_r2 in [
            ("cardinal_calibration", "coordinate_recovery_cardinal_r2", args.coordinate_cardinal_r2_min),
            ("cardinal_step_sweep", "coordinate_recovery_step_sweep_r2", args.coordinate_step_r2_min),
        ]:
            block = rdf[(rdf["basis_type"] == "compact_k10") & (rdf["evaluation_type"] == eval_type)]
            r2x = _r2_score(block["delta_true_px_x"].to_numpy(dtype=np.float64), block["delta_hat_px_x"].to_numpy(dtype=np.float64))
            r2y = _r2_score(block["delta_true_px_y"].to_numpy(dtype=np.float64), block["delta_hat_px_y"].to_numpy(dtype=np.float64))
            mean_r2 = float(np.nanmean([r2x, r2y]))
            rows.append(
                {
                    "test": label,
                    "status": "pass" if np.isfinite(mean_r2) and mean_r2 >= float(min_r2) else "warn",
                    "metric_value": mean_r2,
                    "threshold": f"mean(R2_x,R2_y) >= {float(min_r2):.2f}",
                    "details": f"R2_x={r2x:.4f}; R2_y={r2y:.4f}; median angular error={np.nanmedian(block['angular_error_rad'].to_numpy(dtype=np.float64)):.4f} rad",
                }
            )
    rows.append(
        {
            "test": "cross_image_metric_regularity",
            "status": "pass" if not compact_metric.empty else "not_run",
            "metric_value": "" if compact_metric.empty else float(np.nanmedian(compact_metric["anisotropy"].to_numpy(dtype=np.float64))),
            "threshold": "reported, not required to be universal across images",
            "details": "Summarizes G/trace anisotropy and orientation distributions by image/context.",
        }
    )
    return rows


def _plot_summary(
    out_root: Path,
    local_metric_rows: list[dict[str, Any]],
    quadratic_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    coordinate_rows: list[dict[str, Any]],
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.2))
    local = pd.DataFrame(local_metric_rows)
    compact_local = local[local["basis_type"] == "compact_k10"] if not local.empty else pd.DataFrame()
    if not compact_local.empty:
        axs[0, 0].hist(compact_local["anisotropy"].to_numpy(dtype=np.float64), bins=20, color="#2f5f9f", alpha=0.85)
    axs[0, 0].set_title("Local compact metric anisotropy", loc="left", fontweight="bold", fontsize=9)
    axs[0, 0].set_xlabel("anisotropy")
    axs[0, 0].set_ylabel("contexts")

    qdf = pd.DataFrame(quadratic_rows)
    qblock = qdf[(qdf["basis_type"] == "compact_k10") & (qdf["evaluation_type"] == "cardinal_step_sweep")] if not qdf.empty else pd.DataFrame()
    if not qblock.empty:
        axs[0, 1].scatter(qblock["metric_predicted_squared_distance"], qblock["actual_squared_distance"], s=8, alpha=0.45, color="#7b5ea7")
        maxv = float(np.nanmax([qblock["metric_predicted_squared_distance"].max(), qblock["actual_squared_distance"].max()]))
        axs[0, 1].plot([0, maxv], [0, maxv], color="#8d8d8d", lw=1)
    axs[0, 1].set_title("Metric-predicted distances", loc="left", fontweight="bold", fontsize=9)
    axs[0, 1].set_xlabel("predicted squared distance")
    axs[0, 1].set_ylabel("actual squared distance")

    sdf = pd.DataFrame(scaling_rows)
    sblock = sdf[sdf["basis_type"] == "compact_k10"] if not sdf.empty else pd.DataFrame()
    if not sblock.empty:
        axs[1, 0].hist(sblock["norm_r2"].to_numpy(dtype=np.float64), bins=20, color="#4d8f62", alpha=0.85)
    axs[1, 0].set_title("Step scaling", loc="left", fontweight="bold", fontsize=9)
    axs[1, 0].set_xlabel("norm-vs-step R2")
    axs[1, 0].set_ylabel("object-axis directions")

    rdf = pd.DataFrame(coordinate_rows)
    rblock = rdf[(rdf["basis_type"] == "compact_k10") & (rdf["evaluation_type"] == "cardinal_step_sweep")] if not rdf.empty else pd.DataFrame()
    if not rblock.empty:
        axs[1, 1].scatter(rblock["delta_true_px_x"], rblock["delta_hat_px_x"], s=8, alpha=0.5, label="x", color="#2f5f9f")
        axs[1, 1].scatter(rblock["delta_true_px_y"], rblock["delta_hat_px_y"], s=8, alpha=0.5, label="y", color="#c44e52")
        lim = float(np.nanmax(np.abs(rblock[["delta_true_px_x", "delta_true_px_y", "delta_hat_px_x", "delta_hat_px_y"]].to_numpy(dtype=np.float64))))
        axs[1, 1].plot([-lim, lim], [-lim, lim], color="#8d8d8d", lw=1)
        axs[1, 1].legend(frameon=False, fontsize=7)
    axs[1, 1].set_title("Metric-normalized recovery", loc="left", fontweight="bold", fontsize=9)
    axs[1, 1].set_xlabel("true displacement (px)")
    axs[1, 1].set_ylabel("recovered displacement (px)")

    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.18)
    fig.tight_layout()
    _save_fig(fig, out_root / "figures" / "metric_structure_summary")


def run(args: argparse.Namespace) -> None:
    tfts_root = Path(args.tfts_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    deltas, payload_by_delta = _load_tangent_maps(tfts_root)
    primary_delta = _nearest(deltas, float(args.primary_delta))
    primary_payload = payload_by_delta[primary_delta]
    u_compact = _basis_from_payload(primary_payload, int(args.k))
    bases = _basis_list(u_compact, int(args.seed))

    local_metric_rows, metric_by_basis_object = _local_metrics(primary_payload, bases, args)
    quadratic_rows = _quadratic_prediction(payload_by_delta, metric_by_basis_object, bases, primary_delta)
    opposition_rows = _opposition(primary_payload, bases)
    scaling_rows = _scaling(payload_by_delta, metric_by_basis_object, bases)
    composition_rows = _composition(primary_payload)
    coordinate_rows = _coordinate_recovery(
        payload_by_delta,
        metric_by_basis_object,
        bases,
        primary_delta,
        ridge_fraction=float(args.coordinate_ridge_fraction),
    )
    regularity_rows = _cross_image_regularities(local_metric_rows)
    summary_rows = _summary(
        local_metric_rows=local_metric_rows,
        quadratic_rows=quadratic_rows,
        opposition_rows=opposition_rows,
        scaling_rows=scaling_rows,
        composition_rows=composition_rows,
        coordinate_rows=coordinate_rows,
        args=args,
    )

    outputs = {
        "metric_structure_local_metric.csv": local_metric_rows,
        "metric_structure_quadratic_prediction.csv": quadratic_rows,
        "metric_structure_opposition.csv": opposition_rows,
        "metric_structure_scaling.csv": scaling_rows,
        "metric_structure_composition.csv": composition_rows,
        "metric_structure_coordinate_recovery.csv": coordinate_rows,
        "metric_structure_cross_image_regularities.csv": regularity_rows,
        "metric_structure_summary.csv": summary_rows,
    }
    for name, rows in outputs.items():
        _write_csv(out_root / name, rows)
        _write_csv(out_root / "tables" / name, rows)
    _plot_summary(out_root, local_metric_rows, quadratic_rows, scaling_rows, coordinate_rows)
    manifest = {
        "status": "ok",
        "analysis": "compact_retinal_translation_geometry_metric_structure_validation",
        "tfts_root": str(tfts_root.resolve()),
        "out_root": str(out_root.resolve()),
        "primary_delta_requested_arcmin": float(args.primary_delta),
        "primary_delta_used_arcmin": float(primary_delta),
        "available_deltas_arcmin": [float(v) for v in sorted(deltas)],
        "basis_k": int(args.k),
        "basis_source": "pooled compact tangent basis from primary delta",
        "null_bases": ["random_k10", "unit_shuffled_compact_k10"],
        "rf_readout_preserving_metric_null": "not_run_current_cache_unit_mapping_not_implemented",
        "diagonal_or_arbitrary_heldout_translations": "not_available_current_tangent_map_cache",
        "outputs": sorted(outputs.keys()) + ["figures/metric_structure_summary.png", "figures/metric_structure_summary.pdf"],
    }
    _write_json(out_root / "metric_structure_manifest.json", manifest)
    print(f"Wrote metric-structure validation outputs to {out_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--primary-delta", type=float, default=0.25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-rank-abs-eps", type=float, default=1e-12)
    parser.add_argument("--metric-rank-rel-eps", type=float, default=1e-6)
    parser.add_argument("--rank2-fraction-min", type=float, default=0.50)
    parser.add_argument("--condition-number-warn", type=float, default=1e3)
    parser.add_argument("--quadratic-cardinal-r2-min", type=float, default=0.80)
    parser.add_argument("--quadratic-step-r2-min", type=float, default=0.50)
    parser.add_argument("--scaling-r2-min", type=float, default=0.50)
    parser.add_argument("--coordinate-cardinal-r2-min", type=float, default=0.80)
    parser.add_argument("--coordinate-step-r2-min", type=float, default=0.50)
    parser.add_argument("--coordinate-ridge-fraction", type=float, default=1e-6)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

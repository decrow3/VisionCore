#!/usr/bin/env python3
"""Panel F natural-image spatial-content diagnostic.

This runner asks whether structured natural-image history objects route finite
translation-induced response changes through the compact, image-disjoint
tangent geometry from Panels C-E.

Primary metric
--------------
For each object and displacement scale:

    tangent_subspace_fraction = ||P_B delta r||^2 / ||delta r||^2

where B is built from translation tangents of other image identities.  The
metric is intentionally unweighted: no Fisher/noise model, no decoder.

Typical smoke run
-----------------
    .venv/bin/python declan/fig4_cov_TFTS/run_panelF_natural_structure.py \
        --mode smoke --model-device cuda:0 --max-objects 8

Outputs
-------
    panelF_natural_structure_scale_sweep.csv
    panelF_image_structure_metrics.csv
    panelF_empirical_event_ranges.csv
    panelF_phase_scramble_diagnostic.csv
    panelF_phase_scramble_audit.csv
    panelF_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from VisionCore.paths import VISIONCORE_ROOT
from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import (
    _load_twin_context,
    _movie_to_thw,
    _predict_rate_from_history,
    _shift_movie_subpixel,
)


DEFAULT_TFTS_ROOT = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
DEFAULT_OUT_DIR = VISIONCORE_ROOT / "outputs" / "panelF_natural_structure"
DEFAULT_FEM_RANGES = VISIONCORE_ROOT / "outputs" / "panel_f_covariance_overlap" / "panelF_fem_ranges.json"


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    image_id: int
    payload: dict[str, Any]
    structure: dict[str, float]
    structure_group: str
    structure_quantile: float


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _nearest_key(keys: list[float], target: float) -> float:
    if not keys:
        raise ValueError("No keys available")
    return float(min(keys, key=lambda k: abs(float(k) - float(target))))


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.nanmean(arr))) / sd


def _thin_orthonormal_basis(mat: np.ndarray, k: int, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float64)
    keep = np.all(np.isfinite(x), axis=0) & (np.linalg.norm(x, axis=0) > eps)
    x = x[:, keep]
    if x.size == 0:
        raise ValueError("No finite tangent columns available for basis")
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s * s > eps))
    if rank <= 0:
        raise ValueError("Tangent basis has numerical rank zero")
    return u[:, : min(int(k), rank)].astype(np.float64, copy=False)


def _project_energy_fraction(vec: np.ndarray, basis: np.ndarray, eps: float) -> tuple[float, float, float, bool]:
    v = np.asarray(vec, dtype=np.float64)
    den = float(np.sum(v * v))
    low_signal = (not np.isfinite(den)) or den <= float(eps)
    if low_signal:
        return float("nan"), float("nan"), den, True
    coeff = basis.T @ v
    captured = float(np.sum(coeff * coeff))
    frac = float(captured / den)
    return frac, captured, den, False


def _linear_r2(true_vecs: list[np.ndarray], pred_vecs: list[np.ndarray], eps: float) -> float:
    if not true_vecs or not pred_vecs:
        return float("nan")
    yt = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in true_vecs])
    yp = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in pred_vecs])
    keep = np.isfinite(yt) & np.isfinite(yp)
    if int(np.sum(keep)) < 10:
        return float("nan")
    yt = yt[keep]
    yp = yp[keep]
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= eps:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _history_structure_metrics(history: np.ndarray) -> dict[str, float]:
    """Small, auditable natural-structure scores from the current history frame."""
    h = np.asarray(history, dtype=np.float64)
    frame = h[0] if h.ndim == 3 else np.squeeze(h)
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D current frame, got {frame.shape}")
    x = frame - float(np.mean(frame))
    rms = float(np.std(x))

    gx = np.diff(x, axis=1, prepend=x[:, :1])
    gy = np.diff(x, axis=0, prepend=x[:1, :])
    grad2 = gx * gx + gy * gy
    gradient_rms = float(np.sqrt(np.mean(grad2)))

    lap = (
        -4.0 * x
        + np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
    )
    laplacian_rms = float(np.sqrt(np.mean(lap * lap)))

    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    det_term = np.sqrt(max((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    orientation_coherence = float(det_term / (trace + 1e-12))

    fy = np.fft.fftfreq(x.shape[0])
    fx = np.fft.fftfreq(x.shape[1])
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    power = np.abs(np.fft.fft2(x)) ** 2
    total_power = float(np.sum(power) + 1e-12)
    high_freq_fraction = float(np.sum(power[rr > np.percentile(rr, 70)]) / total_power)
    mid_freq_fraction = float(np.sum(power[(rr > np.percentile(rr, 35)) & (rr <= np.percentile(rr, 70))]) / total_power)

    # Primary proxy: edge/phase organization beyond contrast-energy covariates.
    raw_structure_score = float(
        np.log1p(gradient_rms)
        + np.log1p(laplacian_rms)
        + orientation_coherence
    )

    return {
        "mean_luminance": float(np.mean(frame)),
        "rms_contrast": rms,
        "gradient_rms": gradient_rms,
        "laplacian_rms": laplacian_rms,
        "orientation_coherence": orientation_coherence,
        "high_freq_fraction": high_freq_fraction,
        "mid_freq_fraction": mid_freq_fraction,
        "raw_structure_score": raw_structure_score,
    }


def _amplitude_spectrum_relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    fa = np.abs(np.fft.fft2(aa))
    fb = np.abs(np.fft.fft2(bb))
    return float(np.linalg.norm(fa - fb) / (np.linalg.norm(fa) + float(eps)))


def _phase_scramble_frame(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global Fourier phase scramble preserving a frame's amplitude spectrum."""
    x = np.asarray(frame, dtype=np.float64)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    centered = x - mu
    amp = np.abs(np.fft.fft2(centered))
    random_phase = np.angle(np.fft.fft2(rng.normal(size=x.shape)))
    random_phase[0, 0] = 0.0
    y = np.fft.ifft2(amp * np.exp(1j * random_phase)).real
    y = y - float(np.mean(y))
    y = y / (float(np.std(y)) + 1e-12) * sigma
    y = y + mu
    return y.astype(np.float32, copy=False)


def _phase_scramble_history(
    history: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Scramble spatial phase independently per lag while preserving per-frame spectra."""
    h = np.asarray(history, dtype=np.float32)
    if h.ndim != 3:
        raise ValueError(f"Expected history shape (T,H,W), got {h.shape}")
    out = np.empty_like(h, dtype=np.float32)
    amp_errors: list[float] = []
    mean_errors: list[float] = []
    std_errors: list[float] = []
    for t in range(h.shape[0]):
        out[t] = _phase_scramble_frame(h[t], rng)
        amp_errors.append(_amplitude_spectrum_relative_error(h[t], out[t]))
        mean_errors.append(abs(float(np.mean(h[t])) - float(np.mean(out[t]))))
        std_errors.append(abs(float(np.std(h[t])) - float(np.std(out[t]))))
    audit = {
        "n_frames": int(h.shape[0]),
        "amplitude_relative_error_mean": float(np.nanmean(amp_errors)),
        "amplitude_relative_error_max": float(np.nanmax(amp_errors)),
        "mean_abs_error_mean": float(np.nanmean(mean_errors)),
        "std_abs_error_mean": float(np.nanmean(std_errors)),
        "intact_mean": float(np.mean(h)),
        "scrambled_mean": float(np.mean(out)),
        "intact_std": float(np.std(h)),
        "scrambled_std": float(np.std(out)),
    }
    return out, audit


def _residualize_structure(metrics: list[dict[str, float]]) -> np.ndarray:
    raw = np.asarray([m["raw_structure_score"] for m in metrics], dtype=np.float64)
    covars = np.column_stack(
        [
            np.ones(len(metrics), dtype=np.float64),
            _zscore(np.log1p([m["rms_contrast"] for m in metrics])),
            _zscore(np.log1p([m["gradient_rms"] for m in metrics])),
            _zscore([m["high_freq_fraction"] for m in metrics]),
            _zscore([m["mid_freq_fraction"] for m in metrics]),
        ]
    )
    keep = np.isfinite(raw) & np.all(np.isfinite(covars), axis=1)
    resid = np.full_like(raw, np.nan, dtype=np.float64)
    if int(np.sum(keep)) < covars.shape[1] + 2:
        resid[keep] = _zscore(raw[keep])
        return resid
    beta, *_ = np.linalg.lstsq(covars[keep], raw[keep], rcond=None)
    resid[keep] = raw[keep] - covars[keep] @ beta
    return resid


def _load_objects(tfts_root: Path, basis_delta: float, max_objects: int | None, seed: int, group_quantile: float) -> tuple[float, list[ObjectRecord], dict[str, Any]]:
    pkl_path = tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing tangent maps: {pkl_path}")
    with pkl_path.open("rb") as handle:
        cache = pickle.load(handle)
    delta_keys = [float(v) for v in cache["object_payload"].keys()]
    delta_key = _nearest_key(delta_keys, basis_delta)
    payload = cache["object_payload"][delta_key]
    object_ids = sorted(str(oid) for oid in payload.keys())

    metrics = [_history_structure_metrics(payload[oid]["history"]) for oid in object_ids]
    resid = _residualize_structure(metrics)
    finite = np.isfinite(resid)
    if not np.any(finite):
        raise ValueError("No finite residual structure scores")
    ranks = np.full_like(resid, np.nan, dtype=np.float64)
    order = np.argsort(resid[finite])
    finite_idx = np.where(finite)[0]
    denom = max(len(finite_idx) - 1, 1)
    ranks[finite_idx[order]] = np.arange(len(finite_idx), dtype=np.float64) / denom

    q = float(group_quantile)
    records: list[ObjectRecord] = []
    for idx, oid in enumerate(object_ids):
        m = dict(metrics[idx])
        m["residual_structure_score"] = float(resid[idx])
        m["structure_quantile"] = float(ranks[idx])
        group = "middle_structure"
        if ranks[idx] <= q:
            group = "low_structure_matched"
        elif ranks[idx] >= 1.0 - q:
            group = "high_structure"
        records.append(
            ObjectRecord(
                object_id=oid,
                image_id=int(payload[oid]["image_id"]),
                payload=payload[oid],
                structure=m,
                structure_group=group,
                structure_quantile=float(ranks[idx]),
            )
        )

    # Keep the extreme groups first, then middle as fill.  This makes smoke runs
    # include both groups with deterministic coverage.
    rng = np.random.default_rng(int(seed))
    high = [r for r in records if r.structure_group == "high_structure"]
    low = [r for r in records if r.structure_group == "low_structure_matched"]
    mid = [r for r in records if r.structure_group == "middle_structure"]
    rng.shuffle(high)
    rng.shuffle(low)
    rng.shuffle(mid)
    if max_objects is not None and len(records) > int(max_objects):
        target_each = max(1, int(max_objects) // 2)
        selected = high[:target_each] + low[:target_each]
        remaining_slots = int(max_objects) - len(selected)
        selected += (high[target_each:] + low[target_each:] + mid)[: max(0, remaining_slots)]
        records = sorted(selected, key=lambda r: r.object_id)

    meta = {
        "tangent_maps_path": pkl_path,
        "requested_basis_delta_arcmin": float(basis_delta),
        "resolved_basis_delta_arcmin": float(delta_key),
        "n_objects_available": int(len(object_ids)),
        "n_objects_selected": int(len(records)),
        "structure_group_quantile": q,
        "structure_score": "residualized raw_structure_score after RMS/gradient/frequency covariates",
    }
    return delta_key, records, meta


def _basis_for_record(record: ObjectRecord, records: list[ObjectRecord], basis_k: int) -> np.ndarray:
    train = [r for r in records if int(r.image_id) != int(record.image_id)]
    if len(train) < 2:
        train = [r for r in records if r.object_id != record.object_id]
    bx = np.stack([np.asarray(r.payload["bx"], dtype=np.float64) for r in train], axis=1)
    by = np.stack([np.asarray(r.payload["by"], dtype=np.float64) for r in train], axis=1)
    return _thin_orthonormal_basis(np.concatenate([bx, by], axis=1), k=basis_k)


def _unit_shuffle_basis(record: ObjectRecord, records: list[ObjectRecord], basis_k: int, rng: np.random.Generator) -> np.ndarray:
    train = [r for r in records if int(r.image_id) != int(record.image_id)]
    if len(train) < 2:
        train = [r for r in records if r.object_id != record.object_id]
    bx = np.stack([np.asarray(r.payload["bx"], dtype=np.float64) for r in train], axis=1)
    by = np.stack([np.asarray(r.payload["by"], dtype=np.float64) for r in train], axis=1)
    mat = np.concatenate([bx, by], axis=1)
    mat_shuf = np.stack([col[rng.permutation(col.shape[0])] for col in mat.T], axis=1)
    return _thin_orthonormal_basis(mat_shuf, k=basis_k)


def _random_basis(n_units: int, basis_k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(int(n_units), int(basis_k))))
    return q[:, : int(basis_k)].astype(np.float64, copy=False)


def _directions(n: int) -> list[tuple[float, float]]:
    if int(n) <= 0:
        raise ValueError("n_directions must be positive")
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return [(float(np.cos(t)), float(np.sin(t))) for t in theta]


def _evaluate_record(
    ctx: Any,
    record: ObjectRecord,
    records: list[ObjectRecord],
    *,
    model_device: str,
    ppd: float,
    displacement_arcmin: float,
    n_directions: int,
    basis_k: int,
    basis_types: list[str],
    null_seed: int,
    low_signal_eps: float,
    history_override: np.ndarray | None = None,
    image_condition: str = "intact_natural",
    phase_scramble_seed: int | None = None,
    include_local_linear_r2: bool = True,
) -> list[dict[str, Any]]:
    history = np.asarray(
        record.payload["history"] if history_override is None else history_override,
        dtype=np.float32,
    )
    h_torch = _movie_to_thw(history).to(str(model_device))
    r0 = np.asarray(record.payload.get("r0"), dtype=np.float64) if history_override is None else None
    if r0 is None or not np.isfinite(r0).all():
        r0 = _predict_rate_from_history(ctx, history, model_device=str(model_device))
    bx = np.asarray(record.payload["bx"], dtype=np.float64)
    by = np.asarray(record.payload["by"], dtype=np.float64)
    scale_px = float(displacement_arcmin) * float(ppd) / 60.0

    dr_by_direction: list[np.ndarray] = []
    pred_by_direction: list[np.ndarray] = []
    for ux, uy in _directions(n_directions):
        dx_px = scale_px * ux
        dy_px = scale_px * uy
        shifted = _shift_movie_subpixel(h_torch, dx_px=dx_px, dy_px=dy_px).detach().cpu().numpy()
        r_shift = _predict_rate_from_history(ctx, shifted, model_device=str(model_device))
        dr_by_direction.append(np.asarray(r_shift - r0, dtype=np.float64))
        pred_by_direction.append(bx * dx_px + by * dy_px)

    n_units = int(dr_by_direction[0].shape[0])
    rng = np.random.default_rng(int(null_seed))
    bases: dict[str, np.ndarray] = {}
    if "true_tangent" in basis_types:
        bases["true_tangent"] = _basis_for_record(record, records, basis_k=basis_k)
    if "unit_shuffle" in basis_types:
        bases["unit_shuffle"] = _unit_shuffle_basis(record, records, basis_k=basis_k, rng=rng)
    if "random_subspace" in basis_types:
        bases["random_subspace"] = _random_basis(n_units, basis_k=basis_k, rng=rng)

    rows: list[dict[str, Any]] = []
    for basis_type, basis in bases.items():
        fracs: list[float] = []
        captured_vals: list[float] = []
        den_vals: list[float] = []
        low_flags: list[bool] = []
        for dr in dr_by_direction:
            frac, captured, den, low = _project_energy_fraction(dr, basis, eps=low_signal_eps)
            fracs.append(frac)
            captured_vals.append(captured)
            den_vals.append(den)
            low_flags.append(bool(low))

        valid_frac = np.asarray(fracs, dtype=np.float64)
        captured_arr = np.asarray(captured_vals, dtype=np.float64)
        den_arr = np.asarray(den_vals, dtype=np.float64)
        low_arr = np.asarray(low_flags, dtype=bool)
        mean_frac = float(np.nanmean(valid_frac)) if np.any(np.isfinite(valid_frac)) else float("nan")
        mean_captured = float(np.nanmean(captured_arr)) if np.any(np.isfinite(captured_arr)) else float("nan")
        mean_den = float(np.nanmean(den_arr)) if np.any(np.isfinite(den_arr)) else float("nan")
        low_rate = float(np.mean(low_arr)) if low_arr.size else float("nan")

        common = {
            "image_id": int(record.image_id),
            "object_id": record.object_id,
            "split": "image_disjoint_eval",
            "structure_group": record.structure_group,
            "structure_quantile": float(record.structure_quantile),
            "image_condition": str(image_condition),
            "displacement_arcmin": float(displacement_arcmin),
            "basis_type": basis_type,
            "basis_k": int(basis.shape[1]),
            "bootstrap_id_or_fold": "object",
            "n_directions": int(n_directions),
            "low_signal_fraction": low_rate,
            "delta_r_norm": mean_den,
            "low_signal_threshold": float(low_signal_eps),
            "phase_scramble_seed": "" if phase_scramble_seed is None else int(phase_scramble_seed),
        }
        rows.append({**common, "metric_name": "tangent_subspace_fraction", "metric_value": mean_frac})
        rows.append({**common, "metric_name": "orthogonal_fraction", "metric_value": float(1.0 - mean_frac) if np.isfinite(mean_frac) else float("nan")})
        rows.append({
            **common,
            "metric_name": "raw_tangent_subspace_sensitivity",
            "metric_value": float(mean_captured / max(float(displacement_arcmin) ** 2, 1e-12)) if np.isfinite(mean_captured) else float("nan"),
        })

    if not include_local_linear_r2:
        return rows

    lin_r2 = _linear_r2(dr_by_direction, pred_by_direction, eps=low_signal_eps)
    rows.append(
        {
            "image_id": int(record.image_id),
            "object_id": record.object_id,
            "split": "image_disjoint_eval",
            "structure_group": record.structure_group,
            "structure_quantile": float(record.structure_quantile),
            "image_condition": str(image_condition),
            "displacement_arcmin": float(displacement_arcmin),
            "basis_type": "local_xy_tangent_prediction",
            "basis_k": 2,
            "bootstrap_id_or_fold": "object",
            "n_directions": int(n_directions),
            "low_signal_fraction": float("nan"),
            "delta_r_norm": float("nan"),
            "low_signal_threshold": float(low_signal_eps),
            "metric_name": "unweighted_tangent_R2",
            "metric_value": lin_r2,
            "phase_scramble_seed": "" if phase_scramble_seed is None else int(phase_scramble_seed),
        }
    )
    return rows


def _bootstrap_high_minus_low(
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    base = [
        r for r in rows
        if r.get("metric_name") == "tangent_subspace_fraction"
        and r.get("structure_group") in {"high_structure", "low_structure_matched"}
        and np.isfinite(float(r.get("metric_value", float("nan"))))
    ]
    if not base:
        return []
    rng = np.random.default_rng(int(seed) + 500_000)
    out: list[dict[str, Any]] = []
    keys = sorted({(float(r["displacement_arcmin"]), str(r["basis_type"]), str(r["image_condition"])) for r in base})
    for disp, basis_type, image_condition in keys:
        block = [r for r in base if float(r["displacement_arcmin"]) == disp and str(r["basis_type"]) == basis_type and str(r["image_condition"]) == image_condition]
        high_ids = sorted({int(r["image_id"]) for r in block if r["structure_group"] == "high_structure"})
        low_ids = sorted({int(r["image_id"]) for r in block if r["structure_group"] == "low_structure_matched"})
        if not high_ids or not low_ids:
            continue

        values_by_image: dict[tuple[str, int], list[float]] = {}
        for r in block:
            group = str(r["structure_group"])
            img = int(r["image_id"])
            values_by_image.setdefault((group, img), []).append(float(r["metric_value"]))

        def _mean_for(ids: list[int], group: str) -> float:
            vals: list[float] = []
            for img in ids:
                per_image = values_by_image.get((group, int(img)), [])
                if per_image:
                    vals.append(float(np.nanmean(per_image)))
            return float(np.nanmean(vals)) if vals else float("nan")

        observed = _mean_for(high_ids, "high_structure") - _mean_for(low_ids, "low_structure_matched")
        common = {
            "image_id": -1,
            "object_id": "summary",
            "split": "image_bootstrap",
            "structure_group": "high_minus_low",
            "structure_quantile": float("nan"),
            "image_condition": image_condition,
            "displacement_arcmin": disp,
            "basis_type": basis_type,
            "basis_k": int(next((r.get("basis_k", -1) for r in block if str(r["basis_type"]) == basis_type), -1)),
            "n_high_images": int(len(high_ids)),
            "n_low_images": int(len(low_ids)),
            "metric_name": "high_minus_low_fraction",
        }
        out.append({**common, "bootstrap_id_or_fold": "observed", "metric_value": observed})
        for b in range(int(n_bootstrap)):
            hi_sample = rng.choice(high_ids, size=len(high_ids), replace=True).astype(int).tolist()
            lo_sample = rng.choice(low_ids, size=len(low_ids), replace=True).astype(int).tolist()
            diff = _mean_for(hi_sample, "high_structure") - _mean_for(lo_sample, "low_structure_matched")
            out.append({**common, "bootstrap_id_or_fold": int(b), "metric_value": diff})
    return out


def _bootstrap_phase_scramble_contrasts(
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    base = [
        r for r in rows
        if r.get("metric_name") == "tangent_subspace_fraction"
        and r.get("bootstrap_id_or_fold") == "object"
        and r.get("image_condition") in {"intact_natural", "phase_scrambled"}
        and r.get("structure_group") in {"high_structure", "low_structure_matched"}
        and np.isfinite(float(r.get("metric_value", float("nan"))))
    ]
    if not base:
        return []
    rng = np.random.default_rng(int(seed) + 900_000)
    out: list[dict[str, Any]] = []
    keys = sorted({(float(r["displacement_arcmin"]), str(r["basis_type"])) for r in base})

    def _image_values(block: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[float]]:
        values: dict[tuple[str, str, int], list[float]] = {}
        for r in block:
            key = (str(r["image_condition"]), str(r["structure_group"]), int(r["image_id"]))
            values.setdefault(key, []).append(float(r["metric_value"]))
        return values

    def _mean_for(values: dict[tuple[str, str, int], list[float]], ids: list[int], condition: str, group: str) -> float:
        vals: list[float] = []
        for img in ids:
            per_image = values.get((condition, group, int(img)), [])
            if per_image:
                vals.append(float(np.nanmean(per_image)))
        return float(np.nanmean(vals)) if vals else float("nan")

    for disp, basis_type in keys:
        block = [
            r for r in base
            if float(r["displacement_arcmin"]) == disp
            and str(r["basis_type"]) == basis_type
        ]
        values = _image_values(block)
        groups = ["high_structure", "low_structure_matched"]

        for group in groups:
            ids = sorted({
                img for condition, g, img in values
                if g == group and condition == "intact_natural"
                and (("phase_scrambled", group, img) in values)
            })
            if not ids:
                continue
            observed = _mean_for(values, ids, "intact_natural", group) - _mean_for(values, ids, "phase_scrambled", group)
            common = {
                "image_id": -1,
                "object_id": "summary",
                "split": "image_bootstrap",
                "structure_group": group,
                "structure_quantile": float("nan"),
                "image_condition": "intact_minus_phase_scrambled",
                "displacement_arcmin": float(disp),
                "basis_type": basis_type,
                "basis_k": int(next((r.get("basis_k", -1) for r in block if str(r["basis_type"]) == basis_type), -1)),
                "n_images": int(len(ids)),
                "metric_name": "intact_minus_phase_fraction",
            }
            out.append({**common, "bootstrap_id_or_fold": "observed", "metric_value": observed})
            for b in range(int(n_bootstrap)):
                sample = rng.choice(ids, size=len(ids), replace=True).astype(int).tolist()
                diff = _mean_for(values, sample, "intact_natural", group) - _mean_for(values, sample, "phase_scrambled", group)
                out.append({**common, "bootstrap_id_or_fold": int(b), "metric_value": diff})

        high_ids = sorted({
            img for condition, group, img in values
            if group == "high_structure" and condition == "intact_natural"
            and (("phase_scrambled", "high_structure", img) in values)
        })
        low_ids = sorted({
            img for condition, group, img in values
            if group == "low_structure_matched" and condition == "intact_natural"
            and (("phase_scrambled", "low_structure_matched", img) in values)
        })
        if not high_ids or not low_ids:
            continue

        def _attenuation(hi_ids: list[int], lo_ids: list[int]) -> float:
            intact_diff = (
                _mean_for(values, hi_ids, "intact_natural", "high_structure")
                - _mean_for(values, lo_ids, "intact_natural", "low_structure_matched")
            )
            phase_diff = (
                _mean_for(values, hi_ids, "phase_scrambled", "high_structure")
                - _mean_for(values, lo_ids, "phase_scrambled", "low_structure_matched")
            )
            return float(intact_diff - phase_diff)

        observed = _attenuation(high_ids, low_ids)
        common = {
            "image_id": -1,
            "object_id": "summary",
            "split": "image_bootstrap",
            "structure_group": "high_minus_low",
            "structure_quantile": float("nan"),
            "image_condition": "intact_minus_phase_scrambled",
            "displacement_arcmin": float(disp),
            "basis_type": basis_type,
            "basis_k": int(next((r.get("basis_k", -1) for r in block if str(r["basis_type"]) == basis_type), -1)),
            "n_high_images": int(len(high_ids)),
            "n_low_images": int(len(low_ids)),
            "metric_name": "phase_attenuation_high_minus_low_fraction",
        }
        out.append({**common, "bootstrap_id_or_fold": "observed", "metric_value": observed})
        for b in range(int(n_bootstrap)):
            hi_sample = rng.choice(high_ids, size=len(high_ids), replace=True).astype(int).tolist()
            lo_sample = rng.choice(low_ids, size=len(low_ids), replace=True).astype(int).tolist()
            out.append({**common, "bootstrap_id_or_fold": int(b), "metric_value": _attenuation(hi_sample, lo_sample)})
    return out


def _image_weighted_diff(rows: list[dict[str, Any]], value_key: str = "metric_value") -> tuple[float, int, int]:
    values_by_image: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        group = str(r.get("structure_group"))
        if group not in {"high_structure", "low_structure_matched"}:
            continue
        val = float(r.get(value_key, float("nan")))
        if not np.isfinite(val):
            continue
        values_by_image.setdefault((group, int(r["image_id"])), []).append(val)

    def _group_mean(group: str) -> tuple[float, int]:
        vals = [
            float(np.nanmean(v))
            for (g, _img), v in values_by_image.items()
            if g == group and len(v)
        ]
        return (float(np.nanmean(vals)), len(vals)) if vals else (float("nan"), 0)

    high_mean, n_high = _group_mean("high_structure")
    low_mean, n_low = _group_mean("low_structure_matched")
    return float(high_mean - low_mean), int(n_high), int(n_low)


def _write_robustness_summaries(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Posthoc summaries for the small checks before locking Panel F."""
    base = [
        dict(r)
        for r in rows
        if r.get("metric_name") == "tangent_subspace_fraction"
        and r.get("bootstrap_id_or_fold") == "object"
        and r.get("structure_group") in {"high_structure", "low_structure_matched"}
        and np.isfinite(float(r.get("metric_value", float("nan"))))
    ]
    sensitivity_rows: list[dict[str, Any]] = []
    loio_rows: list[dict[str, Any]] = []
    keys = sorted({(float(r["displacement_arcmin"]), str(r["basis_type"])) for r in base})
    for disp, basis_type in keys:
        block = [
            r for r in base
            if float(r["displacement_arcmin"]) == disp
            and str(r["basis_type"]) == basis_type
        ]
        if not block:
            continue

        observed, n_high, n_low = _image_weighted_diff(block)
        sensitivity_rows.append(
            {
                "basis_type": basis_type,
                "displacement_arcmin": disp,
                "policy": "observed_exclude_low_signal_directions",
                "high_minus_low_fraction": observed,
                "n_high_images": n_high,
                "n_low_images": n_low,
                "n_objects": len({str(r["object_id"]) for r in block}),
            }
        )

        zero_low_signal = []
        for r in block:
            rr = dict(r)
            low_rate = float(rr.get("low_signal_fraction", 0.0))
            if np.isfinite(low_rate):
                rr["metric_value_low_signal_as_zero"] = float(rr["metric_value"]) * max(0.0, 1.0 - low_rate)
            else:
                rr["metric_value_low_signal_as_zero"] = float(rr["metric_value"])
            zero_low_signal.append(rr)
        adjusted, n_high_adj, n_low_adj = _image_weighted_diff(
            zero_low_signal,
            value_key="metric_value_low_signal_as_zero",
        )
        sensitivity_rows.append(
            {
                "basis_type": basis_type,
                "displacement_arcmin": disp,
                "policy": "low_signal_directions_count_as_zero",
                "high_minus_low_fraction": adjusted,
                "n_high_images": n_high_adj,
                "n_low_images": n_low_adj,
                "n_objects": len({str(r["object_id"]) for r in zero_low_signal}),
            }
        )

        no_low_signal = [
            r for r in block
            if float(r.get("low_signal_fraction", 0.0)) <= 0.0
        ]
        clean, n_high_clean, n_low_clean = _image_weighted_diff(no_low_signal)
        sensitivity_rows.append(
            {
                "basis_type": basis_type,
                "displacement_arcmin": disp,
                "policy": "drop_any_object_with_low_signal_direction",
                "high_minus_low_fraction": clean,
                "n_high_images": n_high_clean,
                "n_low_images": n_low_clean,
                "n_objects": len({str(r["object_id"]) for r in no_low_signal}),
            }
        )

        image_ids = sorted({int(r["image_id"]) for r in block})
        diffs: list[float] = []
        for image_id in image_ids:
            kept = [r for r in block if int(r["image_id"]) != int(image_id)]
            diff, n_hi, n_lo = _image_weighted_diff(kept)
            diffs.append(diff)
            loio_rows.append(
                {
                    "basis_type": basis_type,
                    "displacement_arcmin": disp,
                    "left_out_image_id": int(image_id),
                    "high_minus_low_fraction": diff,
                    "n_high_images": n_hi,
                    "n_low_images": n_lo,
                }
            )
        finite = np.asarray([v for v in diffs if np.isfinite(v)], dtype=np.float64)
        if finite.size:
            loio_rows.append(
                {
                    "basis_type": basis_type,
                    "displacement_arcmin": disp,
                    "left_out_image_id": "summary",
                    "high_minus_low_fraction": observed,
                    "min_leave_one_out": float(np.min(finite)),
                    "max_leave_one_out": float(np.max(finite)),
                    "all_leave_one_out_positive": bool(np.all(finite > 0)),
                    "n_leave_one_out": int(finite.size),
                    "n_high_images": n_high,
                    "n_low_images": n_low,
                }
            )

    low_signal_path = out_dir / "panelF_low_signal_sensitivity.csv"
    loio_path = out_dir / "panelF_leave_one_image_out.csv"
    _write_csv(low_signal_path, sensitivity_rows)
    _write_csv(loio_path, loio_rows)
    return {
        "low_signal_sensitivity": str(low_signal_path),
        "leave_one_image_out": str(loio_path),
    }


def _write_empirical_ranges(out_csv: Path, ranges_path: Path | None) -> dict[str, Any] | None:
    if ranges_path is None or not ranges_path.exists():
        _write_csv(out_csv, [])
        return None
    payload = json.loads(ranges_path.read_text(encoding="utf-8"))
    rows = []
    for label, key in [("drift", "drift_band_arcmin"), ("microsaccade", "msac_band_arcmin")]:
        vals = payload.get(key)
        if vals is None:
            continue
        rows.append(
            {
                "event_type": label,
                "band_low_arcmin": float(vals[0]),
                "band_high_arcmin": float(vals[1]),
                "source_json": str(ranges_path),
                "subject": payload.get("subject", ""),
                "date": payload.get("date", ""),
                "n_events": payload.get("n_drift_epochs" if label == "drift" else "n_microsaccades", ""),
            }
        )
    _write_csv(out_csv, rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Panel F natural-image-structure scale sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    p.add_argument("--model-device", default="cuda")
    p.add_argument("--model-ppd", type=float, default=37.5)
    p.add_argument("--basis-delta-arcmin", type=float, default=0.25)
    p.add_argument("--basis-k", type=int, default=10)
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--displacement-arcmin", default=None)
    p.add_argument("--n-directions", type=int, default=None)
    p.add_argument("--basis-types", default="true_tangent,unit_shuffle,random_subspace")
    p.add_argument("--group-quantile", type=float, default=0.33)
    p.add_argument("--low-signal-threshold", type=float, default=1e-8)
    p.add_argument("--bootstrap-repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empirical-ranges-json", type=Path, default=DEFAULT_FEM_RANGES)
    p.add_argument("--include-phase-scramble", action="store_true", default=False,
                   help="Run spectrum-preserving global Fourier phase-scramble diagnostic.")
    p.add_argument("--phase-scramble-seeds", type=int, default=None,
                   help="Number of phase-scrambled controls per object when --include-phase-scramble is set.")
    p.add_argument("--phase-basis-types", default="true_tangent",
                   help="Comma-separated basis types to evaluate for phase-scrambled controls.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.displacement_arcmin is None:
        displacement_scales = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0] if args.mode == "smoke" else [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    else:
        displacement_scales = _parse_float_list(args.displacement_arcmin)
    n_directions = int(args.n_directions if args.n_directions is not None else (4 if args.mode == "smoke" else 8))
    max_objects = args.max_objects
    if max_objects is None:
        max_objects = 12 if args.mode == "smoke" else None
    n_bootstrap = int(args.bootstrap_repeats if args.bootstrap_repeats is not None else (100 if args.mode == "smoke" else 1000))
    basis_types = [s.strip() for s in str(args.basis_types).split(",") if s.strip()]
    phase_basis_types = [s.strip() for s in str(args.phase_basis_types).split(",") if s.strip()]
    if args.include_phase_scramble:
        for basis_type in phase_basis_types:
            if basis_type not in basis_types:
                basis_types.append(basis_type)
    phase_scramble_seeds = int(
        args.phase_scramble_seeds
        if args.phase_scramble_seeds is not None
        else (1 if args.include_phase_scramble else 0)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basis_delta, records, object_meta = _load_objects(
        Path(args.tfts_root),
        basis_delta=float(args.basis_delta_arcmin),
        max_objects=max_objects,
        seed=int(args.seed),
        group_quantile=float(args.group_quantile),
    )
    structure_rows = []
    for r in records:
        structure_rows.append(
            {
                "object_id": r.object_id,
                "image_id": int(r.image_id),
                "structure_group": r.structure_group,
                **r.structure,
            }
        )
    _write_csv(out_dir / "panelF_image_structure_metrics.csv", structure_rows)

    fem_ranges = _write_empirical_ranges(out_dir / "panelF_empirical_event_ranges.csv", args.empirical_ranges_json)

    print(f"Loaded {len(records)} objects from {args.tfts_root}")
    print(f"Displacements: {displacement_scales}; directions={n_directions}; basis_types={basis_types}", flush=True)
    if args.include_phase_scramble:
        print(f"Phase scramble: seeds={phase_scramble_seeds}; basis_types={phase_basis_types}", flush=True)
    ctx = _load_twin_context(model_device=str(args.model_device))

    all_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    phase_audit_rows: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        print(f"[{i + 1}/{len(records)}] object={record.object_id} group={record.structure_group}", flush=True)
        phase_histories: list[tuple[int, np.ndarray]] = []
        if args.include_phase_scramble:
            history = np.asarray(record.payload["history"], dtype=np.float32)
            for phase_seed_idx in range(max(phase_scramble_seeds, 0)):
                scramble_seed = int(args.seed) + 7_000_000 + i * 10_000 + phase_seed_idx
                scrambled, audit = _phase_scramble_history(history, np.random.default_rng(scramble_seed))
                phase_histories.append((phase_seed_idx, scrambled))
                phase_audit_rows.append(
                    {
                        "image_id": int(record.image_id),
                        "object_id": record.object_id,
                        "structure_group": record.structure_group,
                        "structure_quantile": float(record.structure_quantile),
                        "phase_scramble_seed": int(phase_seed_idx),
                        "phase_scramble_rng_seed": int(scramble_seed),
                        **audit,
                    }
                )
        for disp in displacement_scales:
            rows = _evaluate_record(
                ctx,
                record,
                records,
                model_device=str(args.model_device),
                ppd=float(args.model_ppd),
                displacement_arcmin=float(disp),
                n_directions=n_directions,
                basis_k=int(args.basis_k),
                basis_types=basis_types,
                null_seed=int(args.seed) + i * 1000 + int(round(float(disp) * 1000)),
                low_signal_eps=float(args.low_signal_threshold),
            )
            all_rows.extend(rows)
            for phase_seed_idx, phase_history in phase_histories:
                phase_rows.extend(
                    _evaluate_record(
                        ctx,
                        record,
                        records,
                        model_device=str(args.model_device),
                        ppd=float(args.model_ppd),
                        displacement_arcmin=float(disp),
                        n_directions=n_directions,
                        basis_k=int(args.basis_k),
                        basis_types=phase_basis_types,
                        null_seed=int(args.seed) + 3_000_000 + i * 1000 + phase_seed_idx * 100 + int(round(float(disp) * 1000)),
                        low_signal_eps=float(args.low_signal_threshold),
                        history_override=phase_history,
                        image_condition="phase_scrambled",
                        phase_scramble_seed=phase_seed_idx,
                        include_local_linear_r2=False,
                    )
                )

    all_rows.extend(_bootstrap_high_minus_low(all_rows, n_bootstrap=n_bootstrap, seed=int(args.seed)))
    _write_csv(out_dir / "panelF_natural_structure_scale_sweep.csv", all_rows)
    robustness_files = _write_robustness_summaries(out_dir, all_rows)

    phase_output_rows: list[dict[str, Any]] = []
    if args.include_phase_scramble and phase_rows:
        intact_for_phase = [
            r for r in all_rows
            if r.get("bootstrap_id_or_fold") == "object"
            and r.get("metric_name") in {"tangent_subspace_fraction", "orthogonal_fraction", "raw_tangent_subspace_sensitivity"}
            and r.get("basis_type") in set(phase_basis_types)
        ]
        phase_output_rows.extend(intact_for_phase)
        phase_output_rows.extend(phase_rows)
        phase_output_rows.extend(
            _bootstrap_high_minus_low(
                intact_for_phase + phase_rows,
                n_bootstrap=n_bootstrap,
                seed=int(args.seed) + 31,
            )
        )
        phase_output_rows.extend(
            _bootstrap_phase_scramble_contrasts(
                intact_for_phase + phase_rows,
                n_bootstrap=n_bootstrap,
                seed=int(args.seed) + 67,
            )
        )
    elif args.include_phase_scramble:
        phase_output_rows.append(
            {
                "status": "no_phase_rows_written",
                "reason": "No phase-scrambled object rows were produced.",
            }
        )
    _write_csv(out_dir / "panelF_phase_scramble_diagnostic.csv", phase_output_rows)
    _write_csv(out_dir / "panelF_phase_scramble_audit.csv", phase_audit_rows)

    low_signal_audit = []
    for disp in displacement_scales:
        for group in sorted({r.structure_group for r in records}):
            block = [
                row for row in all_rows
                if row.get("metric_name") == "tangent_subspace_fraction"
                and row.get("basis_type") == "true_tangent"
                and row.get("bootstrap_id_or_fold") == "object"
                and row.get("structure_group") == group
                and np.isclose(float(row.get("displacement_arcmin", np.nan)), float(disp))
            ]
            if block:
                low_signal_audit.append(
                    {
                        "displacement_arcmin": float(disp),
                        "structure_group": group,
                        "mean_low_signal_fraction": float(np.nanmean([float(r.get("low_signal_fraction", np.nan)) for r in block])),
                        "n_objects": int(len({str(r.get("object_id")) for r in block})),
                    }
                )
    _write_csv(out_dir / "panelF_low_signal_audit.csv", low_signal_audit)

    manifest = {
        "status": "completed",
        "script": str(Path(__file__).resolve()),
        "tfts_root": str(args.tfts_root),
        "output_dir": str(out_dir),
        "mode": args.mode,
        "model_device": str(args.model_device),
        "model_ppd": float(args.model_ppd),
        "basis_delta_arcmin": float(basis_delta),
        "basis_k": int(args.basis_k),
        "basis_types": basis_types,
        "displacement_arcmin": displacement_scales,
        "n_directions": int(n_directions),
        "low_signal_threshold": float(args.low_signal_threshold),
        "bootstrap_repeats": int(n_bootstrap),
        "phase_scramble_seeds": int(phase_scramble_seeds),
        "phase_basis_types": phase_basis_types,
        "object_source": object_meta,
        "n_selected_by_group": {
            group: int(sum(1 for r in records if r.structure_group == group))
            for group in sorted({r.structure_group for r in records})
        },
        "primary_metric": "unweighted tangent_subspace_fraction = ||P_B delta r||^2 / ||delta r||^2",
        "low_signal_policy": "rows with delta_r_norm <= low_signal_threshold are excluded from object-level fractions; exclusion rates are written to panelF_low_signal_audit.csv",
        "phase_scramble": "global_fourier_per_frame" if args.include_phase_scramble else "not_run",
        "empirical_ranges": fem_ranges,
        "output_files": {
            "scale_sweep": str(out_dir / "panelF_natural_structure_scale_sweep.csv"),
            "image_structure": str(out_dir / "panelF_image_structure_metrics.csv"),
            "empirical_ranges": str(out_dir / "panelF_empirical_event_ranges.csv"),
            "phase_scramble": str(out_dir / "panelF_phase_scramble_diagnostic.csv"),
            "phase_scramble_audit": str(out_dir / "panelF_phase_scramble_audit.csv"),
            "low_signal_audit": str(out_dir / "panelF_low_signal_audit.csv"),
            **robustness_files,
        },
    }
    _write_json(out_dir / "panelF_manifest.json", manifest)
    print(f"Saved Panel F natural-structure outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

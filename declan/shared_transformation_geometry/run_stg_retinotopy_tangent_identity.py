from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT
from eval.fixrsvp import get_fixrsvp_data
from eval.sta_ste import load_cached_sta_ste

from .utils import DEFAULT_OUT_ROOT


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(keep)) < 4:
        return float("nan")
    aa = aa[keep]
    bb = bb[keep]
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def _bootstrap_ci(vals: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boots = []
    for _ in range(int(n_bootstrap)):
        samp = rng.choice(v, size=v.size, replace=True)
        boots.append(float(np.mean(samp)))
    b = np.asarray(boots, dtype=np.float64)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _parse_k_list(raw: str) -> list[int]:
    parts = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    return [int(piece) for piece in parts] if parts else [0]


def _rf_from_sta_cache(session_id: str, expected_n_units: int) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    arrs = load_cached_sta_ste(session_id)
    if arrs is None:
        return None, None, "missing_sta_cache"
    stes = np.asarray(arrs.get("stes"), dtype=np.float64)
    if stes.ndim != 4:
        return None, None, "invalid_sta_cache_shape"
    n_units, _n_lags, h, w = stes.shape
    if int(n_units) != int(expected_n_units):
        return None, None, f"rf_unit_mismatch_cache_{n_units}_expected_{expected_n_units}"

    x_coords = np.arange(w, dtype=np.float64)
    y_coords = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x_coords, y_coords)
    rf_x = np.full(n_units, np.nan, dtype=np.float64)
    rf_y = np.full(n_units, np.nan, dtype=np.float64)
    for u in range(n_units):
        ste_u = stes[u]
        lag = int(np.nanargmax(np.nanstd(ste_u, axis=(1, 2))))
        im = ste_u[lag]
        ww = np.abs(im - np.nanmedian(im))
        ww = np.where(np.isfinite(ww), ww, 0.0)
        s = float(np.sum(ww))
        if s <= 1e-12:
            continue
        rf_x[u] = float(np.sum(ww * xx) / s)
        rf_y[u] = float(np.sum(ww * yy) / s)
    return rf_x, rf_y, f"ok_sta_cache_pixels_h{h}_w{w}"


def _load_twin_rf_positions(
    *,
    subject: str,
    date: str,
    session_root: Path,
    expected_n_units: int,
    model_device: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    session_id = f"{subject}_{date}"
    rf_csv = session_root / "source_twin" / "twin_unit_rf_positions.csv"
    if rf_csv.exists():
        with rf_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) == int(expected_n_units):
            # Backward-compatible read: prefer pixel columns, otherwise use existing normalized columns.
            if rows and ("rf_x_pixel" in rows[0]) and ("rf_y_pixel" in rows[0]):
                x = np.asarray([float(r["rf_x_pixel"]) for r in rows], dtype=np.float64)
                y = np.asarray([float(r["rf_y_pixel"]) for r in rows], dtype=np.float64)
                return x, y, "ok_twin_rf_csv_pixels"
            if rows and ("rf_x_norm" in rows[0]) and ("rf_y_norm" in rows[0]):
                x = np.asarray([float(r["rf_x_norm"]) for r in rows], dtype=np.float64)
                y = np.asarray([float(r["rf_y_norm"]) for r in rows], dtype=np.float64)
                return x, y, "ok_twin_rf_csv_norm_legacy"

    scripts_dir = VISIONCORE_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        import dill
        from scripts.spatial_info import get_spatial_readout
        from scripts.utils import get_model_and_dataset_configs
    except Exception as exc:
        return None, None, f"twin_rf_import_error_{type(exc).__name__}"

    outputs_path = VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl"
    if not outputs_path.exists():
        return None, None, "twin_rf_missing_mcfarland_outputs"

    model, _ = get_model_and_dataset_configs(mode="standard")
    model = model.to(str(model_device))
    model.model.eval()
    if hasattr(model, "names") and session_id not in model.names:
        return None, None, "twin_rf_session_not_in_model_names"

    with outputs_path.open("rb") as handle:
        outputs = dill.load(handle)

    readout = get_spatial_readout(model, outputs).to(str(model_device))
    if not hasattr(readout, "space_weights"):
        return None, None, "twin_rf_missing_space_weights"

    masks = np.asarray(readout.space_weights.detach().cpu().numpy(), dtype=np.float64)
    if masks.ndim != 4 or masks.shape[1] != 1:
        return None, None, "twin_rf_invalid_mask_shape"
    masks = masks[:, 0, :, :]
    n_units, h, w = masks.shape
    if int(n_units) != int(expected_n_units):
        return None, None, f"rf_unit_mismatch_twin_{n_units}_expected_{expected_n_units}"

    xx_px, yy_px = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rf_x_px = np.full(n_units, np.nan, dtype=np.float64)
    rf_y_px = np.full(n_units, np.nan, dtype=np.float64)
    rows: list[dict[str, object]] = []

    for u in range(n_units):
        ww = np.where(np.isfinite(masks[u]), masks[u], 0.0)
        mass = float(np.sum(ww))
        if mass <= 1e-12:
            rows.append(
                {
                    "unit_index": int(u),
                    "rf_x_pixel": float("nan"),
                    "rf_y_pixel": float("nan"),
                    "rf_x_deg": float("nan"),
                    "rf_y_deg": float("nan"),
                    "rf_space": "pixel",
                    "mask_mass": float(mass),
                }
            )
            continue

        x_px = float(np.sum(ww * xx_px) / mass)
        y_px = float(np.sum(ww * yy_px) / mass)
        rf_x_px[u] = x_px
        rf_y_px[u] = y_px
        rows.append(
            {
                "unit_index": int(u),
                "rf_x_pixel": float(x_px),
                "rf_y_pixel": float(y_px),
                "rf_x_deg": float("nan"),
                "rf_y_deg": float("nan"),
                "rf_space": "pixel",
                "mask_mass": float(mass),
            }
        )

    _write_csv(rf_csv, rows)
    return rf_x_px, rf_y_px, f"ok_twin_readout_mask_pixels_h{h}_w{w}"


def _load_rf_positions(
    *,
    source: str,
    subject: str,
    date: str,
    dataset_configs_path: Path,
    expected_n_units: int,
    use_cached_data: bool,
    session_root: Path,
    model_device: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if str(source) == "twin":
        return _load_twin_rf_positions(
            subject=subject,
            date=date,
            session_root=session_root,
            expected_n_units=expected_n_units,
            model_device=str(model_device),
        )

    data = get_fixrsvp_data(
        subject=subject,
        date=date,
        dataset_configs_path=str(dataset_configs_path),
        use_cached_data=bool(use_cached_data),
    )

    # Prefer explicit RF arrays if already present in canonical fixrsvp payload.
    key_pairs = [
        ("rf_x", "rf_y"),
        ("rf_center_x", "rf_center_y"),
        ("unit_rf_x", "unit_rf_y"),
        ("xpos", "ypos"),
    ]
    for kx, ky in key_pairs:
        if kx in data and ky in data:
            x = np.asarray(data[kx], dtype=np.float64).ravel()
            y = np.asarray(data[ky], dtype=np.float64).ravel()
            if x.size == expected_n_units and y.size == expected_n_units:
                return x, y, f"ok_fixrsvp_keys_{kx}_{ky}"

    session_id = f"{subject}_{date}"
    return _rf_from_sta_cache(session_id, expected_n_units=expected_n_units)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retinotopy identity test for STG tangent maps")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--source", type=str, choices=("recorded", "twin"), default="recorded")
    p.add_argument("--dataset-configs-path", type=Path, default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml")
    p.add_argument("--projection-k", type=str, default="0,1,2,3,5,10")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--use-cached-data", action="store_true", default=True)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-device", type=str, default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    session_id = f"{args.subject}_{args.date}"
    session_root = Path(args.out_dir) / session_id
    source_tag = str(args.source)
    per_image_out = session_root / (
        "stg_retinotopy_tangent_identity.csv"
        if source_tag == "recorded"
        else f"stg_retinotopy_tangent_identity_{source_tag}.csv"
    )
    summary_out = session_root / (
        "stg_retinotopy_tangent_identity_summary.csv"
        if source_tag == "recorded"
        else f"stg_retinotopy_tangent_identity_summary_{source_tag}.csv"
    )

    ks = _parse_k_list(args.projection_k)
    maps_by_k: dict[int, dict[str, Any]] = {}
    n_units_expected: int | None = None
    for k in ks:
        pkl = session_root / f"source_{args.source}" / f"projection_k{k}" / "stg_tangent_maps.pkl"
        if not pkl.exists() and int(k) == 0 and str(args.source) == "recorded":
            legacy = session_root / "source_recorded" / "stg_tangent_maps.pkl"
            if legacy.exists():
                pkl = legacy
        if not pkl.exists():
            continue
        with pkl.open("rb") as handle:
            payload = pickle.load(handle)
        images = payload.get("images", {})
        if not images:
            continue
        maps_by_k[int(k)] = images
        any_img = next(iter(images.values()))
        n_units_expected = int(any_img["bx"].shape[0])

    if not maps_by_k or n_units_expected is None:
        summary_rows = [
            {
                "session_id": session_id,
                "projection_k": -1,
                "n_images": 0,
                "n_units_with_rf": 0,
                "rf_coordinate_space": "unknown",
                "mean_corr_bx_rfx": float("nan"),
                "mean_corr_bx_rfy": float("nan"),
                "mean_corr_by_rfx": float("nan"),
                "mean_corr_by_rfy": float("nan"),
                "mean_rf_x": float("nan"),
                "mean_rf_y": float("nan"),
                "mean_abs_corr_bx_rfx": float("nan"),
                "mean_abs_corr_bx_rfy": float("nan"),
                "mean_abs_corr_by_rfx": float("nan"),
                "mean_abs_corr_by_rfy": float("nan"),
                "axis_selectivity_bx": float("nan"),
                "axis_selectivity_by": float("nan"),
                "bootstrap_ci_axis_selectivity_bx": json.dumps([float("nan"), float("nan")]),
                "bootstrap_ci_axis_selectivity_by": json.dumps([float("nan"), float("nan")]),
                "interpretation_label": "not_run_missing_tangent_maps",
            }
        ]
        _write_csv(summary_out, summary_rows)
        print(str(summary_out))
        return

    rf_x, rf_y, rf_status = _load_rf_positions(
        source=str(args.source),
        subject=str(args.subject),
        date=str(args.date),
        dataset_configs_path=Path(args.dataset_configs_path),
        expected_n_units=int(n_units_expected),
        use_cached_data=bool(args.use_cached_data),
        session_root=session_root,
        model_device=str(args.model_device),
    )

    per_image_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    if rf_x is None or rf_y is None:
        for k in sorted(maps_by_k):
            summary_rows.append(
                {
                    "session_id": session_id,
                    "projection_k": int(k),
                    "n_images": 0,
                    "n_units_with_rf": 0,
                    "rf_coordinate_space": "unknown",
                    "mean_corr_bx_rfx": float("nan"),
                    "mean_corr_bx_rfy": float("nan"),
                    "mean_corr_by_rfx": float("nan"),
                    "mean_corr_by_rfy": float("nan"),
                    "mean_rf_x": float("nan"),
                    "mean_rf_y": float("nan"),
                    "mean_abs_corr_bx_rfx": float("nan"),
                    "mean_abs_corr_bx_rfy": float("nan"),
                    "mean_abs_corr_by_rfx": float("nan"),
                    "mean_abs_corr_by_rfy": float("nan"),
                    "axis_selectivity_bx": float("nan"),
                    "axis_selectivity_by": float("nan"),
                    "bootstrap_ci_axis_selectivity_bx": json.dumps([float("nan"), float("nan")]),
                    "bootstrap_ci_axis_selectivity_by": json.dumps([float("nan"), float("nan")]),
                    "interpretation_label": "not_run_missing_rf_positions",
                    "retinotopy_test_status": str(rf_status),
                }
            )
        _write_csv(per_image_out, [])
        _write_csv(summary_out, summary_rows)
        print(str(summary_out))
        return

    for k in sorted(maps_by_k):
        images = maps_by_k[int(k)]
        k_rows: list[dict[str, object]] = []
        sel_bx_vals: list[float] = []
        sel_by_vals: list[float] = []

        for img_id in sorted(images):
            bx = np.asarray(images[img_id]["bx"], dtype=np.float64).ravel()
            by = np.asarray(images[img_id]["by"], dtype=np.float64).ravel()
            keep = np.isfinite(bx) & np.isfinite(by) & np.isfinite(rf_x) & np.isfinite(rf_y)
            if int(np.sum(keep)) < 4:
                continue

            bxk = bx[keep]
            byk = by[keep]
            xk = rf_x[keep]
            yk = rf_y[keep]

            corr_bx_rfx = _pearson(bxk, xk)
            corr_bx_rfy = _pearson(bxk, yk)
            corr_by_rfx = _pearson(byk, xk)
            corr_by_rfy = _pearson(byk, yk)

            abs_corr_bx_rfx = float(abs(corr_bx_rfx)) if np.isfinite(corr_bx_rfx) else float("nan")
            abs_corr_bx_rfy = float(abs(corr_bx_rfy)) if np.isfinite(corr_bx_rfy) else float("nan")
            abs_corr_by_rfx = float(abs(corr_by_rfx)) if np.isfinite(corr_by_rfx) else float("nan")
            abs_corr_by_rfy = float(abs(corr_by_rfy)) if np.isfinite(corr_by_rfy) else float("nan")

            if np.isfinite(abs_corr_bx_rfx) and np.isfinite(abs_corr_bx_rfy):
                dominant_axis_bx = "rf_x" if abs_corr_bx_rfx >= abs_corr_bx_rfy else "rf_y"
            else:
                dominant_axis_bx = "unknown"
            if np.isfinite(abs_corr_by_rfx) and np.isfinite(abs_corr_by_rfy):
                dominant_axis_by = "rf_y" if abs_corr_by_rfy >= abs_corr_by_rfx else "rf_x"
            else:
                dominant_axis_by = "unknown"

            sel_bx = abs_corr_bx_rfx - abs_corr_bx_rfy
            sel_by = abs_corr_by_rfy - abs_corr_by_rfx
            if np.isfinite(sel_bx):
                sel_bx_vals.append(float(sel_bx))
            if np.isfinite(sel_by):
                sel_by_vals.append(float(sel_by))

            row = {
                "session_id": session_id,
                "image_id": int(img_id),
                "projection_k": int(k),
                "n_units_with_rf": int(np.sum(keep)),
                "rf_coordinate_space": "pixel" if ("pixel" in str(rf_status)) else ("normalized" if ("norm" in str(rf_status)) else "raw"),
                "mean_rf_x": float(np.nanmean(xk)),
                "mean_rf_y": float(np.nanmean(yk)),
                "corr_bx_rfx": float(corr_bx_rfx),
                "corr_bx_rfy": float(corr_bx_rfy),
                "corr_by_rfx": float(corr_by_rfx),
                "corr_by_rfy": float(corr_by_rfy),
                "abs_corr_bx_rfx": float(abs_corr_bx_rfx),
                "abs_corr_bx_rfy": float(abs_corr_bx_rfy),
                "abs_corr_by_rfx": float(abs_corr_by_rfx),
                "abs_corr_by_rfy": float(abs_corr_by_rfy),
                "dominant_axis_bx": str(dominant_axis_bx),
                "dominant_axis_by": str(dominant_axis_by),
            }
            per_image_rows.append(row)
            k_rows.append(row)

        if not k_rows:
            summary_rows.append(
                {
                    "session_id": session_id,
                    "projection_k": int(k),
                    "n_images": 0,
                    "n_units_with_rf": 0,
                    "rf_coordinate_space": "unknown",
                    "mean_corr_bx_rfx": float("nan"),
                    "mean_corr_bx_rfy": float("nan"),
                    "mean_corr_by_rfx": float("nan"),
                    "mean_corr_by_rfy": float("nan"),
                    "mean_rf_x": float("nan"),
                    "mean_rf_y": float("nan"),
                    "mean_abs_corr_bx_rfx": float("nan"),
                    "mean_abs_corr_bx_rfy": float("nan"),
                    "mean_abs_corr_by_rfx": float("nan"),
                    "mean_abs_corr_by_rfy": float("nan"),
                    "axis_selectivity_bx": float("nan"),
                    "axis_selectivity_by": float("nan"),
                    "bootstrap_ci_axis_selectivity_bx": json.dumps([float("nan"), float("nan")]),
                    "bootstrap_ci_axis_selectivity_by": json.dumps([float("nan"), float("nan")]),
                    "interpretation_label": "not_retinotopic",
                    "retinotopy_test_status": str(rf_status),
                }
            )
            continue

        mean = lambda key: float(np.nanmean(np.asarray([float(r[key]) for r in k_rows], dtype=np.float64)))
        sel_bx_arr = np.asarray(sel_bx_vals, dtype=np.float64)
        sel_by_arr = np.asarray(sel_by_vals, dtype=np.float64)
        ci_bx = _bootstrap_ci(sel_bx_arr, seed=int(args.seed) + 10 + int(k), n_bootstrap=int(args.bootstrap_repeats))
        ci_by = _bootstrap_ci(sel_by_arr, seed=int(args.seed) + 100 + int(k), n_bootstrap=int(args.bootstrap_repeats))

        if np.isfinite(ci_bx[0]) and np.isfinite(ci_by[0]) and ci_bx[0] > 0.0 and ci_by[0] > 0.0:
            label = "retinotopic_shift_supported"
        elif (np.isfinite(ci_bx[0]) and ci_bx[0] > 0.0) or (np.isfinite(ci_by[0]) and ci_by[0] > 0.0):
            label = "axis_mixed"
        else:
            label = "not_retinotopic"

        summary_rows.append(
            {
                "session_id": session_id,
                "projection_k": int(k),
                "n_images": int(len(k_rows)),
                "n_units_with_rf": int(np.median(np.asarray([int(r["n_units_with_rf"]) for r in k_rows], dtype=np.int64))),
                "rf_coordinate_space": str(k_rows[0].get("rf_coordinate_space", "raw")) if k_rows else "raw",
                "mean_corr_bx_rfx": mean("corr_bx_rfx"),
                "mean_corr_bx_rfy": mean("corr_bx_rfy"),
                "mean_corr_by_rfx": mean("corr_by_rfx"),
                "mean_corr_by_rfy": mean("corr_by_rfy"),
                "mean_rf_x": mean("mean_rf_x"),
                "mean_rf_y": mean("mean_rf_y"),
                "mean_abs_corr_bx_rfx": mean("abs_corr_bx_rfx"),
                "mean_abs_corr_bx_rfy": mean("abs_corr_bx_rfy"),
                "mean_abs_corr_by_rfx": mean("abs_corr_by_rfx"),
                "mean_abs_corr_by_rfy": mean("abs_corr_by_rfy"),
                "axis_selectivity_bx": float(np.nanmean(sel_bx_arr)) if sel_bx_arr.size else float("nan"),
                "axis_selectivity_by": float(np.nanmean(sel_by_arr)) if sel_by_arr.size else float("nan"),
                "bootstrap_ci_axis_selectivity_bx": json.dumps([float(ci_bx[0]), float(ci_bx[1])]),
                "bootstrap_ci_axis_selectivity_by": json.dumps([float(ci_by[0]), float(ci_by[1])]),
                "interpretation_label": str(label),
                "retinotopy_test_status": str(rf_status),
            }
        )

    _write_csv(per_image_out, per_image_rows)
    _write_csv(summary_out, summary_rows)

    print(str(per_image_out))
    print(str(summary_out))


if __name__ == "__main__":
    main()

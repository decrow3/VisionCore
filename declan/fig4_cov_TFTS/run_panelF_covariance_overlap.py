#!/usr/bin/env python3
"""
Panel F: Tangent-subspace validity vs. eye-position cloud scale.

Four metrics (select via --metric, or comma-separated list):

  covariance_overlap   Cosine-like overlap between top-k subspaces of the full
                       FEM response covariance C_full and the linear (tangent)
                       prediction C_lin = J Sigma_eye J^T.  Reads the existing
                       covariance_approx CSV -- no forward passes required.

  variance_fraction    Fraction of actual FEM response variance that lives in
                       the top-k tangent subspace: trace(P_lin C_full)/trace(C_full).
                       Also reads covariance_approx CSV -- no forward passes.

  fisher_r2            Fisher-weighted cosine² between the tangent linear
                       prediction (dmu_pred = bx*dx + by*dy) and the actual
                       response change (dmu_actual = mu(r0+delta) - r0).
                       Requires model forward passes; uses r0/bx/by from the
                       tangent maps pickle.  This is the most scientifically
                       direct measure: at small scales (tangent regime) it
                       should be ~1; it decays as displacements grow large
                       enough for nonlinearities to dominate.

  fem_ranges           Classify fixational eye movements into drift and
                       microsaccades using the Engbert-Kliegl (2003) bivariate
                       velocity criterion.  Outputs empirical amplitude
                       distributions for both event types in arcmin (and as
                       cloud_scale multipliers), saved to panelF_fem_ranges.json.
                       Requires --dataset-configs-path (no GPU needed).

Usage
-----
# Fast (reads existing data, no GPU):
uv run python -m declan.fig4_cov_TFTS.run_panelF_covariance_overlap \\
    --tfts-root outputs/twin_feature_tangent_structure_prod_v2 \\
    --metric covariance_overlap,variance_fraction

# Expensive (needs model + GPU):
uv run python -m declan.fig4_cov_TFTS.run_panelF_covariance_overlap \\
    --tfts-root outputs/twin_feature_tangent_structure_prod_v2 \\
    --metric fisher_r2 \\
    --model-device cuda:1 \\
    --n-samples 30
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from VisionCore.paths import VISIONCORE_ROOT

# ---------------------------------------------------------------------------
# Cheap metrics -- aggregate existing covariance_approx CSV
# ---------------------------------------------------------------------------

_K_VALUES = (1, 2, 3)

_CHEAP_COLS = {
    "covariance_overlap": "subspace_overlap_k{k}",
    "variance_fraction":  "fraction_full_variance_in_lin_subspace_k{k}",
}


def _aggregate_csv(cov_file: Path, col_template: str, k_values=_K_VALUES) -> pd.DataFrame:
    df = pd.read_csv(cov_file)
    rows: list[dict[str, Any]] = []
    for k in k_values:
        col = col_template.format(k=k)
        if col not in df.columns:
            print(f"  column {col!r} not found in CSV, skipping k={k}")
            continue
        for (delta, cs), grp in df.groupby(["delta", "cloud_scale"]):
            vals = pd.to_numeric(grp[col], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            rows.append({
                "delta":       float(delta),
                "cloud_scale": float(cs),
                "k":           int(k),
                "n_objects":   int(len(vals)),
                "median":      float(np.median(vals)),
                "ci_low":      float(np.percentile(vals, 2.5)),
                "ci_high":     float(np.percentile(vals, 97.5)),
            })
    return pd.DataFrame(rows)


def compute_covariance_overlap(tfts_root: Path) -> pd.DataFrame:
    cov_file = tfts_root / "covariance_approx" / "twin_linear_covariance_approx.csv"
    if not cov_file.exists():
        raise FileNotFoundError(f"covariance_approx CSV not found: {cov_file}")
    print(f"  reading {cov_file.name}")
    return _aggregate_csv(cov_file, _CHEAP_COLS["covariance_overlap"])


def compute_variance_fraction(tfts_root: Path) -> pd.DataFrame:
    cov_file = tfts_root / "covariance_approx" / "twin_linear_covariance_approx.csv"
    if not cov_file.exists():
        raise FileNotFoundError(f"covariance_approx CSV not found: {cov_file}")
    print(f"  reading {cov_file.name}")
    return _aggregate_csv(cov_file, _CHEAP_COLS["variance_fraction"])


# ---------------------------------------------------------------------------
# Expensive metric -- Fisher-weighted R² via model forward passes
# ---------------------------------------------------------------------------


def compute_fisher_r2(
    tfts_root:    Path,
    cloud_scales: list[float],
    delta_target: float,
    n_samples:    int,
    model_device: str,
    seed:         int,
    ppd:          float,
) -> pd.DataFrame:
    """
    Fisher-weighted cosine² between tangent prediction and actual response.

    For each (object, cloud_scale, sampled displacement delta):
      dmu_pred   = bx*dx + by*dy            (linear / tangent prediction)
      dmu_actual = mu(history shifted by delta) - r0

    metric = <dmu_pred, dmu_actual>_F^2 / (||dmu_pred||_F^2 * ||dmu_actual||_F^2)

    where the Fisher inner product <u,v>_F = sum_n u_n v_n / max(r0_n, eps).
    Bounded [0, 1].  At small scales (good linear regime) → 1.

    cloud_scale (arcmin) sets the Gaussian sigma:  sigma_px = cloud_scale * ppd / 60.
    """
    # --- load tangent maps ---
    pkl_path = tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"tangent maps pickle not found: {pkl_path}")
    print(f"  loading tangent maps from {pkl_path.name}")
    with open(pkl_path, "rb") as f:
        tm = pickle.load(f)

    available = sorted(tm["object_payload"].keys())
    delta_key = min(available, key=lambda d: abs(d - delta_target))
    print(f"  using delta={delta_key} arcmin (requested {delta_target})")
    payload = tm["object_payload"][delta_key]

    # verify r0 is present (prod_v2 onwards)
    sample_oid = next(iter(payload))
    if "r0" not in payload[sample_oid]:
        raise KeyError(
            "r0 not found in tangent maps; re-run run_twin_feature_tangent_structure.py "
            "from the updated code to store r0 in the pickle."
        )

    # --- load model ---
    print(f"  loading model on {model_device}")
    from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import (
        _load_twin_context,
        _movie_to_thw,
        _shift_movie_subpixel,
        _predict_rate_from_history,
    )
    ctx = _load_twin_context(model_device=model_device)

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []

    n_objects = len(payload)
    for cs_idx, cs in enumerate(cloud_scales):
        sigma_px = float(cs) * float(ppd) / 60.0  # arcmin → pixels
        print(f"  cloud_scale={cs} arcmin  sigma={sigma_px:.4f} px  "
              f"({cs_idx+1}/{len(cloud_scales)})", flush=True)

        per_object: list[float] = []

        for oid_idx, (oid, obj) in enumerate(payload.items()):
            r0 = np.asarray(obj["r0"], dtype=np.float64)
            bx = np.asarray(obj["bx"], dtype=np.float64)
            by = np.asarray(obj["by"], dtype=np.float64)
            history = np.asarray(obj["history"], dtype=np.float32)

            valid = np.isfinite(r0).all() and np.isfinite(bx).all() and np.isfinite(by).all()
            if not valid or float(np.min(r0)) <= 0:
                continue

            w = 1.0 / np.maximum(r0, 1e-8)   # Poisson Fisher weights at base rate

            offsets = rng.normal(0.0, sigma_px, size=(n_samples, 2))

            cosines_sq: list[float] = []
            h0 = _movie_to_thw(history).to(model_device)
            for dx, dy in offsets:
                # Linear prediction
                dmu_pred = bx * float(dx) + by * float(dy)

                # Actual response via forward pass
                hs = _shift_movie_subpixel(h0, dx_px=float(dx), dy_px=float(dy)).detach().cpu().numpy()
                mu_actual = _predict_rate_from_history(ctx, hs, model_device=model_device)
                dmu_actual = mu_actual.astype(np.float64) - r0

                # Fisher inner products
                ip     = float(np.sum(w * dmu_pred * dmu_actual))
                n_pred = float(np.sum(w * dmu_pred**2))
                n_act  = float(np.sum(w * dmu_actual**2))

                if n_pred > 1e-12 and n_act > 1e-12:
                    cosines_sq.append(ip**2 / (n_pred * n_act))

            if cosines_sq:
                per_object.append(float(np.mean(cosines_sq)))

            if (oid_idx + 1) % 8 == 0:
                print(f"    {oid_idx+1}/{n_objects} objects done", flush=True)

        if per_object:
            vals = np.array(per_object)
            rows.append({
                "delta":       float(delta_key),
                "cloud_scale": float(cs),
                "k":           2,
                "n_objects":   int(len(vals)),
                "median":      float(np.median(vals)),
                "ci_low":      float(np.percentile(vals, 2.5)),
                "ci_high":     float(np.percentile(vals, 97.5)),
            })
            print(f"    → median Fisher R²={rows[-1]['median']:.3f}  "
                  f"[{rows[-1]['ci_low']:.3f}, {rows[-1]['ci_high']:.3f}]  "
                  f"n={rows[-1]['n_objects']}", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FEM range estimation — Engbert & Kliegl (2003)
# ---------------------------------------------------------------------------


def _detect_microsaccades_ek2003(
    pos_deg: np.ndarray,
    frame_rate_hz: float = 240.0,
    lambda_thresh: float = 6.0,
    min_dur_frames: int = 3,
) -> list[tuple[int, int]]:
    """
    Detect microsaccades with the bivariate velocity ellipse criterion.

    Uses the two-sample velocity estimator (EK2003 Eq. 1):
        v_i = (pos[i+2] - pos[i-2]) / (4 * dt)   (centered finite difference)

    sigma_x = sqrt(median(v_x²) - median(v_x)²)
    Event: (v_x / (lambda * sigma_x))² + (v_y / (lambda * sigma_y))² > 1

    Returns list of (start, end) index pairs (end exclusive) in pos_deg coords.
    """
    n = len(pos_deg)
    if n < 6:
        return []
    dt = 1.0 / frame_rate_hz

    # Centered 4-sample velocity (valid at indices 2 .. n-3)
    vel = (pos_deg[4:] - pos_deg[:-4]) / (4.0 * dt)   # shape (n-4, 2)
    vx, vy = vel[:, 0], vel[:, 1]

    sigma_x = np.sqrt(max(float(np.median(vx**2) - np.median(vx)**2), 0.0))
    sigma_y = np.sqrt(max(float(np.median(vy**2) - np.median(vy)**2), 0.0))

    if sigma_x < 1e-8 or sigma_y < 1e-8:
        return []

    ellipse = (vx / (lambda_thresh * sigma_x))**2 + (vy / (lambda_thresh * sigma_y))**2
    above = ellipse > 1.0  # shape (n-4,); index i here → pos index i+2

    events: list[tuple[int, int]] = []
    in_event = False
    ev_start = 0
    for i, flag in enumerate(above):
        pos_i = i + 2   # shift to pos index
        if flag and not in_event:
            in_event, ev_start = True, pos_i
        elif not flag and in_event:
            in_event = False
            ev_end = pos_i
            if ev_end - ev_start >= min_dur_frames:
                events.append((ev_start, ev_end))
    if in_event:
        ev_end = n - 2
        if ev_end - ev_start >= min_dur_frames:
            events.append((ev_start, ev_end))

    return events


def compute_fem_ranges(
    dataset_configs_path: str,
    subject: str = "Allen",
    date: str = "2022-02-16",
    frame_rate_hz: float = 240.0,
    lambda_thresh: float = 6.0,
    min_dur_frames: int = 3,
    band_percentiles: tuple[float, float] = (10.0, 90.0),
) -> dict:
    """
    Load fixrsvp eye position data, classify FEM events, compute amplitude ranges.

    Returns a dict suitable for saving as panelF_fem_ranges.json:
      drift_band_arcmin         : [p_lo, p_hi] of drift-epoch SD in arcmin
      msac_band_arcmin          : [p_lo, p_hi] of microsaccade amplitude in arcmin
      local_eye_sd_arcmin       : median trial-level eye-position SD (what cloud_scale=1 maps to)
      drift_band_cloud_scale    : drift_band_arcmin / local_eye_sd_arcmin
      msac_band_cloud_scale     : msac_band_arcmin  / local_eye_sd_arcmin
      n_drift_epochs            : total drift epochs detected
      n_microsaccades           : total microsaccades detected
      lambda_thresh             : EK2003 lambda used
      band_percentiles          : percentiles used for [lo, hi]
    """
    from eval.fixrsvp import get_fixrsvp_data
    from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import (
        _harmonize_fixrsvp_arrays,
    )

    print(f"  loading fixrsvp data: subject={subject} date={date}", flush=True)
    data = get_fixrsvp_data(
        subject=subject, date=date,
        dataset_configs_path=dataset_configs_path,
        use_cached_data=True,
    )
    _, _, eyepos = _harmonize_fixrsvp_arrays(data)
    # eyepos: (n_trials, n_time, 2) in degrees

    msac_amplitudes_arcmin: list[float] = []
    drift_sds_arcmin: list[float] = []
    local_sds_arcmin: list[float] = []

    for tr in range(eyepos.shape[0]):
        pos_deg = eyepos[tr]  # (n_time, 2)
        valid   = np.isfinite(pos_deg).all(axis=1)
        pos_v   = pos_deg[valid]
        if len(pos_v) < 20:
            continue

        # Trial-level SD (what cloud_scale=1 corresponds to in synthetic_local_gaussian mode)
        pos_v_arcmin = pos_v * 60.0
        centered = pos_v_arcmin - pos_v_arcmin.mean(axis=0, keepdims=True)
        trial_sd = float(np.sqrt(np.mean(np.var(centered, axis=0))))
        local_sds_arcmin.append(trial_sd)

        # Microsaccade detection
        events = _detect_microsaccades_ek2003(pos_v, frame_rate_hz, lambda_thresh, min_dur_frames)

        # Microsaccade amplitudes: displacement from onset to offset
        msac_mask = np.zeros(len(pos_v), dtype=bool)
        for start, end in events:
            msac_mask[start:end] = True
            disp = pos_v[end - 1] - pos_v[start]
            amplitude_arcmin = float(np.linalg.norm(disp) * 60.0)
            if amplitude_arcmin > 0.1:   # filter near-zero detections
                msac_amplitudes_arcmin.append(amplitude_arcmin)

        # Drift epochs: contiguous non-microsaccade runs ≥ 10 frames
        drift_start: int | None = None
        for i in range(len(msac_mask)):
            if not msac_mask[i] and drift_start is None:
                drift_start = i
            elif msac_mask[i] and drift_start is not None:
                if i - drift_start >= 10:
                    ep = pos_v_arcmin[drift_start:i]
                    ep_c = ep - ep.mean(axis=0, keepdims=True)
                    drift_sds_arcmin.append(float(np.sqrt(np.mean(np.var(ep_c, axis=0)))))
                drift_start = None
        if drift_start is not None and len(pos_v) - drift_start >= 10:
            ep = pos_v_arcmin[drift_start:]
            ep_c = ep - ep.mean(axis=0, keepdims=True)
            drift_sds_arcmin.append(float(np.sqrt(np.mean(np.var(ep_c, axis=0)))))

    lo_p, hi_p = band_percentiles

    def _band(vals: list[float]) -> list[float]:
        if not vals:
            return [float("nan"), float("nan")]
        return [float(np.percentile(vals, lo_p)), float(np.percentile(vals, hi_p))]

    local_sd_med = float(np.median(local_sds_arcmin)) if local_sds_arcmin else 1.0

    drift_band_arcmin = _band(drift_sds_arcmin)
    msac_band_arcmin  = _band(msac_amplitudes_arcmin)

    result = {
        "drift_band_arcmin":      drift_band_arcmin,
        "msac_band_arcmin":       msac_band_arcmin,
        "drift_band_cloud_scale": [v / local_sd_med for v in drift_band_arcmin],
        "msac_band_cloud_scale":  [v / local_sd_med for v in msac_band_arcmin],
        "local_eye_sd_arcmin":    local_sd_med,
        "n_drift_epochs":         len(drift_sds_arcmin),
        "n_microsaccades":        len(msac_amplitudes_arcmin),
        "lambda_thresh":          lambda_thresh,
        "band_percentiles":       list(band_percentiles),
        "subject":                subject,
        "date":                   date,
    }
    print(f"  detected {result['n_microsaccades']} microsaccades, "
          f"{result['n_drift_epochs']} drift epochs", flush=True)
    print(f"  drift band (arcmin):  [{drift_band_arcmin[0]:.2f}, {drift_band_arcmin[1]:.2f}]")
    print(f"  microsaccade band (arcmin): [{msac_band_arcmin[0]:.2f}, {msac_band_arcmin[1]:.2f}]")
    print(f"  local eye SD (cloud_scale=1): {local_sd_med:.2f} arcmin")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VALID_METRICS = ("covariance_overlap", "variance_fraction", "fisher_r2", "fem_ranges")

DEFAULT_TFTS    = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
DEFAULT_OUT_DIR = VISIONCORE_ROOT / "outputs" / "panel_f_covariance_overlap"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Panel F: tangent validity vs cloud scale (3 metrics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS,
                   help="Path to a completed TFTS production output directory.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Root directory for output CSV files (one subdir per metric).")
    p.add_argument("--metric", default="covariance_overlap",
                   help=f"Comma-separated metric(s): {', '.join(VALID_METRICS)}.  "
                        "Use 'all' for all three.")

    # fisher_r2 parameters
    p.add_argument("--delta-arcmin", type=float, default=0.25,
                   help="Tangent delta (arcmin) to load from pickle for fisher_r2.")
    p.add_argument("--cloud-scales", default="0.125,0.25,0.5,1.0,2.0,4.0",
                   help="Cloud scale values (arcmin) for fisher_r2 x-axis sweep.")
    p.add_argument("--n-samples", type=int, default=30,
                   help="Displacement samples per object per cloud scale (fisher_r2).")
    p.add_argument("--model-device", default="cuda:1",
                   help="Torch device for model forward passes (fisher_r2).")
    p.add_argument("--model-ppd", type=float, default=37.5,
                   help="Pixels per degree for the model (used to convert arcmin→px).")
    p.add_argument("--seed", type=int, default=42)

    # fem_ranges parameters
    p.add_argument("--dataset-configs-path",
                   default="experiments/dataset_configs/multi_basic_240_rsvp.yaml",
                   help="Dataset configs YAML (for fem_ranges eye data loading).")
    p.add_argument("--subject", default="Allen",
                   help="Subject name for fem_ranges eye data loading.")
    p.add_argument("--date", default="2022-02-16",
                   help="Session date for fem_ranges eye data loading.")
    p.add_argument("--fem-lambda", type=float, default=6.0,
                   help="EK2003 velocity threshold multiplier (lambda).")
    p.add_argument("--fem-min-dur-frames", type=int, default=3,
                   help="Minimum microsaccade duration in frames (EK2003).")
    p.add_argument("--band-percentiles", default="10,90",
                   help="Lo,hi percentiles for drift/microsaccade band edges.")
    p.add_argument("--frame-rate-hz", type=float, default=240.0,
                   help="Eye-tracker frame rate in Hz.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    metrics_raw = args.metric.lower().strip()
    if metrics_raw == "all":
        metrics = list(VALID_METRICS)
    else:
        metrics = [m.strip() for m in metrics_raw.split(",")]

    for m in metrics:
        if m not in VALID_METRICS:
            raise ValueError(f"Unknown metric {m!r}. Choose from: {', '.join(VALID_METRICS)}")

    cloud_scales = [float(x) for x in args.cloud_scales.split(",")]

    all_frames: list[pd.DataFrame] = []

    for metric in metrics:
        print(f"\n=== metric: {metric} ===", flush=True)
        out_subdir = args.output_dir / metric
        out_subdir.mkdir(parents=True, exist_ok=True)

        if metric == "fem_ranges":
            band_lo, band_hi = [float(x) for x in args.band_percentiles.split(",")]
            ranges = compute_fem_ranges(
                dataset_configs_path=str(args.dataset_configs_path),
                subject=str(args.subject),
                date=str(args.date),
                frame_rate_hz=float(args.frame_rate_hz),
                lambda_thresh=float(args.fem_lambda),
                min_dur_frames=int(args.fem_min_dur_frames),
                band_percentiles=(band_lo, band_hi),
            )
            # fem_ranges writes directly to the shared output root (not a subdir),
            # so both metrics and the figure generator can find a single JSON.
            fem_json = args.output_dir / "panelF_fem_ranges.json"
            fem_json.parent.mkdir(parents=True, exist_ok=True)
            fem_json.write_text(json.dumps(ranges, indent=2), encoding="utf-8")
            print(f"\nSaved FEM ranges to {fem_json}", flush=True)
            print(json.dumps(ranges, indent=2), flush=True)
            continue   # no CSV to write for fem_ranges

        if metric == "covariance_overlap":
            df = compute_covariance_overlap(args.tfts_root)
        elif metric == "variance_fraction":
            df = compute_variance_fraction(args.tfts_root)
        else:  # fisher_r2
            df = compute_fisher_r2(
                tfts_root=args.tfts_root,
                cloud_scales=cloud_scales,
                delta_target=float(args.delta_arcmin),
                n_samples=int(args.n_samples),
                model_device=str(args.model_device),
                seed=int(args.seed),
                ppd=float(args.model_ppd),
            )

        df["metric"] = metric
        out_csv = out_subdir / "panelF_summary.csv"
        df.to_csv(out_csv, index=False)

        manifest: dict[str, Any] = {
            "metric":       metric,
            "tfts_root":    str(args.tfts_root),
            "n_rows":       int(len(df)),
            "deltas":       sorted(df["delta"].unique().tolist()) if len(df) else [],
            "cloud_scales": sorted(df["cloud_scale"].unique().tolist()) if len(df) else [],
            "n_objects_range": (
                [int(df["n_objects"].min()), int(df["n_objects"].max())]
                if len(df) else [0, 0]
            ),
            "status": "completed",
        }
        (out_subdir / "panelF_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
        all_frames.append(df)

    # Write combined CSV (all metrics together) for easy figure loading
    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_csv = args.output_dir / "panelF_combined_summary.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nCombined summary: {combined_csv}")


if __name__ == "__main__":
    main()

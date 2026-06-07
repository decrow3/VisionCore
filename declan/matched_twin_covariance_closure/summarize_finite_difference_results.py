from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .run_cache_closure import _write_csv, _write_json


DEFAULT_ROOT = Path("outputs/matched_twin_covariance_closure_finite_difference")


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _sign_test_p_two_sided(n_positive: int, n_total: int) -> float:
    if n_total <= 0:
        return float("nan")
    k = int(n_positive)
    n = int(n_total)
    if k == n / 2:
        return 1.0
    if k > n / 2:
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    else:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    vals = _finite(values)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    if vals.size == 1 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    boot = np.mean(vals[idx], axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _summarize_grouped(metrics: pd.DataFrame, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    group_cols = ["target_variant", "projection_control", "basis_source", "k"]
    rows: list[dict[str, Any]] = []
    ok = metrics[metrics["row_status"] == "ok"].copy()
    for key, g in ok.groupby(group_cols, dropna=False):
        capture_mean, capture_lo, capture_hi = _bootstrap_mean_ci(
            g["capture"].to_numpy(), rng=rng, n_boot=n_boot
        )
        effect_mean, effect_lo, effect_hi = _bootstrap_mean_ci(
            g["effect_minus_unit_shuffle_median"].to_numpy(), rng=rng, n_boot=n_boot
        )
        random_effect_mean, random_lo, random_hi = _bootstrap_mean_ci(
            g["effect_minus_random_subspace_median"].to_numpy(), rng=rng, n_boot=n_boot
        )
        eff = _finite(g["effect_minus_unit_shuffle_median"].to_numpy())
        n_pos = int(np.sum(eff > 0.0))
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                "n_sessions": int(g["session"].nunique()),
                "capture_mean": capture_mean,
                "capture_boot_ci_low": capture_lo,
                "capture_boot_ci_high": capture_hi,
                "effect_unit_mean": effect_mean,
                "effect_unit_boot_ci_low": effect_lo,
                "effect_unit_boot_ci_high": effect_hi,
                "effect_random_mean": random_effect_mean,
                "effect_random_boot_ci_low": random_lo,
                "effect_random_boot_ci_high": random_hi,
                "n_effect_positive": n_pos,
                "n_effect_nonzero": int(eff.size),
                "sign_test_p_two_sided": _sign_test_p_two_sided(n_pos, int(eff.size)),
                "effect_unit_min": float(np.min(eff)) if eff.size else float("nan"),
                "effect_unit_max": float(np.max(eff)) if eff.size else float("nan"),
            }
        )
    return sorted(rows, key=lambda r: (str(r["target_variant"]), str(r["projection_control"]), str(r["basis_source"]), int(r["k"])))


def _headline_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headline_sources = {
        ("fd_mean_tangent_matrix", 2),
        ("fd_sample_eye_trace_cov", 2),
        ("fd_sample_eye_trace_cov", 10),
        ("fd_sample_eye_trace_xfit_compact_k10_cov", 2),
        ("fd_sample_eye_trace_xfit_compact_k10_cov", 10),
        ("fd_tangent_gram_cov", 2),
        ("fd_tangent_gram_cov", 10),
    }
    controls = {"none", "global_rate", "target_pc1", "global_rate+target_pc1"}
    out = [
        row
        for row in summary_rows
        if (str(row["basis_source"]), int(row["k"])) in headline_sources
        and str(row["projection_control"]) in controls
        and str(row["target_variant"]) in {"raw", "psd"}
    ]
    return sorted(out, key=lambda r: (str(r["target_variant"]), str(r["basis_source"]), int(r["k"]), str(r["projection_control"])))


def _audit(root: Path, manifest: dict[str, Any], sessions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    ok_sessions = sessions[sessions["status"] == "ok"].copy()
    gain_min = ok_sessions["gain_min"].to_numpy(dtype=np.float64) if "gain_min" in ok_sessions else np.array([])
    return {
        "root": str(root.resolve()),
        "manifest_status": manifest.get("status"),
        "manifest_device": manifest.get("device"),
        "manifest_step_px": manifest.get("step_px"),
        "manifest_max_samples": manifest.get("max_samples"),
        "manifest_n_nulls": manifest.get("n_nulls"),
        "manifest_rescale_mode": manifest.get("rescale_mode"),
        "checkpoint": manifest.get("checkpoint"),
        "model_config": manifest.get("model_config"),
        "dataset_config": manifest.get("dataset_config"),
        "fig2_cache": manifest.get("fig2_cache"),
        "fig3_cache": manifest.get("fig3_cache"),
        "n_sessions_manifest": manifest.get("n_sessions_ok"),
        "n_sessions_csv": int(len(sessions)),
        "session_status_counts": sessions["status"].value_counts(dropna=False).to_dict(),
        "n_metric_rows_manifest": manifest.get("n_metric_rows"),
        "n_metric_rows_csv": int(len(metrics)),
        "metric_row_status_counts": metrics["row_status"].value_counts(dropna=False).to_dict(),
        "window_idx_values": sorted(int(v) for v in metrics["window_idx"].dropna().unique().tolist()),
        "target_variant_values": sorted(str(v) for v in metrics["target_variant"].dropna().unique().tolist()),
        "basis_source_values": sorted(str(v) for v in metrics["basis_source"].dropna().unique().tolist()),
        "projection_control_values": sorted(str(v) for v in metrics["projection_control"].dropna().unique().tolist()),
        "n_common_units_min": int(ok_sessions["n_common_units"].min()) if len(ok_sessions) else None,
        "n_common_units_median": float(ok_sessions["n_common_units"].median()) if len(ok_sessions) else None,
        "n_common_units_max": int(ok_sessions["n_common_units"].max()) if len(ok_sessions) else None,
        "n_samples_used_min": int(ok_sessions["n_samples_used"].min()) if len(ok_sessions) else None,
        "n_samples_used_median": float(ok_sessions["n_samples_used"].median()) if len(ok_sessions) else None,
        "n_samples_used_max": int(ok_sessions["n_samples_used"].max()) if len(ok_sessions) else None,
        "rescale_status_counts": ok_sessions["rescale_status"].value_counts(dropna=False).to_dict()
        if "rescale_status" in ok_sessions
        else {},
        "sessions_with_gain_min_below_1e_minus_6": int(np.sum(gain_min < 1e-6)) if gain_min.size else 0,
        "target_negative_eigenvalue_mass_total_raw": float(ok_sessions["target_negative_eigenvalue_mass_raw"].sum())
        if "target_negative_eigenvalue_mass_raw" in ok_sessions
        else float("nan"),
        "target_trace_raw_total": float(ok_sessions["target_trace_raw"].sum()) if "target_trace_raw" in ok_sessions else float("nan"),
        "target_trace_psd_total": float(ok_sessions["target_trace_psd"].sum()) if "target_trace_psd" in ok_sessions else float("nan"),
        "notes": [
            "Metric CIs in finite_difference_bootstrap_summary.csv are session-cluster bootstrap CIs for the mean.",
            "Existing finite_difference_metric_summary.csv CI columns are session percentiles, not bootstrap CIs.",
            "PSD targets are eigenvalue-clipped recorded FEM covariance; raw target rows are also reported.",
            "Unit-shuffle null breaks unit identity while preserving source loading structure; random-subspace null is also reported.",
        ],
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    sessions = pd.read_csv(root / "finite_difference_session_summary.csv")
    metrics = pd.read_csv(root / "finite_difference_capture_metrics.csv")

    summary_rows = _summarize_grouped(metrics, n_boot=int(args.n_boot), seed=int(args.seed))
    headline_rows = _headline_rows(summary_rows)
    _write_csv(root / "finite_difference_bootstrap_summary.csv", summary_rows)
    _write_csv(root / "finite_difference_headline_raw_psd_bootstrap.csv", headline_rows)
    _write_json(root / "finite_difference_provenance_audit.json", _audit(root, manifest, sessions, metrics))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap/sign-test summary for finite-difference closure outputs")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

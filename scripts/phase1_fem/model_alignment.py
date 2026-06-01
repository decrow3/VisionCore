"""
Phase 1 FEM/V1 covariance analysis: cached model-alignment loading.

Loads precomputed model-empirical alignment summaries from the existing
jacobian_predictive_framework outputs and converts them into session-level and
basis-level metrics compatible with Phase 1 reporting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _session_slug(session: str) -> str:
    """Convert Allen_2022-02-16 -> allen_2022_02_16."""
    return session.lower().replace("-", "_")


def _pick_summary_path(base_dir: Path, slug: str) -> Path | None:
    """
    Find best cached model alignment summary path for a session slug.

    Preference order:
      1) *_model_empirical_alignment_iter1/model_empirical_alignment_summary.json
      2) *_model_empirical_alignment_*/model_empirical_alignment_summary.json
         excluding smoke/test variants where possible.
    """
    exact = sorted(base_dir.glob(f"{slug}_model_empirical_alignment_iter1/model_empirical_alignment_summary.json"))
    if exact:
        return exact[0]

    all_candidates = sorted(base_dir.glob(f"{slug}_model_empirical_alignment_*/model_empirical_alignment_summary.json"))
    if not all_candidates:
        return None

    preferred = [
        p for p in all_candidates
        if not any(x in str(p.parent.name) for x in ("smoke", "onewindow", "twowindow", "quick"))
    ]
    if preferred:
        return preferred[0]
    return all_candidates[0]


def _norm_alignment(alignment: float, shuffle: float, ceiling: float, min_ceiling: float = 0.20) -> tuple[float, str]:
    if not np.isfinite(ceiling) or ceiling < min_ceiling:
        return float("nan"), "low_reliability_ceiling"
    if not (np.isfinite(alignment) and np.isfinite(shuffle)):
        return float("nan"), "alignment_not_finite"
    return float((alignment - shuffle) / ceiling), "ok"


def load_cached_model_alignment(
    session: str,
    base_dir: Path,
    preferred_basis_order: tuple[str, ...] = ("FEM_PCs", "B_model", "J_local"),
) -> dict:
    """
    Load cached model alignment summary for a session.

    Returns dict with:
      - available: bool
      - source_path: str
      - basis_rows: list[dict]
      - primary_basis: str
      - model_alignment: float (matched alignment)
      - model_shuffle_alignment: float (shuffled alignment)
      - reliability_ceiling_model: float
      - ceiling_normalized_alignment: float
      - alignment_norm_status: str
      - n_windows_model_alignment: int
    """
    summary_path = _pick_summary_path(base_dir, _session_slug(session))
    if summary_path is None or not summary_path.exists():
        return {
            "available": False,
            "source_path": "",
            "basis_rows": [],
            "status": "not_found",
        }

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_by_basis = payload.get("summary_by_basis", {})
    if not summary_by_basis:
        return {
            "available": False,
            "source_path": str(summary_path),
            "basis_rows": [],
            "status": "missing_summary_by_basis",
        }

    basis_rows: list[dict] = []
    for basis_name, b in summary_by_basis.items():
        matched = float(b.get("median_align_to_emp_2d_matched", np.nan))
        shuffled = float(b.get("median_align_to_emp_2d_shuffled", np.nan))
        ceiling = float(b.get("median_emp_split_alignment_2d", np.nan))
        norm, norm_status = _norm_alignment(matched, shuffled, ceiling)

        basis_rows.append(
            {
                "basis_name": basis_name,
                "n_windows": int(b.get("n", payload.get("n_windows", 0))),
                "model_alignment": matched,
                "model_shuffle_alignment": shuffled,
                "model_alignment_delta": float(matched - shuffled) if np.isfinite(matched) and np.isfinite(shuffled) else float("nan"),
                "reliability_ceiling_model": ceiling,
                "ceiling_normalized_alignment": norm,
                "alignment_norm_status": norm_status,
                "capture_emp_2d_matched": float(b.get("median_capture_emp_2d_matched", np.nan)),
                "capture_emp_2d_shuffled": float(b.get("median_capture_emp_2d_shuffled", np.nan)),
                "alignment_fraction_of_ceiling_2d_matched": float(
                    b.get("median_alignment_fraction_of_ceiling_2d_matched", np.nan)
                ),
                "alignment_fraction_of_ceiling_2d_shuffled": float(
                    b.get("median_alignment_fraction_of_ceiling_2d_shuffled", np.nan)
                ),
            }
        )

    by_name = {r["basis_name"]: r for r in basis_rows}
    primary_row = None
    for name in preferred_basis_order:
        if name in by_name:
            primary_row = by_name[name]
            break
    if primary_row is None:
        primary_row = max(
            basis_rows,
            key=lambda r: r["model_alignment_delta"] if np.isfinite(r["model_alignment_delta"]) else -np.inf,
        )

    return {
        "available": True,
        "source_path": str(summary_path),
        "status": "ok",
        "basis_rows": basis_rows,
        "primary_basis": primary_row["basis_name"],
        "model_alignment": primary_row["model_alignment"],
        "model_shuffle_alignment": primary_row["model_shuffle_alignment"],
        "reliability_ceiling_model": primary_row["reliability_ceiling_model"],
        "ceiling_normalized_alignment": primary_row["ceiling_normalized_alignment"],
        "alignment_norm_status": primary_row["alignment_norm_status"],
        "n_windows_model_alignment": primary_row["n_windows"],
        "n_windows_total": int(payload.get("n_windows", 0)),
        "subject": payload.get("subject", ""),
        "date": payload.get("date", ""),
    }

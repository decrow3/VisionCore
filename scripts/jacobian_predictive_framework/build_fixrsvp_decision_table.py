#!/usr/bin/env python3
"""Build a cross-session decision table for the fixRSVP Figure 4 pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework")
DEFAULT_OUTPUT_DIR = ROOT_OUTPUT_DIR / "cross_session_decision_table"
DEFAULT_SESSIONS = (
    ("Allen", "2022-02-16"),
    ("Allen", "2022-02-24"),
    ("Allen", "2022-03-04"),
    ("Allen", "2022-04-08"),
)


def _parse_sessions(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_SESSIONS)
    sessions: list[tuple[str, str]] = []
    for value in values:
        subject, date = value.split(":", maxsplit=1)
        sessions.append((subject, date))
    return sessions


def _session_prefix(subject: str, date: str) -> str:
    return f"{subject.lower()}_{date.replace('-', '_')}"


def _latest_summary_path_any(subject: str, date: str, contains_values: list[str], filename: str) -> Path | None:
    prefix = _session_prefix(subject, date)
    candidates = []
    for directory in ROOT_OUTPUT_DIR.glob(f"{prefix}*"):
        if not directory.is_dir() or not any(value in directory.name for value in contains_values):
            continue
        path = directory / filename
        if path.exists():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_manifest(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _load_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_get(mapping: dict, *keys: str) -> float:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return float("nan")
        value = value.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _paired_delta_summary(rows: list[dict], matched_key: str, shuffled_key: str) -> dict:
    if not rows:
        return {
            "n": 0,
            "matched_median": float("nan"),
            "shuffled_median": float("nan"),
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "positive_count": 0,
            "nonnegative_count": 0,
        }

    matched = np.array([_safe_get(row, matched_key) for row in rows], dtype=np.float64)
    shuffled = np.array([_safe_get(row, shuffled_key) for row in rows], dtype=np.float64)
    valid = np.isfinite(matched) & np.isfinite(shuffled)
    matched = matched[valid]
    shuffled = shuffled[valid]
    if matched.size == 0:
        return {
            "n": 0,
            "matched_median": float("nan"),
            "shuffled_median": float("nan"),
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "positive_count": 0,
            "nonnegative_count": 0,
        }

    delta = matched - shuffled
    rng = np.random.default_rng(0)
    boot = np.empty(10000, dtype=np.float64)
    for idx in range(boot.size):
        sample_idx = rng.integers(0, delta.size, size=delta.size)
        boot[idx] = np.nanmedian(delta[sample_idx])
    return {
        "n": int(delta.size),
        "matched_median": float(np.nanmedian(matched)),
        "shuffled_median": float(np.nanmedian(shuffled)),
        "median_delta": float(np.nanmedian(delta)),
        "mean_delta": float(np.nanmean(delta)),
        "ci95_low": float(np.nanpercentile(boot, 2.5)),
        "ci95_high": float(np.nanpercentile(boot, 97.5)),
        "positive_count": int(np.sum(delta > 0.0)),
        "nonnegative_count": int(np.sum(delta >= 0.0)),
    }


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _manifest_path(manifest: dict[str, dict[str, str]], subject: str, date: str, key: str) -> Path | None:
    session_key = f"{subject}_{date}"
    value = manifest.get(session_key, {}).get(key)
    return Path(value) if value else None


def _phase3_basis_rows(rows: list[dict[str, str]], basis_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("basis_name") == basis_name]


def _build_row(subject: str, date: str, manifest: dict[str, dict[str, str]]) -> dict:
    step2_summary_path = _manifest_path(manifest, subject, date, "step2_summary") or _latest_summary_path_any(subject, date, ["step2"], "step2_summary.json")
    step2_residual_path = _manifest_path(manifest, subject, date, "step2_residualized") or _latest_summary_path_any(subject, date, ["step2"], "step2_residualized_summary.json")
    ceiling_summary_path = _manifest_path(manifest, subject, date, "ceiling_summary") or _latest_summary_path_any(subject, date, ["empirical_ceiling", "empirical_geometry_ceiling"], "empirical_geometry_ceiling_summary.json")
    chart_summary_path = _manifest_path(manifest, subject, date, "chart_summary") or _latest_summary_path_any(subject, date, ["translation_chart"], "translation_chart_summary.json")
    phase3_summary_path = _manifest_path(manifest, subject, date, "phase3_summary") or _latest_summary_path_any(subject, date, ["model_empirical_alignment"], "model_empirical_alignment_summary.json")

    step2 = _load_json(step2_summary_path)
    step2_resid = _load_json(step2_residual_path)
    ceiling = _load_json(ceiling_summary_path)
    chart = _load_json(chart_summary_path)
    phase3 = _load_json(phase3_summary_path)
    phase3_csv_rows = _load_csv_rows(phase3_summary_path.parent / "model_empirical_alignment.csv") if phase3_summary_path else []

    paired_2d = ceiling.get("paired_alignment2_vs_eyeperm", {})
    paired_1d = ceiling.get("paired_alignment1_vs_eyeperm", {})
    paired_chart = chart.get("paired_matched_vs_shuffled_coord_R2_total", {})
    paired_fem = chart.get("paired_matched_vs_shuffled_fem_capture", {})
    join_diag = chart.get("step2_join_diagnostics", {})
    phase3_summary_by_basis = phase3.get("summary_by_basis", {})

    def phase3_metric(basis_name: str, *keys: str) -> float:
        return _safe_get(phase3_summary_by_basis, basis_name, *keys)

    def phase3_top1_ci(basis_name: str) -> dict:
        return _paired_delta_summary(
            _phase3_basis_rows(phase3_csv_rows, basis_name),
            "align_to_emp_top1_matched",
            "align_to_emp_top1_shuffled",
        )

    phase3_top1_ci_map = {
        "B_model": phase3_top1_ci("B_model"),
        "FEM_PCs": phase3_top1_ci("FEM_PCs"),
        "J_local": phase3_top1_ci("J_local"),
    }

    row = {
        "subject": subject,
        "date": date,
        "step2_summary_path": str(step2_summary_path) if step2_summary_path else "",
        "step2_residualized_path": str(step2_residual_path) if step2_residual_path else "",
        "ceiling_summary_path": str(ceiling_summary_path) if ceiling_summary_path else "",
        "chart_summary_path": str(chart_summary_path) if chart_summary_path else "",
        "phase3_summary_path": str(phase3_summary_path) if phase3_summary_path else "",
        "n_step2_windows": _safe_get(step2, "n_finite"),
        "n_ceiling_windows": _safe_get(ceiling, "n_windows"),
        "n_chart_windows": _safe_get(chart, "n_windows"),
        "n_windows_joined_to_step2": _safe_get(join_diag, "n_windows_joined_to_step2"),
        "fraction_joined_to_step2": _safe_get(join_diag, "fraction_joined_to_step2"),
        "n_joined_missing_trace_cov_model_fem": _safe_get(join_diag, "n_joined_missing_trace_cov_model_fem"),
        "n_joined_missing_e_fem_cv_median": _safe_get(join_diag, "n_joined_missing_e_fem_cv_median"),
        "median_emp_2d_alignment": _safe_get(ceiling, "median_emp_split_alignment_2d"),
        "median_eyeperm_2d_alignment": _safe_get(ceiling, "median_eye_perm_alignment_2d"),
        "delta_2d": _safe_get(ceiling, "median_alignment_delta_2d"),
        "frac_windows_delta_2d_positive": _safe_ratio(_safe_get(paired_2d, "positive_count"), _safe_get(paired_2d, "n")),
        "median_emp_top1_alignment": _safe_get(ceiling, "median_emp_split_alignment_top1"),
        "median_eyeperm_top1_alignment": _safe_get(ceiling, "median_eye_perm_alignment_top1"),
        "delta_top1": _safe_get(ceiling, "median_alignment_delta_top1"),
        "frac_windows_delta_top1_positive": _safe_ratio(_safe_get(paired_1d, "positive_count"), _safe_get(paired_1d, "n")),
        "r_trace_model_vs_e_fem_cv": _safe_get(step2, "spearman_model_trace_vs_e_fem_cv"),
        "r_trace_model_vs_e_fem_cv_residualized": _safe_get(step2_resid, "rho_trace_vs_Ecv_resid_full"),
        "matched_chart_coord_r2": _safe_get(chart, "summary_by_basis", "jacobian", "median_coord_R2_total"),
        "shuffled_chart_coord_r2": _safe_get(chart, "summary_by_basis", "shuffled", "median_coord_R2_total"),
        "matched_minus_shuffled_chart_r2": _safe_get(paired_chart, "median_delta"),
        "fem_chart_capture_matched": _safe_get(chart, "fem_sampling_summary_by_basis", "jacobian", "median_fem_chart_capture"),
        "fem_chart_capture_shuffled": _safe_get(chart, "fem_sampling_summary_by_basis", "shuffled", "median_fem_chart_capture"),
        "fem_chart_capture_delta": _safe_get(paired_fem, "median_delta"),
        "chart_coord_r2_matched_vs_e_fem_cv": _safe_get(chart, "paired_jacobian_vs_shuffled_chart_correlations", "spearman_chart_coord_R2_matched_vs_e_fem_cv_median"),
        "chart_coord_r2_delta_vs_e_fem_cv": _safe_get(chart, "paired_jacobian_vs_shuffled_chart_correlations", "spearman_chart_coord_R2_delta_vs_e_fem_cv_median"),
        "chart_coord_r2_delta_vs_trace_cov_model_fem": _safe_get(chart, "paired_jacobian_vs_shuffled_chart_correlations", "spearman_chart_coord_R2_delta_vs_trace_cov_model_fem"),
        "fem_chart_capture_delta_vs_e_fem_cv": _safe_get(chart, "paired_jacobian_vs_shuffled_fem_correlations", "spearman_fem_chart_capture_delta_vs_e_fem_cv_median"),
        "trace_cov_model_fem_vs_e_fem_cv": _safe_get(step2, "spearman_model_trace_vs_e_fem_cv"),
        "phase3_b_model_2d_delta": phase3_metric("B_model", "median_align_to_emp_2d_delta"),
        "phase3_b_model_2d_ci_low": phase3_metric("B_model", "paired_align_to_emp_2d", "ci95_low"),
        "phase3_b_model_2d_ci_high": phase3_metric("B_model", "paired_align_to_emp_2d", "ci95_high"),
        "phase3_b_model_top1_delta": phase3_metric("B_model", "median_align_to_emp_top1_delta"),
        "phase3_b_model_top1_ci_low": _safe_get(phase3_top1_ci_map, "B_model", "ci95_low"),
        "phase3_b_model_top1_ci_high": _safe_get(phase3_top1_ci_map, "B_model", "ci95_high"),
        "phase3_fem_pcs_2d_delta": phase3_metric("FEM_PCs", "median_align_to_emp_2d_delta"),
        "phase3_fem_pcs_2d_ci_low": phase3_metric("FEM_PCs", "paired_align_to_emp_2d", "ci95_low"),
        "phase3_fem_pcs_2d_ci_high": phase3_metric("FEM_PCs", "paired_align_to_emp_2d", "ci95_high"),
        "phase3_fem_pcs_top1_delta": phase3_metric("FEM_PCs", "median_align_to_emp_top1_delta"),
        "phase3_fem_pcs_top1_ci_low": _safe_get(phase3_top1_ci_map, "FEM_PCs", "ci95_low"),
        "phase3_fem_pcs_top1_ci_high": _safe_get(phase3_top1_ci_map, "FEM_PCs", "ci95_high"),
        "phase3_j_local_2d_delta": phase3_metric("J_local", "median_align_to_emp_2d_delta"),
        "phase3_j_local_2d_ci_low": phase3_metric("J_local", "paired_align_to_emp_2d", "ci95_low"),
        "phase3_j_local_2d_ci_high": phase3_metric("J_local", "paired_align_to_emp_2d", "ci95_high"),
        "phase3_j_local_top1_delta": phase3_metric("J_local", "median_align_to_emp_top1_delta"),
        "phase3_j_local_top1_ci_low": _safe_get(phase3_top1_ci_map, "J_local", "ci95_low"),
        "phase3_j_local_top1_ci_high": _safe_get(phase3_top1_ci_map, "J_local", "ci95_high"),
    }
    return row


def _write_csv(rows: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "cross_session_decision_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_f):
        return "NA"
    return f"{value_f:.3f}"


def _write_markdown(rows: list[dict], output_dir: Path) -> None:
    md_path = output_dir / "cross_session_decision_table.md"
    headers = [
        "Session",
        "Emp 2D",
        "Eye-perm 2D",
        "Delta 2D",
        "Frac + 2D",
        "Emp top1",
        "Eye-perm top1",
        "Delta top1",
        "Trace vs Ecv",
        "Trace vs Ecv resid",
        "Chart matched",
        "Chart shuffled",
        "Chart delta",
        "FEM capture matched",
        "FEM capture shuffled",
        "FEM capture delta",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        session = row["date"]
        lines.append(
            "| " + " | ".join([
                session,
                _format_value(row["median_emp_2d_alignment"]),
                _format_value(row["median_eyeperm_2d_alignment"]),
                _format_value(row["delta_2d"]),
                _format_value(row["frac_windows_delta_2d_positive"]),
                _format_value(row["median_emp_top1_alignment"]),
                _format_value(row["median_eyeperm_top1_alignment"]),
                _format_value(row["delta_top1"]),
                _format_value(row["r_trace_model_vs_e_fem_cv"]),
                _format_value(row["r_trace_model_vs_e_fem_cv_residualized"]),
                _format_value(row["matched_chart_coord_r2"]),
                _format_value(row["shuffled_chart_coord_r2"]),
                _format_value(row["matched_minus_shuffled_chart_r2"]),
                _format_value(row["fem_chart_capture_matched"]),
                _format_value(row["fem_chart_capture_shuffled"]),
                _format_value(row["fem_chart_capture_delta"]),
            ]) + " |"
        )
    lines.append("")
    lines.append("## Phase 3 Summary")
    lines.append("- Matched finite-displacement model geometries beat image-shuffled controls in multiple sessions.")
    lines.append("- FEM_PCs are the most consistent object.")
    lines.append("- B_model is clearly positive in 02-24 and 04-08.")
    lines.append("- J_local does not carry the empirical bridge.")
    lines.append("")
    phase3_headers = [
        "Session",
        "B_model 2D delta",
        "B_model 2D CI",
        "B_model top1 delta",
        "B_model top1 CI",
        "FEM_PCs 2D delta",
        "FEM_PCs 2D CI",
        "FEM_PCs top1 delta",
        "FEM_PCs top1 CI",
        "J_local 2D delta",
        "J_local 2D CI",
        "J_local top1 delta",
        "J_local top1 CI",
    ]
    lines.append("| " + " | ".join(phase3_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(phase3_headers)) + "|")
    for row in rows:
        lines.append(
            "| " + " | ".join([
                row["date"],
                _format_value(row["phase3_b_model_2d_delta"]),
                f"[{_format_value(row['phase3_b_model_2d_ci_low'])}, {_format_value(row['phase3_b_model_2d_ci_high'])}]",
                _format_value(row["phase3_b_model_top1_delta"]),
                f"[{_format_value(row['phase3_b_model_top1_ci_low'])}, {_format_value(row['phase3_b_model_top1_ci_high'])}]",
                _format_value(row["phase3_fem_pcs_2d_delta"]),
                f"[{_format_value(row['phase3_fem_pcs_2d_ci_low'])}, {_format_value(row['phase3_fem_pcs_2d_ci_high'])}]",
                _format_value(row["phase3_fem_pcs_top1_delta"]),
                f"[{_format_value(row['phase3_fem_pcs_top1_ci_low'])}, {_format_value(row['phase3_fem_pcs_top1_ci_high'])}]",
                _format_value(row["phase3_j_local_2d_delta"]),
                f"[{_format_value(row['phase3_j_local_2d_ci_low'])}, {_format_value(row['phase3_j_local_2d_ci_high'])}]",
                _format_value(row["phase3_j_local_top1_delta"]),
                f"[{_format_value(row['phase3_j_local_top1_ci_low'])}, {_format_value(row['phase3_j_local_top1_ci_high'])}]",
            ]) + " |"
        )
    lines.append("")
    lines.append("## Validation")
    for row in rows:
        lines.append(f"### {row['subject']}_{row['date']}")
        lines.append(f"- step2_summary_path: {row['step2_summary_path'] or 'NA'}")
        lines.append(f"- step2_residualized_path: {row['step2_residualized_path'] or 'NA'}")
        lines.append(f"- ceiling_summary_path: {row['ceiling_summary_path'] or 'NA'}")
        lines.append(f"- chart_summary_path: {row['chart_summary_path'] or 'NA'}")
        lines.append(f"- phase3_summary_path: {row['phase3_summary_path'] or 'NA'}")
        lines.append(f"- n_step2_windows: {_format_value(row['n_step2_windows'])}")
        lines.append(f"- n_ceiling_windows: {_format_value(row['n_ceiling_windows'])}")
        lines.append(f"- n_chart_windows: {_format_value(row['n_chart_windows'])}")
        lines.append(f"- n_windows_joined_to_step2: {_format_value(row['n_windows_joined_to_step2'])}")
        lines.append(f"- fraction_joined_to_step2: {_format_value(row['fraction_joined_to_step2'])}")
        lines.append(f"- n_joined_missing_trace_cov_model_fem: {_format_value(row['n_joined_missing_trace_cov_model_fem'])}")
        lines.append(f"- n_joined_missing_e_fem_cv_median: {_format_value(row['n_joined_missing_e_fem_cv_median'])}")
    md_path.write_text("\n".join(lines) + "\n")


def _write_json(rows: list[dict], output_dir: Path) -> None:
    (output_dir / "cross_session_decision_table.json").write_text(json.dumps(rows, indent=2) + "\n")


def _write_figure(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    sessions = [row["date"] for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    axes[0, 0].bar(x, [row["delta_2d"] for row in rows], color="#2c7fb8")
    axes[0, 0].set_title("Empirical Geometry Delta (2D)")
    axes[0, 0].set_xticks(x, sessions, rotation=20)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)

    axes[0, 1].bar(x, [row["delta_top1"] for row in rows], color="#41ab5d")
    axes[0, 1].set_title("Empirical Geometry Delta (Top-1)")
    axes[0, 1].set_xticks(x, sessions, rotation=20)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)

    axes[1, 0].bar(x, [row["r_trace_model_vs_e_fem_cv"] for row in rows], color="#f16913")
    axes[1, 0].set_title("Scalar Bridge: Trace vs E_FEM_cv")
    axes[1, 0].set_xticks(x, sessions, rotation=20)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)

    axes[1, 1].bar(x, [row["matched_minus_shuffled_chart_r2"] for row in rows], color="#756bb1")
    axes[1, 1].set_title("Chart Recovery Delta")
    axes[1, 1].set_xticks(x, sessions, rotation=20)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)

    fig.savefig(output_dir / "cross_session_decision_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest mapping Session -> explicit step2/ceiling/chart summary paths.")
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session to include, formatted as Subject:YYYY-MM-DD. Defaults to the four Allen sessions.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.manifest)
    rows = [_build_row(subject, date, manifest) for subject, date in _parse_sessions(args.session)]
    _write_csv(rows, output_dir)
    _write_json(rows, output_dir)
    _write_markdown(rows, output_dir)
    _write_figure(rows, output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
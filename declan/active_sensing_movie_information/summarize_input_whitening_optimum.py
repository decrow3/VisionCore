#!/usr/bin/env python3
"""Regenerate input-whitening summaries from cached runner outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from non_circular_fem_common import DEFAULT_STACK_OUT_DIR, parse_float, read_csv_rows, write_csv_rows, write_json
from run_input_whitening_optimum import (
    passband_sensitivity,
    summarize_scale,
    write_bootstrap_placeholder,
    write_figures,
    write_summary_md,
)


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "input_whitening"


def pooled_drift_summary(out_dir: Path) -> dict[str, float]:
    rows = read_csv_rows(out_dir / "drift_diffusion_estimates.csv")
    for row in rows:
        if row.get("scope") == "pooled":
            return {
                "D_eye_deg2_per_s": parse_float(row.get("D_eye_deg2_per_s")),
                "D_eye_arcmin2_per_s": parse_float(row.get("D_eye_arcmin2_per_s")),
                "fit_r2": parse_float(row.get("fit_r2")),
                "n_trace_windows": parse_float(row.get("n_trace_windows")),
            }
    return {
        "D_eye_deg2_per_s": float("nan"),
        "D_eye_arcmin2_per_s": float("nan"),
        "fit_r2": float("nan"),
        "n_trace_windows": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metrics_rows = read_csv_rows(out_dir / "whitening_movie_metrics.csv")
    if not metrics_rows:
        raise FileNotFoundError(f"No whitening_movie_metrics.csv found under {out_dir}")
    scale_summary = summarize_scale(metrics_rows)
    sensitivity = passband_sensitivity(scale_summary)
    write_csv_rows(out_dir / "whitening_scale_summary.csv", scale_summary)
    write_csv_rows(out_dir / "whitening_passband_sensitivity.csv", sensitivity)
    write_bootstrap_placeholder(out_dir)
    write_figures(out_dir, metrics_rows, scale_summary)
    write_summary_md(out_dir, pooled_drift_summary(out_dir), sensitivity)
    write_json(
        out_dir / "input_whitening_summary_manifest.json",
        {
            "out_dir": out_dir,
            "n_metric_rows": len(metrics_rows),
            "n_scale_summary_rows": len(scale_summary),
            "n_passband_sensitivity_rows": len(sensitivity),
            "bootstrap_status": "not_computed",
        },
    )
    print(f"Regenerated input-whitening summaries in {out_dir}")


if __name__ == "__main__":
    main()

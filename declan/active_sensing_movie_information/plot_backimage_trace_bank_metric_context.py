#!/usr/bin/env python3
"""Regenerate trace-bank metric plots with path-length context bands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    plot_trace_bank_diffusion_distribution,
    plot_trace_bank_metric_summary_panel,
    plot_trace_bank_microsaccade_diffusion_histogram,
    trace_bank_path_length_context_windows,
    write_csv_rows,
)


DEFAULT_TRACE_BANK_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/"
    "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-bank-csv", type=Path, default=DEFAULT_TRACE_BANK_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--scale-metric", type=str, default="rendered_diffusion_constant_deg2_s")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_existing_bin_summary(out_dir: Path) -> list[dict[str, Any]] | None:
    for name in (
        "trace_bank_filtered_summary.json",
        "trace_bank_large_sample_summary.json",
        "run_metadata.json",
    ):
        path = out_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        bin_summary = payload.get("bin_summary")
        if isinstance(bin_summary, list):
            return [dict(row) for row in bin_summary if isinstance(row, dict)]
    return None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    trace_bank_csv = Path(args.trace_bank_csv)
    out_dir = Path(args.out_dir) if args.out_dir is not None else trace_bank_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_bank_rows = list(csv.DictReader(trace_bank_csv.open(newline="", encoding="utf-8")))
    if not trace_bank_rows:
        raise ValueError(f"No trace-bank rows in {trace_bank_csv}")

    context_rows = trace_bank_path_length_context_windows(trace_bank_rows)
    context_csv = out_dir / "trace_bank_path_length_context_windows.csv"
    write_csv_rows(context_csv, context_rows)
    bin_summary = load_existing_bin_summary(out_dir)
    metric_panel = plot_trace_bank_metric_summary_panel(out_dir, trace_bank_rows, dpi=int(args.dpi))
    diffusion = plot_trace_bank_diffusion_distribution(
        out_dir,
        trace_bank_rows,
        scale_metric=str(args.scale_metric),
        bin_summary=bin_summary,
        dpi=int(args.dpi),
    )
    microsaccade = plot_trace_bank_microsaccade_diffusion_histogram(out_dir, trace_bank_rows, dpi=int(args.dpi))
    summary = {
        "trace_bank_csv": trace_bank_csv,
        "out_dir": out_dir,
        "n_trace_bank_rows": len(trace_bank_rows),
        "context_windows_csv": context_csv,
        "existing_bin_summary_reused": bool(bin_summary),
        "metric_summary_panel_png": metric_panel[0] if metric_panel is not None else None,
        "metric_summary_panel_pdf": metric_panel[1] if metric_panel is not None else None,
        "diffusion_distribution_png": diffusion[0] if diffusion is not None else None,
        "diffusion_distribution_pdf": diffusion[1] if diffusion is not None else None,
        "diffusion_by_microsaccade_png": microsaccade[0] if microsaccade is not None else None,
        "diffusion_by_microsaccade_pdf": microsaccade[1] if microsaccade is not None else None,
    }
    summary_path = out_dir / "trace_bank_metric_context_plot_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

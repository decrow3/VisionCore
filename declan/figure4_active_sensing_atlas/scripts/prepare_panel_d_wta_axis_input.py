#!/usr/bin/env python3
"""Prepare a paired WTA-axis input table for Figure 4D decoding reruns."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

try:
    from declan.figure4_active_sensing_atlas.scripts.run_panel_d_wta_behavior_diagnostic import (
        _axis_delta_deg,
        _crop_patch,
        _wta_axis_from_patch,
    )
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.figure4_active_sensing_atlas.scripts.run_panel_d_wta_behavior_diagnostic import (
        _axis_delta_deg,
        _crop_patch,
        _wta_axis_from_patch,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_INPUT = BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
DEFAULT_MANIFEST = BASE / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1" / "selected_windows.csv"
DEFAULT_OUT_DIR = BASE / "backimage_wta_orientation_axis_input_v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def build(args: argparse.Namespace) -> Path:
    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.input)
    windows["source_row"] = np.arange(windows.shape[0], dtype=int)
    manifest = pd.read_csv(args.manifest)
    if "source_row" not in manifest.columns:
        raise ValueError("--manifest must contain source_row for paired WTA-axis preparation")
    requested = manifest["source_row"].astype(int).drop_duplicates().to_list()
    missing = sorted(set(requested).difference(set(windows["source_row"].astype(int))))
    if missing:
        preview = ", ".join(str(v) for v in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"manifest source_row values are not present in input: {preview}{suffix}")

    for col in [
        "wta_edge_axis_deg",
        "wta_edge_axis_array_deg",
        "wta_peak_fraction",
        "wta_hist_peak_bin",
        "wta_total_gradient_energy",
        "wta_n_edge_pixels",
        "wta_average_axis_delta_deg",
    ]:
        if col not in windows.columns:
            windows[col] = np.nan
    if "wta_ok" not in windows.columns:
        windows["wta_ok"] = False
    if "wta_error" not in windows.columns:
        windows["wta_error"] = ""

    records: list[dict[str, Any]] = []
    lookup = windows.set_index("source_row", drop=False)
    for idx, source_row in enumerate(requested, start=1):
        row = lookup.loc[int(source_row)]
        patch, _ppd, _center = _crop_patch(row)
        wta = _wta_axis_from_patch(
            patch,
            n_bins=int(args.n_bins),
            energy_quantile=float(args.energy_quantile),
        )
        for key, value in wta.items():
            if key == "wta_ok":
                windows.loc[windows["source_row"] == int(source_row), key] = bool(value)
            else:
                windows.loc[windows["source_row"] == int(source_row), key] = value
        if bool(wta.get("wta_ok", 0.0)):
            delta = abs(
                float(
                    _axis_delta_deg(
                        float(wta["wta_edge_axis_deg"]),
                        float(row["image_edge_axis_deg"]),
                    )
                )
            )
            windows.loc[windows["source_row"] == int(source_row), "wta_average_axis_delta_deg"] = delta
        rec = {
            "source_row": int(source_row),
            "session": str(row["session"]),
            "trial_idx": int(row["trial_idx"]),
            "image_edge_axis_deg": float(row["image_edge_axis_deg"]),
            "wta_ok": bool(wta.get("wta_ok", 0.0)),
            "wta_edge_axis_deg": float(wta.get("wta_edge_axis_deg", np.nan)),
            "wta_peak_fraction": float(wta.get("wta_peak_fraction", np.nan)),
            "wta_average_axis_delta_deg": float(
                windows.loc[windows["source_row"] == int(source_row), "wta_average_axis_delta_deg"].iloc[0]
            ),
            "wta_error": str(wta.get("wta_error", "")),
        }
        records.append(rec)
        if idx == len(requested) or idx % int(args.progress_every) == 0:
            print(f"prepared WTA axes for {idx}/{len(requested)} manifest windows", flush=True)

    selected = windows[windows["source_row"].isin(requested)].copy()
    bad = selected[~selected["wta_ok"].astype(bool) | ~np.isfinite(pd.to_numeric(selected["wta_edge_axis_deg"], errors="coerce"))]
    if not bad.empty:
        raise ValueError(f"WTA axis failed for {bad.shape[0]} selected rows; see selected_wta_axis_values.csv")

    out_csv = out_dir / "backimage_image_fem_windows_wta_axis.csv"
    values_csv = out_dir / "selected_wta_axis_values.csv"
    windows.to_csv(out_csv, index=False)
    pd.DataFrame(records).to_csv(values_csv, index=False)

    metadata = {
        "input": Path(args.input),
        "manifest": Path(args.manifest),
        "output_csv": out_csv,
        "selected_values_csv": values_csv,
        "n_input_rows": int(windows.shape[0]),
        "n_manifest_source_rows": int(len(requested)),
        "n_wta_ok": int(selected["wta_ok"].astype(bool).sum()),
        "n_bins": int(args.n_bins),
        "energy_quantile": float(args.energy_quantile),
        "elapsed_s": float(time.time() - start),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_csv, flush=True)
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--energy-quantile", type=float, default=0.75)
    parser.add_argument("--progress-every", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()

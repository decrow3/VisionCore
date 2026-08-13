#!/usr/bin/env python3
"""Extract one small aperture for the original-matrix Panel G audit.

This intentionally runs one selected trial per process. BackImage canvas
construction has process-level caches large enough that extracting all selected
trials in one plotting process can exhaust host memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import _extract_patch


ROOT = Path(__file__).resolve().parents[4]
BANK_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
AUDIT_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_original_matrix_pair_rotation_audit_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_index", type=int)
    args = parser.parse_args()
    selection = pd.read_csv(AUDIT_ROOT / "frozen_pair_selection.csv").set_index("selection_index")
    selected = selection.loc[int(args.selection_index)]
    images = pd.read_csv(BANK_DIR / "image_feature_table.csv").set_index("image_index", drop=False)
    image_row = images.loc[int(selected["image_index"])]
    patch, meta = _extract_patch(image_row, canvas_cache={}, patch_size_px=540)
    ppd = float(meta["patch_ppd"])
    half = int(round(ppd / 2.0))
    cy, cx = np.asarray(patch.shape) // 2
    aperture = np.asarray(patch[cy - half : cy + half, cx - half : cx + half]).copy()
    out_dir = AUDIT_ROOT / "input_aperture_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"selection_{int(args.selection_index):02d}.npz",
        aperture=aperture,
        ppd=np.asarray([ppd], dtype=np.float32),
        image_index=np.asarray([int(selected["image_index"])], dtype=np.int32),
        image_source_row=np.asarray([int(selected["image_source_row"])], dtype=np.int32),
    )
    print(f"extracted selection {int(args.selection_index):02d}: {aperture.shape}", flush=True)


if __name__ == "__main__":
    main()

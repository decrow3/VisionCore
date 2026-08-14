#!/usr/bin/env python3
"""Map-first actual-data validation of the lag-zero BackImage renderer.

The checkpoint makes two distinct comparisons on matched recorded conditions:

1. direct lag-zero renderer versus the validated lag-embedded helper;
2. corrected reconstruction versus the exact saved 51x51 BackImage frames.

It scores no neural model and stops at concrete input maps and per-frame errors.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    centered,
    corrected_crop_xy_deg,
    load_dset,
    model_aligned_indices,
)
from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import corrected_patch
from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, _cached_session
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)


ROOT = Path(__file__).resolve().parents[2]
IMAGE_AUDIT = ROOT / "outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1"
SOURCE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_input_only_renderer_actual_validation_checkpoint_31_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-audit-dir", type=Path, default=IMAGE_AUDIT)
    parser.add_argument("--source-csv", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def select_examples(frame: pd.DataFrame, n_examples: int) -> pd.DataFrame:
    valid = frame.loc[frame["corrected_crop_valid"].astype(bool)].sort_values("reconstruction_exact_pixel_r")
    if len(valid) < int(n_examples):
        raise ValueError(f"Only {len(valid)} crop-valid images for {n_examples} examples")
    positions = np.linspace(0, len(valid) - 1, int(n_examples)).round().astype(int)
    selected = valid.iloc[positions].copy().drop_duplicates("image_index")
    if len(selected) != int(n_examples):
        raise AssertionError("Quantile example selection produced duplicate identities")
    selected["selection_role"] = [f"renderer_agreement_rank_{index + 1}_of_{len(selected)}" for index in range(len(selected))]
    selected["selection_criterion"] = "evenly spaced reconstruction-versus-exact pixel-agreement ranks among crop-valid images"
    selected["selection_is_algorithmic"] = True
    return selected


def framewise_metrics(direct: np.ndarray, exact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correlations = []
    maes = []
    for reconstructed, observed in zip(direct, exact, strict=True):
        correlations.append(float(np.corrcoef(reconstructed.ravel(), observed.ravel())[0, 1]))
        maes.append(float(np.mean(np.abs(reconstructed - observed))))
    return np.asarray(correlations), np.asarray(maes)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation checkpoint: {args.out_dir}")
    crosswalk_path = args.image_audit_dir / "corrected_image_crosswalk.csv"
    crosswalk = pd.read_csv(crosswalk_path)
    selected = select_examples(crosswalk, int(args.n_examples))
    sources = load_source_rows(args.source_csv)
    common = _load_twin_common()
    records: list[dict[str, object]] = []
    arrays: dict[int, dict[str, np.ndarray]] = {}

    # Process by session so no two multi-gigabyte BackImage datasets coexist.
    for session, session_rows in selected.groupby("session", sort=True):
        dset_cache: dict = {}
        canvas_cache: dict = {}
        dset = load_dset(str(session), dset_cache)
        for row in session_rows.itertuples(index=False):
            source = source_row_by_id(sources, int(row.source_row))
            patch, patch_meta, indices_from_patch = corrected_patch(source, dset, canvas_cache)
            indices = model_aligned_indices(int(source.global_start), int(source.global_stop))
            if not np.array_equal(indices, indices_from_patch):
                raise AssertionError("Corrected patch and exact-frame index contracts disagree")
            crop = centered(corrected_crop_xy_deg(dset)[indices]).astype(np.float32)
            retinal_trace = -crop
            standardized = _standardize_uint_like(patch)
            direct_tensor = render_retinal_frames_lag_zero(
                common,
                standardized,
                retinal_trace,
                ppd=float(patch_meta["patch_ppd"]),
                device=str(args.device),
            )
            direct = direct_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            full_stack = np.broadcast_to(
                standardized[None],
                (len(retinal_trace) + int(common.N_LAGS) + 1, *standardized.shape),
            ).copy()
            eye = torch.from_numpy(_trace_xy_to_twin_helper_order(retinal_trace))
            helper = common.make_counterfactual_stim(
                full_stack,
                eye,
                ppd=float(patch_meta["patch_ppd"]),
                n_lags=int(common.N_LAGS),
                out_size=(51, 51),
            )[1 : 1 + len(retinal_trace), 0, 0].detach().cpu().numpy()
            helper_error = float(np.max(np.abs(direct - helper)))
            if helper_error > 1e-6:
                raise AssertionError(f"Actual-data helper mismatch for image {row.image_index}: {helper_error}")
            exact = np.asarray(dset["stim"][indices], dtype=np.float32)
            if exact.shape != direct.shape or not np.isfinite(exact).all():
                raise ValueError(f"Invalid exact saved frames for image {row.image_index}: {exact.shape}")
            per_frame_r, per_frame_mae = framewise_metrics(direct, exact)
            record = {
                "image_index": int(row.image_index),
                "source_row": int(row.source_row),
                "session": str(session),
                "trial_idx": int(row.trial_idx),
                "n_frames": int(len(direct)),
                "direct_helper_max_abs_error": helper_error,
                "direct_exact_pixel_r": float(np.corrcoef(direct.ravel(), exact.ravel())[0, 1]),
                "direct_exact_mae": float(np.mean(np.abs(direct - exact))),
                "median_frame_pixel_r": float(np.median(per_frame_r)),
                "minimum_frame_pixel_r": float(np.min(per_frame_r)),
                "median_frame_mae": float(np.median(per_frame_mae)),
                "selection_role": str(row.selection_role),
                "selection_criterion": str(row.selection_criterion),
                "selection_is_algorithmic": True,
            }
            records.append(record)
            arrays[int(row.image_index)] = {
                "patch": np.asarray(standardized, dtype=np.float32),
                "trace": retinal_trace,
                "direct": direct,
                "exact": exact,
                "absolute_difference": np.abs(direct - exact),
                "per_frame_pixel_r": per_frame_r,
                "per_frame_mae": per_frame_mae,
            }
            del direct_tensor, direct, helper, exact, full_stack
        del dset
        dset_cache.clear()
        canvas_cache.clear()
        _backimage_canvas.cache_clear()
        _cached_session.cache_clear()
        gc.collect()

    metrics = pd.DataFrame(records).sort_values("direct_exact_pixel_r").reset_index(drop=True)
    if len(metrics) != int(args.n_examples):
        raise AssertionError("Actual validation did not produce every selected example")
    if float(metrics["direct_helper_max_abs_error"].max()) > 1e-6:
        raise AssertionError("Lag-zero renderer failed helper equivalence gate")
    if float(metrics["direct_exact_pixel_r"].median()) < 0.80:
        raise AssertionError("Median reconstruction-versus-exact correlation fell below 0.80")

    args.out_dir.mkdir(parents=True)
    metrics.to_csv(args.out_dir / "actual_input_validation_metrics.csv", index=False)
    selected.merge(
        metrics[["image_index", "direct_helper_max_abs_error", "direct_exact_pixel_r", "direct_exact_mae"]],
        on="image_index",
        validate="one_to_one",
    ).to_csv(args.out_dir / "selected_actual_input_examples.csv", index=False)
    for image_index, data in arrays.items():
        np.savez_compressed(args.out_dir / f"actual_input_example_{image_index:03d}.npz", **data)

    fig, axes = plt.subplots(len(metrics), 5, figsize=(14.5, 2.55 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    for axis_row, row in zip(axes, metrics.itertuples(index=False), strict=True):
        data = arrays[int(row.image_index)]
        patch = data["patch"]
        center = np.asarray(patch.shape) // 2
        view = patch[center[0] - 80 : center[0] + 80, center[1] - 80 : center[1] + 80]
        axis_row[0].imshow(view, cmap="gray", vmin=0, vmax=255)
        trace_arcmin = data["trace"] * 60.0
        axis_row[1].plot(trace_arcmin[:, 0], trace_arcmin[:, 1], color="#333333", lw=1.3)
        axis_row[1].scatter(trace_arcmin[0, 0], trace_arcmin[0, 1], s=22, color="#009E73")
        axis_row[1].set_aspect("equal", adjustable="datalim")
        axis_row[2].imshow(data["direct"].mean(0), cmap="gray", vmin=0, vmax=255)
        axis_row[3].imshow(data["exact"].mean(0), cmap="gray", vmin=0, vmax=255)
        vmax = max(float(np.percentile(data["absolute_difference"].mean(0), 99)), 1e-6)
        difference_image = axis_row[4].imshow(
            data["absolute_difference"].mean(0), cmap="magma", vmin=0, vmax=vmax
        )
        fig.colorbar(difference_image, ax=axis_row[4], shrink=0.62, pad=0.01)
        axis_row[0].set_ylabel(
            f"image {row.image_index}\nhelper err={row.direct_helper_max_abs_error:.1g}\nexact r={row.direct_exact_pixel_r:.3f}",
            fontsize=8,
        )
        for axis in (axis_row[0], axis_row[2], axis_row[3], axis_row[4]):
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(
        axes[0],
        (
            "corrected source patch",
            "recorded retinal path",
            "direct reconstruction",
            "exact saved input",
            "mean absolute difference",
        ),
        strict=True,
    ):
        axis.set_title(title, fontsize=9)
    fig.suptitle(
        "Actual-input checkpoint: lag-zero renderer matches the validated helper exactly\n"
        "and retains the known reconstruction-versus-saved-input residual",
        fontweight="bold",
    )
    fig.savefig(args.out_dir / "actual_input_renderer_validation.png", dpi=int(args.dpi))
    fig.savefig(args.out_dir / "actual_input_renderer_validation.pdf")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "actual_input_map_checkpoint_complete_stop_before_population_cache",
        "scope": f"{len(metrics)} crop-valid matched recorded BackImage conditions; no neural model",
        "gates": {
            "direct_helper_max_abs_error": float(metrics["direct_helper_max_abs_error"].max()),
            "median_direct_exact_pixel_r": float(metrics["direct_exact_pixel_r"].median()),
            "minimum_direct_exact_pixel_r": float(metrics["direct_exact_pixel_r"].min()),
            "helper_equivalence_pass": True,
            "median_exact_input_agreement_pass_ge_0p80": True,
        },
        "sources": {
            "image_crosswalk": file_identity(crosswalk_path),
            "source_rows": file_identity(args.source_csv),
            "renderer": file_identity(ROOT / "declan/fig4_active_sensing/input_only_retinal_renderer.py"),
        },
        "next_gate": "review concrete input maps, then run a bounded streaming spectral-cache smoke test",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

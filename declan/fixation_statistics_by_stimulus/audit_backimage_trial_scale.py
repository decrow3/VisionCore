"""Audit BackImage source-to-display scaling on a trial-by-trial basis.

This is a stimulus/QC diagnostic.  It records the native image dimensions and
the recorded Psychtoolbox destination rectangle for every distinct trial in a
window manifest, then renders both a trial raster and concrete image examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from DataYatesV1 import MAT_DIR
from DataYatesV1.exp.support import get_backimage_directory


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/"
    "backimage_image_fem_windows.csv"
)
DEFAULT_OUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_trial_scale_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def collect_trials(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, usecols=["session", "trial_idx"])
    manifest = (
        manifest.drop_duplicates()
        .sort_values(["session", "trial_idx"], kind="stable")
        .reset_index(drop=True)
    )
    image_root = get_backimage_directory()
    size_cache: dict[str, tuple[int, int]] = {}
    rows: list[dict[str, object]] = []

    for session_name, block in manifest.groupby("session", sort=True):
        print(f"Loading {session_name} ({len(block)} BackImage trials)", flush=True)
        mat_path = Path(MAT_DIR) / f"{session_name}_struct.mat"
        # Read only the recorded stimulus fields.  Full get_session() loads
        # hundreds of MB per session and is unnecessary for this geometry QC.
        with h5py.File(mat_path, "r") as mat:
            screen_rect = np.asarray(mat["S"]["screenRect"], dtype=float).ravel()
            screen_width = float(screen_rect[2] - screen_rect[0])
            screen_height = float(screen_rect[3] - screen_rect[1])
            ppd = float(np.asarray(mat["S"]["pixPerDeg"]).squeeze())

            for session_order, trial_idx in enumerate(block["trial_idx"].astype(int)):
                trial_group = mat[mat["D"][trial_idx, 0]]
                pr = trial_group["PR"]
                parameters = trial_group["P"]
                raw_name = "".join(chr(int(value)) for value in np.asarray(pr["imagefile"]).ravel())
                image_file = raw_name.replace("\\", "/").split("/")[-1]
                if image_file not in size_cache:
                    with Image.open(image_root / image_file) as image:
                        size_cache[image_file] = tuple(map(int, image.size))
                source_width, source_height = size_cache[image_file]
                x0, y0, x1, y1 = map(int, np.asarray(pr["destRect"]).ravel())
                dest_width = x1 - x0
                dest_height = y1 - y0
                scale_x = dest_width / source_width
                scale_y = dest_height / source_height
                relative_horizontal_magnification = scale_x / scale_y
                configured_sizes = (
                    np.asarray(parameters["imageSizes"], dtype=float).ravel()
                    if "imageSizes" in parameters
                    else np.asarray([], dtype=float)
                )
                display_width_deg = dest_width / ppd
                nominal_size_deg = (
                    float(configured_sizes[np.argmin(np.abs(configured_sizes - display_width_deg))])
                    if configured_sizes.size
                    else np.nan
                )
                rows.append({
                    "session": session_name,
                    "trial_idx": trial_idx,
                    "session_trial_order": session_order,
                    "image_file": image_file,
                    "source_width_px": source_width,
                    "source_height_px": source_height,
                    "source_aspect": source_width / source_height,
                    "dest_x0_px": x0,
                    "dest_y0_px": y0,
                    "dest_x1_px": x1,
                    "dest_y1_px": y1,
                    "dest_width_px": dest_width,
                    "dest_height_px": dest_height,
                    "dest_aspect": dest_width / dest_height,
                    "screen_width_px": screen_width,
                    "screen_height_px": screen_height,
                    "pix_per_deg": ppd,
                    "display_width_deg": display_width_deg,
                    "display_height_deg": dest_height / ppd,
                    "configured_image_sizes_deg": ",".join(f"{x:g}" for x in configured_sizes),
                    "nominal_size_deg": nominal_size_deg,
                    "screen_area_fraction": (dest_width * dest_height)
                    / (screen_width * screen_height),
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "relative_horizontal_magnification": relative_horizontal_magnification,
                    "aspect_preserved": bool(
                        np.isclose(relative_horizontal_magnification, 1.0, rtol=0.0, atol=0.01)
                    ),
                })
    return pd.DataFrame(rows)


def summarize_images(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "image_file",
        "source_width_px",
        "source_height_px",
        "source_aspect",
    ]
    summary = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            n_trials=("trial_idx", "size"),
            n_sessions=("session", "nunique"),
            n_dest_rects=("dest_aspect", "size"),
            median_dest_aspect=("dest_aspect", "median"),
            median_relative_horizontal_magnification=(
                "relative_horizontal_magnification",
                "median",
            ),
            min_relative_horizontal_magnification=(
                "relative_horizontal_magnification",
                "min",
            ),
            max_relative_horizontal_magnification=(
                "relative_horizontal_magnification",
                "max",
            ),
        )
        .sort_values(
            ["median_relative_horizontal_magnification", "image_file"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    # Correct the count above: this is explicitly the number of distinct rectangles.
    rect_counts = trials.groupby("image_file").apply(
        lambda x: x[["dest_x0_px", "dest_y0_px", "dest_x1_px", "dest_y1_px"]]
        .drop_duplicates()
        .shape[0]
    )
    summary["n_dest_rects"] = summary["image_file"].map(rect_counts).astype(int)
    return summary


def select_examples(trials: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    candidates = images.copy()
    candidates["distance_from_unity"] = np.abs(
        candidates["median_relative_horizontal_magnification"] - 1.0
    )
    roles = [
        (
            "native_aspect_control",
            candidates.sort_values(["distance_from_unity", "n_trials"], ascending=[True, False]).iloc[0],
            "minimum absolute deviation of horizontal/vertical magnification from 1",
        ),
        (
            "common_4_to_3_stretch",
            candidates[np.isclose(candidates["source_aspect"], 4 / 3, atol=0.01)]
            .sort_values("n_trials", ascending=False)
            .iloc[0],
            "most frequently presented native 4:3 image",
        ),
        (
            "maximum_aspect_stretch",
            candidates.sort_values(
                ["median_relative_horizontal_magnification", "n_trials"],
                ascending=[False, False],
            ).iloc[0],
            "maximum median horizontal/vertical magnification ratio",
        ),
    ]
    rows = []
    for role, image_row, criterion in roles:
        example = trials[trials["image_file"] == image_row["image_file"]].iloc[0]
        rows.append(
            {
                "selection_role": role,
                "criterion": criterion,
                "session": example["session"],
                "trial_idx": int(example["trial_idx"]),
                "image_file": example["image_file"],
                "source_width_px": int(example["source_width_px"]),
                "source_height_px": int(example["source_height_px"]),
                "dest_width_px": int(example["dest_width_px"]),
                "dest_height_px": int(example["dest_height_px"]),
                "relative_horizontal_magnification": float(
                    example["relative_horizontal_magnification"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_trial_raster(trials: pd.DataFrame, output_path: Path) -> None:
    sessions = sorted(trials["session"].unique())
    width = int(trials["session_trial_order"].max()) + 1
    values = np.full((len(sessions), width), np.nan)
    for row_idx, session_name in enumerate(sessions):
        block = trials[trials["session"] == session_name]
        values[row_idx, block["session_trial_order"].to_numpy(dtype=int)] = block[
            "relative_horizontal_magnification"
        ]

    fig, ax = plt.subplots(figsize=(12, 8.5), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=1.0, vmax=1.8, cmap="magma")
    ax.set_yticks(np.arange(len(sessions)), labels=sessions, fontsize=7)
    ax.set_xlabel("BackImage trial order within session")
    ax.set_ylabel("Session")
    ax.set_title(
        "Recorded BackImage aspect scaling, trial by trial\n"
        "color = horizontal magnification / vertical magnification"
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label("relative horizontal magnification (1 = shape preserved)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_display_size_raster(trials: pd.DataFrame, output_path: Path) -> None:
    sessions = sorted(trials["session"].unique())
    width = int(trials["session_trial_order"].max()) + 1
    values = np.full((len(sessions), width), np.nan)
    for row_idx, session_name in enumerate(sessions):
        block = trials[trials["session"] == session_name]
        values[row_idx, block["session_trial_order"].to_numpy(dtype=int)] = block[
            "display_width_deg"
        ]

    fig, ax = plt.subplots(figsize=(12, 8.5), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=4.0, vmax=35.0, cmap="viridis")
    ax.set_yticks(np.arange(len(sessions)), labels=sessions, fontsize=7)
    ax.set_xlabel("BackImage trial order within session")
    ax.set_ylabel("Session")
    ax.set_title(
        "Recorded BackImage display width, trial by trial\n"
        "small terminal blocks are configured 4°, 8°, 16°, and 32° size conditions"
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label("display width (degrees)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _aspect_preserving_canvas(image: Image.Image, size: tuple[int, int], background: int = 127) -> Image.Image:
    canvas = Image.new("RGB", size, color=(background, background, background))
    fit = image.copy()
    fit.thumbnail(size, resample=Image.Resampling.BILINEAR)
    x0 = (size[0] - fit.width) // 2
    y0 = (size[1] - fit.height) // 2
    canvas.paste(fit, (x0, y0))
    return canvas


def plot_examples(selected: pd.DataFrame, output_path: Path) -> None:
    image_root = get_backimage_directory()
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 8), constrained_layout=True)
    column_titles = ["Native image (aspect preserved)", "Recorded display", "Aspect-preserving fit control"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row_idx, row in selected.reset_index(drop=True).iterrows():
        with Image.open(image_root / row["image_file"]) as opened:
            source = opened.convert("RGB")
            display_size = (int(row["dest_width_px"]), int(row["dest_height_px"]))
            recorded = source.resize(display_size, resample=Image.Resampling.BILINEAR)
            fitted = _aspect_preserving_canvas(source, display_size)
            panels = [source, recorded, fitted]
            for col, panel in enumerate(panels):
                axes[row_idx, col].imshow(panel)
                axes[row_idx, col].set_axis_off()
        axes[row_idx, 0].set_ylabel(str(row["selection_role"]), fontsize=9)
        axes[row_idx, 1].text(
            0.5,
            -0.06,
            f"{row['session']} trial {int(row['trial_idx'])}; "
            f"horizontal/vertical magnification = {row['relative_horizontal_magnification']:.3f}x",
            transform=axes[row_idx, 1].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    fig.suptitle("Concrete BackImage source-to-display scaling examples", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = collect_trials(args.input)
    images = summarize_images(trials)
    selected = select_examples(trials, images)

    trials.to_csv(args.out_dir / "backimage_trial_scale_audit.csv", index=False)
    images.to_csv(args.out_dir / "backimage_image_scale_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_examples.csv", index=False)
    plot_trial_raster(trials, args.out_dir / "backimage_trial_scale_raster.png")
    plot_display_size_raster(trials, args.out_dir / "backimage_trial_display_size_raster.png")
    plot_examples(selected, args.out_dir / "backimage_source_vs_display_examples.png")

    metadata = {
        "analysis": "backimage_trial_source_to_display_scale_audit",
        "input_manifest": str(args.input),
        "n_trials": int(len(trials)),
        "n_sessions": int(trials["session"].nunique()),
        "n_images": int(trials["image_file"].nunique()),
        "n_aspect_preserved_trials_atol_0p01": int(trials["aspect_preserved"].sum()),
        "n_aspect_changed_trials_atol_0p01": int((~trials["aspect_preserved"]).sum()),
        "n_full_screen_trials": int(np.isclose(trials["screen_area_fraction"], 1.0).sum()),
        "n_reduced_size_trials": int((trials["screen_area_fraction"] < 0.99).sum()),
        "n_oversized_trials": int((trials["screen_area_fraction"] > 1.01).sum()),
        "recorded_dest_rect_contract": "BackImageTrial.trial['PR']['destRect']",
        "source_image_contract": "PIL native size before BackImageTrial.get_image resize",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

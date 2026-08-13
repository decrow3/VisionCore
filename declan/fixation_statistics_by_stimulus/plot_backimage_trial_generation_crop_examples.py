"""Render exact BackImage source -> display -> local-contour crop examples."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from DataYatesV1 import MAT_DIR
from DataYatesV1.exp.support import get_backimage_directory


BASE = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
WINDOWS = BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
TRIALS = BASE / "backimage_trial_scale_audit/backimage_trial_scale_audit.csv"
OUT = BASE / "backimage_trial_generation_crop_checkpoint"
EXAMPLES = [
    ("full_screen_4x3", "Allen_2022-03-02", 343),
    ("reduced_8deg_4x3", "Allen_2022-03-02", 651),
]


def _raw_dest_rect(session: str, trial_idx: int) -> np.ndarray:
    with h5py.File(Path(MAT_DIR) / f"{session}_struct.mat", "r") as mat:
        trial = mat[mat["D"][int(trial_idx), 0]]
        return np.asarray(trial["PR"]["destRect"], dtype=float).ravel()


def _draw_axis(ax: plt.Axes, angle_deg: float) -> None:
    theta = np.radians(angle_deg)
    cx = cy = 38.0
    length = 31.0
    ax.plot(
        [cx - length * np.cos(theta), cx + length * np.cos(theta)],
        [cy + length * np.sin(theta), cy - length * np.sin(theta)],
        color="#00ffff",
        linewidth=2.0,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(WINDOWS)
    trials = pd.read_csv(TRIALS)
    image_root = get_backimage_directory()
    records: list[dict[str, object]] = []
    rendered: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []

    for role, session, trial_idx in EXAMPLES:
        trial = trials[(trials["session"] == session) & (trials["trial_idx"] == trial_idx)].iloc[0]
        window = (
            windows[(windows["session"] == session) & (windows["trial_idx"] == trial_idx)]
            .sort_values("global_start")
            .iloc[0]
        )
        with Image.open(image_root / trial["image_file"]) as opened:
            source_rgb = opened.convert("RGB")
            source = np.mean(np.asarray(source_rgb), axis=2)
            dest_w = int(trial["dest_width_px"])
            dest_h = int(trial["dest_height_px"])
            resized = np.mean(
                np.asarray(source_rgb.resize((dest_w, dest_h), resample=Image.Resampling.BILINEAR)),
                axis=2,
            )
        screen_h = int(trial["screen_height_px"])
        screen_w = int(trial["screen_width_px"])
        canvas = np.full((screen_h, screen_w), 127.0, dtype=np.float64)
        x0 = int(trial["dest_x0_px"])
        y0 = int(trial["dest_y0_px"])
        x1 = int(trial["dest_x1_px"])
        y1 = int(trial["dest_y1_px"])
        canvas[y0:y1, x0:x1] = resized

        ppd = float(trial["pix_per_deg"])
        center_x = screen_w / 2.0 + float(window["mean_x_deg"]) * ppd
        center_y = screen_h / 2.0 - float(window["mean_y_deg"]) * ppd
        radius = max(2, int(round(ppd)))
        crop_x0 = max(0, int(round(center_x)) - radius)
        crop_x1 = min(screen_w, int(round(center_x)) + radius + 1)
        crop_y0 = max(0, int(round(center_y)) - radius)
        crop_y1 = min(screen_h, int(round(center_y)) + radius + 1)
        patch = canvas[crop_y0:crop_y1, crop_x0:crop_x1]

        source_x0 = (crop_x0 - x0) * float(trial["source_width_px"]) / dest_w
        source_x1 = (crop_x1 - x0) * float(trial["source_width_px"]) / dest_w
        source_y0 = (crop_y0 - y0) * float(trial["source_height_px"]) / dest_h
        source_y1 = (crop_y1 - y0) * float(trial["source_height_px"]) / dest_h
        raw_dest = _raw_dest_rect(session, trial_idx)
        record = {
            "selection_role": role,
            "session": session,
            "trial_idx": trial_idx,
            "global_start": int(window["global_start"]),
            "global_stop": int(window["global_stop"]),
            "image_file": trial["image_file"],
            "source_width_px": int(trial["source_width_px"]),
            "source_height_px": int(trial["source_height_px"]),
            "raw_dest_x0": raw_dest[0],
            "raw_dest_y0": raw_dest[1],
            "raw_dest_x1": raw_dest[2],
            "raw_dest_y1": raw_dest[3],
            "analysis_dest_x0": x0,
            "analysis_dest_y0": y0,
            "analysis_dest_x1": x1,
            "analysis_dest_y1": y1,
            "mean_gaze_x_deg": float(window["mean_x_deg"]),
            "mean_gaze_y_deg": float(window["mean_y_deg"]),
            "crop_center_x_screen_px": center_x,
            "crop_center_y_screen_px": center_y,
            "crop_radius_px": radius,
            "crop_x0_screen_px": crop_x0,
            "crop_y0_screen_px": crop_y0,
            "crop_x1_screen_px_exclusive": crop_x1,
            "crop_y1_screen_px_exclusive": crop_y1,
            "patch_width_px": int(patch.shape[1]),
            "patch_height_px": int(patch.shape[0]),
            "source_footprint_x0_px": source_x0,
            "source_footprint_y0_px": source_y0,
            "source_footprint_x1_px": source_x1,
            "source_footprint_y1_px": source_y1,
            "image_edge_axis_deg": float(window["image_edge_axis_deg"]),
        }
        records.append(record)
        rendered.append((source, canvas, patch, record))

    pd.DataFrame(records).to_csv(OUT / "selected_trial_generation_crop_values.csv", index=False)

    fig, axes = plt.subplots(len(rendered), 3, figsize=(13, 7.5), constrained_layout=True)
    for col, title in enumerate(
        ["Native source (analysis footprint boxed)", "Reconstructed screen canvas", "Exact local contour crop"]
    ):
        axes[0, col].set_title(title)
    for row_idx, (source, canvas, patch, rec) in enumerate(rendered):
        axes[row_idx, 0].imshow(source, cmap="gray")
        axes[row_idx, 0].add_patch(
            Rectangle(
                (rec["source_footprint_x0_px"], rec["source_footprint_y0_px"]),
                rec["source_footprint_x1_px"] - rec["source_footprint_x0_px"],
                rec["source_footprint_y1_px"] - rec["source_footprint_y0_px"],
                fill=False,
                edgecolor="#ff00ff",
                linewidth=2,
            )
        )
        axes[row_idx, 1].imshow(canvas, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 1].add_patch(
            Rectangle(
                (rec["crop_x0_screen_px"], rec["crop_y0_screen_px"]),
                rec["patch_width_px"],
                rec["patch_height_px"],
                fill=False,
                edgecolor="#ff00ff",
                linewidth=2,
            )
        )
        axes[row_idx, 2].imshow(patch, cmap="gray", vmin=0, vmax=255)
        _draw_axis(axes[row_idx, 2], rec["image_edge_axis_deg"])
        axes[row_idx, 0].set_ylabel(str(rec["selection_role"]), fontsize=9)
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 2].text(
            1.03,
            0.5,
            f"{rec['session']} trial {rec['trial_idx']}\n{rec['image_file']}\n"
            f"raw destRect: [{rec['raw_dest_x0']:.1f}, {rec['raw_dest_y0']:.1f}, "
            f"{rec['raw_dest_x1']:.1f}, {rec['raw_dest_y1']:.1f}]\n"
            f"analysis destRect: [{rec['analysis_dest_x0']}, {rec['analysis_dest_y0']}, "
            f"{rec['analysis_dest_x1']}, {rec['analysis_dest_y1']}]\n"
            f"screen crop: x[{rec['crop_x0_screen_px']}:{rec['crop_x1_screen_px_exclusive']}], "
            f"y[{rec['crop_y0_screen_px']}:{rec['crop_y1_screen_px_exclusive']}]",
            transform=axes[row_idx, 2].transAxes,
            va="center",
            fontsize=8,
        )
    fig.suptitle("BackImage per-trial generation and local-contour crop (magenta box; cyan axis)")
    fig.savefig(OUT / "backimage_trial_generation_crop_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "analysis": "targeted_backimage_trial_generation_and_crop_checkpoint",
        "source_contract": "entire native file; no source crop before resize",
        "display_contract": "bilinear resize to integer-truncated recorded destRect, paste on bkgd=127 screen canvas",
        "local_crop_contract": "77x77 pixels centered on mean gaze for a 1-degree radius at 37.504766 px/deg",
        "examples": [f"{session}:{trial}" for _, session, trial in EXAMPLES],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

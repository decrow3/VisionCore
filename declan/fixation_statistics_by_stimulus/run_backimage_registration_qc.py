"""Create per-image BackImage gaze-registration heatmaps.

The key convention audited here matches Ryan's free-viewing browser:

    x_px = W / 2 + x_deg * pixPerDeg
    y_px = H / 2 - y_deg * pixPerDeg

Jake's raw DPI conversion uses the inverse sign convention when converting
screen pixels back into degrees, so these overlays are the direct visual check
that the image, trial index, screen center, and y-axis direction agree.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

try:
    from .image_features import _backimage_canvas, backimage_trial_geometry, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, backimage_trial_geometry, gaze_deg_to_screen_px


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def _select_trials(df: pd.DataFrame, *, max_trials: int, max_per_session: int) -> pd.DataFrame:
    counts = (
        df.groupby(["session", "trial_idx"], as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
        .sort_values(["n_windows", "session", "trial_idx"], ascending=[False, True, True])
    )
    chosen = []
    session_counts: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        n_for_session = session_counts.get(str(row.session), 0)
        if n_for_session >= max_per_session:
            continue
        chosen.append({"session": row.session, "trial_idx": int(row.trial_idx), "n_windows": int(row.n_windows)})
        session_counts[str(row.session)] = n_for_session + 1
        if len(chosen) >= max_trials:
            break
    return pd.DataFrame(chosen)


def _heatmap(x_px: np.ndarray, y_px: np.ndarray, *, width: int, height: int, bins: int, sigma: float) -> np.ndarray:
    y_bins = max(16, int(round(bins * height / max(width, 1))))
    x_bins = max(16, int(bins))
    hist, _, _ = np.histogram2d(y_px, x_px, bins=(y_bins, x_bins), range=((0, height), (0, width)))
    if sigma > 0:
        hist = gaussian_filter(hist, sigma=float(sigma))
    return hist


def _trial_summary(rows: pd.DataFrame, geometry: dict[str, object], px: np.ndarray) -> dict[str, object]:
    height = int(geometry["screen_height_px"])
    width = int(geometry["screen_width_px"])
    x0, y0, x1, y1 = geometry["dest_rect"]
    x_px = px[:, 0]
    y_px = px[:, 1]
    on_screen = (x_px >= 0) & (x_px < width) & (y_px >= 0) & (y_px < height)
    in_dest = (x_px >= x0) & (x_px < x1) & (y_px >= y0) & (y_px < y1)
    return {
        "session": rows["session"].iloc[0],
        "trial_idx": int(rows["trial_idx"].iloc[0]),
        "n_windows": int(len(rows)),
        "screen_width_px": width,
        "screen_height_px": height,
        "pix_per_deg": float(geometry["ppd"]),
        "dest_x0_px": int(x0),
        "dest_y0_px": int(y0),
        "dest_x1_px": int(x1),
        "dest_y1_px": int(y1),
        "fraction_centroids_on_screen": float(np.mean(on_screen)),
        "fraction_centroids_inside_image_rect": float(np.mean(in_dest)),
        "x_px_min": float(np.nanmin(x_px)),
        "x_px_max": float(np.nanmax(x_px)),
        "y_px_min": float(np.nanmin(y_px)),
        "y_px_max": float(np.nanmax(y_px)),
        "mean_x_deg": float(rows["mean_x_deg"].mean()),
        "mean_y_deg": float(rows["mean_y_deg"].mean()),
        "std_x_deg": float(rows["mean_x_deg"].std(ddof=0)),
        "std_y_deg": float(rows["mean_y_deg"].std(ddof=0)),
    }


def _plot_trial(
    rows: pd.DataFrame,
    *,
    out_file: Path,
    heatmap_bins: int,
    heatmap_sigma: float,
    max_scatter: int,
    seed: int,
) -> dict[str, object]:
    session = str(rows["session"].iloc[0])
    trial_idx = int(rows["trial_idx"].iloc[0])
    canvas, ppd, (height, width) = _backimage_canvas(session, trial_idx)
    geometry = backimage_trial_geometry(session, trial_idx)
    xy_deg = rows[["mean_x_deg", "mean_y_deg"]].to_numpy(dtype=np.float64)
    px = gaze_deg_to_screen_px(xy_deg, ppd=ppd, screen_shape=(height, width))
    x_px = px[:, 0]
    y_px = px[:, 1]
    hist = _heatmap(x_px, y_px, width=width, height=height, bins=heatmap_bins, sigma=heatmap_sigma)

    fig_w = 7.0
    fig_h = fig_w * height / width
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=255, origin="upper")
    finite_hist = hist[np.isfinite(hist)]
    if finite_hist.size and float(np.nanmax(finite_hist)) > 0:
        vmax = np.nanpercentile(finite_hist[finite_hist > 0], 99.0) if np.any(finite_hist > 0) else np.nanmax(finite_hist)
        ax.imshow(
            hist,
            cmap="magma",
            alpha=0.58,
            origin="upper",
            extent=(0, width, height, 0),
            vmin=0,
            vmax=max(float(vmax), 1e-6),
        )
    if len(px) > 0:
        rng = np.random.default_rng(seed)
        keep = np.arange(len(px))
        if len(keep) > max_scatter:
            keep = rng.choice(keep, size=max_scatter, replace=False)
        ax.scatter(px[keep, 0], px[keep, 1], s=9, c="#7df9ff", edgecolors="black", linewidths=0.25, alpha=0.82)
    x0, y0, x1, y1 = geometry["dest_rect"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#55ff55", linewidth=1.0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    ax.set_title(f"{session} trial {trial_idx} | n={len(rows)}", fontsize=8)
    fig.subplots_adjust(0, 0, 1, 0.94)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return _trial_summary(rows, geometry, px)


def _make_montage(panel_files: list[Path], summaries: pd.DataFrame, out_file: Path, *, columns: int) -> None:
    if not panel_files:
        return
    images = [plt.imread(str(path)) for path in panel_files]
    n = len(images)
    cols = max(1, min(columns, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.1), dpi=140)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, image, path in zip(axes_arr, images, panel_files):
        ax.imshow(image)
        ax.set_axis_off()
        match = summaries[summaries["panel_file"] == path.name]
        if not match.empty:
            row = match.iloc[0]
            ax.set_title(
                f"{row['session']} t{int(row['trial_idx'])} "
                f"in-img={row['fraction_centroids_inside_image_rect']:.2f}",
                fontsize=6,
            )
    for ax in axes_arr[n:]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure/backimage_image_fem_windows.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_registration_qc"),
    )
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--max-per-session", type=int, default=1)
    parser.add_argument("--heatmap-bins", type=int, default=180)
    parser.add_argument("--heatmap-sigma", type=float, default=1.4)
    parser.add_argument("--max-scatter", type=int, default=160)
    parser.add_argument("--montage-columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.out_dir / "per_image_heatmaps"
    panels_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.windows_csv)
    needed = {"session", "trial_idx", "mean_x_deg", "mean_y_deg"}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {args.windows_csv}: {missing}")
    df = df.dropna(subset=["session", "trial_idx", "mean_x_deg", "mean_y_deg"]).copy()
    selected = _select_trials(df, max_trials=args.max_trials, max_per_session=args.max_per_session)

    summaries = []
    panel_files = []
    for i, row in enumerate(selected.itertuples(index=False)):
        rows = df[(df["session"] == row.session) & (df["trial_idx"] == int(row.trial_idx))].copy()
        panel_file = panels_dir / f"{i:02d}_{_slug(row.session)}_trial{int(row.trial_idx)}.png"
        summary = _plot_trial(
            rows,
            out_file=panel_file,
            heatmap_bins=args.heatmap_bins,
            heatmap_sigma=args.heatmap_sigma,
            max_scatter=args.max_scatter,
            seed=args.seed + i,
        )
        summary["panel_file"] = panel_file.name
        summaries.append(summary)
        panel_files.append(panel_file)

    summary_df = pd.DataFrame(summaries)
    summary_path = args.out_dir / "registration_qc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    montage_path = args.out_dir / "registration_qc_montage.png"
    _make_montage(panel_files, summary_df, montage_path, columns=args.montage_columns)

    metadata = {
        "windows_csv": str(args.windows_csv),
        "out_dir": str(args.out_dir),
        "n_input_windows": int(len(df)),
        "n_selected_trials": int(len(selected)),
        "mapping_formula": {
            "x_px": "screen_width / 2 + gaze_x_deg * pixPerDeg",
            "y_px": "screen_height / 2 - gaze_y_deg * pixPerDeg",
        },
        "code_cross_checks": [
            "ryan/fig4/browse_freeview_segments.py uses the same degree-to-screen-pixel overlay formula",
            "jake/detect_saccades.py converts raw DPI pixels to degrees with the inverse y sign convention",
        ],
        "summary_csv": str(summary_path),
        "montage_png": str(montage_path),
        "panels_dir": str(panels_dir),
    }
    with (args.out_dir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {len(summary_df)} BackImage registration QC panels to {args.out_dir}")
    return args.out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

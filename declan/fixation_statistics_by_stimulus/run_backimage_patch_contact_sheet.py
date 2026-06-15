"""Create a contact sheet auditing BackImage patch extraction provenance."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px


DEFAULT_INPUT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure/backimage_image_fem_windows.csv")
DEFAULT_OUT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_patch_contact_sheet.png")


def _select_examples(df: pd.DataFrame, n_examples: int) -> pd.DataFrame:
    counts = df.groupby(["session", "trial_idx"], as_index=False).size().rename(columns={"size": "n_windows"})
    counts = counts.sort_values(["n_windows", "session", "trial_idx"], ascending=[False, True, True]).head(n_examples)
    examples = []
    for row in counts.itertuples(index=False):
        sub = df[(df["session"] == row.session) & (df["trial_idx"] == int(row.trial_idx))]
        examples.append(sub.iloc[len(sub) // 2])
    return pd.DataFrame(examples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--patch-radius-deg", type=float, default=1.0)
    parser.add_argument("--n-examples", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> Path:
    df = pd.read_csv(args.input).dropna(subset=["session", "trial_idx", "mean_x_deg", "mean_y_deg"]).copy()
    examples = _select_examples(df, int(args.n_examples))
    rows = len(examples)
    fig, axes = plt.subplots(rows, 2, figsize=(9, max(2.2, rows * 2.2)), dpi=140)
    axes = np.atleast_2d(axes)
    summary_rows = []
    for ax_pair, row in zip(axes, examples.itertuples(index=False)):
        session = str(row.session)
        trial_idx = int(row.trial_idx)
        canvas, ppd, (height, width) = _backimage_canvas(session, trial_idx)
        xy_deg = np.asarray([float(row.mean_x_deg), float(row.mean_y_deg)], dtype=np.float64)
        cx, cy = gaze_deg_to_screen_px(xy_deg, ppd=ppd, screen_shape=(height, width))
        rad = max(2, int(round(float(args.patch_radius_deg) * ppd)))
        x0, x1 = max(0, int(round(cx)) - rad), min(width, int(round(cx)) + rad + 1)
        y0, y1 = max(0, int(round(cy)) - rad), min(height, int(round(cy)) + rad + 1)
        patch = canvas[y0:y1, x0:x1]

        ax = ax_pair[0]
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.scatter([cx], [cy], s=18, c="#7df9ff", edgecolors="black", linewidths=0.5)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff4fd8", linewidth=0.9))
        ax.set_xlim(max(0, cx - 5 * rad), min(width, cx + 5 * rad))
        ax.set_ylim(min(height, cy + 5 * rad), max(0, cy - 5 * rad))
        ax.set_axis_off()
        ax.set_title(f"{session} trial {trial_idx}", fontsize=7)

        ax = ax_pair[1]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.set_axis_off()
        ax.set_title(f"patch x={xy_deg[0]:.2f} y={xy_deg[1]:.2f} deg", fontsize=7)
        summary_rows.append({
            "session": session,
            "trial_idx": trial_idx,
            "mean_x_deg": float(xy_deg[0]),
            "mean_y_deg": float(xy_deg[1]),
            "center_x_px": float(cx),
            "center_y_px": float(cy),
            "patch_x0_px": int(x0),
            "patch_y0_px": int(y0),
            "patch_x1_px": int(x1),
            "patch_y1_px": int(y1),
        })
    fig.tight_layout(pad=0.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Wrote BackImage patch contact sheet to {args.out}")
    return args.out


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

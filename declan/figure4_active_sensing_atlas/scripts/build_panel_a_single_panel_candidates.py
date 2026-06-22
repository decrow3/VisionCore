"""Build single-panel promotion candidates for Figure 4A.

These are not multipanel layouts.  Each output is a complete draft candidate
for the one 4A panel.  The main purpose is to test whether the clear A1 design
survives when the thumbnail and trace come from the real BackImage image set
and recorded fixation traces.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

try:
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _session_dataset_cache
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _session_dataset_cache


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
PANEL_A = ATLAS / "figures" / "panel_A"
OUT_DIR = PANEL_A / "promotion_candidates"
WINDOWS = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)
ORIGINAL_A1 = PANEL_A / "A1_retinal_movie_transform.png"


COLORS = {
    "dark": "#242a2f",
    "muted": "#65717a",
    "grid": "#d8dde3",
    "green": "#2f8f6a",
    "blue": "#244f7a",
    "orange": "#d07a22",
    "purple": "#8064a2",
    "brown": "#765b35",
}


@dataclass(frozen=True)
class Candidate:
    slug: str
    title: str
    row_query: dict[str, Any]
    note: str
    mode: str


CANDIDATES = (
    Candidate(
        slug="4A_candidate_1_real_backimage_a1_proportions",
        title="Real BackImage A1, current proportions",
        row_query={
            "session": "Logan_2020-02-29",
            "trial_idx": 404,
            "global_start": 208092,
            "global_stop": 208220,
        },
        note="Best default candidate: real BackImage canvas and recorded fixation trace, while keeping the clear A1 screen -> trace -> retinal crops grammar.",
        mode="a1",
    ),
    Candidate(
        slug="4A_candidate_2_real_backimage_context",
        title="Real BackImage A1, larger image context",
        row_query={
            "session": "Allen_2022-02-24",
            "trial_idx": 529,
            "global_start": 281281,
            "global_stop": 281409,
        },
        note="More provenance-forward: the screen-image crop shows more of the BackImage context before zooming into retinal samples.",
        mode="context",
    ),
    Candidate(
        slug="4A_candidate_3_real_high_contrast_positive",
        title="One image becomes a changing retinal movie",
        row_query={
            "session": "Logan_2020-01-10",
            "trial_idx": 407,
            "global_start": 397359,
            "global_stop": 397487,
        },
        note="High-contrast real-image option with positive drift-edge alignment metadata; useful if candidate 1 is too dark at print scale.",
        mode="a1",
    ),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _norm_image(image: np.ndarray, *, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _match_row(windows: pd.DataFrame, query: dict[str, Any]) -> pd.Series:
    mask = np.ones(windows.shape[0], dtype=bool)
    for key, value in query.items():
        if isinstance(value, str):
            mask &= windows[key].astype(str).to_numpy() == value
        else:
            mask &= windows[key].astype(int).to_numpy() == int(value)
    rows = windows.loc[mask]
    if rows.shape[0] != 1:
        raise ValueError(f"Expected one row for {query}, found {rows.shape[0]}")
    return rows.iloc[0]


def _load_trace(row: pd.Series) -> np.ndarray:
    session = str(row["session"])
    cache = _session_dataset_cache([session])
    eye = np.asarray(cache[session], dtype=np.float64)
    start = int(row["global_start"])
    stop = int(row["global_stop"])
    trace = eye[start:stop]
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 8:
        raise ValueError(f"Bad eyepos slice for {session} {start}:{stop}: {trace.shape}")
    finite = np.isfinite(trace).all(axis=1)
    trace = trace[finite]
    if trace.shape[0] < 8:
        raise ValueError(f"Too few finite eyepos samples for {session} {start}:{stop}")
    return trace


def _crop_centered(image: np.ndarray, center_xy: tuple[float, float], size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    cx, cy = center_xy
    half = int(size) // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(width - size, x0))
    y0 = max(0, min(height - size, y0))
    return image[y0 : y0 + size, x0 : x0 + size], (x0, y0)


def _crop_variable(image: np.ndarray, center_xy: tuple[float, float], size: int) -> np.ndarray:
    crop, _ = _crop_centered(image, center_xy, size)
    return crop


def _axis_off(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c5ccd2")
        spine.set_linewidth(0.8)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#5d6871") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=color,
            transform=ax.transAxes,
        )
    )


def _representative_indices(trace_xy_px: np.ndarray) -> list[int]:
    n = trace_xy_px.shape[0]
    raw = [int(round(v * (n - 1))) for v in (0.18, 0.52, 0.84)]
    return sorted(set(max(0, min(n - 1, i)) for i in raw))


def _plot_real_a1(row: pd.Series, out_path: Path, *, title: str, mode: str) -> dict[str, Any]:
    canvas, ppd, screen_shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    trace_deg = _load_trace(row)
    trace_px = gaze_deg_to_screen_px(trace_deg, ppd=ppd, screen_shape=screen_shape)
    center_xy = (float(row["image_patch_center_x_px"]), float(row["image_patch_center_y_px"]))
    idxs = _representative_indices(trace_px)
    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]

    context_size = 300 if mode == "a1" else 430
    retinal_size = 86 if mode == "a1" else 96
    context, origin = _crop_centered(canvas, center_xy, context_size)
    trace_context = trace_px - np.asarray(origin)[None, :]
    center_context = np.asarray(center_xy) - np.asarray(origin)

    fig, ax = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title(title, pad=8)

    screen_ax = ax.inset_axes([0.035, 0.13, 0.29 if mode == "context" else 0.25, 0.74])
    screen_ax.imshow(_norm_image(context), cmap="gray", vmin=0, vmax=1)
    screen_ax.plot(trace_context[:, 0], trace_context[:, 1], color=COLORS["green"], lw=1.15)
    screen_ax.scatter(trace_context[0, 0], trace_context[0, 1], s=16, color=COLORS["green"], zorder=3)
    for idx, color in zip(idxs, colors):
        x, y = trace_context[idx]
        screen_ax.add_patch(
            Rectangle(
                (x - retinal_size / 2, y - retinal_size / 2),
                retinal_size,
                retinal_size,
                fill=False,
                edgecolor=color,
                lw=1.25,
            )
        )
    screen_ax.scatter(center_context[0], center_context[1], s=10, color="white", edgecolor=COLORS["dark"], lw=0.4)
    _axis_off(screen_ax)
    screen_ax.set_xlabel("natural image + recorded eye trace", fontsize=7.5)

    trace_ax = ax.inset_axes([0.36, 0.20, 0.20, 0.58])
    centered = trace_deg - np.nanmean(trace_deg, axis=0, keepdims=True)
    trace_ax.plot(centered[:, 0], centered[:, 1], color=COLORS["green"], lw=1.35)
    for idx, color in zip(idxs, colors):
        trace_ax.scatter(centered[idx, 0], centered[idx, 1], color=color, s=24, zorder=3)
    pad_x = max(0.015, float(np.nanmax(np.abs(centered[:, 0]))) * 1.15)
    pad_y = max(0.015, float(np.nanmax(np.abs(centered[:, 1]))) * 1.15)
    trace_ax.set_xlim(-pad_x, pad_x)
    trace_ax.set_ylim(-pad_y, pad_y)
    trace_ax.axhline(0, color=COLORS["grid"], lw=0.7)
    trace_ax.axvline(0, color=COLORS["grid"], lw=0.7)
    trace_ax.set_aspect("equal", adjustable="box")
    trace_ax.set_title("eye drift", fontsize=8.5)
    trace_ax.set_xticks([])
    trace_ax.set_yticks([])
    for spine in trace_ax.spines.values():
        spine.set_color("#c5ccd2")

    crop_x0 = 0.64 if mode == "a1" else 0.66
    crop_w = 0.095 if mode == "a1" else 0.088
    for j, (idx, color) in enumerate(zip(idxs, colors)):
        crop_ax = ax.inset_axes([crop_x0 + 0.11 * j, 0.26, crop_w, 0.44])
        crop = _crop_variable(canvas, tuple(trace_px[idx]), retinal_size)
        crop_ax.imshow(_norm_image(crop), cmap="gray", vmin=0, vmax=1)
        crop_ax.set_xticks([])
        crop_ax.set_yticks([])
        for spine in crop_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.3)
        crop_ax.set_title(f"t{j + 1}", fontsize=7.5, color=color)

    _arrow(ax, (0.30 if mode == "a1" else 0.34, 0.50), (0.34, 0.50))
    _arrow(ax, (0.57, 0.50), (0.62, 0.50))
    ax.text(
        0.78,
        0.12,
        "same image, shifted retinal views",
        ha="center",
        fontsize=7.8,
        transform=ax.transAxes,
    )
    ax.text(
        0.035,
        0.015,
        f"{row['session']} trial {int(row['trial_idx'])}; "
        f"{int(row['global_start'])}:{int(row['global_stop'])}",
        fontsize=6.6,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "candidate": out_path.stem,
        "session": str(row["session"]),
        "trial_idx": int(row["trial_idx"]),
        "global_start": int(row["global_start"]),
        "global_stop": int(row["global_stop"]),
        "mean_x_deg": float(row["mean_x_deg"]),
        "mean_y_deg": float(row["mean_y_deg"]),
        "image_patch_center_x_px": float(row["image_patch_center_x_px"]),
        "image_patch_center_y_px": float(row["image_patch_center_y_px"]),
        "image_orientation_coherence": float(row["image_orientation_coherence"]),
        "anisotropy": float(row["anisotropy"]),
        "image_patch_rms_contrast": float(row["image_patch_rms_contrast"]),
        "path_length_deg": float(row["path_length_deg"]),
        "drift_edge_cos2": float(row["drift_edge_cos2"]),
        "ppd": float(ppd),
        "source": "real_backimage_canvas_and_recorded_backimage_dset_eyepos",
    }


def _copy_original_reference() -> Path:
    out = OUT_DIR / "4A_candidate_0_current_A1_reference.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ORIGINAL_A1, out)
    return out


def _make_contact_sheet(paths: list[Path], values: pd.DataFrame) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
        scale = min(box[0] / image.width, box[1] / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))
        return image.resize(size, resample)

    def wrap(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, max_width: int) -> int:
        x, y = xy
        words = text.split()
        line = ""
        line_h = int(getattr(fnt, "size", 18) * 1.18)
        for word in words:
            cand = word if not line else f"{line} {word}"
            width = fnt.getbbox(cand)[2] - fnt.getbbox(cand)[0] if hasattr(fnt, "getbbox") else fnt.getsize(cand)[0]
            if width <= max_width:
                line = cand
            else:
                if line:
                    draw.text((x, y), line, font=fnt, fill=(45, 49, 54))
                    y += line_h
                line = word
        if line:
            draw.text((x, y), line, font=fnt, fill=(45, 49, 54))
            y += line_h
        return y

    width, height = 2600, 2100
    margin, gap = 62, 36
    thumb_w, thumb_h = 1210, 650
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 45), "Figure 4A Single-Panel Promotion Candidates", font=font(46, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Each option is one possible 4A panel. The real variants use BackImage canvas pixels and recorded eyepos traces.",
        font=font(26),
        fill=(73, 80, 88),
    )
    draw.line((margin, 170, width - margin, 170), fill=(183, 190, 198), width=2)

    captions = [
        ("0. Current A1 reference", "Synthetic patch/trace reference. Keep only as a proportions/style benchmark."),
        ("1. Real BackImage, A1 proportions", "Good A1 proportions, but centered on a dark patch."),
        ("2. Real BackImage, more context", "More provenance-forward, slightly less compact."),
        ("3. Real high-contrast positive", "Selected provisional 4A: clearer retinal samples with positive drift-edge alignment metadata."),
    ]
    for i, path in enumerate(paths):
        row = i // 2
        col = i % 2
        x = margin + col * (thumb_w + gap)
        y = 215 + row * (thumb_h + 330)
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(191, 199, 207), fill="white")
        image = Image.open(path).convert("RGBA")
        image = contain(image, (thumb_w - 34, thumb_h - 34))
        sheet.paste(image, (x + 17 + (thumb_w - 34 - image.width) // 2, y + 17), image)
        draw.text((x, y + thumb_h + 24), captions[i][0], font=font(25, True), fill=(20, 26, 32))
        wrap(draw, (x, y + thumb_h + 64), captions[i][1], font(22), thumb_w)
        if i > 0:
            value = values.iloc[i - 1]
            meta = (
                f"{value.session}, trial {int(value.trial_idx)}; "
                f"coherence {value.image_orientation_coherence:.2f}, "
                f"anisotropy {value.anisotropy:.2f}, "
                f"edge cos2 {value.drift_edge_cos2:.2f}"
            )
            wrap(draw, (x, y + thumb_h + 136), meta, font(20), thumb_w)

    out = OUT_DIR / "4A_single_panel_candidate_sheet.png"
    sheet.save(out, optimize=True)
    return out


def _write_readme(paths: list[Path], sheet: Path, values: pd.DataFrame) -> None:
    readme = OUT_DIR / "README.md"
    lines = [
        "# Figure 4A Single-Panel Promotion Candidates",
        "",
        "Status: draft candidates for choosing one promoted 4A panel.",
        "",
        "![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/4A_single_panel_candidate_sheet.png)",
        "",
        "## Recommendation",
        "",
        "Selected provisional 4A: candidate 3, `4A_candidate_3_real_high_contrast_positive.png`.",
        "",
        "Rationale: candidate 1 preserves A1's proportions but is centered on a dark patch. Candidate 3 keeps the single-panel A1 grammar, uses a real BackImage canvas crop and recorded fixation trace, and has a clearer high-contrast retinal sample with positive drift-edge alignment metadata. Additional image/fixation pairs can be screened later.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `4A_single_panel_candidate_sheet.png`",
            "- `4A_single_panel_candidate_values.csv`",
            "",
            "## Real-Data Provenance",
            "",
            "The real candidates call `_backimage_canvas(session, trial_idx)` and use the recorded `backimage.dset` eyepos slice indexed by `global_start:global_stop` from `backimage_image_fem_windows.csv`.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    _configure_matplotlib()
    windows = pd.read_csv(WINDOWS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [_copy_original_reference()]
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        row = _match_row(windows, candidate.row_query)
        out = OUT_DIR / f"{candidate.slug}.png"
        values = _plot_real_a1(row, out, title=candidate.title, mode=candidate.mode)
        values["note"] = candidate.note
        rows.append(values)
        paths.append(out)
    values_df = pd.DataFrame(rows)
    values_df.to_csv(OUT_DIR / "4A_single_panel_candidate_values.csv", index=False)
    sheet = _make_contact_sheet(paths, values_df)
    _write_readme(paths, sheet, values_df)
    for path in paths + [sheet, OUT_DIR / "README.md", OUT_DIR / "4A_single_panel_candidate_values.csv"]:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()

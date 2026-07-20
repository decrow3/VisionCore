#!/usr/bin/env python3
"""Plot every-frame activation-map panels for representative microsaccade units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/example_instantaneous_maps"
)
DEFAULT_OUT_DIR = DEFAULT_EXAMPLE_DIR / "all_frame_activation_map_panels"
GROUP_COLORS = {
    "large_scale_preferring": "#1f77b4",
    "small_scale_preferring": "#d62728",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, default=DEFAULT_EXAMPLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--units", type=str, default="", help="Optional comma-separated unit indices/labels, e.g. u081,34.")
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--map-vmin-percentile", type=float, default=1.0)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.0)
    parser.add_argument(
        "--label-ssi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay instantaneous spatial SSI bits/spike on every activation map tile.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_units(text: str) -> list[int]:
    if not str(text).strip():
        return []
    units: list[int] = []
    for part in str(text).split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.startswith("u"):
            token = token[1:]
        units.append(int(token))
    return units


def safe_slug(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "unnamed"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def image_scale(images: np.ndarray, vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    vals = np.asarray(images, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.nanmean(finite))
        spread = float(np.nanstd(finite))
        if not np.isfinite(spread) or spread <= 0:
            spread = max(abs(center) * 0.1, 1e-6)
        return center - spread, center + spread
    return vmin, vmax


def instantaneous_ssi_bits(unit_maps: np.ndarray) -> np.ndarray:
    maps = np.maximum(np.asarray(unit_maps, dtype=np.float64), 0.0)
    mean_rate = np.mean(maps, axis=(-2, -1))
    gain = maps / (mean_rate[..., None, None] + EPS)
    bits = np.mean(gain * np.log2(gain + EPS), axis=(-2, -1))
    return bits.astype(np.float32, copy=False)


def load_cache(example_dir: Path) -> dict[str, np.ndarray]:
    cache_path = Path(example_dir) / "instantaneous_example_maps_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing rendered instantaneous map cache: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def select_units(unit_table: pd.DataFrame, cache_units: np.ndarray, requested: list[int], max_units: int) -> pd.DataFrame:
    table = unit_table.copy()
    table["unit_index"] = table["unit_index"].astype(int)
    table = table[table["unit_index"].isin([int(v) for v in cache_units])].copy()
    if requested:
        requested_set = set(int(v) for v in requested)
        table = table[table["unit_index"].isin(requested_set)].copy()
    elif int(max_units) > 0:
        table = table.head(int(max_units)).copy()
    if table.empty:
        raise ValueError("No requested units are available in the instantaneous map cache.")
    return table.reset_index(drop=True)


def plot_unit_panel(
    *,
    unit_row: pd.Series,
    cache: dict[str, np.ndarray],
    out_dir: Path,
    vmin_percentile: float,
    vmax_percentile: float,
    label_ssi: bool,
    dpi: int,
) -> Path:
    maps = np.asarray(cache["maps"], dtype=np.float32)
    movie_indices = np.asarray(cache["movie_indices"], dtype=int)
    selected_units = np.asarray(cache["selected_units"], dtype=int)
    event_masks = np.asarray(cache["event_masks"], dtype=bool)
    condition_labels = np.asarray(cache["condition_label"]).astype(str)
    motion_scale = np.asarray(cache["motion_scale"], dtype=float)

    unit = int(unit_row["unit_index"])
    unit_pos = int(np.flatnonzero(selected_units == unit)[0])
    movie_idx = int(unit_row["representative_movie_index"])
    movie_pos = int(np.flatnonzero(movie_indices == movie_idx)[0])
    unit_maps = maps[movie_pos, :, :, unit_pos]
    ssi_bits = instantaneous_ssi_bits(unit_maps)
    event_mask = event_masks[movie_pos]
    n_conditions, n_frames, _height, _width = unit_maps.shape
    vmin, vmax = image_scale(unit_maps, vmin_percentile, vmax_percentile)

    group = str(unit_row.get("curve_group", ""))
    color = GROUP_COLORS.get(group, "0.25")
    unit_label = str(unit_row.get("unit_label", f"u{unit:03d}"))
    source_row = int(unit_row.get("representative_source_row", movie_idx))
    group_label = str(unit_row.get("curve_group_label", group)).replace("_", " ")
    sf_label = str(unit_row.get("sf_group_label", "")).split("(")[0].strip()

    fig_w = max(18.0, 0.46 * n_frames + 2.6)
    fig_h = max(5.5, 0.62 * n_conditions + 1.35)
    fig, axes = plt.subplots(n_conditions, n_frames, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.065, top=0.84, wspace=0.015, hspace=0.055)
    fig.suptitle(
        f"{unit_label}: every-frame raw RR100 activation maps",
        fontsize=14,
        color=color,
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        (
            f"{group_label}; {sf_label}; representative movie {movie_idx}, source row {source_row}; "
            "orange borders mark detected microsaccade event bins"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.35",
    )
    for cond_idx in range(n_conditions):
        for frame_idx in range(n_frames):
            ax = axes[cond_idx, frame_idx]
            ax.imshow(unit_maps[cond_idx, frame_idx], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if cond_idx == 0:
                ax.set_title(str(frame_idx), fontsize=5.8, pad=1.2)
            if frame_idx == 0:
                ax.set_ylabel(
                    f"{condition_labels[cond_idx]}\n{motion_scale[cond_idx]:g}x",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=7.0,
                    labelpad=18,
                    color=color,
                )
            if label_ssi:
                ax.text(
                    0.045,
                    0.055,
                    f"{float(ssi_bits[cond_idx, frame_idx]):.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    color="white",
                    fontsize=3.9,
                    path_effects=[path_effects.withStroke(linewidth=0.85, foreground="black")],
                )
            for spine in ax.spines.values():
                spine.set_visible(True)
                if frame_idx < event_mask.size and bool(event_mask[frame_idx]):
                    spine.set_edgecolor("#f59e0b")
                    spine.set_linewidth(1.15)
                else:
                    spine.set_edgecolor("0.72")
                    spine.set_linewidth(0.22)
    png = out_dir / f"{unit_label}_movie{movie_idx}_all_frame_activation_maps.png"
    fig.savefig(png, dpi=int(dpi))
    plt.close(fig)
    return png


def main() -> None:
    args = parse_args()
    example_dir = Path(args.example_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(example_dir)
    unit_table = pd.read_csv(example_dir / "selected_example_units_with_representative_movies.csv")
    selected = select_units(unit_table, cache["selected_units"], parse_units(args.units), int(args.max_units))
    pngs = []
    for _, row in selected.iterrows():
        pngs.append(
            plot_unit_panel(
                unit_row=row,
                cache=cache,
                out_dir=out_dir,
                vmin_percentile=float(args.map_vmin_percentile),
                vmax_percentile=float(args.map_vmax_percentile),
                label_ssi=bool(args.label_ssi),
                dpi=int(args.dpi),
            )
        )
    pdf = out_dir / "selected_units_all_frame_activation_maps.pdf"
    with PdfPages(pdf) as pages:
        for png in pngs:
            image = plt.imread(png)
            fig_w = max(8.0, image.shape[1] / float(args.dpi))
            fig_h = max(4.0, image.shape[0] / float(args.dpi))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.imshow(image)
            ax.axis("off")
            pages.savefig(fig, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)
    selected.to_csv(out_dir / "selected_units_all_frame_activation_maps.csv", index=False)
    metadata = {
        "analysis": "backimage_microsaccade_unit_all_frame_activation_maps",
        "example_dir": example_dir,
        "cache_path": example_dir / "instantaneous_example_maps_cache.npz",
        "out_dir": out_dir,
        "pngs": pngs,
        "pdf": pdf,
        "selected_units": selected["unit_index"].astype(int).to_list(),
        "map_contract": (
            "Panels display raw instantaneous RR100 activation maps, not SSI-density maps. "
            "Rows are microsaccade event scale conditions and columns are all cached time bins. "
            "Grayscale limits are shared within each unit panel using the requested percentiles. "
            "When enabled, tile labels show instantaneous spatial SSI bits/spike computed from that raw activation map."
        ),
        "label_ssi": bool(args.label_ssi),
    }
    (out_dir / "all_frame_activation_map_metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(pngs)} unit panels to {out_dir}")
    print(f"Wrote {pdf}")
    for png in pngs:
        print(png)


if __name__ == "__main__":
    main()

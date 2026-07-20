#!/usr/bin/env python3
"""Plot per-frame SSI and mean activation heatmaps for representative microsaccade units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/example_instantaneous_maps"
)
DEFAULT_OUT_DIR = DEFAULT_EXAMPLE_DIR / "activation_ssi_rate_heatmaps"
GROUP_COLORS = {
    "large_scale_preferring": "#1f77b4",
    "small_scale_preferring": "#d62728",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, default=DEFAULT_EXAMPLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--units", type=str, default="", help="Optional comma-separated unit indices/labels, e.g. u081,30.")
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=220)
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
        table = table[table["unit_index"].isin(set(int(v) for v in requested))].copy()
    elif int(max_units) > 0:
        table = table.head(int(max_units)).copy()
    if table.empty:
        raise ValueError("No requested units are available in the instantaneous map cache.")
    return table.reset_index(drop=True)


def activation_metrics(unit_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = np.maximum(np.asarray(unit_maps, dtype=np.float64), 0.0)
    mean_activation = np.mean(maps, axis=(-2, -1))
    gain = maps / (mean_activation[..., None, None] + EPS)
    ssi_bits = np.mean(gain * np.log2(gain + EPS), axis=(-2, -1))
    return ssi_bits.astype(np.float32, copy=False), mean_activation.astype(np.float32, copy=False)


def build_metric_table(
    cache: dict[str, np.ndarray],
    unit_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    maps = np.asarray(cache["maps"], dtype=np.float32)
    movie_indices = np.asarray(cache["movie_indices"], dtype=int)
    selected_units = np.asarray(cache["selected_units"], dtype=int)
    event_masks = np.asarray(cache["event_masks"], dtype=bool)
    condition_labels = np.asarray(cache["condition_label"]).astype(str)
    motion_scale = np.asarray(cache["motion_scale"], dtype=float)
    rows: list[dict[str, Any]] = []
    unit_metric_arrays: dict[int, dict[str, np.ndarray]] = {}
    for _, row in unit_table.iterrows():
        unit = int(row["unit_index"])
        unit_pos = int(np.flatnonzero(selected_units == unit)[0])
        movie_idx = int(row["representative_movie_index"])
        movie_pos = int(np.flatnonzero(movie_indices == movie_idx)[0])
        unit_maps = maps[movie_pos, :, :, unit_pos]
        ssi_bits, mean_activation = activation_metrics(unit_maps)
        event_mask = event_masks[movie_pos]
        unit_metric_arrays[unit] = {
            "ssi_bits": ssi_bits,
            "mean_activation": mean_activation,
            "event_mask": event_mask,
        }
        for cond_idx, label in enumerate(condition_labels):
            for frame_idx in range(ssi_bits.shape[1]):
                rows.append(
                    {
                        "unit_index": unit,
                        "unit_label": str(row.get("unit_label", f"u{unit:03d}")),
                        "curve_group": str(row.get("curve_group", "")),
                        "curve_group_label": str(row.get("curve_group_label", "")),
                        "sf_group": str(row.get("sf_group", "")),
                        "sf_group_label": str(row.get("sf_group_label", "")),
                        "representative_movie_index": movie_idx,
                        "representative_source_row": int(row.get("representative_source_row", movie_idx)),
                        "condition_index": int(cond_idx),
                        "condition_label": str(label),
                        "motion_scale": float(motion_scale[cond_idx]),
                        "frame_index": int(frame_idx),
                        "is_microsaccade_event_frame": bool(frame_idx < event_mask.size and event_mask[frame_idx]),
                        "instantaneous_ssi_bits_per_spike": float(ssi_bits[cond_idx, frame_idx]),
                        "mean_activation": float(mean_activation[cond_idx, frame_idx]),
                    }
                )
    return pd.DataFrame(rows), unit_metric_arrays


def robust_limits(values: list[np.ndarray], *, lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, lower))
    vmax = float(np.nanpercentile(finite, upper))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def plot_heatmaps(
    out_dir: Path,
    cache: dict[str, np.ndarray],
    unit_table: pd.DataFrame,
    unit_metric_arrays: dict[int, dict[str, np.ndarray]],
    *,
    dpi: int,
) -> Path:
    condition_labels = np.asarray(cache["condition_label"]).astype(str)
    motion_scale = np.asarray(cache["motion_scale"], dtype=float)
    n_units = int(unit_table.shape[0])
    n_frames = int(next(iter(unit_metric_arrays.values()))["ssi_bits"].shape[1])
    fig, axes = plt.subplots(n_units, 2, figsize=(14.5, max(2.05 * n_units, 6.5)), squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.91, bottom=0.065, top=0.9, hspace=0.58, wspace=0.12)
    fig.suptitle("Example units: per-frame SSI and mean activation from raw activation maps", y=0.975, fontsize=15)
    fig.text(
        0.5,
        0.938,
        "Rows within each heatmap are microsaccade scale conditions; columns are 40 time bins; orange lines mark detected event bins",
        ha="center",
        fontsize=10.2,
        color="0.35",
    )
    ssi_limits = robust_limits([payload["ssi_bits"] for payload in unit_metric_arrays.values()])
    activation_limits = robust_limits([payload["mean_activation"] for payload in unit_metric_arrays.values()])
    last_ims = {}
    for row_idx, (_, unit_row) in enumerate(unit_table.iterrows()):
        unit = int(unit_row["unit_index"])
        payload = unit_metric_arrays[unit]
        group = str(unit_row.get("curve_group", ""))
        color = GROUP_COLORS.get(group, "0.25")
        unit_label = str(unit_row.get("unit_label", f"u{unit:03d}"))
        group_label = str(unit_row.get("curve_group_label", group)).replace("_", " ")
        sf_label = str(unit_row.get("sf_group_label", "")).split("(")[0].strip()
        for col_idx, (metric_key, title, limits, cmap) in enumerate(
            [
                ("ssi_bits", "instantaneous SSI (bits/spike)", ssi_limits, "magma"),
                ("mean_activation", "mean activation", activation_limits, "viridis"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            image = np.asarray(payload[metric_key], dtype=float)
            im = ax.imshow(
                image,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
            last_ims[metric_key] = im
            event_frames = np.flatnonzero(np.asarray(payload["event_mask"], dtype=bool))
            for frame in event_frames:
                ax.axvline(float(frame), color="#f59e0b", lw=0.7, alpha=0.85)
            ax.set_yticks(np.arange(len(condition_labels)))
            if col_idx == 0:
                ax.set_yticklabels([f"{label}" for label in condition_labels], fontsize=7.3)
                ax.set_ylabel(
                    f"{unit_label}\n{group_label}\n{sf_label}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=50,
                    fontsize=7.5,
                    color=color,
                )
            else:
                ax.set_yticklabels([])
            ax.set_xlim(-0.5, n_frames - 0.5)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 39])
            if row_idx == n_units - 1:
                ax.set_xlabel("frame index")
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(0.8)
    cax1 = fig.add_axes([0.925, 0.54, 0.012, 0.30])
    fig.colorbar(last_ims["ssi_bits"], cax=cax1, label="bits/spike")
    cax2 = fig.add_axes([0.925, 0.16, 0.012, 0.30])
    fig.colorbar(last_ims["mean_activation"], cax=cax2, label="mean activation")
    png = out_dir / "example_units_activation_ssi_and_mean_activation_heatmaps.png"
    pdf = out_dir / "example_units_activation_ssi_and_mean_activation_heatmaps.pdf"
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
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
    metric_df, unit_metric_arrays = build_metric_table(cache, selected)
    metric_csv = out_dir / "example_units_activation_ssi_and_mean_activation_by_frame_condition.csv"
    selected_csv = out_dir / "selected_units_activation_ssi_rate_heatmaps.csv"
    metric_df.to_csv(metric_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    png = plot_heatmaps(out_dir, cache, selected, unit_metric_arrays, dpi=int(args.dpi))
    metadata = {
        "analysis": "backimage_microsaccade_unit_activation_ssi_rate_heatmaps",
        "example_dir": example_dir,
        "cache_path": example_dir / "instantaneous_example_maps_cache.npz",
        "selected_units": selected["unit_index"].astype(int).to_list(),
        "metric_csv": metric_csv,
        "selected_units_csv": selected_csv,
        "plot_png": png,
        "plot_pdf": out_dir / "example_units_activation_ssi_and_mean_activation_heatmaps.pdf",
        "metric_contract": (
            "instantaneous_ssi_bits_per_spike is computed directly from each raw activation map as "
            "mean_spatial(gain * log2(gain)), where gain is activation divided by the map mean. "
            "mean_activation is the spatial mean of the same raw activation map."
        ),
    }
    (out_dir / "activation_ssi_rate_heatmap_metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {png}")
    print(f"Wrote {metric_csv}")


if __name__ == "__main__":
    main()

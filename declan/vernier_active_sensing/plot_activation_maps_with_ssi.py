#!/usr/bin/env python3
"""Print-ready Vernier activation maps with one SSI number per map.

This script reads the final-history map caches produced by the Vernier
walkthrough, collapses each population activation tensor to a 2-D image, and
stamps the matching spatial spiking information (SSI) on the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


DEFAULT_CACHE_DIR = Path("outputs") / "notebook_vernier_walkthrough" / "ssi_final_history_map"
DEFAULT_OUT_DIR = DEFAULT_CACHE_DIR / "activation_map_prints"
DEFAULT_CONDITIONS = (
    "static_center",
    "real_fem",
    "order_shuffled_positions",
    "static_phase_cloud_matched_positions",
    "axis_horizontal",
    "axis_vertical",
)
DEFAULT_POPULATIONS = ("full756", "rr100_medoid")
RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)

CONDITION_LABELS = {
    "static_center": "static center",
    "real_fem": "real FEM",
    "order_shuffled_positions": "order shuffled",
    "static_phase_cloud_matched_positions": "phase cloud",
    "axis_horizontal": "horizontal only\n(across contour)",
    "axis_vertical": "vertical only\n(along contour)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--populations", type=str, default=",".join(DEFAULT_POPULATIONS))
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--map-kind", choices=("zero", "plus", "minus"), default="zero")
    parser.add_argument("--collapse", choices=("mean", "sum", "max", "ssi_density"), default="mean")
    parser.add_argument("--vmin-percentile", type=float, default=0.5)
    parser.add_argument("--vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def safe_slug(value: object, max_len: int = 120) -> str:
    text = str(value)
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return (slug or "unnamed")[:max_len]


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(str(condition), str(condition).replace("_", " "))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (str(row.get("condition", "")), str(row.get("population_key", ""))): row
        for row in rows
    }


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def ssi_single_frame(rate_maps: np.ndarray, eps: float = 1e-8) -> dict[str, Any]:
    """Spatial Spiking Information from one ``(unit, H, W)`` rate map."""
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W), got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    unit_mean_rate = flat.mean(axis=1)
    gain = flat / (unit_mean_rate[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = unit_mean_rate / max(float(unit_mean_rate.sum()), eps)
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_mean_rate": unit_mean_rate.astype(np.float32),
        "population_bits_per_spike": float(np.sum(weights * unit_bits)),
    }


def total_rate(rate_maps: np.ndarray) -> float:
    y = np.asarray(rate_maps, dtype=np.float32)
    return float(y.reshape(y.shape[0], -1).mean(axis=1).sum())


def ssi_density_map(rate_maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rate-weighted SSI contribution per spatial bin, summed over units."""
    y = np.asarray(rate_maps, dtype=np.float64)
    flat = y.reshape(y.shape[0], -1)
    unit_mean_rate = flat.mean(axis=1)
    gain = flat / (unit_mean_rate[:, None] + eps)
    unit_bits_xy = gain * np.log2(gain + eps)
    weights = unit_mean_rate / max(float(unit_mean_rate.sum()), eps)
    density = np.sum(weights[:, None] * unit_bits_xy, axis=0)
    return density.reshape(y.shape[1], y.shape[2]).astype(np.float32)


def collapse_activation_map(rate_maps: np.ndarray, method: str) -> np.ndarray:
    y = np.asarray(rate_maps, dtype=np.float32)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W), got {y.shape}")
    if method == "mean":
        return np.mean(y, axis=0)
    if method == "sum":
        return np.sum(y, axis=0)
    if method == "max":
        return np.max(y, axis=0)
    if method == "ssi_density":
        return ssi_density_map(y)
    raise ValueError(f"Unsupported collapse method: {method}")


def final_map_key(map_kind: str) -> str:
    return {
        "zero": "final_spatial_zero",
        "plus": "final_spatial_plus",
        "minus": "final_spatial_minus",
    }[str(map_kind)]


def load_final_map(path: Path, map_kind: str) -> np.ndarray:
    key = final_map_key(map_kind)
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=np.float32)


def find_condition_cache(cache_dir: Path, condition: str) -> Path:
    slug = safe_slug(condition)
    matches = sorted(cache_dir.glob(f"final_history_full_map_{slug}_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No final-history cache found for condition {condition!r} in {cache_dir}")
    if len(matches) > 1:
        fd_matches = [path for path in matches if "_fd0.2500arcmin" in path.name]
        if len(fd_matches) == 1:
            return fd_matches[0]
        choices = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(f"Ambiguous cache files for {condition!r}:\n{choices}")
    return matches[0]


def population_specs(population_keys: list[str], rr100_version: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    rr100_view = None
    for key in population_keys:
        if key == "full756":
            specs.append({"key": "full756", "label": "full 756", "view": None, "version": "full 756"})
        elif key == "rr100_medoid":
            if rr100_view is None:
                rr100_view = load_population_view(version_name=rr100_version)
            specs.append(
                {
                    "key": "rr100_medoid",
                    "label": "RR100 movie-medoid",
                    "view": rr100_view,
                    "version": rr100_view.name,
                }
            )
        else:
            raise ValueError(f"Unsupported population key: {key}")
    return specs


def apply_population(full_map: np.ndarray, view: Any | None) -> np.ndarray:
    if view is None:
        return np.asarray(full_map, dtype=np.float32)
    return np.asarray(apply_population_view(full_map, view), dtype=np.float32)


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def draw_single_map(
    image: np.ndarray,
    *,
    title: str,
    subtitle: str,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 3.35), dpi=dpi, constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.025)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_gallery(
    items: list[dict[str, Any]],
    *,
    population_label: str,
    collapse: str,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
    suptitle: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(items)
    fig_width = max(2.35 * n, 7.5)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 2.95), dpi=dpi, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    last_im = None
    for ax, item in zip(axes_arr, items, strict=True):
        last_im = ax.imshow(
            item["image"],
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"{item['condition_label']}\nSSI {item['ssi_bits_per_spike']:.5f}",
            fontsize=7.8,
            pad=5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#666666")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), fraction=0.018, pad=0.012)
        cbar.ax.tick_params(labelsize=6.8, length=2)
        cbar.set_label(f"{collapse} activation", fontsize=7.5)
    fig.suptitle(suptitle or f"Vernier final-history activation maps: {population_label}", fontsize=10.5, y=1.04)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    conditions = parse_csv_list(args.conditions)
    populations = population_specs(parse_csv_list(args.populations), str(args.rr100_version))
    summary_path = args.summary_csv or (args.cache_dir / "vernier_ssi_final_history_map_summary.csv")
    summary = read_summary(summary_path)

    items: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    condition_sources: list[dict[str, Any]] = []

    full_maps: dict[str, tuple[Path, np.ndarray]] = {}
    for condition in conditions:
        source = find_condition_cache(args.cache_dir, condition)
        full_maps[condition] = (source, load_final_map(source, args.map_kind))
        condition_sources.append({"condition": condition, "source_npz": source})

    for pop in populations:
        pop_items: list[dict[str, Any]] = []
        for condition in conditions:
            source, full_map = full_maps[condition]
            pop_map = apply_population(full_map, pop["view"])
            ssi = ssi_single_frame(pop_map)
            image = collapse_activation_map(pop_map, str(args.collapse))
            summary_row = summary.get((condition, str(pop["key"])), {})
            summary_ssi = fnum(summary_row.get("final_history_ssi_bits_per_spike"))
            item = {
                "condition": condition,
                "condition_label": condition_label(condition),
                "population_key": pop["key"],
                "population_label": pop["label"],
                "population_version": pop["version"],
                "source_npz": source,
                "rate_map": pop_map,
                "image": image,
                "ssi_bits_per_spike": float(ssi["population_bits_per_spike"]),
                "summary_ssi_bits_per_spike": summary_ssi,
                "total_rate": total_rate(pop_map),
            }
            pop_items.append(item)
            items.append(item)
        vmin, vmax = image_scale(
            [item["image"] for item in pop_items],
            float(args.vmin_percentile),
            float(args.vmax_percentile),
        )
        gallery_path = args.out_dir / f"vernier_activation_map_gallery_{pop['key']}_{args.map_kind}_{args.collapse}.png"
        draw_gallery(
            pop_items,
            population_label=str(pop["label"]),
            collapse=str(args.collapse),
            vmin=vmin,
            vmax=vmax,
            path=gallery_path,
            dpi=int(args.dpi),
        )
        for item in pop_items:
            image_path = (
                args.out_dir
                / "individual_maps"
                / f"activation_map_{item['population_key']}_{safe_slug(item['condition'])}_{args.map_kind}_{args.collapse}.png"
            )
            draw_single_map(
                item["image"],
                title=str(item["condition_label"]).replace("\n", " "),
                subtitle=f"SSI {item['ssi_bits_per_spike']:.5f} bits/spike",
                vmin=vmin,
                vmax=vmax,
                path=image_path,
                dpi=int(args.dpi),
            )
            image = np.asarray(item["image"], dtype=np.float32)
            manifest_rows.append(
                {
                    "condition": item["condition"],
                    "condition_label": item["condition_label"].replace("\n", " "),
                    "population_key": item["population_key"],
                    "population_label": item["population_label"],
                    "population_version": item["population_version"],
                    "map_kind": args.map_kind,
                    "collapse": args.collapse,
                    "n_units": int(item["rate_map"].shape[0]),
                    "height": int(item["rate_map"].shape[1]),
                    "width": int(item["rate_map"].shape[2]),
                    "ssi_bits_per_spike": item["ssi_bits_per_spike"],
                    "summary_ssi_bits_per_spike": item["summary_ssi_bits_per_spike"],
                    "summary_ssi_delta": item["ssi_bits_per_spike"] - item["summary_ssi_bits_per_spike"]
                    if np.isfinite(item["summary_ssi_bits_per_spike"])
                    else float("nan"),
                    "total_rate": item["total_rate"],
                    "image_min": float(np.nanmin(image)),
                    "image_mean": float(np.nanmean(image)),
                    "image_max": float(np.nanmax(image)),
                    "image_vmin": vmin,
                    "image_vmax": vmax,
                    "source_npz": item["source_npz"],
                    "image_png": image_path,
                    "gallery_png": gallery_path,
                }
            )
    return items, manifest_rows, condition_sources


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items, manifest_rows, condition_sources = build_items(args)
    manifest_path = args.out_dir / "vernier_activation_map_ssi_manifest.csv"
    write_csv_rows(manifest_path, manifest_rows)
    write_json(
        args.out_dir / "vernier_activation_map_ssi_manifest.json",
        {
            "cache_dir": args.cache_dir,
            "out_dir": args.out_dir,
            "summary_csv": args.summary_csv or (args.cache_dir / "vernier_ssi_final_history_map_summary.csv"),
            "conditions": parse_csv_list(args.conditions),
            "populations": parse_csv_list(args.populations),
            "map_kind": args.map_kind,
            "collapse": args.collapse,
            "vmin_percentile": float(args.vmin_percentile),
            "vmax_percentile": float(args.vmax_percentile),
            "n_maps": len(items),
            "condition_sources": condition_sources,
            "manifest_csv": manifest_path,
        },
    )
    galleries = sorted(args.out_dir.glob(f"vernier_activation_map_gallery_*_{args.map_kind}_{args.collapse}.png"))
    print(f"Wrote {len(manifest_rows)} activation-map images to {args.out_dir}")
    print(f"Wrote SSI manifest: {manifest_path}")
    for gallery in galleries:
        print(f"Wrote gallery: {gallery}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print RR100 unit activation-map sheets across Vernier conditions."""

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
from declan.vernier_active_sensing.plot_activation_maps_with_ssi import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONDITIONS,
    RR100_MOVIE_MEDOID_VERSION,
    condition_label,
    final_map_key,
    find_condition_cache,
    image_scale,
    load_final_map,
    safe_slug,
    ssi_single_frame,
    write_json,
)


DEFAULT_OUT_DIR = DEFAULT_CACHE_DIR / "rr100_unit_condition_map_prints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--map-kind", choices=("zero", "plus", "minus"), default="zero")
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument(
        "--sheet-mode",
        choices=("unit", "condition", "both"),
        default="unit",
        help="Save one sheet per unit, one sheet per condition, or both.",
    )
    parser.add_argument("--sort-units", choices=("index", "ssi_desc", "mean_rate_desc"), default="index")
    parser.add_argument(
        "--scale-scope",
        choices=("unit", "global", "condition"),
        default="unit",
        help=(
            "Color scaling. Unit mode shares scale across conditions for each unit; "
            "global shares one scale everywhere; condition is for condition-grouped sheets."
        ),
    )
    parser.add_argument("--vmin-percentile", type=float, default=0.5)
    parser.add_argument("--vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--write-individual",
        action="store_true",
        help="Also save one PNG per unit per condition. Contact sheets are always written.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


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


def _sort_indices(unit_bits: np.ndarray, mean_rates: np.ndarray, sort_mode: str) -> np.ndarray:
    if sort_mode == "index":
        return np.arange(unit_bits.size, dtype=np.int64)
    if sort_mode == "ssi_desc":
        return np.argsort(-np.asarray(unit_bits, dtype=np.float64), kind="mergesort")
    if sort_mode == "mean_rate_desc":
        return np.argsort(-np.asarray(mean_rates, dtype=np.float64), kind="mergesort")
    raise ValueError(f"Unsupported sort mode: {sort_mode}")


def _unit_title(unit_index: int, ssi_bits: float) -> str:
    return f"u{unit_index:03d}\nSSI {ssi_bits:.4f}"


def draw_unit_sheet(
    maps: np.ndarray,
    *,
    unit_bits: np.ndarray,
    mean_rates: np.ndarray,
    condition: str,
    sort_mode: str,
    vmin: float,
    vmax: float,
    cols: int,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_units = int(maps.shape[0])
    cols = max(1, int(cols))
    rows = int(np.ceil(n_units / cols))
    order = _sort_indices(unit_bits, mean_rates, sort_mode)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(1.45 * cols, 1.55 * rows + 0.35),
        dpi=int(dpi),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(rows, cols)
    last_im = None
    for pos, unit_index in enumerate(order):
        ax = axes_arr[pos // cols, pos % cols]
        last_im = ax.imshow(
            maps[int(unit_index)],
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(_unit_title(int(unit_index), float(unit_bits[int(unit_index)])), fontsize=4.9, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)
            spine.set_color("#777777")
    for pos in range(n_units, rows * cols):
        axes_arr[pos // cols, pos % cols].axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.ravel().tolist(), fraction=0.015, pad=0.01)
        cbar.ax.tick_params(labelsize=6.0, length=2)
        cbar.set_label("activation", fontsize=6.5)
    fig.suptitle(
        f"RR100 unit activation maps: {condition_label(condition).replace(chr(10), ' ')}",
        fontsize=10.0,
        y=1.01,
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_individual_unit_map(
    unit_map: np.ndarray,
    *,
    unit_index: int,
    unit_bits: float,
    condition: str,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.25, 2.55), dpi=int(dpi), constrained_layout=True)
    im = ax.imshow(unit_map, origin="lower", cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"{condition_label(condition).replace(chr(10), ' ')}\nRR100 u{unit_index:03d}  SSI {unit_bits:.5f}",
        fontsize=7.2,
        pad=4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_unit_condition_sheet(
    condition_rows: list[dict[str, Any]],
    *,
    unit_index: int,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_conditions = len(condition_rows)
    cols = 3 if n_conditions > 3 else max(1, n_conditions)
    rows = int(np.ceil(n_conditions / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.35 * cols, 2.65 * rows + 0.25),
        dpi=int(dpi),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(rows, cols)
    last_im = None
    for pos, row in enumerate(condition_rows):
        condition = str(row["condition"])
        unit_map = np.asarray(row["rr100_map"], dtype=np.float32)[int(unit_index)]
        unit_bits = float(np.asarray(row["unit_bits"], dtype=np.float32)[int(unit_index)])
        unit_rate = float(np.asarray(row["unit_mean_rates"], dtype=np.float32)[int(unit_index)])
        ax = axes_arr[pos // cols, pos % cols]
        last_im = ax.imshow(
            unit_map,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"{condition_label(condition).replace(chr(10), ' ')}\nSSI {unit_bits:.5f}  rate {unit_rate:.3f}",
            fontsize=7.0,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#777777")
    for pos in range(n_conditions, rows * cols):
        axes_arr[pos // cols, pos % cols].axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.ravel().tolist(), fraction=0.027, pad=0.012)
        cbar.ax.tick_params(labelsize=6.5, length=2)
        cbar.set_label("activation", fontsize=7.0)
    fig.suptitle(f"RR100 unit {int(unit_index):03d}: Vernier activation maps by condition", fontsize=10.0, y=1.02)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_rr100_condition_maps(args: argparse.Namespace) -> list[dict[str, Any]]:
    view = load_population_view(version_name=str(args.rr100_version))
    rows: list[dict[str, Any]] = []
    for condition in parse_csv_list(args.conditions):
        source = find_condition_cache(args.cache_dir, condition)
        full_map = load_final_map(source, args.map_kind)
        rr100_map = np.asarray(apply_population_view(full_map, view), dtype=np.float32)
        if rr100_map.ndim != 3:
            raise ValueError(f"Expected RR100 map (unit, H, W), got {rr100_map.shape}")
        ssi = ssi_single_frame(rr100_map)
        rows.append(
            {
                "condition": condition,
                "source_npz": source,
                "rr100_map": rr100_map,
                "unit_bits": np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32),
                "unit_mean_rates": np.asarray(ssi["unit_mean_rate"], dtype=np.float32),
                "population_ssi_bits_per_spike": float(ssi["population_bits_per_spike"]),
                "population_version": view.name,
            }
        )
    return rows


def scales_for_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    if args.scale_scope in {"global", "unit"}:
        vmin, vmax = image_scale(
            [np.asarray(row["rr100_map"], dtype=np.float32) for row in rows],
            float(args.vmin_percentile),
            float(args.vmax_percentile),
        )
        return {str(row["condition"]): (vmin, vmax) for row in rows}
    return {
        str(row["condition"]): image_scale(
            [np.asarray(row["rr100_map"], dtype=np.float32)],
            float(args.vmin_percentile),
            float(args.vmax_percentile),
        )
        for row in rows
    }


def unit_scale_for_rows(rows: list[dict[str, Any]], args: argparse.Namespace, unit_index: int) -> tuple[float, float]:
    if args.scale_scope == "global":
        return image_scale(
            [np.asarray(row["rr100_map"], dtype=np.float32) for row in rows],
            float(args.vmin_percentile),
            float(args.vmax_percentile),
        )
    return image_scale(
        [np.asarray(row["rr100_map"], dtype=np.float32)[int(unit_index)] for row in rows],
        float(args.vmin_percentile),
        float(args.vmax_percentile),
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rr100_condition_maps(args)
    scales = scales_for_rows(rows, args)

    manifest_rows: list[dict[str, Any]] = []
    condition_sheet_paths: dict[str, Path] = {}
    if args.sheet_mode in {"condition", "both"}:
        for row in rows:
            condition = str(row["condition"])
            rr100_map = np.asarray(row["rr100_map"], dtype=np.float32)
            unit_bits = np.asarray(row["unit_bits"], dtype=np.float32)
            unit_mean_rates = np.asarray(row["unit_mean_rates"], dtype=np.float32)
            vmin, vmax = scales[condition]
            sheet_path = args.out_dir / "condition_sheets" / f"rr100_unit_activation_maps_{safe_slug(condition)}_{args.map_kind}.png"
            condition_sheet_paths[condition] = sheet_path
            draw_unit_sheet(
                rr100_map,
                unit_bits=unit_bits,
                mean_rates=unit_mean_rates,
                condition=condition,
                sort_mode=str(args.sort_units),
                vmin=vmin,
                vmax=vmax,
                cols=int(args.cols),
                path=sheet_path,
                dpi=int(args.dpi),
            )

    n_units = int(rows[0]["rr100_map"].shape[0]) if rows else 0
    unit_sheet_paths: dict[int, Path] = {}
    if args.sheet_mode in {"unit", "both"}:
        for unit_index in range(n_units):
            vmin, vmax = unit_scale_for_rows(rows, args, unit_index)
            sheet_path = args.out_dir / "unit_condition_sheets" / f"rr100_unit_{unit_index:03d}_conditions_{args.map_kind}.png"
            unit_sheet_paths[int(unit_index)] = sheet_path
            draw_unit_condition_sheet(
                rows,
                unit_index=int(unit_index),
                vmin=vmin,
                vmax=vmax,
                path=sheet_path,
                dpi=int(args.dpi),
            )

    for row in rows:
        condition = str(row["condition"])
        rr100_map = np.asarray(row["rr100_map"], dtype=np.float32)
        unit_bits = np.asarray(row["unit_bits"], dtype=np.float32)
        unit_mean_rates = np.asarray(row["unit_mean_rates"], dtype=np.float32)
        condition_vmin, condition_vmax = scales[condition]
        for unit_index in range(rr100_map.shape[0]):
            unit_vmin, unit_vmax = unit_scale_for_rows(rows, args, int(unit_index))
            individual_path = ""
            if args.write_individual:
                path = (
                    args.out_dir
                    / "individual_units"
                    / safe_slug(condition)
                    / f"rr100_unit_{unit_index:03d}_{safe_slug(condition)}_{args.map_kind}.png"
                )
                draw_individual_unit_map(
                    rr100_map[unit_index],
                    unit_index=int(unit_index),
                    unit_bits=float(unit_bits[unit_index]),
                    condition=condition,
                    vmin=condition_vmin if args.scale_scope == "condition" else unit_vmin,
                    vmax=condition_vmax if args.scale_scope == "condition" else unit_vmax,
                    path=path,
                    dpi=int(args.dpi),
                )
                individual_path = str(path)
            manifest_rows.append(
                {
                    "condition": condition,
                    "condition_label": condition_label(condition).replace("\n", " "),
                    "population_key": "rr100_medoid",
                    "population_version": row["population_version"],
                    "map_kind": args.map_kind,
                    "source_npz": row["source_npz"],
                    "unit_index": int(unit_index),
                    "unit_ssi_bits_per_spike": float(unit_bits[unit_index]),
                    "unit_mean_rate": float(unit_mean_rates[unit_index]),
                    "unit_activation_min": float(np.nanmin(rr100_map[unit_index])),
                    "unit_activation_mean": float(np.nanmean(rr100_map[unit_index])),
                    "unit_activation_max": float(np.nanmax(rr100_map[unit_index])),
                    "population_ssi_bits_per_spike": row["population_ssi_bits_per_spike"],
                    "scale_scope": args.scale_scope,
                    "condition_sheet_vmin": float(condition_vmin),
                    "condition_sheet_vmax": float(condition_vmax),
                    "unit_sheet_vmin": float(unit_vmin),
                    "unit_sheet_vmax": float(unit_vmax),
                    "condition_contact_sheet_png": condition_sheet_paths.get(condition, ""),
                    "unit_condition_sheet_png": unit_sheet_paths.get(int(unit_index), ""),
                    "individual_png": individual_path,
                }
            )

    manifest_csv = args.out_dir / "rr100_unit_activation_map_manifest.csv"
    write_csv_rows(manifest_csv, manifest_rows)
    write_json(
        args.out_dir / "rr100_unit_activation_map_manifest.json",
        {
            "cache_dir": args.cache_dir,
            "out_dir": args.out_dir,
            "conditions": parse_csv_list(args.conditions),
            "rr100_version": args.rr100_version,
            "map_kind": args.map_kind,
            "map_npz_key": final_map_key(args.map_kind),
            "sheet_mode": args.sheet_mode,
            "sort_units": args.sort_units,
            "scale_scope": args.scale_scope,
            "vmin_percentile": float(args.vmin_percentile),
            "vmax_percentile": float(args.vmax_percentile),
            "n_conditions": len(rows),
            "n_units_per_condition": int(rows[0]["rr100_map"].shape[0]) if rows else 0,
            "write_individual": bool(args.write_individual),
            "manifest_csv": manifest_csv,
        },
    )
    if args.sheet_mode in {"unit", "both"}:
        print(f"Wrote {n_units} RR100 unit-by-condition sheets to {args.out_dir / 'unit_condition_sheets'}")
    if args.sheet_mode in {"condition", "both"}:
        print(f"Wrote {len(rows)} condition contact sheets to {args.out_dir / 'condition_sheets'}")
    print(f"Wrote {len(manifest_rows)} unit rows to {manifest_csv}")
    if args.write_individual:
        print(f"Wrote individual unit PNGs under {args.out_dir / 'individual_units'}")


if __name__ == "__main__":
    main()

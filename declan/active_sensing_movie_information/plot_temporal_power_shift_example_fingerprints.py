#!/usr/bin/env python3
"""Map-first checkpoint 4: compact fingerprints across cached examples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import DEFAULT_RUN_DIR


DEFAULT_STAGE2_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_02_activation_maps_v1"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_04_example_fingerprints_v1"
DEFAULT_EXAMPLES = (
    "largest_positive_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_largest_positive_delta' / 'checkpoint_02_selected_activation_maps.npz'},"
    "distinct_positive_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_distinct_positive_delta' / 'checkpoint_02_selected_activation_maps.npz'},"
    "moderate_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_moderate_delta' / 'checkpoint_02_selected_activation_maps.npz'},"
    "near_zero_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_near_zero_delta' / 'checkpoint_02_selected_activation_maps.npz'},"
    "most_negative_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_most_negative_delta' / 'checkpoint_02_selected_activation_maps.npz'}"
)
EPS = 1e-8

# Okabe-Ito colors.
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_RED = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_BLACK = "#000000"
EXAMPLE_COLORS = [OI_BLUE, OI_GREEN, OI_ORANGE, OI_PURPLE, OI_RED, OI_SKY, OI_YELLOW]
EXAMPLE_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=str, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--units", type=str, default="8,92,19")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_units(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def safe_slug(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "unnamed"


def parse_examples(text: str) -> list[tuple[str, Path]]:
    examples: list[tuple[str, Path]] = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Example spec must be label:path, got {item!r}.")
        label, path = item.split(":", 1)
        examples.append((safe_slug(label.strip()), Path(path.strip())))
    if not examples:
        raise ValueError("No examples provided.")
    return examples


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_metadata(cache_path: Path) -> dict[str, Any]:
    path = cache_path.parent / "checkpoint_02_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def bits_for_stack(stack: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(stack, dtype=np.float64), 0.0)
    flat = y.reshape(y.shape[0], -1)
    rbar = np.mean(flat, axis=1)
    gain = flat / (rbar[:, None] + EPS)
    return np.mean(gain * np.log2(gain + EPS), axis=1)


def mean_rate_for_stack(stack: np.ndarray) -> np.ndarray:
    return np.mean(np.maximum(np.asarray(stack, dtype=np.float64), 0.0), axis=(1, 2))


def display_label(label: str) -> str:
    lookup = {
        "largest_positive_delta": "largest +",
        "distinct_positive_delta": "distinct +",
        "moderate_delta": "moderate",
        "near_zero_delta": "near zero",
        "most_negative_delta": "most negative",
    }
    return lookup.get(label, label.replace("_", " "))


def unit_title(unit_info: dict[str, Any]) -> str:
    unit = int(unit_info["unit_index"])
    label = str(unit_info.get("unit_label", f"u{unit:03d}"))
    sf = finite_float(unit_info.get("preferred_sf_cpd"))
    tf = finite_float(unit_info.get("dense_fit_pref_tf_hz"))
    parts = [label]
    if math.isfinite(sf):
        parts.append(f"SF {sf:.3g} cpd")
    if math.isfinite(tf):
        parts.append(f"TF {tf:.2g} Hz")
    return " | ".join(parts)


def collect_payloads(example_specs: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    unit_infos: dict[int, dict[str, Any]] = {}
    for label, cache_path in example_specs:
        cache = load_npz(cache_path)
        selected_units = [int(unit) for unit in np.asarray(cache["selected_units"]).tolist()]
        selected_labels = [str(value) for value in np.asarray(cache["selected_unit_labels"]).tolist()]
        selected_rows = read_csv_rows(cache_path.parent / "checkpoint_02_selected_units.csv")
        row_by_unit = {int(row["unit_index"]): row for row in selected_rows}
        for unit, unit_label in zip(selected_units, selected_labels, strict=True):
            info = dict(row_by_unit.get(int(unit), {}))
            info.setdefault("unit_index", int(unit))
            info.setdefault("unit_label", unit_label)
            unit_infos.setdefault(int(unit), info)
        meta = load_metadata(cache_path)
        pop_static = finite_float(meta.get("population_static_ssi_bits_per_spike"))
        pop_normal = finite_float(meta.get("population_original_ssi_bits_per_spike"))
        pop_delta = pop_normal - pop_static if math.isfinite(pop_static) and math.isfinite(pop_normal) else float("nan")
        payloads.append(
            {
                "label": label,
                "display_label": display_label(label),
                "cache_path": cache_path,
                "selected_units": selected_units,
                "selected_unit_labels": selected_labels,
                "static_maps": np.asarray(cache["static_maps"], dtype=np.float32),
                "normal_maps": np.asarray(cache["normal_maps"], dtype=np.float32),
                "speed_deg_s": np.asarray(cache["speed_deg_s"], dtype=np.float32),
                "population_static_ssi_bits_per_spike": pop_static,
                "population_normal_ssi_bits_per_spike": pop_normal,
                "population_delta_ssi_bits_per_spike": pop_delta,
                "metadata": meta,
            }
        )
    return payloads, unit_infos


def metric_rows(payloads: list[dict[str, Any]], units: list[int], unit_infos: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for unit in units:
            unit_pos = payload["selected_units"].index(int(unit))
            static = np.asarray(payload["static_maps"][:, unit_pos], dtype=np.float32)
            normal = np.asarray(payload["normal_maps"][:, unit_pos], dtype=np.float32)
            diff = normal - static
            static_rate = mean_rate_for_stack(static)
            normal_rate = mean_rate_for_stack(normal)
            static_bits = bits_for_stack(static)
            normal_bits = bits_for_stack(normal)
            delta_bits = normal_bits - static_bits
            diff_rms = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
            max_abs_frame = int(np.nanargmax(np.abs(delta_bits)))
            info = unit_infos[int(unit)]
            rows.append(
                {
                    "example_label": str(payload["label"]),
                    "example_display_label": str(payload["display_label"]),
                    "unit_index": int(unit),
                    "unit_label": str(info.get("unit_label", f"u{int(unit):03d}")),
                    "preferred_sf_cpd": finite_float(info.get("preferred_sf_cpd")),
                    "dense_fit_pref_tf_hz": finite_float(info.get("dense_fit_pref_tf_hz")),
                    "population_delta_ssi_bits_per_spike": float(payload["population_delta_ssi_bits_per_spike"]),
                    "mean_delta_ssi_bits_per_spike": float(np.nanmean(delta_bits)),
                    "min_delta_ssi_bits_per_spike": float(np.nanmin(delta_bits)),
                    "max_delta_ssi_bits_per_spike": float(np.nanmax(delta_bits)),
                    "mean_static_ssi_bits_per_spike": float(np.nanmean(static_bits)),
                    "mean_normal_ssi_bits_per_spike": float(np.nanmean(normal_bits)),
                    "mean_static_rate": float(np.nanmean(static_rate)),
                    "mean_normal_rate": float(np.nanmean(normal_rate)),
                    "mean_delta_rate": float(np.nanmean(normal_rate - static_rate)),
                    "mean_diff_abs_rate": float(np.nanmean(np.abs(diff))),
                    "mean_diff_rms_rate": float(np.nanmean(diff_rms)),
                    "max_diff_rms_rate": float(np.nanmax(diff_rms)),
                    "peak_abs_dssi_frame": max_abs_frame,
                    "peak_abs_dssi_value": float(delta_bits[max_abs_frame]),
                    "peak_abs_dssi_speed_deg_s": float(payload["speed_deg_s"][max_abs_frame]),
                    "mean_speed_deg_s": float(np.nanmean(payload["speed_deg_s"])),
                    "peak_speed_deg_s": float(np.nanmax(payload["speed_deg_s"])),
                }
            )
    return rows


def axis_limits(values: list[float], *, zero: bool = False, pad_frac: float = 0.18) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return -1.0, 1.0
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * float(pad_frac)
    return lo - pad, hi + pad


def render_fingerprint_scatter(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    units: list[int],
    unit_infos: dict[int, dict[str, Any]],
    dpi: int,
) -> tuple[Path, Path]:
    n_units = len(units)
    fig, axes = plt.subplots(
        n_units,
        3,
        figsize=(13.6, max(3.1, 3.0 * n_units)),
        constrained_layout=True,
        squeeze=False,
    )
    y_all = [float(row["mean_delta_ssi_bits_per_spike"]) for row in rows]
    pop_all = [float(row["population_delta_ssi_bits_per_spike"]) for row in rows]
    rate_all = [float(row["mean_delta_rate"]) for row in rows]
    rms_all = [float(row["mean_diff_rms_rate"]) for row in rows]
    y_lim = axis_limits(y_all, zero=True)
    pop_lim = axis_limits(pop_all, zero=True)
    rate_lim = axis_limits(rate_all, zero=True)
    rms_lim = axis_limits(rms_all, zero=True)
    rows_by_unit = {unit: [row for row in rows if int(row["unit_index"]) == int(unit)] for unit in units}

    for unit_idx, unit in enumerate(units):
        unit_rows = rows_by_unit[int(unit)]
        panels = [
            ("mean rate change", "mean_delta_rate", rate_lim),
            ("map-diff RMS", "mean_diff_rms_rate", rms_lim),
            ("population dSSI", "population_delta_ssi_bits_per_spike", pop_lim),
        ]
        for panel_idx, (xlabel, xkey, xlim) in enumerate(panels):
            ax = axes[unit_idx, panel_idx]
            ax.axhline(0.0, color="0.35", lw=0.85)
            ax.axvline(0.0, color="0.35", lw=0.85)
            for example_idx, row in enumerate(unit_rows):
                color = EXAMPLE_COLORS[example_idx % len(EXAMPLE_COLORS)]
                marker = EXAMPLE_MARKERS[example_idx % len(EXAMPLE_MARKERS)]
                x = float(row[xkey])
                y = float(row["mean_delta_ssi_bits_per_spike"])
                size = 68.0 + 1200.0 * max(0.0, float(row["mean_diff_rms_rate"]))
                ax.scatter(
                    [x],
                    [y],
                    s=size,
                    marker=marker,
                    color=color,
                    edgecolor=OI_BLACK,
                    linewidth=0.45,
                    alpha=0.9,
                    label=str(row["example_display_label"]),
                )
                ax.text(x, y, " " + str(row["example_display_label"]), fontsize=7, va="center")
            ax.set_xlim(*xlim)
            ax.set_ylim(*y_lim)
            ax.grid(True, color="0.9", lw=0.7)
            if unit_idx == 0:
                ax.set_title(f"unit dSSI vs {xlabel}", fontsize=10)
            if unit_idx == n_units - 1:
                ax.set_xlabel(xlabel)
            if panel_idx == 0:
                ax.set_ylabel(f"{unit_title(unit_infos[int(unit)])}\nmean dSSI", rotation=0, ha="right", va="center", labelpad=66)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=min(5, len(labels)),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
    )
    fig.suptitle(
        "Checkpoint 4: example fingerprints from cached maps\n"
        "Each point is one image/trace example for one unit; point size follows map-difference RMS",
        fontsize=12,
    )
    png = out_dir / "checkpoint_04_example_unit_fingerprint_scatter.png"
    pdf = out_dir / "checkpoint_04_example_unit_fingerprint_scatter.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def render_difference_gallery(
    *,
    out_dir: Path,
    payloads: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    units: list[int],
    unit_infos: dict[int, dict[str, Any]],
    mode: str,
    dpi: int,
) -> tuple[Path, Path]:
    if mode not in {"mean", "peak"}:
        raise ValueError(f"Unknown gallery mode {mode!r}.")
    n_examples = len(payloads)
    n_units = len(units)
    row_lookup = {
        (str(row["example_label"]), int(row["unit_index"])): row
        for row in rows
    }
    images: dict[tuple[str, int], np.ndarray] = {}
    for payload in payloads:
        for unit in units:
            unit_pos = payload["selected_units"].index(int(unit))
            diff = np.asarray(payload["normal_maps"][:, unit_pos] - payload["static_maps"][:, unit_pos], dtype=np.float32)
            if mode == "mean":
                image = np.nanmean(diff, axis=0)
            else:
                frame = int(row_lookup[(str(payload["label"]), int(unit))]["peak_abs_dssi_frame"])
                image = diff[frame]
            images[(str(payload["label"]), int(unit))] = image

    finite = np.concatenate([np.abs(img).ravel() for img in images.values()])
    finite = finite[np.isfinite(finite)]
    vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
    vmax = max(vmax, EPS)

    fig, axes = plt.subplots(
        n_examples,
        n_units,
        figsize=(3.15 * n_units + 1.2, 2.7 * n_examples),
        constrained_layout=True,
        squeeze=False,
    )
    image_handle = None
    for ex_idx, payload in enumerate(payloads):
        for unit_idx, unit in enumerate(units):
            ax = axes[ex_idx, unit_idx]
            image = images[(str(payload["label"]), int(unit))]
            image_handle = ax.imshow(image, cmap="PuOr_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            key = (str(payload["label"]), int(unit))
            row = row_lookup[key]
            if mode == "mean":
                text = (
                    f"mean dSSI {float(row['mean_delta_ssi_bits_per_spike']):+.3f}\n"
                    f"dRate {float(row['mean_delta_rate']):+.3f}"
                )
            else:
                text = (
                    f"f{int(row['peak_abs_dssi_frame']):02d}\n"
                    f"dSSI {float(row['peak_abs_dssi_value']):+.3f}"
                )
            ax.text(
                0.02,
                0.04,
                text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color=OI_BLACK,
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.78},
            )
            if ex_idx == 0:
                ax.set_title(unit_title(unit_infos[int(unit)]), fontsize=9)
            if unit_idx == 0:
                ax.set_ylabel(
                    f"{payload['display_label']}\npop dSSI {float(payload['population_delta_ssi_bits_per_spike']):+.4f}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=54,
                    fontsize=8,
                )
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_edgecolor("0.55")
    if image_handle is not None:
        cbar = fig.colorbar(image_handle, ax=axes.ravel().tolist(), shrink=0.72, pad=0.012)
        label = "temporal mean normal-static rate" if mode == "mean" else "peak-frame normal-static rate"
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_ticks([-vmax, 0.0, vmax])
        cbar.set_ticklabels([f"{-vmax:.2g}", "0", f"{vmax:.2g}"])
    title = (
        "temporal mean normal-static maps"
        if mode == "mean"
        else "normal-static maps at each example/unit's largest absolute dSSI frame"
    )
    fig.suptitle(f"Checkpoint 4: {title}", fontsize=12)
    stem = "checkpoint_04_mean_difference_map_gallery" if mode == "mean" else "checkpoint_04_peak_dssi_frame_difference_gallery"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    example_specs = parse_examples(str(args.examples))
    units = parse_units(str(args.units))
    payloads, unit_infos = collect_payloads(example_specs)
    missing = [unit for unit in units if int(unit) not in unit_infos]
    if missing:
        raise ValueError(f"Requested units not present in cached examples: {missing}")

    rows = metric_rows(payloads, units, unit_infos)
    metrics_csv = out_dir / "checkpoint_04_example_unit_metrics.csv"
    selection_csv = out_dir / "checkpoint_04_example_selection.csv"
    write_csv(metrics_csv, rows)
    write_csv(
        selection_csv,
        [
            {
                "example_label": str(payload["label"]),
                "example_display_label": str(payload["display_label"]),
                "cache_path": Path(payload["cache_path"]),
                "population_static_ssi_bits_per_spike": float(payload["population_static_ssi_bits_per_spike"]),
                "population_normal_ssi_bits_per_spike": float(payload["population_normal_ssi_bits_per_spike"]),
                "population_delta_ssi_bits_per_spike": float(payload["population_delta_ssi_bits_per_spike"]),
            }
            for payload in payloads
        ],
    )
    scatter_png, scatter_pdf = render_fingerprint_scatter(
        out_dir=out_dir,
        rows=rows,
        units=units,
        unit_infos=unit_infos,
        dpi=int(args.dpi),
    )
    mean_png, mean_pdf = render_difference_gallery(
        out_dir=out_dir,
        payloads=payloads,
        rows=rows,
        units=units,
        unit_infos=unit_infos,
        mode="mean",
        dpi=int(args.dpi),
    )
    peak_png, peak_pdf = render_difference_gallery(
        out_dir=out_dir,
        payloads=payloads,
        rows=rows,
        units=units,
        unit_infos=unit_infos,
        mode="peak",
        dpi=int(args.dpi),
    )
    metadata_json = out_dir / "checkpoint_04_metadata.json"
    write_json(
        metadata_json,
        {
            "analysis": "temporal_power_shift_map_first_checkpoint_04_example_fingerprints",
            "render_type": "cached_targeted_visualization_fingerprint",
            "examples": [{"label": label, "cache_path": path} for label, path in example_specs],
            "units": units,
            "color_contract": (
                "Okabe-Ito categorical markers; PuOr_r signed difference maps; "
                "difference map galleries use one global symmetric scale within the figure."
            ),
            "outputs": {
                "example_selection_csv": selection_csv,
                "example_unit_metrics_csv": metrics_csv,
                "fingerprint_scatter_png": scatter_png,
                "fingerprint_scatter_pdf": scatter_pdf,
                "mean_difference_map_gallery_png": mean_png,
                "mean_difference_map_gallery_pdf": mean_pdf,
                "peak_dssi_frame_difference_gallery_png": peak_png,
                "peak_dssi_frame_difference_gallery_pdf": peak_pdf,
            },
            "checkpoint_policy": "Stop after compact multi-example fingerprints before population modeling.",
        },
    )
    print(f"Wrote {scatter_png}")
    print(f"Wrote {mean_png}")
    print(f"Wrote {peak_png}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {metadata_json}")


if __name__ == "__main__":
    main()

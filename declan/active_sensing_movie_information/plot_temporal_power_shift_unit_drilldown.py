#!/usr/bin/env python3
"""Map-first checkpoint 3: all-frame unit drill-down from cached maps."""

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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import DEFAULT_RUN_DIR
from declan.active_sensing_movie_information.temporal_remapping import MODEL_RATE_HZ


DEFAULT_STAGE2_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_02_activation_maps_v1"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_03_unit_drilldown_v1"
DEFAULT_EXAMPLES = (
    "largest_positive_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_largest_positive_delta' / 'checkpoint_02_selected_activation_maps.npz'},"
    "near_zero_delta:"
    f"{DEFAULT_STAGE2_DIR / 'example_near_zero_delta' / 'checkpoint_02_selected_activation_maps.npz'}"
)
EPS = 1e-8

# Okabe-Ito palette.
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_RED = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_BLACK = "#000000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=str, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--units", type=str, default="8,92")
    parser.add_argument("--frame-rate-hz", type=float, default=MODEL_RATE_HZ)
    parser.add_argument("--ncols", type=int, default=16)
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


def load_metadata(path: Path) -> dict[str, Any]:
    meta_path = path.parent / "checkpoint_02_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def bits_for_stack(stack: np.ndarray) -> np.ndarray:
    """Framewise spatial SSI for a stack with shape T x H x W."""
    y = np.maximum(np.asarray(stack, dtype=np.float64), 0.0)
    flat = y.reshape(y.shape[0], -1)
    rbar = np.mean(flat, axis=1)
    gain = flat / (rbar[:, None] + EPS)
    return np.mean(gain * np.log2(gain + EPS), axis=1)


def mean_rate_for_stack(stack: np.ndarray) -> np.ndarray:
    return np.mean(np.maximum(np.asarray(stack, dtype=np.float64), 0.0), axis=(1, 2))


def robust_scale(values: list[np.ndarray], lo: float, hi: float) -> tuple[float, float]:
    arr = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(arr, lo))
    vmax = float(np.nanpercentile(arr, hi))
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(arr))
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def tile_sheet(stack: np.ndarray, *, ncols: int, pad: int = 2) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    n, h, w = arr.shape
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    sheet_h = nrows * h + (nrows - 1) * pad
    sheet_w = ncols * w + (ncols - 1) * pad
    sheet = np.full((sheet_h, sheet_w), np.nan, dtype=np.float32)
    for idx in range(n):
        row = idx // ncols
        col = idx % ncols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        sheet[y0 : y0 + h, x0 : x0 + w] = arr[idx]
    return sheet


def cmap_with_bad(name: str, bad: str = "#f4f4f4") -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad)
    return cmap


def frame_text_effects(color: str) -> list[Any]:
    stroke = "white" if color == "black" else "black"
    return [path_effects.Stroke(linewidth=0.9, foreground=stroke), path_effects.Normal()]


def annotate_sheet(
    ax: plt.Axes,
    *,
    nframes: int,
    tile_shape: tuple[int, int],
    ncols: int,
    pad: int,
    labels: list[str],
    color: str,
    fontsize: float,
) -> None:
    h, w = tile_shape
    for idx in range(nframes):
        row = idx // ncols
        col = idx % ncols
        x = col * (w + pad) + 2.0
        y = row * (h + pad) + 3.0
        text = ax.text(
            x,
            y,
            labels[idx],
            ha="left",
            va="top",
            color=color,
            fontsize=fontsize,
            family="monospace",
            linespacing=0.95,
        )
        text.set_path_effects(frame_text_effects(color))


def set_sheet_ticks(ax: plt.Axes, *, nframes: int, tile_shape: tuple[int, int], ncols: int, pad: int) -> None:
    h, w = tile_shape
    nrows = int(math.ceil(nframes / ncols))
    row_centers = [row * (h + pad) + h / 2.0 for row in range(nrows)]
    row_labels = []
    for row in range(nrows):
        first = row * ncols
        last = min(nframes - 1, first + ncols - 1)
        row_labels.append(f"f{first:02d}-f{last:02d}")
    ax.set_yticks(row_centers, row_labels, fontsize=7)
    xtick_frames = list(range(0, min(nframes, ncols)))
    x_centers = [col * (w + pad) + w / 2.0 for col in xtick_frames]
    ax.set_xticks(x_centers, [str(frame) for frame in xtick_frames], fontsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def unit_title(unit_info: dict[str, Any]) -> str:
    label = str(unit_info.get("unit_label", f"u{int(unit_info['unit_index']):03d}"))
    sf = finite_float(unit_info.get("preferred_sf_cpd"))
    tf = finite_float(unit_info.get("dense_fit_pref_tf_hz"))
    parts = [label]
    if math.isfinite(sf):
        parts.append(f"SF {sf:.3g} cpd")
    if math.isfinite(tf):
        parts.append(f"TF pref {tf:.2g} Hz")
    return " | ".join(parts)


def frame_metrics_for_unit(
    *,
    example_label: str,
    unit_index: int,
    unit_label: str,
    unit_pos: int,
    static_maps: np.ndarray,
    normal_maps: np.ndarray,
    speed: np.ndarray,
    frame_rate_hz: float,
) -> list[dict[str, Any]]:
    static = np.asarray(static_maps[:, unit_pos], dtype=np.float32)
    normal = np.asarray(normal_maps[:, unit_pos], dtype=np.float32)
    diff = normal - static
    static_rate = mean_rate_for_stack(static)
    normal_rate = mean_rate_for_stack(normal)
    static_bits = bits_for_stack(static)
    normal_bits = bits_for_stack(normal)
    rows: list[dict[str, Any]] = []
    for frame in range(static.shape[0]):
        delta = diff[frame]
        rows.append(
            {
                "example_label": example_label,
                "unit_index": int(unit_index),
                "unit_label": unit_label,
                "frame_index": int(frame),
                "time_ms": float(frame * 1000.0 / float(frame_rate_hz)),
                "speed_deg_s": float(speed[frame]),
                "static_mean_rate": float(static_rate[frame]),
                "normal_mean_rate": float(normal_rate[frame]),
                "delta_mean_rate": float(normal_rate[frame] - static_rate[frame]),
                "static_ssi_bits_per_spike": float(static_bits[frame]),
                "normal_ssi_bits_per_spike": float(normal_bits[frame]),
                "delta_ssi_bits_per_spike": float(normal_bits[frame] - static_bits[frame]),
                "diff_abs_mean_rate": float(np.nanmean(np.abs(delta))),
                "diff_rms_rate": float(np.sqrt(np.nanmean(delta * delta))),
                "diff_positive_fraction": float(np.nanmean(delta > 0.0)),
                "diff_negative_fraction": float(np.nanmean(delta < 0.0)),
            }
        )
    return rows


def render_unit_example_sheet(
    *,
    out_dir: Path,
    example_label: str,
    unit_info: dict[str, Any],
    unit_pos: int,
    static_maps: np.ndarray,
    normal_maps: np.ndarray,
    speed: np.ndarray,
    frame_rate_hz: float,
    ncols: int,
    act_scale: tuple[float, float],
    diff_scale_abs: float,
    population_delta: float | None,
    dpi: int,
) -> tuple[Path, Path]:
    unit_index = int(unit_info["unit_index"])
    unit_label = str(unit_info.get("unit_label", f"u{unit_index:03d}"))
    static = np.asarray(static_maps[:, unit_pos], dtype=np.float32)
    normal = np.asarray(normal_maps[:, unit_pos], dtype=np.float32)
    diff = normal - static
    nframes, h, w = static.shape
    frames = np.arange(nframes)
    time_ms = frames * 1000.0 / float(frame_rate_hz)
    static_rate = mean_rate_for_stack(static)
    normal_rate = mean_rate_for_stack(normal)
    static_bits = bits_for_stack(static)
    normal_bits = bits_for_stack(normal)
    delta_bits = normal_bits - static_bits
    act_vmin, act_vmax = act_scale
    diff_abs = max(float(diff_scale_abs), EPS)
    pad = 2

    fig = plt.figure(figsize=(17.0, 15.0), constrained_layout=False)
    gs = fig.add_gridspec(
        7,
        2,
        width_ratios=[1.0, 0.028],
        height_ratios=[0.55, 0.55, 0.55, 0.55, 2.15, 2.15, 2.15],
        left=0.06,
        right=0.95,
        top=0.93,
        bottom=0.045,
        hspace=0.34,
        wspace=0.04,
    )

    ax_speed = fig.add_subplot(gs[0, 0])
    ax_rate = fig.add_subplot(gs[1, 0], sharex=ax_speed)
    ax_bits = fig.add_subplot(gs[2, 0], sharex=ax_speed)
    ax_delta = fig.add_subplot(gs[3, 0], sharex=ax_speed)
    for ax in (ax_speed, ax_rate, ax_bits, ax_delta):
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))

    ax_speed.plot(time_ms, speed, color=OI_BLUE, lw=1.6)
    ax_speed.set_ylabel("speed\ndeg/s", rotation=0, ha="right", va="center", labelpad=28)

    ax_rate.plot(time_ms, static_rate, color="0.55", linestyle="--", lw=1.35, label="static")
    ax_rate.plot(time_ms, normal_rate, color=OI_BLUE, lw=1.45, label="normal")
    ax_rate.set_ylabel("mean\nrate", rotation=0, ha="right", va="center", labelpad=28)
    ax_rate.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)

    ax_bits.plot(time_ms, static_bits, color="0.55", linestyle="--", lw=1.35)
    ax_bits.plot(time_ms, normal_bits, color=OI_GREEN, lw=1.45)
    ax_bits.set_ylabel("SSI\nbits/spk", rotation=0, ha="right", va="center", labelpad=28)

    ax_delta.axhline(0.0, color="0.35", lw=0.9)
    ax_delta.plot(time_ms, delta_bits, color=OI_ORANGE, lw=1.55)
    ax_delta.fill_between(time_ms, 0.0, delta_bits, where=delta_bits >= 0.0, color=OI_ORANGE, alpha=0.18)
    ax_delta.fill_between(time_ms, 0.0, delta_bits, where=delta_bits < 0.0, color=OI_PURPLE, alpha=0.18)
    ax_delta.set_ylabel("normal-static\nSSI", rotation=0, ha="right", va="center", labelpad=28)
    ax_delta.set_xlabel("time (ms)")

    for ax in (ax_speed, ax_rate, ax_bits):
        plt.setp(ax.get_xticklabels(), visible=False)

    map_specs = [
        (
            "static maps",
            static,
            "cividis",
            act_vmin,
            act_vmax,
            [f"f{idx:02d}\nS {static_bits[idx]:.3f}" for idx in range(nframes)],
            "white",
        ),
        (
            "normal maps",
            normal,
            "cividis",
            act_vmin,
            act_vmax,
            [f"f{idx:02d}\nN {normal_bits[idx]:.3f}" for idx in range(nframes)],
            "white",
        ),
        (
            "normal - static",
            diff,
            "PuOr_r",
            -diff_abs,
            diff_abs,
            [f"f{idx:02d}\ndSSI {delta_bits[idx]:+.3f}" for idx in range(nframes)],
            "black",
        ),
    ]
    act_image = None
    diff_image = None
    for row_idx, (title, stack, cmap_name, vmin, vmax, labels, text_color) in enumerate(map_specs, start=4):
        ax = fig.add_subplot(gs[row_idx, 0])
        sheet = tile_sheet(stack, ncols=ncols, pad=pad)
        image = ax.imshow(
            sheet,
            cmap=cmap_with_bad(cmap_name),
            vmin=float(vmin),
            vmax=float(vmax),
            interpolation="nearest",
        )
        annotate_sheet(
            ax,
            nframes=nframes,
            tile_shape=(h, w),
            ncols=ncols,
            pad=pad,
            labels=labels,
            color=text_color,
            fontsize=4.6,
        )
        set_sheet_ticks(ax, nframes=nframes, tile_shape=(h, w), ncols=ncols, pad=pad)
        ax.set_title(title, fontsize=10, pad=4)
        if row_idx == 4:
            act_image = image
        if row_idx == 6:
            diff_image = image

    if act_image is not None:
        cax = fig.add_subplot(gs[4:6, 1])
        cbar = fig.colorbar(act_image, cax=cax)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.set_label("model rate\nstatic & normal", fontsize=8)
        cbar.set_ticks([float(act_vmin), float(act_vmax)])
        cbar.set_ticklabels([f"{float(act_vmin):.2g}", f"{float(act_vmax):.2g}"])
    if diff_image is not None:
        cax = fig.add_subplot(gs[6, 1])
        cbar = fig.colorbar(diff_image, cax=cax)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.set_label("normal-static\nmodel rate", fontsize=8)
        cbar.set_ticks([-diff_abs, 0.0, diff_abs])
        cbar.set_ticklabels([f"{-diff_abs:.2g}", "0", f"{diff_abs:.2g}"])

    pop_text = "" if population_delta is None or not math.isfinite(population_delta) else f"; population dSSI {population_delta:+.4f}"
    fig.suptitle(
        f"Checkpoint 3 all-frame drill-down: {unit_title(unit_info)} | {example_label}{pop_text}\n"
        "tile text shows frame and instantaneous map SSI; difference row text shows normal-static SSI",
        fontsize=13,
    )

    png = out_dir / f"checkpoint_03_{unit_label}_{example_label}_all_frame_maps.png"
    pdf = out_dir / f"checkpoint_03_{unit_label}_{example_label}_all_frame_maps.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def render_unit_metric_comparison(
    *,
    out_dir: Path,
    unit_info: dict[str, Any],
    example_payloads: list[dict[str, Any]],
    unit_pos_by_example: dict[str, int],
    frame_rate_hz: float,
    dpi: int,
) -> tuple[Path, Path]:
    unit_index = int(unit_info["unit_index"])
    unit_label = str(unit_info.get("unit_label", f"u{unit_index:03d}"))
    n_examples = len(example_payloads)
    fig, axes = plt.subplots(
        4,
        n_examples,
        figsize=(6.0 * n_examples, 8.6),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    if n_examples == 1:
        axes = np.asarray(axes)[:, None]

    for col, payload in enumerate(example_payloads):
        label = str(payload["label"])
        unit_pos = int(unit_pos_by_example[label])
        static = np.asarray(payload["static_maps"][:, unit_pos], dtype=np.float32)
        normal = np.asarray(payload["normal_maps"][:, unit_pos], dtype=np.float32)
        speed = np.asarray(payload["speed_deg_s"], dtype=np.float64)
        frames = np.arange(static.shape[0])
        time_ms = frames * 1000.0 / float(frame_rate_hz)
        static_rate = mean_rate_for_stack(static)
        normal_rate = mean_rate_for_stack(normal)
        static_bits = bits_for_stack(static)
        normal_bits = bits_for_stack(normal)
        delta_bits = normal_bits - static_bits
        diff = normal - static
        diff_rms = np.sqrt(np.mean(diff * diff, axis=(1, 2)))

        axes[0, col].plot(time_ms, speed, color=OI_BLUE, lw=1.6)
        axes[0, col].set_title(label.replace("_", " "), fontsize=10)
        axes[1, col].plot(time_ms, static_rate, color="0.55", linestyle="--", lw=1.35, label="static")
        axes[1, col].plot(time_ms, normal_rate, color=OI_BLUE, lw=1.45, label="normal")
        axes[2, col].plot(time_ms, static_bits, color="0.55", linestyle="--", lw=1.35)
        axes[2, col].plot(time_ms, normal_bits, color=OI_GREEN, lw=1.45)
        axes[3, col].axhline(0.0, color="0.35", lw=0.9)
        axes[3, col].plot(time_ms, delta_bits, color=OI_ORANGE, lw=1.55, label="dSSI")
        ax2 = axes[3, col].twinx()
        ax2.plot(time_ms, diff_rms, color=OI_PURPLE, lw=1.1, alpha=0.85, label="diff RMS")
        ax2.tick_params(axis="y", labelsize=7, colors=OI_PURPLE)
        ax2.set_ylabel("diff RMS", color=OI_PURPLE, fontsize=8)
        for row in range(4):
            axes[row, col].grid(True, color="0.9", linewidth=0.7)
            axes[row, col].set_xlim(float(time_ms[0]), float(time_ms[-1]))
        axes[3, col].set_xlabel("time (ms)")

    ylabels = ["speed\ndeg/s", "mean\nrate", "SSI\nbits/spk", "normal-static\nSSI"]
    for row, ylabel in enumerate(ylabels):
        axes[row, 0].set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=34)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
    fig.suptitle(
        f"Checkpoint 3 metric comparison: {unit_title(unit_info)}\n"
        "Bottom row overlays dSSI (orange) with map-difference RMS (purple, right axis)",
        fontsize=12,
    )
    png = out_dir / f"checkpoint_03_{unit_label}_metric_comparison.png"
    pdf = out_dir / f"checkpoint_03_{unit_label}_metric_comparison.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def drilled_unit_rows(unit_infos: dict[int, dict[str, Any]], units: list[int]) -> list[dict[str, Any]]:
    role_lookup = {
        8: (
            "high_sf_mechanistic_candidate",
            "High-SF representative with dense TF preference in the low-Hz range; tests the direct power-shift prediction.",
        ),
        92: (
            "middle_sf_mixed_candidate",
            "Middle-SF representative with visible map changes; useful for checking gain versus spatial-information dissociation.",
        ),
        19: (
            "low_sf_control_candidate",
            "Low-SF representative with edge TF preference; useful as a mechanistically distinct control.",
        ),
    }
    rows: list[dict[str, Any]] = []
    for unit in units:
        info = unit_infos[int(unit)]
        role, rationale = role_lookup.get(
            int(unit),
            ("requested_unit", "User or script requested this unit for all-frame map drill-down."),
        )
        rows.append(
            {
                "unit_index": int(unit),
                "unit_label": str(info.get("unit_label", f"u{int(unit):03d}")),
                "drilldown_role": role,
                "drilldown_rationale": rationale,
                "preferred_sf_cpd": finite_float(info.get("preferred_sf_cpd")),
                "dense_fit_pref_tf_hz": finite_float(info.get("dense_fit_pref_tf_hz")),
                "checkpoint_02_selection_role": str(info.get("selection_role", "")),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    example_specs = parse_examples(str(args.examples))
    units = parse_units(str(args.units))
    payloads: list[dict[str, Any]] = []
    unit_infos: dict[int, dict[str, Any]] = {}
    for label, cache_path in example_specs:
        cache = load_npz(cache_path)
        selected_units = [int(unit) for unit in np.asarray(cache["selected_units"]).tolist()]
        unit_labels = [str(label_value) for label_value in np.asarray(cache["selected_unit_labels"]).tolist()]
        selected_rows = read_csv_rows(cache_path.parent / "checkpoint_02_selected_units.csv")
        row_by_unit = {int(row["unit_index"]): row for row in selected_rows}
        for unit, unit_label in zip(selected_units, unit_labels, strict=True):
            row = dict(row_by_unit.get(int(unit), {}))
            row.setdefault("unit_index", int(unit))
            row.setdefault("unit_label", unit_label)
            unit_infos.setdefault(int(unit), row)
        meta = load_metadata(cache_path)
        population_static = finite_float(meta.get("population_static_ssi_bits_per_spike"))
        population_original = finite_float(meta.get("population_original_ssi_bits_per_spike"))
        population_delta = (
            population_original - population_static
            if math.isfinite(population_static) and math.isfinite(population_original)
            else float("nan")
        )
        payloads.append(
            {
                "label": label,
                "cache_path": cache_path,
                "selected_units": selected_units,
                "selected_unit_labels": unit_labels,
                "static_maps": np.asarray(cache["static_maps"], dtype=np.float32),
                "normal_maps": np.asarray(cache["normal_maps"], dtype=np.float32),
                "speed_deg_s": np.asarray(cache["speed_deg_s"], dtype=np.float32),
                "population_delta": population_delta,
                "metadata": meta,
            }
        )

    missing = [unit for unit in units if int(unit) not in unit_infos]
    if missing:
        raise ValueError(f"Requested units not present in cached examples: {missing}")

    frame_rows: list[dict[str, Any]] = []
    map_outputs: list[dict[str, Any]] = []
    metric_outputs: list[dict[str, Any]] = []
    for unit in units:
        unit = int(unit)
        unit_info = unit_infos[unit]
        unit_pos_by_example: dict[str, int] = {}
        act_values: list[np.ndarray] = []
        diff_values: list[np.ndarray] = []
        for payload in payloads:
            label = str(payload["label"])
            unit_pos = payload["selected_units"].index(unit)
            unit_pos_by_example[label] = unit_pos
            static = np.asarray(payload["static_maps"][:, unit_pos], dtype=np.float32)
            normal = np.asarray(payload["normal_maps"][:, unit_pos], dtype=np.float32)
            act_values.extend([static, normal])
            diff_values.append(normal - static)
            frame_rows.extend(
                frame_metrics_for_unit(
                    example_label=label,
                    unit_index=unit,
                    unit_label=str(unit_info.get("unit_label", f"u{unit:03d}")),
                    unit_pos=unit_pos,
                    static_maps=np.asarray(payload["static_maps"], dtype=np.float32),
                    normal_maps=np.asarray(payload["normal_maps"], dtype=np.float32),
                    speed=np.asarray(payload["speed_deg_s"], dtype=np.float32),
                    frame_rate_hz=float(args.frame_rate_hz),
                )
            )
        act_scale = robust_scale(act_values, 1.0, 99.5)
        diff_abs_values = np.concatenate([np.abs(value).ravel() for value in diff_values])
        diff_abs = float(np.nanpercentile(diff_abs_values[np.isfinite(diff_abs_values)], 99.0))
        diff_abs = max(diff_abs, EPS)

        metric_png, metric_pdf = render_unit_metric_comparison(
            out_dir=out_dir,
            unit_info=unit_info,
            example_payloads=payloads,
            unit_pos_by_example=unit_pos_by_example,
            frame_rate_hz=float(args.frame_rate_hz),
            dpi=int(args.dpi),
        )
        metric_outputs.append({"unit_index": unit, "png": metric_png, "pdf": metric_pdf})

        for payload in payloads:
            label = str(payload["label"])
            unit_pos = unit_pos_by_example[label]
            png, pdf = render_unit_example_sheet(
                out_dir=out_dir,
                example_label=label,
                unit_info=unit_info,
                unit_pos=unit_pos,
                static_maps=np.asarray(payload["static_maps"], dtype=np.float32),
                normal_maps=np.asarray(payload["normal_maps"], dtype=np.float32),
                speed=np.asarray(payload["speed_deg_s"], dtype=np.float32),
                frame_rate_hz=float(args.frame_rate_hz),
                ncols=int(args.ncols),
                act_scale=act_scale,
                diff_scale_abs=diff_abs,
                population_delta=finite_float(payload.get("population_delta")),
                dpi=int(args.dpi),
            )
            map_outputs.append({"unit_index": unit, "example_label": label, "png": png, "pdf": pdf})

    drilled_csv = out_dir / "checkpoint_03_drilled_units.csv"
    frame_csv = out_dir / "checkpoint_03_unit_frame_metrics.csv"
    metadata_json = out_dir / "checkpoint_03_metadata.json"
    write_csv(drilled_csv, drilled_unit_rows(unit_infos, units))
    write_csv(frame_csv, frame_rows)
    write_json(
        metadata_json,
        {
            "analysis": "temporal_power_shift_map_first_checkpoint_03_unit_drilldown",
            "render_type": "cached_targeted_visualization_drilldown",
            "examples": [{"label": label, "cache_path": path} for label, path in example_specs],
            "units": units,
            "frame_rate_hz": float(args.frame_rate_hz),
            "color_contract": (
                "Okabe-Ito categorical colors; cividis activation maps; PuOr_r signed difference maps; "
                "activation and difference color scales are shared across examples within each unit."
            ),
            "outputs": {
                "drilled_units_csv": drilled_csv,
                "unit_frame_metrics_csv": frame_csv,
                "metric_comparison_figures": metric_outputs,
                "all_frame_map_figures": map_outputs,
            },
            "checkpoint_policy": "Stop after selected-unit all-frame drill-down before population summaries.",
        },
    )
    for row in metric_outputs:
        print(f"Wrote {row['png']}")
    for row in map_outputs:
        print(f"Wrote {row['png']}")
    print(f"Wrote {frame_csv}")
    print(f"Wrote {metadata_json}")


if __name__ == "__main__":
    main()

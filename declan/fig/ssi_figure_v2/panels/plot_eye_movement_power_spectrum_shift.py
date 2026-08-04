#!/usr/bin/env python3
"""Plot how SSI figure-v3 FEM banks redistribute image power.

Pure translations do not change a static image's Fourier magnitude.  What the
fixation trajectory changes is the *retinal modulation* spectrum: for each
spatial frequency, eye motion converts static image power into frame-to-frame
luminance change.  This diagnostic uses the exact SSI figure-v3 image and trace
banks, then estimates that modulation spectrum with the Fourier shift theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas  # noqa: E402


SSI_V3_BANK_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "merged"
)
DEFAULT_IMAGE_TABLE = SSI_V3_BANK_DIR / "image_feature_table.csv"
DEFAULT_TRACE_TABLE = SSI_V3_BANK_DIR / "trace_feature_table.csv"
DEFAULT_TRACE_XY = SSI_V3_BANK_DIR / "trace_xy.npy"
DEFAULT_OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "power_spectrum_shift"

DT_S = 1.0 / 120.0
PATCH_SIZE_PX = 151
FIT_LOW_CPD = 2.0
FIT_HIGH_CPD = 16.0
PLOT_LOW_CPD = 0.5
PLOT_HIGH_CPD = 18.0

COLORS = {
    "all_real_fem": "#222222",
    "short_drift": "#276fbf",
    "mid_drift": "#559f76",
    "long_drift": "#c36d1d",
    "microsaccade": "#b83b5e",
}
LABELS = {
    "all_real_fem": "all real FEM",
    "short_drift": "short drift",
    "mid_drift": "mid drift",
    "long_drift": "long drift",
    "microsaccade": "microsaccade windows",
}
DOT_LABELS = {
    "source_image": "source",
    "all_real_fem": "all FEM",
    "short_drift": "short",
    "mid_drift": "mid",
    "long_drift": "long",
    "microsaccade": "microsaccade",
}


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean_sem(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(float(arr.size))), int(arr.size)


def clip_patch_subpixel(canvas: np.ndarray, center_xy_px: tuple[float, float], size_px: int) -> np.ndarray:
    half = int(size_px) // 2
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    x = cx + np.arange(int(size_px), dtype=np.float64) - float(half)
    y = cy + np.arange(int(size_px), dtype=np.float64) - float(half)
    xx, yy = np.meshgrid(x, y)
    fill = float(np.nanmean(canvas))
    patch = map_coordinates(
        np.asarray(canvas, dtype=np.float32),
        [yy, xx],
        order=1,
        mode="constant",
        cval=fill,
        prefilter=False,
    )
    return np.asarray(patch, dtype=np.float32)


def frequency_grid(size_px: int, ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.fft.fftfreq(int(size_px), d=1.0 / float(ppd))
    fx, fy = np.meshgrid(freq, freq)
    rr = np.hypot(fx, fy)
    return fx.astype(np.float32), fy.astype(np.float32), rr.astype(np.float32)


def radial_edges(rr: np.ndarray, n_bins: int) -> np.ndarray:
    high = min(float(PLOT_HIGH_CPD), float(np.nanmax(rr)))
    return np.geomspace(float(PLOT_LOW_CPD), high, int(n_bins) + 1)


def radial_bin(power2d: np.ndarray, rr: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers: list[float] = []
    values: list[float] = []
    arr = np.asarray(power2d, dtype=np.float64)
    radius = np.asarray(rr, dtype=np.float64)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (radius >= float(lo)) & (radius < float(hi))
        centers.append(float(math.sqrt(float(lo) * float(hi))))
        values.append(float(np.nanmean(arr[mask])) if np.any(mask) else float("nan"))
    return np.asarray(centers, dtype=np.float64), np.asarray(values, dtype=np.float64)


def loglog_slope(freq: np.ndarray, power: np.ndarray, *, low_cpd: float, high_cpd: float) -> float:
    f = np.asarray(freq, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    keep = np.isfinite(f) & np.isfinite(p) & (f >= float(low_cpd)) & (f <= float(high_cpd)) & (f > 0.0) & (p > 0.0)
    if int(np.sum(keep)) < 3:
        return float("nan")
    return float(np.polyfit(np.log10(f[keep]), np.log10(p[keep]), 1)[0])


def spectral_flatness(power: np.ndarray, fit_mask: np.ndarray) -> float:
    vals = np.asarray(power, dtype=np.float64)[np.asarray(fit_mask, dtype=bool)]
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(vals + 1e-30))) / max(float(np.mean(vals)), 1e-30))


def geometric_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def normalize_to_fit_band(power: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(power, dtype=np.float64)
    scale = geometric_mean(arr[np.asarray(fit_mask, dtype=bool)])
    if not math.isfinite(scale) or scale <= 0.0:
        return arr * np.nan
    return arr / scale


def patch_power2d(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float32)
    arr = arr - float(np.nanmean(arr))
    win = np.hanning(arr.shape[0]).astype(np.float32)
    window = np.outer(win, win).astype(np.float32)
    weighted = arr * window
    norm = float(np.mean(window * window))
    fft = np.fft.fft2(weighted, norm="ortho")
    return (np.abs(fft) ** 2 / max(norm, 1e-12)).astype(np.float32)


def motion_kernel2d(
    traces_xy_deg: np.ndarray,
    fx_cpd: np.ndarray,
    fy_cpd: np.ndarray,
    *,
    dt_s: float,
    chunk_size: int,
) -> np.ndarray:
    traces = np.asarray(traces_xy_deg, dtype=np.float32)
    if traces.ndim != 3 or traces.shape[1] < 2 or traces.shape[2] != 2:
        raise ValueError(f"Expected traces with shape (n,T,2); got {traces.shape}")
    flat_fx = np.asarray(fx_cpd, dtype=np.float32).ravel()
    flat_fy = np.asarray(fy_cpd, dtype=np.float32).ravel()
    acc = np.zeros(flat_fx.shape[0], dtype=np.float64)
    n = 0
    for start in range(0, traces.shape[0], int(chunk_size)):
        chunk = traces[start : start + int(chunk_size)].astype(np.float32, copy=True)
        chunk -= np.nanmean(chunk, axis=1, keepdims=True)
        # The BackImage twin helper samples screen columns opposite gaze x and
        # screen rows with gaze y; signs are irrelevant for isotropic averages
        # but matter for anisotropic patches, so keep the same convention.
        screen_x = -chunk[:, :, 0]
        screen_y = chunk[:, :, 1]
        phase_arg = screen_x[:, :, None] * flat_fx[None, None, :] + screen_y[:, :, None] * flat_fy[None, None, :]
        phase = np.exp((-2j * np.pi) * phase_arg)
        diff = np.diff(phase, axis=1)
        kernel = np.mean(np.abs(diff) ** 2, axis=1) / max(float(dt_s) ** 2, 1e-12)
        acc += np.sum(kernel, axis=0)
        n += int(kernel.shape[0])
    if n == 0:
        return np.zeros_like(fx_cpd, dtype=np.float32)
    return (acc / float(n)).reshape(fx_cpd.shape).astype(np.float32)


def trace_path_arcmin(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)) * 60.0)


def trace_rms_arcmin(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))) * 60.0)


def select_evenly(df: pd.DataFrame, *, sort_col: str, max_rows: int) -> pd.DataFrame:
    if int(max_rows) <= 0 or df.shape[0] <= int(max_rows):
        return df.copy()
    ordered = df.sort_values(sort_col).reset_index(drop=True)
    positions = np.linspace(0, ordered.shape[0] - 1, int(max_rows)).round().astype(int)
    positions = np.unique(positions)
    return ordered.iloc[positions].copy()


def sample_indices(indices: np.ndarray, *, max_n: int, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if int(max_n) <= 0 or values.size <= int(max_n):
        return np.sort(values)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(values, size=int(max_n), replace=False))


def select_trace_groups(trace_table: pd.DataFrame, *, max_traces_per_condition: int, seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    trace_table = trace_table.copy()
    trace_table["trace_bank_index"] = pd.to_numeric(trace_table["trace_bank_index"], errors="coerce").astype(int)
    trace_table["path_length_arcmin"] = pd.to_numeric(trace_table["rendered_path_length_arcmin"], errors="coerce")
    trace_table["n_ms"] = pd.to_numeric(trace_table["rendered_n_microsaccade_events"], errors="coerce").fillna(0).astype(int)
    no_ms = trace_table[(trace_table["n_ms"] == 0) & np.isfinite(trace_table["path_length_arcmin"])].copy()
    with_ms = trace_table[(trace_table["n_ms"] > 0) & np.isfinite(trace_table["path_length_arcmin"])].copy()
    q25 = float(no_ms["path_length_arcmin"].quantile(0.25))
    q75 = float(no_ms["path_length_arcmin"].quantile(0.75))
    masks = {
        "all_real_fem": trace_table[np.isfinite(trace_table["path_length_arcmin"])],
        "short_drift": no_ms[no_ms["path_length_arcmin"] <= q25],
        "mid_drift": no_ms[(no_ms["path_length_arcmin"] > q25) & (no_ms["path_length_arcmin"] < q75)],
        "long_drift": no_ms[no_ms["path_length_arcmin"] >= q75],
        "microsaccade": with_ms,
    }
    groups: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for offset, (condition, frame) in enumerate(masks.items()):
        available = frame["trace_bank_index"].to_numpy(dtype=int)
        selected = sample_indices(available, max_n=int(max_traces_per_condition), seed=int(seed) + 1009 * offset)
        groups[condition] = selected
        selected_frame = trace_table[trace_table["trace_bank_index"].isin(selected)]
        rows.append(
            {
                "condition": condition,
                "label": LABELS.get(condition, condition),
                "n_available_traces": int(available.size),
                "n_selected_traces": int(selected.size),
                "path_q25_arcmin": float(selected_frame["path_length_arcmin"].quantile(0.25)) if not selected_frame.empty else float("nan"),
                "path_median_arcmin": float(selected_frame["path_length_arcmin"].median()) if not selected_frame.empty else float("nan"),
                "path_q75_arcmin": float(selected_frame["path_length_arcmin"].quantile(0.75)) if not selected_frame.empty else float("nan"),
                "microsaccade_fraction": float(np.mean(selected_frame["n_ms"].to_numpy(dtype=float) > 0.0)) if not selected_frame.empty else float("nan"),
            }
        )
    return groups, rows


def load_image_powers(image_table: pd.DataFrame, *, max_images: int, patch_size_px: int) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    selected = select_evenly(image_table, sort_col="image_orientation_coherence", max_rows=int(max_images)).reset_index(drop=True)
    powers: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    example_patch: np.ndarray | None = None
    example_score = -np.inf
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    ppds: list[float] = []
    for image_pos, (_, row) in enumerate(selected.iterrows(), start=1):
        key = (str(row["session"]), int(row["trial_idx"]))
        print(f"[images] {image_pos}/{selected.shape[0]} {key[0]} trial {key[1]}", flush=True)
        if key not in canvas_cache:
            canvas_cache[key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
        canvas, ppd, _shape = canvas_cache[key]
        ppds.append(float(ppd))
        center = (float(row["image_patch_center_x_px"]), float(row["image_patch_center_y_px"]))
        patch = clip_patch_subpixel(canvas, center, int(patch_size_px))
        patch_score = float(np.nanstd(patch))
        if patch_score > example_score:
            example_patch = np.asarray(patch, dtype=np.float32)
            example_score = patch_score
        powers.append(patch_power2d(patch))
        rows.append(
            {
                "image_index": int(row["image_index"]),
                "source_row": int(row["source_row"]) if "source_row" in row.index else -1,
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "image_edge_axis_deg": float(row["image_edge_axis_deg"]),
                "image_patch_center_x_px": float(row["image_patch_center_x_px"]),
                "image_patch_center_y_px": float(row["image_patch_center_y_px"]),
                "image_patch_std": float(row["image_patch_std"]) if "image_patch_std" in row.index else patch_score,
                "image_patch_fraction_background": (
                    float(row["image_patch_fraction_background"]) if "image_patch_fraction_background" in row.index else float("nan")
                ),
                "image_patch_fraction_inside_image": (
                    float(row["image_patch_fraction_inside_image"]) if "image_patch_fraction_inside_image" in row.index else float("nan")
                ),
                "ppd": float(ppd),
            }
        )
    if not powers:
        raise RuntimeError("No image patches were loaded.")
    return np.stack(powers, axis=0), rows, np.asarray(example_patch, dtype=np.float32)


def image_level_rows(
    source_powers: np.ndarray,
    kernels: dict[str, np.ndarray],
    rr: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    freq, _ = radial_bin(source_powers[0], rr, edges)
    fit_mask = (freq >= FIT_LOW_CPD) & (freq <= FIT_HIGH_CPD)
    radial_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for image_pos in range(source_powers.shape[0]):
        src_freq, src_power = radial_bin(source_powers[image_pos], rr, edges)
        source_slope = loglog_slope(src_freq, src_power, low_cpd=FIT_LOW_CPD, high_cpd=FIT_HIGH_CPD)
        radial_rows.extend(
            {
                "image_pos": image_pos,
                "condition": "source_image",
                "frequency_cpd": float(f),
                "source_power": float(p),
                "modulation_power": float("nan"),
                "transfer_power_ratio": float("nan"),
                "in_fit_band": bool(in_fit),
            }
            for f, p, in_fit in zip(src_freq, src_power, fit_mask, strict=True)
        )
        metric_rows.append(
            {
                "image_pos": image_pos,
                "condition": "source_image",
                "source_power_slope": source_slope,
                "modulation_power_slope": float("nan"),
                "transfer_slope": float("nan"),
                "modulation_total_power": 0.0,
                "modulation_flatness": float("nan"),
            }
        )
        for condition, kernel in kernels.items():
            mod_power2d = source_powers[image_pos] * np.asarray(kernel, dtype=np.float32)
            _, mod_power = radial_bin(mod_power2d, rr, edges)
            ratio = mod_power / np.maximum(src_power, 1e-30)
            radial_rows.extend(
                {
                    "image_pos": image_pos,
                    "condition": condition,
                    "frequency_cpd": float(f),
                    "source_power": float(sp),
                    "modulation_power": float(mp),
                    "transfer_power_ratio": float(r),
                    "in_fit_band": bool(in_fit),
                }
                for f, sp, mp, r, in_fit in zip(src_freq, src_power, mod_power, ratio, fit_mask, strict=True)
            )
            metric_rows.append(
                {
                    "image_pos": image_pos,
                    "condition": condition,
                    "source_power_slope": source_slope,
                    "modulation_power_slope": loglog_slope(src_freq, mod_power, low_cpd=FIT_LOW_CPD, high_cpd=FIT_HIGH_CPD),
                    "transfer_slope": loglog_slope(src_freq, ratio, low_cpd=FIT_LOW_CPD, high_cpd=FIT_HIGH_CPD),
                    "modulation_total_power": float(np.nansum(mod_power[fit_mask])),
                    "modulation_flatness": spectral_flatness(mod_power, fit_mask),
                }
            )
    return freq, radial_rows, metric_rows


def aggregate_radial(radial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(radial_rows)
    out: list[dict[str, Any]] = []
    for (condition, freq), group in frame.groupby(["condition", "frequency_cpd"], sort=True):
        row: dict[str, Any] = {
            "condition": str(condition),
            "frequency_cpd": float(freq),
            "n_images": int(group["image_pos"].nunique()),
            "in_fit_band": bool(group["in_fit_band"].any()),
        }
        for key in ("source_power", "modulation_power", "transfer_power_ratio"):
            row[f"{key}_geomean"] = geometric_mean(group[key].to_numpy(dtype=np.float64))
            mean, sem, n = finite_mean_sem(group[key].to_numpy(dtype=np.float64))
            row[f"{key}_mean"] = mean
            row[f"{key}_sem"] = sem
            row[f"{key}_n"] = n
        out.append(row)
    return out


def aggregate_metrics(metric_rows: list[dict[str, Any]], trace_group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    trace_meta = {str(row["condition"]): row for row in trace_group_rows}
    out: list[dict[str, Any]] = []
    for condition, group in frame.groupby("condition", sort=False):
        row: dict[str, Any] = {
            "condition": str(condition),
            "label": LABELS.get(str(condition), str(condition).replace("_", " ")),
            "n_images": int(group["image_pos"].nunique()),
        }
        if str(condition) in trace_meta:
            for key, value in trace_meta[str(condition)].items():
                if key not in {"condition", "label"}:
                    row[key] = value
        for key in ("source_power_slope", "modulation_power_slope", "transfer_slope", "modulation_total_power", "modulation_flatness"):
            mean, sem, n = finite_mean_sem(group[key].to_numpy(dtype=np.float64))
            row[f"{key}_mean"] = mean
            row[f"{key}_sem"] = sem
            row[f"{key}_n"] = n
        out.append(row)
    return out


def representative_trace_indices(trace_table: pd.DataFrame, groups: dict[str, np.ndarray]) -> dict[str, int]:
    frame = trace_table.set_index("trace_bank_index")
    reps: dict[str, int] = {}
    for condition, indices in groups.items():
        if condition == "all_real_fem" or len(indices) == 0:
            continue
        sub = frame.loc[[idx for idx in indices if idx in frame.index]]
        if sub.empty:
            continue
        paths = pd.to_numeric(sub["rendered_path_length_arcmin"], errors="coerce")
        target = float(paths.median())
        reps[condition] = int((paths - target).abs().sort_values().index[0])
    return reps


def plot_figure(
    out_dir: Path,
    *,
    example_patch: np.ndarray,
    traces: np.ndarray,
    representative: dict[str, int],
    radial_summary: list[dict[str, Any]],
    metric_summary: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    ppd: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    radial = pd.DataFrame(radial_summary)
    metrics = pd.DataFrame(metric_summary)
    fit_mask_by_freq = radial.drop_duplicates("frequency_cpd").set_index("frequency_cpd")["in_fit_band"].astype(bool).to_dict()
    source_curve = radial[radial["condition"] == "source_image"].sort_values("frequency_cpd")
    source_fit_mask = source_curve["frequency_cpd"].map(fit_mask_by_freq).to_numpy(dtype=bool)
    source_norm = normalize_to_fit_band(source_curve["source_power_geomean"].to_numpy(dtype=np.float64), source_fit_mask)

    fig = plt.figure(figsize=(13.0, 6.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.38, 1.38], height_ratios=[1.0, 0.88], wspace=0.48, hspace=0.45)
    ax_patch = fig.add_subplot(gs[:, 0])
    ax_mod = fig.add_subplot(gs[0, 1])
    ax_transfer = fig.add_subplot(gs[0, 2])
    ax_slope = fig.add_subplot(gs[1, 1])
    ax_total = fig.add_subplot(gs[1, 2])

    patch = np.asarray(example_patch, dtype=np.float32)
    lo, hi = np.nanpercentile(patch, [1, 99])
    ax_patch.imshow(patch, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax_patch.set_title("SSI v3 image bank\nexample patch + traces", loc="left", fontsize=10, fontweight="bold")
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    center = np.array([patch.shape[1] / 2.0, patch.shape[0] / 2.0])
    visual_scale = 5.0
    for condition in ("short_drift", "mid_drift", "long_drift", "microsaccade"):
        idx = representative.get(condition)
        if idx is None:
            continue
        trace = np.asarray(traces[idx], dtype=np.float64)
        xy = np.column_stack([-trace[:, 0], trace[:, 1]]) * float(ppd) * visual_scale
        coords = center[None, :] + xy
        ax_patch.plot(coords[:, 0], coords[:, 1], color=COLORS[condition], linewidth=1.5, label=LABELS[condition])
        ax_patch.scatter(coords[0, 0], coords[0, 1], s=11, color=COLORS[condition], edgecolor="white", linewidth=0.4, zorder=5)
    ax_patch.text(
        0.02,
        0.02,
        f"trajectory overlay {visual_scale:g}x",
        transform=ax_patch.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
    )
    ax_patch.set_xlim(-0.5, patch.shape[1] - 0.5)
    ax_patch.set_ylim(patch.shape[0] - 0.5, -0.5)
    ax_patch.legend(loc="upper left", bbox_to_anchor=(0.0, -0.035), fontsize=6.8, frameon=False)

    ax_mod.plot(
        source_curve["frequency_cpd"],
        source_norm,
        color="#8a8a8a",
        linestyle="--",
        linewidth=1.6,
        label="source image",
    )
    for condition in ("short_drift", "mid_drift", "long_drift", "microsaccade", "all_real_fem"):
        sub = radial[radial["condition"] == condition].sort_values("frequency_cpd")
        if sub.empty:
            continue
        fit_mask = sub["frequency_cpd"].map(fit_mask_by_freq).to_numpy(dtype=bool)
        y = normalize_to_fit_band(sub["modulation_power_geomean"].to_numpy(dtype=np.float64), fit_mask)
        ax_mod.plot(
            sub["frequency_cpd"],
            y,
            color=COLORS[condition],
            linewidth=2.0 if condition == "all_real_fem" else 1.4,
            alpha=0.95 if condition == "all_real_fem" else 0.86,
            label=LABELS[condition],
        )
    ax_mod.axvspan(FIT_LOW_CPD, FIT_HIGH_CPD, color="#efefef", zorder=-10)
    ax_mod.set_xscale("log")
    ax_mod.set_yscale("log")
    ax_mod.set_title("Relative spectral shape", loc="left", fontsize=10, fontweight="bold")
    ax_mod.set_xlabel("spatial frequency (cpd)")
    ax_mod.set_ylabel("power, fit-band normalized")
    ax_mod.legend(fontsize=6.8, frameon=False)

    for condition in ("short_drift", "mid_drift", "long_drift", "microsaccade", "all_real_fem"):
        sub = radial[radial["condition"] == condition].sort_values("frequency_cpd")
        if sub.empty:
            continue
        y = sub["transfer_power_ratio_geomean"].to_numpy(dtype=np.float64)
        ax_transfer.plot(
            sub["frequency_cpd"],
            y,
            color=COLORS[condition],
            linewidth=2.0 if condition == "all_real_fem" else 1.4,
            alpha=0.95 if condition == "all_real_fem" else 0.86,
            label=LABELS[condition],
        )
    ax_transfer.axvspan(FIT_LOW_CPD, FIT_HIGH_CPD, color="#efefef", zorder=-10)
    ax_transfer.set_xscale("log")
    ax_transfer.set_yscale("log")
    ax_transfer.set_title("Motion transfer", loc="left", fontsize=10, fontweight="bold")
    ax_transfer.set_xlabel("spatial frequency (cpd)")
    ax_transfer.set_ylabel("modulation power / image power")

    plot_conditions = ["source_image", "short_drift", "mid_drift", "long_drift", "microsaccade", "all_real_fem"]
    labels = [DOT_LABELS[c] for c in plot_conditions]
    y_pos = np.arange(len(plot_conditions))
    slope_vals = []
    slope_errs = []
    colors = []
    for condition in plot_conditions:
        row = metrics[metrics["condition"] == condition].iloc[0]
        if condition == "source_image":
            slope_vals.append(float(row["source_power_slope_mean"]))
            slope_errs.append(float(row["source_power_slope_sem"]))
            colors.append("#8a8a8a")
        else:
            slope_vals.append(float(row["modulation_power_slope_mean"]))
            slope_errs.append(float(row["modulation_power_slope_sem"]))
            colors.append(COLORS[condition])
    ax_slope.axvline(0.0, color="#444444", linewidth=0.8)
    ax_slope.errorbar(slope_vals, y_pos, xerr=slope_errs, fmt="none", ecolor="#555555", elinewidth=0.8, capsize=2)
    ax_slope.scatter(slope_vals, y_pos, c=colors, s=34, zorder=3)
    ax_slope.set_yticks(y_pos, labels)
    ax_slope.tick_params(axis="y", labelsize=8)
    ax_slope.invert_yaxis()
    ax_slope.set_xlabel(f"log-log slope ({FIT_LOW_CPD:g}-{FIT_HIGH_CPD:g} cpd)")
    ax_slope.set_title("Flattening readout", loc="left", fontsize=10, fontweight="bold")

    total_rows = metrics[metrics["condition"].isin(plot_conditions[1:])].set_index("condition").loc[plot_conditions[1:]]
    total = total_rows["modulation_total_power_mean"].to_numpy(dtype=np.float64)
    total_sem = total_rows["modulation_total_power_sem"].to_numpy(dtype=np.float64)
    baseline = total_rows.loc["all_real_fem", "modulation_total_power_mean"]
    if math.isfinite(float(baseline)) and float(baseline) > 0:
        total = total / float(baseline)
        total_sem = total_sem / float(baseline)
    ax_total.axvline(1.0, color="#444444", linestyle="--", linewidth=0.8)
    ax_total.errorbar(total, np.arange(total.size), xerr=total_sem, fmt="none", ecolor="#555555", elinewidth=0.8, capsize=2)
    ax_total.scatter(total, np.arange(total.size), c=[COLORS[c] for c in plot_conditions[1:]], s=34, zorder=3)
    ax_total.set_yticks(np.arange(total.size), [DOT_LABELS[c] for c in plot_conditions[1:]])
    ax_total.tick_params(axis="y", labelsize=8)
    ax_total.invert_yaxis()
    ax_total.set_xlabel("fit-band modulation power / all-real")
    ax_total.set_title("Power scale", loc="left", fontsize=10, fontweight="bold")

    n_images = len(image_rows)
    fig.suptitle(
        f"Eye movements convert SSI v3 image power into retinal modulation power  (n images={n_images})",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_dir / "eye_movement_power_spectrum_shift.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "eye_movement_power_spectrum_shift.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "eye_movement_power_spectrum_shift.svg", bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, manifest: dict[str, Any], metric_summary: list[dict[str, Any]]) -> None:
    rows = {str(row["condition"]): row for row in metric_summary}
    lines = [
        "# Eye-Movement Power Spectrum Shift",
        "",
        "This diagnostic starts from the banked SSI figure-v3 BackImage inputs:",
        "",
        f"- image bank: `{Path(manifest['image_table']).relative_to(ROOT)}`",
        f"- trajectory bank: `{Path(manifest['trace_xy_npy']).relative_to(ROOT)}`",
        f"- trace metadata: `{Path(manifest['trace_table']).relative_to(ROOT)}`",
        "",
        "A pure translation preserves the static image spatial-power spectrum. The plotted shift is therefore the frame-to-frame retinal modulation spectrum induced by each eye-movement class.",
        "",
        f"Radial slopes are fit over `{FIT_LOW_CPD:g}-{FIT_HIGH_CPD:g} cpd`.",
        "",
        "## Summary",
        "",
    ]
    source = rows.get("source_image", {})
    if source:
        lines.append(f"- Source image power slope: `{float(source.get('source_power_slope_mean', float('nan'))):.3g}`.")
    for condition in ("short_drift", "mid_drift", "long_drift", "microsaccade", "all_real_fem"):
        row = rows.get(condition)
        if not row:
            continue
        lines.append(
            f"- `{LABELS[condition]}`: modulation slope `{float(row.get('modulation_power_slope_mean', float('nan'))):.3g}`, "
            f"transfer slope `{float(row.get('transfer_slope_mean', float('nan'))):.3g}`, "
            f"median path `{float(row.get('path_median_arcmin', float('nan'))):.3g}` arcmin, "
            f"selected traces `{int(row.get('n_selected_traces', 0))}`."
        )
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "For a patch Fourier coefficient `F(k)` and trace displacement `x(t)`, the motion-induced derivative power is estimated as `|F(k)|^2 mean_t |exp(-2 pi i k.x(t+1)) - exp(-2 pi i k.x(t))|^2 / dt^2`. A 2D Hann window is applied to each patch before the FFT to reduce finite-crop wraparound artifacts.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_table = pd.read_csv(args.image_table)
    trace_table = pd.read_csv(args.trace_table)
    traces = np.load(args.trace_xy, mmap_mode="r")
    trace_groups, trace_group_rows = select_trace_groups(
        trace_table,
        max_traces_per_condition=int(args.max_traces_per_condition),
        seed=int(args.seed),
    )
    source_powers, image_rows, example_patch = load_image_powers(
        image_table,
        max_images=int(args.max_images),
        patch_size_px=int(args.patch_size_px),
    )
    ppds = np.asarray([row["ppd"] for row in image_rows], dtype=np.float64)
    ppd = float(np.median(ppds))
    fx, fy, rr = frequency_grid(int(args.patch_size_px), ppd)
    edges = radial_edges(rr, int(args.n_radial_bins))

    kernels: dict[str, np.ndarray] = {}
    for condition, indices in trace_groups.items():
        print(f"[kernel] {condition}: {len(indices)} traces", flush=True)
        selected_traces = np.asarray(traces[indices], dtype=np.float32)
        kernels[condition] = motion_kernel2d(
            selected_traces,
            fx,
            fy,
            dt_s=float(args.dt_s),
            chunk_size=int(args.kernel_chunk_size),
        )

    _freq, radial_rows, metric_rows = image_level_rows(source_powers, kernels, rr, edges)
    radial_summary = aggregate_radial(radial_rows)
    metric_summary = aggregate_metrics(metric_rows, trace_group_rows)
    representative = representative_trace_indices(trace_table, trace_groups)

    write_csv(out_dir / "eye_movement_power_spectrum_shift_radial_by_image.csv", radial_rows)
    write_csv(out_dir / "eye_movement_power_spectrum_shift_radial_summary.csv", radial_summary)
    write_csv(out_dir / "eye_movement_power_spectrum_shift_metric_by_image.csv", metric_rows)
    write_csv(out_dir / "eye_movement_power_spectrum_shift_metric_summary.csv", metric_summary)
    write_csv(out_dir / "eye_movement_power_spectrum_shift_trace_groups.csv", trace_group_rows)
    write_csv(out_dir / "eye_movement_power_spectrum_shift_image_sample.csv", image_rows)

    manifest = {
        "analysis": "eye_movement_power_spectrum_shift",
        "image_table": Path(args.image_table),
        "trace_table": Path(args.trace_table),
        "trace_xy_npy": Path(args.trace_xy),
        "out_dir": out_dir,
        "n_images_available": int(image_table.shape[0]),
        "n_images_selected": int(len(image_rows)),
        "n_traces_available": int(traces.shape[0]),
        "trace_groups": trace_group_rows,
        "patch_size_px": int(args.patch_size_px),
        "ppd_median": ppd,
        "ppd_min": float(np.nanmin(ppds)),
        "ppd_max": float(np.nanmax(ppds)),
        "dt_s": float(args.dt_s),
        "fit_band_cpd": [FIT_LOW_CPD, FIT_HIGH_CPD],
        "radial_bins": int(args.n_radial_bins),
        "method": "Fourier-shift theorem on Hann-windowed image patches; reports retinal modulation power, not a change in static image Fourier magnitude.",
    }
    write_json(out_dir / "eye_movement_power_spectrum_shift_manifest.json", manifest)
    write_readme(out_dir, manifest, metric_summary)
    plot_figure(
        out_dir,
        example_patch=example_patch,
        traces=np.asarray(traces),
        representative=representative,
        radial_summary=radial_summary,
        metric_summary=metric_summary,
        image_rows=image_rows,
        ppd=ppd,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_IMAGE_TABLE)
    parser.add_argument("--trace-table", type=Path, default=DEFAULT_TRACE_TABLE)
    parser.add_argument("--trace-xy", type=Path, default=DEFAULT_TRACE_XY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-images", type=int, default=24, help="Evenly sample this many image-bank rows by contour coherence; 0 means all rows.")
    parser.add_argument("--max-traces-per-condition", type=int, default=192, help="Randomly sample this many trace-bank rows per movement class; 0 means all available.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--n-radial-bins", type=int, default=17)
    parser.add_argument("--kernel-chunk-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(f"Wrote eye-movement power-spectrum shift diagnostic to {manifest['out_dir']}")


if __name__ == "__main__":
    main()

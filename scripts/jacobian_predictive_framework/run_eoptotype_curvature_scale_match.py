#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from VisionCore.paths import STATS_DIR, VISIONCORE_ROOT

SCRIPTS_DIR = VISIONCORE_ROOT / "scripts"
TD_DIR = SCRIPTS_DIR / "temporal_decoding"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(TD_DIR) not in sys.path:
    sys.path.insert(0, str(TD_DIR))

from scripts.spatial_info import get_spatial_readout  # noqa: E402
from scripts.temporal_decoding.stimulus_hires import HiResERenderer, HiResRetina  # noqa: E402
from scripts.utils import get_model_and_dataset_configs  # noqa: E402


RATES_DIR = VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "rates"
EYE_TRACES_PATH = VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz"
PKL_PATH = VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl"
DEFAULT_OUTPUT_DIR = STATS_DIR / "fem_curvature_scale_match"
DEFAULT_LOGMARS = (-0.35, -0.20, 0.00, 0.20, 0.40, 0.60)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_RADII_PX = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
DEFAULT_THRESHOLDS = (0.25, 0.5, 0.75)
DEFAULT_PPD = 37.50476617
DEFAULT_N_LAGS = 32
EPS = 1e-12


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(float(part)) for part in value.split(",") if part.strip())


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _stimulus_size_arcmin(logmar: float) -> float:
    return float(5.0 * (10.0 ** float(logmar)))


def _gap_size_arcmin(logmar: float) -> float:
    return float(10.0 ** float(logmar))


def _px_to_deg(value_px: float, pixels_per_degree: float) -> float:
    return float(value_px) / float(pixels_per_degree)


def _px_to_arcmin(value_px: float, pixels_per_degree: float) -> float:
    return 60.0 * _px_to_deg(value_px, pixels_per_degree)


def _embed_time_lags_pure(movie: torch.Tensor, n_lags: int) -> torch.Tensor:
    total_frames = int(movie.shape[0])
    out_frames = total_frames - n_lags + 1
    if out_frames < 1:
        raise ValueError(f"Need T >= n_lags, got T={total_frames}, n_lags={n_lags}")
    lagged = []
    for lag in range(n_lags):
        lagged.append(movie[n_lags - 1 - lag : total_frames - lag])
    return torch.stack(lagged, dim=1).unsqueeze(1)


def _rates_from_stim(model, readout, stim: torch.Tensor) -> torch.Tensor:
    feats = model.model.core_forward(stim, None)
    readout_map = readout(feats[:, :, -1])
    rates_map = model.model.activation(readout_map)
    return rates_map.mean(dim=(-2, -1))


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["traces"].astype(np.float32), data["durations"].astype(np.int32)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _nanmedian(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanquantile(arr, q))


def _compute_trace_metrics(trace_px: np.ndarray) -> dict[str, float]:
    if trace_px.shape[0] == 0:
        return {
            "fem_rms_px": float("nan"),
            "fem_median_radius_px": float("nan"),
            "fem_p75_radius_px": float("nan"),
            "fem_p90_radius_px": float("nan"),
            "fem_step_rms_px": float("nan"),
        }
    centered = trace_px - np.nanmean(trace_px, axis=0, keepdims=True)
    if centered.shape[0] >= 2:
        cov = np.cov(centered.T)
        fem_rms_px = float(np.sqrt(max(np.trace(cov), 0.0)))
    else:
        fem_rms_px = 0.0
    radii = np.linalg.norm(centered, axis=1)
    if centered.shape[0] >= 2:
        steps = np.diff(trace_px, axis=0)
        fem_step_rms_px = float(np.sqrt(np.mean(np.sum(steps * steps, axis=1))))
    else:
        fem_step_rms_px = 0.0
    return {
        "fem_rms_px": fem_rms_px,
        "fem_median_radius_px": float(np.nanmedian(radii)),
        "fem_p75_radius_px": float(np.nanquantile(radii, 0.75)),
        "fem_p90_radius_px": float(np.nanquantile(radii, 0.90)),
        "fem_step_rms_px": fem_step_rms_px,
    }


def _interpolate_threshold(
    xs: np.ndarray,
    ys: np.ndarray,
    tau: float,
) -> tuple[float, str, str]:
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return float("nan"), "insufficient_bins", "none"
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if ys[0] > tau:
        return float(xs[0]), "below_resolution", "upper_bound_first_bin"
    equal_idx = np.where(np.isclose(ys, tau))[0]
    if equal_idx.size:
        idx = int(equal_idx[0])
        return float(xs[idx]), "crossed", "exact_bin"
    for idx in range(xs.size - 1):
        y0 = float(ys[idx])
        y1 = float(ys[idx + 1])
        x0 = float(xs[idx])
        x1 = float(xs[idx + 1])
        if y0 < tau <= y1:
            if math.isclose(y1, y0):
                return float(x1), "crossed", "flat_segment"
            alpha = (tau - y0) / (y1 - y0)
            return float(x0 + alpha * (x1 - x0)), "crossed", "linear"
    return float("nan"), "not_crossed", "none"


def _estimate_crossing(
    stimulus_sizes_arcmin: list[float],
    delta_star_arcmin: list[float],
    fem_rms_arcmin: float,
) -> tuple[float, float, str]:
    xs = np.asarray(stimulus_sizes_arcmin, dtype=np.float64)
    ys = np.asarray(delta_star_arcmin, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(fem_rms_arcmin)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size < 2:
        return float("nan"), float("nan"), "insufficient_points"
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    diff = ys - float(fem_rms_arcmin)
    zero_idx = np.where(np.isclose(diff, 0.0))[0]
    if zero_idx.size:
        size_arcmin = float(xs[int(zero_idx[0])])
        return size_arcmin, float(np.log10(size_arcmin / 5.0)), "exact"
    for idx in range(xs.size - 1):
        d0 = float(diff[idx])
        d1 = float(diff[idx + 1])
        if d0 == 0.0:
            size_arcmin = float(xs[idx])
            return size_arcmin, float(np.log10(size_arcmin / 5.0)), "exact"
        if d0 * d1 < 0.0:
            x0 = float(xs[idx])
            x1 = float(xs[idx + 1])
            alpha = -d0 / (d1 - d0)
            size_arcmin = float(x0 + alpha * (x1 - x0))
            return size_arcmin, float(np.log10(size_arcmin / 5.0)), "linear"
    return float("nan"), float("nan"), "not_crossed"


def _describe_support(support_fraction: float) -> str:
    if not np.isfinite(support_fraction):
        return "invalid"
    return "ok" if support_fraction >= 0.95 else "edge_support"


class CurvatureScaleMatchRunner:
    def __init__(
        self,
        device: str,
        pixels_per_degree: float,
        n_lags: int,
        jacobian_step_px: float,
        model_batch_size: int,
        load_model: bool = True,
    ):
        self.device = torch.device(device)
        self.pixels_per_degree = float(pixels_per_degree)
        self.n_lags = int(n_lags)
        self.jacobian_step_px = float(jacobian_step_px)
        self.model_batch_size = int(model_batch_size)

        self.model = None
        self.readout = None
        if load_model:
            model, _dataset_configs = get_model_and_dataset_configs()
            with PKL_PATH.open("rb") as handle:
                outputs = dill.load(handle)
            self.model = model.to(self.device)
            self.model.model.eval()
            self.readout = get_spatial_readout(self.model, outputs).to(self.device)
            self.readout.eval()
        self.renderer = HiResERenderer(device=device).to(self.device)
        self.renderer.eval()
        self.retina = HiResRetina().to(self.device)
        self.retina.eval()

    def _position_key(self, position_px: np.ndarray) -> tuple[float, float]:
        pos = np.asarray(position_px, dtype=np.float64)
        return (round(float(pos[0]), 6), round(float(pos[1]), 6))

    def _make_static_trace_deg(self, position_px: np.ndarray) -> torch.Tensor:
        pos_deg = np.asarray(position_px, dtype=np.float32) / np.float32(self.pixels_per_degree)
        trace = np.repeat(pos_deg[None, :], self.n_lags, axis=0)
        return torch.tensor(trace, dtype=torch.float32, device=self.device)

    def _sample_support_only(
        self,
        world_support: torch.Tensor,
        positions_px: list[np.ndarray],
        center_support_sum: float,
    ) -> dict[tuple[float, float], float]:
        out: dict[tuple[float, float], float] = {}
        with torch.no_grad():
            for position_px in positions_px:
                key = self._position_key(position_px)
                trace = self._make_static_trace_deg(position_px)
                movie = self.retina(world_support, trace)[0, 0]
                support_sum = float(movie[-1].sum().item())
                out[key] = support_sum / max(center_support_sum, EPS)
        return out

    def evaluate_condition(
        self,
        logmar: float,
        orientation: int,
        positions_px: list[np.ndarray],
    ) -> tuple[dict[tuple[float, float], np.ndarray], dict[tuple[float, float], float]]:
        if self.model is None or self.readout is None:
            raise RuntimeError("Model/readout not loaded; evaluate_condition requires load_model=True")
        with torch.no_grad():
            world_img = self.renderer(float(orientation), float(logmar)).to(self.device)
            world_gray = 127.0 * (1.0 - world_img)
            world_support = world_img
            center_movie = self.retina(world_support, self._make_static_trace_deg(np.zeros(2, dtype=np.float32)))[0, 0]
            center_support_sum = float(center_movie[-1].sum().item())

        unique_positions: list[np.ndarray] = []
        seen: set[tuple[float, float]] = set()
        for position_px in positions_px:
            key = self._position_key(position_px)
            if key in seen:
                continue
            seen.add(key)
            unique_positions.append(np.asarray(position_px, dtype=np.float64))

        responses: dict[tuple[float, float], np.ndarray] = {}
        support_fractions = self._sample_support_only(world_support, unique_positions, center_support_sum)

        for start in range(0, len(unique_positions), self.model_batch_size):
            batch_positions = unique_positions[start : start + self.model_batch_size]
            stims = []
            keys = []
            with torch.no_grad():
                for position_px in batch_positions:
                    trace = self._make_static_trace_deg(position_px)
                    movie = self.retina(world_gray, trace)[0, 0] / 127.0
                    stims.append(_embed_time_lags_pure(movie, self.n_lags))
                    keys.append(self._position_key(position_px))
                stim_batch = torch.cat(stims, dim=0)
                rates = _rates_from_stim(self.model, self.readout, stim_batch).cpu().numpy()
            for idx, key in enumerate(keys):
                responses[key] = rates[idx].astype(np.float64)
        return responses, support_fractions

    def finite_difference_jacobian(
        self,
        responses: dict[tuple[float, float], np.ndarray],
        anchor_px: np.ndarray,
    ) -> np.ndarray:
        anchor = np.asarray(anchor_px, dtype=np.float64)
        step = float(self.jacobian_step_px)
        pos_x = self._position_key(anchor + np.array([step, 0.0], dtype=np.float64))
        neg_x = self._position_key(anchor + np.array([-step, 0.0], dtype=np.float64))
        pos_y = self._position_key(anchor + np.array([0.0, step], dtype=np.float64))
        neg_y = self._position_key(anchor + np.array([0.0, -step], dtype=np.float64))
        jac_x = (responses[pos_x] - responses[neg_x]) / (2.0 * step)
        jac_y = (responses[pos_y] - responses[neg_y]) / (2.0 * step)
        return np.stack([jac_x, jac_y], axis=1)


def _build_radial_pairs(radii_px: tuple[float, ...], n_directions: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for radius_px in radii_px:
        for direction_idx in range(int(n_directions)):
            theta = 2.0 * math.pi * float(direction_idx) / float(n_directions)
            rows.append(
                {
                    "radius_px": float(radius_px),
                    "direction_idx": int(direction_idx),
                    "theta_rad": float(theta),
                    "dx_px": float(radius_px * math.cos(theta)),
                    "dy_px": float(radius_px * math.sin(theta)),
                }
            )
    return rows


def _pair_key(row: dict) -> tuple:
    return (
        row["dataset"],
        row["session"],
        row["stimulus_family"],
        row["logmar"],
        row["orientation"],
        row["jacobian_mode"],
        row["displacement_px"],
    )


def _bin_rows_from_pairs(pair_rows: list[dict], pixels_per_degree: float) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in pair_rows:
        if row["valid"]:
            grouped[_pair_key(row)].append(row)

    condition_groups: dict[tuple, list[tuple[float, list[dict]]]] = defaultdict(list)
    for key, rows in grouped.items():
        condition_key = key[:-1]
        condition_groups[condition_key].append((float(key[-1]), rows))

    out_rows: list[dict] = []
    for condition_key, displacement_groups in condition_groups.items():
        displacement_groups.sort(key=lambda item: item[0])
        centers = [item[0] for item in displacement_groups]
        lows: list[float] = []
        highs: list[float] = []
        for idx, center in enumerate(centers):
            prev_center = centers[idx - 1] if idx > 0 else None
            next_center = centers[idx + 1] if idx < len(centers) - 1 else None
            low = 0.0 if prev_center is None else 0.5 * (prev_center + center)
            high = center if next_center is None else 0.5 * (center + next_center)
            lows.append(float(low))
            highs.append(float(high))
        for idx, ((center, rows), low, high) in enumerate(zip(displacement_groups, lows, highs, strict=False)):
            err_values = [float(row["err_norm"]) for row in rows]
            frac_values = [float(row["frac_residual_energy"]) for row in rows]
            pred_values = [float(row["predicted_fraction"]) for row in rows]
            cosine_values = [float(row["cosine_true_pred"]) for row in rows]
            dataset, session, stimulus_family, logmar, orientation, jacobian_mode = condition_key
            out_rows.append(
                {
                    "dataset": dataset,
                    "session": session,
                    "stimulus_family": stimulus_family,
                    "logmar": logmar,
                    "orientation": orientation,
                    "image_id": "",
                    "window_key": "",
                    "jacobian_mode": jacobian_mode,
                    "bin_index": idx,
                    "bin_low_px": float(low),
                    "bin_high_px": float(high),
                    "bin_center_px": float(center),
                    "bin_center_deg": _px_to_deg(center, pixels_per_degree),
                    "bin_center_arcmin": _px_to_arcmin(center, pixels_per_degree),
                    "n_pairs": len(rows),
                    "median_err_norm": float(np.nanmedian(err_values)),
                    "mean_err_norm": float(np.nanmean(err_values)),
                    "median_frac_residual_energy": float(np.nanmedian(frac_values)),
                    "median_predicted_fraction": float(np.nanmedian(pred_values)),
                    "median_cosine_true_pred": float(np.nanmedian(cosine_values)),
                }
            )
    return out_rows


def _summaries_from_bins(
    bin_rows: list[dict],
    fem_summary: dict[str, float],
    pixels_per_degree: float,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in bin_rows:
        groups[(row["dataset"], row["session"], row["stimulus_family"], row["logmar"], row["orientation"], row["jacobian_mode"])].append(row)

    summary_rows: list[dict] = []
    for key, rows in groups.items():
        rows.sort(key=lambda item: float(item["bin_center_px"]))
        centers_px = np.asarray([row["bin_center_px"] for row in rows], dtype=np.float64)
        errs = np.asarray([row["median_err_norm"] for row in rows], dtype=np.float64)
        delta_values: dict[str, float] = {}
        status_values: dict[str, str] = {}
        method_values: dict[str, str] = {}
        for tau in DEFAULT_THRESHOLDS:
            delta_px, status, method = _interpolate_threshold(centers_px, errs, tau)
            key_name = f"{int(round(tau * 100)):03d}"
            delta_values[key_name] = delta_px
            status_values[key_name] = status
            method_values[key_name] = method

        dataset, session, stimulus_family, logmar, orientation, jacobian_mode = key
        fem_rms_px = float(fem_summary["fem_rms_px"])
        scale_ratio_050 = fem_rms_px / delta_values["050"] if np.isfinite(delta_values["050"]) and delta_values["050"] > 0 else float("nan")
        scale_ratio_025 = fem_rms_px / delta_values["025"] if np.isfinite(delta_values["025"]) and delta_values["025"] > 0 else float("nan")
        scale_ratio_075 = fem_rms_px / delta_values["075"] if np.isfinite(delta_values["075"]) and delta_values["075"] > 0 else float("nan")
        summary_rows.append(
            {
                "dataset": dataset,
                "session": session,
                "stimulus_family": stimulus_family,
                "logmar": logmar,
                "orientation": orientation,
                "image_id": "",
                "window_key": "",
                "jacobian_mode": jacobian_mode,
                "pixels_per_degree": float(pixels_per_degree),
                "stimulus_size_arcmin": _stimulus_size_arcmin(float(logmar)),
                "gap_size_arcmin": _gap_size_arcmin(float(logmar)),
                "n_pairs": int(sum(int(row["n_pairs"]) for row in rows)),
                "n_bins": int(len(rows)),
                "delta_star_025_px": delta_values["025"],
                "delta_star_050_px": delta_values["050"],
                "delta_star_075_px": delta_values["075"],
                "delta_star_025_interp_method": method_values["025"],
                "delta_star_050_interp_method": method_values["050"],
                "delta_star_075_interp_method": method_values["075"],
                "delta_star_025_arcmin": _px_to_arcmin(delta_values["025"], pixels_per_degree) if np.isfinite(delta_values["025"]) else float("nan"),
                "delta_star_050_arcmin": _px_to_arcmin(delta_values["050"], pixels_per_degree) if np.isfinite(delta_values["050"]) else float("nan"),
                "delta_star_075_arcmin": _px_to_arcmin(delta_values["075"], pixels_per_degree) if np.isfinite(delta_values["075"]) else float("nan"),
                "delta_star_025_status": status_values["025"],
                "delta_star_050_status": status_values["050"],
                "delta_star_075_status": status_values["075"],
                "crossing_logmar_est": float("nan"),
                "crossing_stimulus_size_arcmin_est": float("nan"),
                "crossing_status": "not_computed",
                "fem_rms_px": fem_rms_px,
                "fem_rms_arcmin": _px_to_arcmin(fem_summary["fem_rms_px"], pixels_per_degree),
                "fem_median_radius_px": float(fem_summary["fem_median_radius_px"]),
                "fem_median_radius_arcmin": _px_to_arcmin(fem_summary["fem_median_radius_px"], pixels_per_degree),
                "fem_p75_radius_px": float(fem_summary["fem_p75_radius_px"]),
                "fem_p75_radius_arcmin": _px_to_arcmin(fem_summary["fem_p75_radius_px"], pixels_per_degree),
                "fem_p90_radius_px": float(fem_summary["fem_p90_radius_px"]),
                "fem_p90_radius_arcmin": _px_to_arcmin(fem_summary["fem_p90_radius_px"], pixels_per_degree),
                "fem_step_rms_px": float(fem_summary["fem_step_rms_px"]),
                "fem_step_rms_arcmin": _px_to_arcmin(fem_summary["fem_step_rms_px"], pixels_per_degree),
                "scale_ratio_050": scale_ratio_050,
                "scale_ratio_025": scale_ratio_025,
                "scale_ratio_075": scale_ratio_075,
                "within_factor2_050": bool(np.isfinite(scale_ratio_050) and 0.5 <= scale_ratio_050 <= 2.0),
                "within_factor3_050": bool(np.isfinite(scale_ratio_050) and (1.0 / 3.0) <= scale_ratio_050 <= 3.0),
            }
        )
    return summary_rows


def _annotate_crossings(summary_rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary_rows:
        groups[(row["orientation"], row["jacobian_mode"])].append(row)
    for key, rows in groups.items():
        stimulus_sizes = [float(row["stimulus_size_arcmin"]) for row in rows]
        delta_star = [float(row["delta_star_050_arcmin"]) for row in rows]
        fem_rms = float(rows[0]["fem_rms_arcmin"])
        crossing_size, crossing_logmar, crossing_status = _estimate_crossing(stimulus_sizes, delta_star, fem_rms)
        for row in rows:
            row["crossing_stimulus_size_arcmin_est"] = crossing_size
            row["crossing_logmar_est"] = crossing_logmar
            row["crossing_status"] = crossing_status


def _plot_error_curves(
    output_path: Path,
    bin_rows: list[dict],
    fem_summary: dict[str, float],
) -> None:
    primary_rows = [row for row in bin_rows if row["jacobian_mode"] == "center_jacobian"]
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in primary_rows:
        grouped[float(row["logmar"])].append(row)
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for logmar, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: float(item["bin_center_arcmin"]))
        x = [float(row["bin_center_arcmin"]) for row in rows]
        y = [float(row["median_err_norm"]) for row in rows]
        ax.plot(x, y, marker="o", label=f"LogMAR {logmar:.2f}")
    fem_rms = float(fem_summary["fem_rms_arcmin"])
    fem_p25 = float(fem_summary["fem_rms_p25_arcmin"])
    fem_p75 = float(fem_summary["fem_rms_p75_arcmin"])
    if np.isfinite(fem_p25) and np.isfinite(fem_p75):
        ax.axvspan(fem_p25, fem_p75, color="0.85", alpha=0.75)
    if np.isfinite(fem_rms):
        ax.axvline(fem_rms, color="0.25", linestyle="--", linewidth=1.2)
    ax.axhline(0.5, color="tab:red", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Displacement (arcmin)")
    ax.set_ylabel("Median normalized error")
    ax.set_title("Jacobian error versus displacement")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scale_vs_stimulus(
    output_path: Path,
    summary_rows: list[dict],
    fem_summary: dict[str, float],
) -> None:
    primary_rows = [row for row in summary_rows if row["jacobian_mode"] == "center_jacobian"]
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in primary_rows:
        if np.isfinite(row["delta_star_050_arcmin"]):
            grouped[float(row["stimulus_size_arcmin"])] .append(float(row["delta_star_050_arcmin"]))
    if not grouped:
        return
    xs = sorted(grouped)
    ys = [float(np.nanmedian(grouped[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, ys, marker="o", color="tab:blue", label=r"$\delta_\star$ (tau=0.5)")
    fem_rms = float(fem_summary["fem_rms_arcmin"])
    fem_p25 = float(fem_summary["fem_rms_p25_arcmin"])
    fem_p75 = float(fem_summary["fem_rms_p75_arcmin"])
    if np.isfinite(fem_p25) and np.isfinite(fem_p75):
        ax.axhspan(fem_p25, fem_p75, color="0.88", alpha=0.8, label="FEM RMS IQR")
    if np.isfinite(fem_rms):
        ax.axhline(fem_rms, color="tab:orange", linestyle="--", linewidth=1.2, label="Median FEM RMS")
    crossing_size, _, crossing_status = _estimate_crossing(xs, ys, fem_rms)
    if np.isfinite(crossing_size):
        crossing_y = float(np.interp(crossing_size, xs, ys))
        ax.scatter([crossing_size], [crossing_y], color="black", s=25, zorder=3)
        ax.annotate("crossing", (crossing_size, crossing_y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Stimulus size (arcmin)")
    ax.set_ylabel("Scale (arcmin)")
    title = "Curvature-onset scale and FEM scale"
    if crossing_status == "not_crossed":
        title += " (no crossing in range)"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_delta_vs_fem(output_path: Path, summary_rows: list[dict]) -> None:
    primary_rows = [row for row in summary_rows if row["jacobian_mode"] == "center_jacobian" and np.isfinite(row["delta_star_050_arcmin"]) and np.isfinite(row["fem_rms_arcmin"])]
    if not primary_rows:
        return
    xs = np.asarray([float(row["delta_star_050_arcmin"]) for row in primary_rows], dtype=np.float64)
    ys = np.asarray([float(row["fem_rms_arcmin"]) for row in primary_rows], dtype=np.float64)
    lim = float(np.nanmax(np.concatenate([xs, ys])) * 1.1)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(xs, ys, color="tab:blue", alpha=0.85)
    ax.plot([0.0, lim], [0.0, lim], color="0.3", linestyle="--", linewidth=1.0)
    ax.plot([0.0, lim], [0.0, 2.0 * lim], color="0.7", linestyle=":", linewidth=1.0)
    ax.plot([0.0, lim], [0.0, 0.5 * lim], color="0.7", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r"$\delta_\star$ (arcmin)")
    ax.set_ylabel("FEM RMS (arcmin)")
    ax.set_title("Per-condition scale match")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scale_ratio(output_path: Path, summary_rows: list[dict]) -> None:
    primary_rows = [row for row in summary_rows if row["jacobian_mode"] == "center_jacobian" and np.isfinite(row["scale_ratio_050"])]
    if not primary_rows:
        return
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in primary_rows:
        grouped[float(row["logmar"])].append(float(row["scale_ratio_050"]))
    xs = sorted(grouped)
    ys = [float(np.nanmedian(grouped[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(xs, ys, marker="o", color="tab:green")
    ax.axhspan(0.5, 2.0, color="0.9", alpha=0.8)
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.2)
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("FEM RMS / delta_star")
    ax.set_title("Scale ratio by LogMAR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_overview(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axvspan(0.0, 0.32, color="#d9edf7")
    ax.axvspan(0.32, 0.68, color="#dff0d8")
    ax.axvspan(0.68, 1.0, color="#f2dede")
    ax.text(0.16, 0.55, "Too small\nweak trajectory", ha="center", va="center", fontsize=11)
    ax.text(0.50, 0.55, "Finite-local regime\norganized by local geometry", ha="center", va="center", fontsize=11)
    ax.text(0.84, 0.55, "Too large\ncurved regime", ha="center", va="center", fontsize=11)
    ax.annotate("FEM scale", xy=(0.52, 0.2), xytext=(0.52, 0.05), arrowprops={"arrowstyle": "-|>", "linewidth": 1.2}, ha="center")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    summary_rows: list[dict],
    fem_summary: dict[str, float],
    config: dict,
    feasibility_rows: list[dict],
) -> None:
    def _rows_for_mode(mode: str) -> list[dict]:
        return [row for row in summary_rows if row["jacobian_mode"] == mode]

    def _median_metric(rows: list[dict], key: str) -> float:
        return _nanmedian([float(row[key]) for row in rows])

    def _status_fraction(rows: list[dict], status_key: str, status_value: str) -> float:
        if not rows:
            return float("nan")
        matches = [1.0 if row[status_key] == status_value else 0.0 for row in rows]
        return float(np.mean(matches))

    center_rows = _rows_for_mode("center_jacobian")
    midpoint_rows = _rows_for_mode("midpoint_jacobian")

    primary_rows = center_rows
    med_delta = _median_metric(primary_rows, "delta_star_050_arcmin")
    med_ratio = _median_metric(primary_rows, "scale_ratio_050")
    within2 = _nanmedian([1.0 if row["within_factor2_050"] else 0.0 for row in primary_rows])
    within3 = _nanmedian([1.0 if row["within_factor3_050"] else 0.0 for row in primary_rows])
    crossing_rows = [row for row in primary_rows if np.isfinite(row["crossing_stimulus_size_arcmin_est"])]
    crossing_size = _nanmedian([float(row["crossing_stimulus_size_arcmin_est"]) for row in crossing_rows])
    crossing_logmar = _nanmedian([float(row["crossing_logmar_est"]) for row in crossing_rows])
    center_below_resolution = _status_fraction(center_rows, "delta_star_050_status", "below_resolution")
    midpoint_below_resolution = _status_fraction(midpoint_rows, "delta_star_050_status", "below_resolution")
    center_tau025 = _median_metric(center_rows, "delta_star_025_arcmin")
    center_tau050 = _median_metric(center_rows, "delta_star_050_arcmin")
    center_tau075 = _median_metric(center_rows, "delta_star_075_arcmin")
    midpoint_tau025 = _median_metric(midpoint_rows, "delta_star_025_arcmin")
    midpoint_tau050 = _median_metric(midpoint_rows, "delta_star_050_arcmin")
    midpoint_tau075 = _median_metric(midpoint_rows, "delta_star_075_arcmin")
    midpoint_crossing_rows = [row for row in midpoint_rows if np.isfinite(row["crossing_stimulus_size_arcmin_est"])]
    midpoint_crossing_size = _nanmedian([float(row["crossing_stimulus_size_arcmin_est"]) for row in midpoint_crossing_rows])
    midpoint_crossing_logmar = _nanmedian([float(row["crossing_logmar_est"]) for row in midpoint_crossing_rows])
    finest_logmar = min(float(x) for x in config["logmars"])
    max_radius = max(float(x) for x in config["radii_px"])
    feasibility_target_rows = [
        row
        for row in feasibility_rows
        if math.isclose(float(row["logmar"]), finest_logmar, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(float(row["radius_px"]), max_radius, rel_tol=0.0, abs_tol=1e-9)
    ]
    feasible_fraction = _nanmedian([float(row["support_fraction"]) for row in feasibility_target_rows])

    lines = [
        "# Curvature-onset versus FEM-scale analysis",
        "",
        "## Scope",
        f"- Dataset/stimulus family: E-optotype",
        f"- Model checkpoint: {config['model_source']}",
        f"- LogMARs/images: {', '.join(f'{x:.2f}' for x in config['logmars'])}",
        f"- Translation radii: {', '.join(f'{x:g}' for x in config['radii_px'])} px",
        f"- Jacobian modes: {', '.join(config['jacobian_modes'])}",
        f"- FEM trace source: {config['eye_traces_path']}",
        f"- Eye convention: HiResRetina static-position sampling in model pixel units",
        "",
        "## Primary result",
        f"- Crossing LogMAR: {crossing_logmar:.4f}" if np.isfinite(crossing_logmar) else "- Crossing LogMAR: not crossed in tested range",
        f"- Crossing stimulus size: {crossing_size:.4f} arcmin" if np.isfinite(crossing_size) else "- Crossing stimulus size: not crossed in tested range",
        "- Crossing aggregation: median across per-orientation center_jacobian crossing estimates; per-orientation values are in curvature_scale_match_summary.csv",
        f"- Median delta_star_050: {med_delta:.4f} arcmin" if np.isfinite(med_delta) else "- Median delta_star_050: NaN",
        f"- Median FEM RMS: {float(fem_summary['fem_rms_arcmin']):.4f} arcmin",
        f"- Median scale_ratio_050: {med_ratio:.4f}" if np.isfinite(med_ratio) else "- Median scale_ratio_050: NaN",
        f"- Fraction within factor 2: {within2:.4f}" if np.isfinite(within2) else "- Fraction within factor 2: NaN",
        f"- Fraction within factor 3: {within3:.4f}" if np.isfinite(within3) else "- Fraction within factor 3: NaN",
        "",
        "## Center Vs Midpoint",
        f"- Center crossing LogMAR: {crossing_logmar:.4f}" if np.isfinite(crossing_logmar) else "- Center crossing LogMAR: not crossed in tested range",
        f"- Midpoint crossing LogMAR: {midpoint_crossing_logmar:.4f}" if np.isfinite(midpoint_crossing_logmar) else "- Midpoint crossing LogMAR: not crossed in tested range",
        f"- Center delta_star arcmin at tau 0.25/0.50/0.75: {center_tau025:.4f}, {center_tau050:.4f}, {center_tau075:.4f}" if np.isfinite(center_tau050) else "- Center delta_star arcmin at tau 0.25/0.50/0.75: NaN",
        f"- Midpoint delta_star arcmin at tau 0.25/0.50/0.75: {midpoint_tau025:.4f}, {midpoint_tau050:.4f}, {midpoint_tau075:.4f}" if np.isfinite(midpoint_tau050) else "- Midpoint delta_star arcmin at tau 0.25/0.50/0.75: NaN",
        f"- Center below-resolution fraction at tau 0.50: {center_below_resolution:.4f}" if np.isfinite(center_below_resolution) else "- Center below-resolution fraction at tau 0.50: NaN",
        f"- Midpoint below-resolution fraction at tau 0.50: {midpoint_below_resolution:.4f}" if np.isfinite(midpoint_below_resolution) else "- Midpoint below-resolution fraction at tau 0.50: NaN",
        "",
        "## Stimulus-scale dependence",
        f"- Coarse stimulus delta_star: {float(np.nanmax([row['delta_star_050_arcmin'] for row in primary_rows])):.4f} arcmin" if primary_rows else "- Coarse stimulus delta_star: NaN",
        f"- Fine stimulus delta_star: {float(np.nanmin([row['delta_star_050_arcmin'] for row in primary_rows])):.4f} arcmin" if primary_rows else "- Fine stimulus delta_star: NaN",
        f"- Does the delta_star curve cross the FEM RMS band? {'yes' if np.isfinite(crossing_size) else 'no'}",
        "- Does delta_star shift with stimulus scale? inspect delta_star_and_fem_vs_stimulus_size.png",
        "",
        "## Interpretation",
        "- Strong / partial / no support: inspect crossing status and factor-of-two summaries together",
        "- Does real FEM scale lie near the finite-local transition regime? see crossing estimate and per-condition scatter",
        f"- Are results limited by edge artifacts or sampling range? median feasibility support fraction at finest LogMAR {finest_logmar:.2f}, max radius {max_radius:g} px = {feasible_fraction:.4f}" if np.isfinite(feasible_fraction) else "- Are results limited by edge artifacts or sampling range? feasibility check unavailable or not run at finest/max-radius target",
        "",
        "## Caveats",
        "- Model-internal analysis: yes",
        "- Directionality of tuning not proven: yes",
        "- Biological anchor: trace-position RMS is primary; step RMS is diagnostic only",
        "- Any excluded conditions: invalid edge-support pairs are marked in curvature_by_pair.csv",
        "- A below-resolution delta_star is an upper bound at the smallest tested displacement, not a resolved crossing estimate.",
        "",
        "## Notes",
        "- fem_rms uses the radial RMS of the eye-position cloud, aggregated as the median over traces.",
        "- delta_star is interpolated between displacement bins; it is not snapped to the next tested radius.",
        "- The stimulus-size figure uses the median delta_star curve across orientations, while the summary CSV preserves orientation-specific crossings for spread checks.",
        "- Center and midpoint Jacobians are reported side by side because they answer different biological questions: single-chart extent versus local smoothness along the path.",
    ]
    if np.isfinite(center_below_resolution) and center_below_resolution > 0.5:
        lines.extend(
            [
                "",
                "## Warning",
                f"- Center-jacobian delta_star_050 is below resolution in {center_below_resolution:.4f} of conditions, so many center values are upper bounds rather than resolved crossings.",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curvature-onset versus FEM-scale analysis for cached E-optotype infrastructure.")
    parser.add_argument("--logmars", default=",".join(f"{x:.2f}" for x in DEFAULT_LOGMARS))
    parser.add_argument("--orientations", default=",".join(str(x) for x in DEFAULT_ORIENTATIONS))
    parser.add_argument("--radii-px", default=",".join(f"{x:g}" for x in DEFAULT_RADII_PX))
    parser.add_argument("--n-directions", type=int, default=8)
    parser.add_argument("--thresholds", default=",".join(f"{x:.2f}" for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--pixels-per-degree", type=float, default=DEFAULT_PPD)
    parser.add_argument("--jacobian-step-px", type=float, default=0.125)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--device", default=_pick_device())
    parser.add_argument("--eye-traces-path", type=Path, default=EYE_TRACES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--edge-valid-threshold", type=float, default=0.95)
    parser.add_argument("--feasibility-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--max-traces", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    radii_px = _parse_csv_floats(args.radii_px)
    thresholds = _parse_csv_floats(args.thresholds)
    if tuple(thresholds) != DEFAULT_THRESHOLDS:
        raise ValueError("This implementation currently expects thresholds 0.25,0.50,0.75 to match output schema.")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    traces_deg, durations = _load_eye_traces(args.eye_traces_path)
    if args.max_traces is not None:
        traces_deg = traces_deg[: args.max_traces]
        durations = durations[: args.max_traces]

    pair_specs = _build_radial_pairs(radii_px, args.n_directions)
    runner = CurvatureScaleMatchRunner(
        device=args.device,
        pixels_per_degree=args.pixels_per_degree,
        n_lags=args.n_lags,
        jacobian_step_px=args.jacobian_step_px,
        model_batch_size=args.model_batch_size,
        load_model=not args.feasibility_only,
    )

    pair_rows: list[dict] = []
    feasibility_rows: list[dict] = []
    fem_trace_rows: list[dict] = []

    trace_support_reference_logmar = min(logmars)
    with torch.no_grad():
        support_world = runner.renderer(0.0, float(trace_support_reference_logmar)).to(runner.device)
        center_support_movie = runner.retina(support_world, runner._make_static_trace_deg(np.zeros(2, dtype=np.float32)))[0, 0]
        center_support_sum = float(center_support_movie[-1].sum().item())

    trace_position_support_cache: dict[tuple[float, float], float] = {}
    trace_positions_unique: list[np.ndarray] = []
    for trace_idx in range(len(durations)):
        trace_deg = traces_deg[trace_idx, : int(durations[trace_idx])]
        trace_px = trace_deg.astype(np.float64) * float(args.pixels_per_degree)
        for position_px in trace_px:
            key = runner._position_key(position_px)
            if key in trace_position_support_cache:
                continue
            trace_position_support_cache[key] = float("nan")
            trace_positions_unique.append(np.asarray(position_px, dtype=np.float64))
    trace_position_support_cache.update(runner._sample_support_only(support_world, trace_positions_unique, center_support_sum))

    all_trace_metrics: list[dict[str, float]] = []
    for trace_idx in range(len(durations)):
        trace_deg = traces_deg[trace_idx, : int(durations[trace_idx])]
        finite_mask = np.isfinite(trace_deg).all(axis=1)
        trace_deg = trace_deg[finite_mask]
        trace_px = trace_deg.astype(np.float64) * float(args.pixels_per_degree)
        trace_metrics = _compute_trace_metrics(trace_px)
        edge_support_values = [trace_position_support_cache[runner._position_key(position)] for position in trace_px]
        edge_valid_fraction = float(np.mean(np.asarray(edge_support_values) >= float(args.edge_valid_threshold))) if edge_support_values else float("nan")
        valid_fraction = float(np.mean(finite_mask)) if finite_mask.size else 0.0
        all_trace_metrics.append(trace_metrics)
        fem_trace_rows.append(
            {
                "dataset": "eoptotype",
                "session": "cached_eye_traces",
                "trace_id": trace_idx,
                "condition": "real",
                "logmar": "",
                "orientation": "",
                "n_frames": int(trace_px.shape[0]),
                "pixels_per_degree": float(args.pixels_per_degree),
                "center_mode": "trace_mean",
                "fem_rms_px": float(trace_metrics["fem_rms_px"]),
                "fem_rms_deg": _px_to_deg(trace_metrics["fem_rms_px"], args.pixels_per_degree),
                "fem_rms_arcmin": _px_to_arcmin(trace_metrics["fem_rms_px"], args.pixels_per_degree),
                "fem_median_radius_px": float(trace_metrics["fem_median_radius_px"]),
                "fem_p75_radius_px": float(trace_metrics["fem_p75_radius_px"]),
                "fem_p90_radius_px": float(trace_metrics["fem_p90_radius_px"]),
                "fem_step_rms_px": float(trace_metrics["fem_step_rms_px"]),
                "fem_step_rms_deg": _px_to_deg(trace_metrics["fem_step_rms_px"], args.pixels_per_degree),
                "fem_step_rms_arcmin": _px_to_arcmin(trace_metrics["fem_step_rms_px"], args.pixels_per_degree),
                "valid_fraction": valid_fraction,
                "edge_valid_fraction": edge_valid_fraction,
            }
        )

    fem_summary = {
        "fem_rms_px": _nanmedian([float(item["fem_rms_px"]) for item in all_trace_metrics]),
        "fem_median_radius_px": _nanmedian([float(item["fem_median_radius_px"]) for item in all_trace_metrics]),
        "fem_p75_radius_px": _nanmedian([float(item["fem_p75_radius_px"]) for item in all_trace_metrics]),
        "fem_p90_radius_px": _nanmedian([float(item["fem_p90_radius_px"]) for item in all_trace_metrics]),
        "fem_step_rms_px": _nanmedian([float(item["fem_step_rms_px"]) for item in all_trace_metrics]),
    }
    fem_summary["fem_rms_arcmin"] = _px_to_arcmin(fem_summary["fem_rms_px"], args.pixels_per_degree)
    fem_summary["fem_rms_p25_arcmin"] = _quantile([_px_to_arcmin(float(item["fem_rms_px"]), args.pixels_per_degree) for item in all_trace_metrics], 0.25)
    fem_summary["fem_rms_p75_arcmin"] = _quantile([_px_to_arcmin(float(item["fem_rms_px"]), args.pixels_per_degree) for item in all_trace_metrics], 0.75)

    for logmar in logmars:
        for orientation in orientations:
            max_radius = max(radii_px)
            feasibility_positions = [
                np.array([spec["dx_px"], spec["dy_px"]], dtype=np.float64)
                for spec in pair_specs
                if math.isclose(spec["radius_px"], max_radius, rel_tol=0.0, abs_tol=1e-9)
            ]

            if args.feasibility_only:
                with torch.no_grad():
                    world_support = runner.renderer(float(orientation), float(logmar)).to(runner.device)
                    center_movie = runner.retina(world_support, runner._make_static_trace_deg(np.zeros(2, dtype=np.float32)))[0, 0]
                    center_support_sum = float(center_movie[-1].sum().item())
                support_fractions = runner._sample_support_only(world_support, feasibility_positions, center_support_sum)
                for spec in pair_specs:
                    if math.isclose(spec["radius_px"], max_radius, rel_tol=0.0, abs_tol=1e-9):
                        position_px = np.array([spec["dx_px"], spec["dy_px"]], dtype=np.float64)
                        key = runner._position_key(position_px)
                        feasibility_rows.append(
                            {
                                "logmar": float(logmar),
                                "orientation": int(orientation),
                                "radius_px": float(spec["radius_px"]),
                                "direction_idx": int(spec["direction_idx"]),
                                "support_fraction": float(support_fractions[key]),
                                "status": _describe_support(float(support_fractions[key])),
                            }
                        )
                continue

            positions_needed: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
            for spec in pair_specs:
                p1 = np.array([spec["dx_px"], spec["dy_px"]], dtype=np.float64)
                positions_needed.append(p1)
                for anchor in (np.zeros(2, dtype=np.float64), 0.5 * p1):
                    positions_needed.extend(
                        [
                            anchor + np.array([args.jacobian_step_px, 0.0], dtype=np.float64),
                            anchor + np.array([-args.jacobian_step_px, 0.0], dtype=np.float64),
                            anchor + np.array([0.0, args.jacobian_step_px], dtype=np.float64),
                            anchor + np.array([0.0, -args.jacobian_step_px], dtype=np.float64),
                        ]
                    )
            responses, support_fractions = runner.evaluate_condition(float(logmar), int(orientation), positions_needed)

            for spec in pair_specs:
                if math.isclose(spec["radius_px"], max_radius, rel_tol=0.0, abs_tol=1e-9):
                    position_px = np.array([spec["dx_px"], spec["dy_px"]], dtype=np.float64)
                    key = runner._position_key(position_px)
                    feasibility_rows.append(
                        {
                            "logmar": float(logmar),
                            "orientation": int(orientation),
                            "radius_px": float(spec["radius_px"]),
                            "direction_idx": int(spec["direction_idx"]),
                            "support_fraction": float(support_fractions[key]),
                            "status": _describe_support(float(support_fractions[key])),
                        }
                    )

            center_response = responses[runner._position_key(np.zeros(2, dtype=np.float64))]
            center_jacobian = runner.finite_difference_jacobian(responses, np.zeros(2, dtype=np.float64))
            for spec in pair_specs:
                p0 = np.zeros(2, dtype=np.float64)
                p1 = np.array([spec["dx_px"], spec["dy_px"]], dtype=np.float64)
                delta_p = p1 - p0
                displacement_px = float(np.linalg.norm(delta_p))
                midpoint = 0.5 * (p0 + p1)
                midpoint_jacobian = runner.finite_difference_jacobian(responses, midpoint)
                response_true = responses[runner._position_key(p1)] - center_response
                support_fraction = float(min(support_fractions[runner._position_key(p0)], support_fractions[runner._position_key(p1)]))

                for jacobian_mode, jacobian in (("center_jacobian", center_jacobian), ("midpoint_jacobian", midpoint_jacobian)):
                    response_pred = jacobian @ delta_p
                    residual = response_true - response_pred
                    response_norm = float(np.linalg.norm(response_true))
                    prediction_norm = float(np.linalg.norm(response_pred))
                    residual_norm = float(np.linalg.norm(residual))
                    frac_residual = float((residual_norm ** 2) / (response_norm ** 2 + EPS))
                    cosine = float(np.dot(response_true, response_pred) / (response_norm * prediction_norm + EPS))
                    valid = bool(np.isfinite(response_norm) and support_fraction >= float(args.edge_valid_threshold))
                    pair_rows.append(
                        {
                            "dataset": "eoptotype",
                            "session": "cached_eye_traces",
                            "stimulus_family": "eoptotype",
                            "stimulus_id": f"lm{_format_logmar(logmar)}_ori{orientation}",
                            "logmar": float(logmar),
                            "orientation": int(orientation),
                            "image_id": "",
                            "window_key": "",
                            "center_position_x_px": 0.0,
                            "center_position_y_px": 0.0,
                            "p0_x_px": float(p0[0]),
                            "p0_y_px": float(p0[1]),
                            "p1_x_px": float(p1[0]),
                            "p1_y_px": float(p1[1]),
                            "delta_x_px": float(delta_p[0]),
                            "delta_y_px": float(delta_p[1]),
                            "displacement_px": displacement_px,
                            "displacement_deg": _px_to_deg(displacement_px, args.pixels_per_degree),
                            "displacement_arcmin": _px_to_arcmin(displacement_px, args.pixels_per_degree),
                            "jacobian_mode": jacobian_mode,
                            "response_norm": response_norm,
                            "prediction_norm": prediction_norm,
                            "residual_norm": residual_norm,
                            "err_norm": residual_norm / (response_norm + EPS),
                            "frac_residual_energy": frac_residual,
                            "predicted_fraction": float(1.0 - frac_residual),
                            "cosine_true_pred": cosine,
                            "support_fraction": support_fraction,
                            "padding_fraction": float(1.0 - min(max(support_fraction, 0.0), 1.0)),
                            "valid": valid,
                            "failure_reason": "" if valid else _describe_support(support_fraction),
                        }
                    )

    _write_csv(output_dir / "fem_scale_by_trace.csv", fem_trace_rows)
    _write_csv(output_dir / "feasibility_check.csv", feasibility_rows)

    if args.feasibility_only:
        run_config = {
            "mode": "feasibility_only",
            "logmars": logmars,
            "orientations": orientations,
            "radii_px": radii_px,
            "edge_valid_threshold": args.edge_valid_threshold,
            "pixels_per_degree": args.pixels_per_degree,
            "eye_traces_path": str(args.eye_traces_path),
            "model_source": str(VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl"),
        }
        (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
        print(f"Saved feasibility outputs to {output_dir}")
        return

    bin_rows = _bin_rows_from_pairs(pair_rows, args.pixels_per_degree)
    summary_rows = _summaries_from_bins(bin_rows, fem_summary, args.pixels_per_degree)
    _annotate_crossings(summary_rows)

    _write_csv(output_dir / "curvature_by_pair.csv", pair_rows)
    _write_csv(output_dir / "curvature_by_condition.csv", bin_rows)
    _write_csv(output_dir / "curvature_scale_match_summary.csv", summary_rows)

    run_config = {
        "mode": "full",
        "logmars": logmars,
        "orientations": orientations,
        "radii_px": radii_px,
        "n_directions": args.n_directions,
        "thresholds": thresholds,
        "pixels_per_degree": args.pixels_per_degree,
        "jacobian_step_px": args.jacobian_step_px,
        "n_lags": args.n_lags,
        "device": args.device,
        "edge_valid_threshold": args.edge_valid_threshold,
        "eye_traces_path": str(args.eye_traces_path),
        "output_dir": str(output_dir),
        "jacobian_modes": ["center_jacobian", "midpoint_jacobian"],
        "model_source": str(VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl"),
        "renderer_path": "scripts.temporal_decoding.stimulus_hires.HiResERenderer",
        "retina_path": "scripts.temporal_decoding.stimulus_hires.HiResRetina",
        "eye_convention_helper": "static-position trace through HiResRetina",
        "grid_sample_align_corners": False,
        "padding_mode": "background fill 127.0 gray",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if not args.skip_figures:
        _plot_error_curves(figures_dir / "jacobian_error_vs_displacement_by_logmar.png", bin_rows, fem_summary)
        _plot_scale_vs_stimulus(figures_dir / "delta_star_and_fem_vs_stimulus_size.png", summary_rows, fem_summary)
        _plot_delta_vs_fem(figures_dir / "delta_star_vs_fem_rms.png", summary_rows)
        _plot_scale_ratio(figures_dir / "scale_ratio_by_logmar.png", summary_rows)
        _plot_overview(figures_dir / "curvature_scale_match_overview.png")

    _write_readme(output_dir, summary_rows, fem_summary, run_config, feasibility_rows)
    print(f"Saved curvature-scale-match outputs to {output_dir}")


if __name__ == "__main__":
    main()
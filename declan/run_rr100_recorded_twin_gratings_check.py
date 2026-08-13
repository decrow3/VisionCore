#!/usr/bin/env python3
"""Map-first recorded-versus-twin grating tuning check for the RR100 medoids.

This script uses the held-out grating responses already saved in
``scripts/mcfarland_outputs_mono.pkl``.  It reloads only each session's grating
metadata to recover the exact spatial-frequency/orientation sequence, verifies
that the cached ``robs`` is sample-for-sample identical to the dataset, and then
computes lagged condition-response maps for the 100 selected movie medoids.

The comparison is deliberately matched: the recorded response selects the peak
lag and the twin is evaluated at that same lag.  Outputs are unit-level maps and
curves; no population inference is performed here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from models.config_loader import load_dataset_configs
from models.data import prepare_data


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_CONFIG = ROOT / "experiments/dataset_configs/multi_basic_120_long_legacy.yaml"
DEFAULT_CACHE = ROOT / "scripts/mcfarland_outputs_mono.pkl"
DEFAULT_OUT_DIR = (
    ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1"
)


@dataclass
class UnitResult:
    rr100_index: int
    canonical_channel: int
    session: str
    source_unit_index: int
    ccnorm: float
    group_kind: str
    group_size: int
    peak_lag_bins: int
    peak_lag_ms: float
    real_peak_sf: float
    twin_peak_sf: float
    real_peak_ori: float
    twin_peak_ori: float
    ori_difference_deg: float
    real_tuning_strength: float
    twin_tuning_strength: float
    map_correlation: float
    sf_curve_correlation: float
    ori_curve_correlation: float
    real_mean_rate_hz: float
    twin_mean_rate_hz: float
    min_condition_count: int
    max_condition_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-version", default=RR100_VERSION)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outputs-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-lags", type=int, default=20)
    parser.add_argument("--min-condition-samples", type=int, default=5)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Smoke-test limit; zero processes every selected session.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=0,
        help="Smoke-test limit after RR100 ordering; zero processes all units.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_slug(value: str) -> str:
    return "_".join(
        part for part in "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).split("_") if part
    )


def population_paths(version: str) -> tuple[Path, Path]:
    slug = safe_slug(version)
    json_path = SPEC_DIR / f"population_spec_{slug}.json"
    npz_path = SPEC_DIR / f"population_spec_{slug}.npz"
    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Missing RR100 spec files for {version!r} under {SPEC_DIR}")
    return json_path, npz_path


def load_rr100_rows(version: str, max_units: int) -> tuple[list[dict[str, Any]], dict[str, Any], Path, Path]:
    json_path, npz_path = population_paths(version)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    reps = sorted(meta["representatives"], key=lambda row: int(row["rep_idx"]))
    with np.load(npz_path) as z:
        membership = np.asarray(z["membership"], dtype=np.float32)
    if membership.shape != (len(reps), int(meta["n_input_channels"])):
        raise ValueError(f"Unexpected membership shape {membership.shape}")
    for row in reps:
        ridx = int(row["rep_idx"])
        selected = np.flatnonzero(membership[ridx] != 0)
        if selected.size != 1 or int(selected[0]) != int(row["selected_channel"]):
            raise ValueError(f"RR100 row {ridx} is not the declared one-hot movie medoid")
    if max_units > 0:
        reps = reps[: int(max_units)]
    return reps, meta, json_path, npz_path


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return float("nan")
    xv = np.asarray(x[valid], dtype=np.float64)
    yv = np.asarray(y[valid], dtype=np.float64)
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def tuning_strength(x: np.ndarray) -> float:
    valid = np.isfinite(x)
    if not np.any(valid):
        return float("nan")
    mean = float(np.mean(x[valid]))
    return float(np.std(x[valid]) / max(abs(mean), 1e-8))


def ori_distance(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    delta = abs(float(a) - float(b)) % 180.0
    return float(min(delta, 180.0 - delta))


def lagged_condition_maps(
    response: np.ndarray,
    dfs: np.ndarray,
    sf: np.ndarray,
    ori: np.ndarray,
    sfs: np.ndarray,
    oris: np.ndarray,
    *,
    dt: float,
    n_lags: int,
    min_condition_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rate maps and valid counts with shape U x lag x SF x orientation."""
    response = np.asarray(response, dtype=np.float64)
    dfs = np.asarray(dfs, dtype=np.float64)
    n_time, n_units = response.shape
    maps = np.full((n_units, n_lags, len(sfs), len(oris)), np.nan, dtype=np.float32)
    counts = np.zeros_like(maps, dtype=np.int32)
    for lag in range(int(n_lags)):
        stim_idx = np.arange(0, n_time - lag, dtype=np.int64)
        response_idx = stim_idx + lag
        for isf, sf_value in enumerate(sfs):
            sf_mask = sf[stim_idx] == sf_value
            for iori, ori_value in enumerate(oris):
                base = sf_mask & (ori[stim_idx] == ori_value)
                if not np.any(base):
                    continue
                candidate = response_idx[base]
                values = response[candidate]
                valid = (dfs[candidate] > 0) & np.isfinite(values)
                count = np.sum(valid, axis=0)
                counts[:, lag, isf, iori] = count
                enough = count >= int(min_condition_samples)
                if np.any(enough):
                    sums = np.sum(np.where(valid, values, 0.0), axis=0)
                    maps[enough, lag, isf, iori] = (sums[enough] / count[enough] / float(dt)).astype(np.float32)
    return maps, counts


def standardized_map(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    sd = float(np.std(out[valid]))
    out[valid] = (out[valid] - float(np.mean(out[valid]))) / max(sd, 1e-8)
    return out


def condition_peak(map_: np.ndarray, sfs: np.ndarray, oris: np.ndarray) -> tuple[float, float, int, int]:
    if not np.any(np.isfinite(map_)):
        return float("nan"), float("nan"), -1, -1
    flat = int(np.nanargmax(map_))
    isf, iori = np.unravel_index(flat, map_.shape)
    return float(sfs[isf]), float(oris[iori]), int(isf), int(iori)


def analyze_session_units(
    reps: list[dict[str, Any]],
    cached: dict[str, Any],
    sf: np.ndarray,
    ori: np.ndarray,
    *,
    dt: float,
    n_lags: int,
    min_condition_samples: int,
) -> tuple[list[UnitResult], list[dict[str, Any]], list[dict[str, Any]]]:
    unit_columns = np.asarray([int(row["selected_source_unit_index"]) for row in reps], dtype=np.int64)
    gratings = cached["bps_results"]["gratings"]
    robs = gratings["robs"][:, unit_columns].detach().cpu().numpy()
    rhat = gratings["rhat"][:, unit_columns].detach().cpu().numpy()
    dfs = gratings["dfs"][:, unit_columns].detach().cpu().numpy()
    sfs = np.unique(sf[np.isfinite(sf)])
    oris = np.unique(ori[np.isfinite(ori)])
    real_all, counts_all = lagged_condition_maps(
        robs,
        dfs,
        sf,
        ori,
        sfs,
        oris,
        dt=dt,
        n_lags=n_lags,
        min_condition_samples=min_condition_samples,
    )
    twin_all, _ = lagged_condition_maps(
        rhat,
        dfs,
        sf,
        ori,
        sfs,
        oris,
        dt=dt,
        n_lags=n_lags,
        min_condition_samples=min_condition_samples,
    )

    metrics: list[UnitResult] = []
    map_rows: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    for iu, rep in enumerate(reps):
        temporal_strength = np.asarray(
            [np.nanstd(real_all[iu, lag]) for lag in range(n_lags)], dtype=np.float64
        )
        peak_lag = int(np.nanargmax(temporal_strength)) if np.any(np.isfinite(temporal_strength)) else 0
        real_map = np.asarray(real_all[iu, peak_lag], dtype=np.float64)
        twin_map = np.asarray(twin_all[iu, peak_lag], dtype=np.float64)
        count_map = np.asarray(counts_all[iu, peak_lag], dtype=np.int64)
        real_sf, real_ori, real_isf, real_iori = condition_peak(real_map, sfs, oris)
        twin_sf, twin_ori, _, _ = condition_peak(twin_map, sfs, oris)

        if real_iori >= 0:
            real_sf_curve = real_map[:, real_iori]
            twin_sf_curve = twin_map[:, real_iori]
        else:
            real_sf_curve = np.full(len(sfs), np.nan)
            twin_sf_curve = np.full(len(sfs), np.nan)
        if real_isf >= 0:
            real_ori_curve = real_map[real_isf, :]
            twin_ori_curve = twin_map[real_isf, :]
        else:
            real_ori_curve = np.full(len(oris), np.nan)
            twin_ori_curve = np.full(len(oris), np.nan)

        valid_counts = count_map[count_map > 0]
        metric = UnitResult(
            rr100_index=int(rep["rep_idx"]),
            canonical_channel=int(rep["selected_channel"]),
            session=str(rep["selected_session"]),
            source_unit_index=int(rep["selected_source_unit_index"]),
            ccnorm=float(rep["selected_ccnorm"]),
            group_kind=str(rep["kind"]),
            group_size=int(rep["n_members"]),
            peak_lag_bins=peak_lag,
            peak_lag_ms=float(peak_lag * dt * 1000.0),
            real_peak_sf=real_sf,
            twin_peak_sf=twin_sf,
            real_peak_ori=real_ori,
            twin_peak_ori=twin_ori,
            ori_difference_deg=ori_distance(real_ori, twin_ori),
            real_tuning_strength=tuning_strength(real_map),
            twin_tuning_strength=tuning_strength(twin_map),
            map_correlation=corrcoef_safe(real_map.ravel(), twin_map.ravel()),
            sf_curve_correlation=corrcoef_safe(real_sf_curve, twin_sf_curve),
            ori_curve_correlation=corrcoef_safe(real_ori_curve, twin_ori_curve),
            real_mean_rate_hz=float(np.nanmean(real_map)),
            twin_mean_rate_hz=float(np.nanmean(twin_map)),
            min_condition_count=int(valid_counts.min()) if valid_counts.size else 0,
            max_condition_count=int(valid_counts.max()) if valid_counts.size else 0,
        )
        metrics.append(metric)
        for source, values in (("recorded", real_map), ("twin", twin_map)):
            for isf, sf_value in enumerate(sfs):
                for iori, ori_value in enumerate(oris):
                    map_rows.append(
                        {
                            "rr100_index": metric.rr100_index,
                            "canonical_channel": metric.canonical_channel,
                            "session": metric.session,
                            "source_unit_index": metric.source_unit_index,
                            "source": source,
                            "peak_lag_bins_from_recorded": peak_lag,
                            "spatial_frequency_cpd": float(sf_value),
                            "orientation_deg": float(ori_value),
                            "rate_hz": float(values[isf, iori]),
                            "valid_sample_count": int(count_map[isf, iori]),
                        }
                    )
        plot_records.append(
            {
                "metric": metric,
                "sfs": sfs,
                "oris": oris,
                "real_map": real_map,
                "twin_map": twin_map,
                "real_sf_curve": real_sf_curve,
                "twin_sf_curve": twin_sf_curve,
                "real_ori_curve": real_ori_curve,
                "twin_ori_curve": twin_ori_curve,
            }
        )
    return metrics, map_rows, plot_records


def imshow_tuning(ax: plt.Axes, values: np.ndarray, sfs: np.ndarray, oris: np.ndarray, **kwargs: Any) -> Any:
    im = ax.imshow(values, origin="lower", aspect="auto", **kwargs)
    ax.set_xticks(np.arange(len(oris)))
    ax.set_xticklabels([f"{v:g}" for v in oris], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(sfs)))
    ax.set_yticklabels([f"{v:g}" for v in sfs], fontsize=6)
    return im


def plot_unit_row(fig: plt.Figure, axes: np.ndarray, record: dict[str, Any], *, title_prefix: str = "") -> None:
    metric: UnitResult = record["metric"]
    real_map = record["real_map"]
    twin_map = record["twin_map"]
    sfs = record["sfs"]
    oris = record["oris"]
    finite = np.concatenate([real_map[np.isfinite(real_map)], twin_map[np.isfinite(twin_map)]])
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    im0 = imshow_tuning(axes[0], real_map, sfs, oris, cmap="viridis", vmin=vmin, vmax=vmax)
    imshow_tuning(axes[1], twin_map, sfs, oris, cmap="viridis", vmin=vmin, vmax=vmax)
    diff = standardized_map(twin_map) - standardized_map(real_map)
    diff_lim = max(float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else 1.0, 1e-6)
    im2 = imshow_tuning(axes[2], diff, sfs, oris, cmap="coolwarm", vmin=-diff_lim, vmax=diff_lim)
    fig.colorbar(im0, ax=axes[:2].tolist(), fraction=0.018, pad=0.01)
    fig.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.02)

    axes[3].plot(sfs, record["real_sf_curve"], "o-k", lw=1.4, ms=3, label="recorded")
    axes[3].plot(sfs, record["twin_sf_curve"], "o-", color="#d62728", lw=1.4, ms=3, label="twin")
    axes[3].set_xlabel("spatial frequency (cpd)")
    axes[3].set_ylabel("spikes/s")
    axes[3].grid(alpha=0.2)
    axes[3].legend(fontsize=6, frameon=False)
    axes[4].plot(oris, record["real_ori_curve"], "o-k", lw=1.4, ms=3)
    axes[4].plot(oris, record["twin_ori_curve"], "o-", color="#d62728", lw=1.4, ms=3)
    axes[4].set_xlabel("orientation (deg)")
    axes[4].set_ylabel("spikes/s")
    axes[4].grid(alpha=0.2)

    axes[0].set_title("recorded", fontsize=8)
    axes[1].set_title("twin @ recorded lag", fontsize=8)
    axes[2].set_title("normalized difference", fontsize=8)
    title_lines = []
    if title_prefix:
        title_lines.append(title_prefix)
    title_lines.extend(
        [
            f"RR100 {metric.rr100_index:03d}",
            f"{metric.session}",
            f"source unit {metric.source_unit_index}",
            f"lag {metric.peak_lag_ms:.1f} ms",
            f"map r={metric.map_correlation:.2f}",
            f"ccnorm={metric.ccnorm:.2f}",
        ]
    )
    unit_title = "\n".join(title_lines)
    axes[0].text(
        -0.48,
        0.5,
        unit_title,
        transform=axes[0].transAxes,
        va="center",
        ha="right",
        fontsize=7,
    )


def write_atlas(records: list[dict[str, Any]], path: Path, rows_per_page: int = 4) -> None:
    with PdfPages(path) as pdf:
        for start in range(0, len(records), rows_per_page):
            page = records[start : start + rows_per_page]
            fig, axes = plt.subplots(len(page), 5, figsize=(16, 3.15 * len(page)), squeeze=False)
            for row_idx, record in enumerate(page):
                plot_unit_row(fig, axes[row_idx], record)
            fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.95, wspace=0.55, hspace=0.85)
            fig.suptitle(
                "RR100 recorded vs fitted-twin grating tuning (matched recorded peak lag)",
                fontsize=13,
            )
            pdf.savefig(fig, dpi=160)
            plt.close(fig)


def choose_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply predefined auditable roles; avoid duplicate units when possible."""
    finite_strength = metrics[np.isfinite(metrics["real_tuning_strength"])].copy()
    if finite_strength.empty:
        return pd.DataFrame()
    median_strength = float(finite_strength["real_tuning_strength"].median())
    tuned = finite_strength[
        (finite_strength["real_tuning_strength"] >= median_strength)
        & np.isfinite(finite_strength["map_correlation"])
    ].copy()
    role_specs = [
        ("strongest_recorded_tuning", finite_strength.sort_values("real_tuning_strength", ascending=False), "real_tuning_strength"),
        ("best_matched_among_tuned", tuned.sort_values("map_correlation", ascending=False), "map_correlation"),
        ("mismatch_among_tuned", tuned.sort_values("map_correlation", ascending=True), "map_correlation"),
        ("weakly_tuned_control", finite_strength.sort_values("real_tuning_strength", ascending=True), "real_tuning_strength"),
    ]
    chosen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for role, candidates, criterion in role_specs:
        if candidates.empty:
            continue
        available = candidates[~candidates["rr100_index"].astype(int).isin(chosen)]
        row = (available if not available.empty else candidates).iloc[0]
        chosen.add(int(row["rr100_index"]))
        out = row.to_dict()
        out.update(
            {
                "selection_role": role,
                "criterion_name": criterion,
                "criterion_value": float(row[criterion]),
                "selection_method": "predefined_algorithmic_role",
                "tuned_threshold_median_real_tuning_strength": median_strength,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def write_example_figure(records: list[dict[str, Any]], examples: pd.DataFrame, path: Path) -> None:
    lookup = {int(record["metric"].rr100_index): record for record in records}
    selected = [lookup[int(v)] for v in examples["rr100_index"] if int(v) in lookup]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3.25 * len(selected)), squeeze=False)
    role_lookup = {int(row.rr100_index): str(row.selection_role) for _, row in examples.iterrows()}
    for row_idx, record in enumerate(selected):
        role = role_lookup[int(record["metric"].rr100_index)]
        plot_unit_row(fig, axes[row_idx], record, title_prefix=role)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.07, top=0.94, wspace=0.55, hspace=0.9)
    fig.suptitle(
        "RR100 grating checkpoint: predefined positive, match, mismatch, and weak-tuning roles",
        fontsize=13,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_support_figure(support: pd.DataFrame, path: Path) -> None:
    sessions = list(dict.fromkeys(support["session"].astype(str)))
    ncols = 5
    nrows = int(math.ceil(len(sessions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.8 * nrows), squeeze=False)
    for ax, session in zip(axes.ravel(), sessions):
        sub = support[support["session"] == session]
        pivot = sub.pivot(index="spatial_frequency_cpd", columns="orientation_deg", values="n_validation_frames")
        im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(session, fontsize=8)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    for ax in axes.ravel()[len(sessions) :]:
        ax.axis("off")
    fig.supxlabel("orientation (deg)")
    fig.supylabel("spatial frequency (cpd)")
    fig.suptitle("Held-out grating stimulus support used for the RR100 tuning check", fontsize=13)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def main() -> None:
    args = parse_args()
    if args.n_lags <= 0:
        raise ValueError("--n-lags must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "rr100_recorded_twin_gratings_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {manifest_path}. Pass --force to rerun.")

    reps, population_meta, spec_json, spec_npz = load_rr100_rows(args.population_version, args.max_units)
    sessions = list(dict.fromkeys(str(row["selected_session"]) for row in reps))
    if args.max_sessions > 0:
        allowed = set(sessions[: int(args.max_sessions)])
        reps = [row for row in reps if str(row["selected_session"]) in allowed]
        sessions = sessions[: int(args.max_sessions)]
    print(f"RR100 rows selected: {len(reps)} across {len(sessions)} sessions", flush=True)

    mapping_rows = []
    for row in reps:
        mapping_rows.append(
            {
                "rr100_index": int(row["rep_idx"]),
                "canonical_channel": int(row["selected_channel"]),
                "session": str(row["selected_session"]),
                "source_unit_index": int(row["selected_source_unit_index"]),
                "ccnorm": float(row["selected_ccnorm"]),
                "group_kind": str(row["kind"]),
                "group_label": int(row["group_label"]),
                "group_size": int(row["n_members"]),
                "member_channels": ",".join(str(v) for v in row["members"]),
                "selection_role": "rr100_movie_medoid_fixed_before_grating_check",
            }
        )
    pd.DataFrame(mapping_rows).to_csv(args.out_dir / "rr100_unit_mapping.csv", index=False)

    print(f"Loading held-out response cache: {args.outputs_cache}", flush=True)
    with args.outputs_cache.open("rb") as f:
        outputs = dill.load(f)
    outputs_by_session = {str(row["sess"]): row for row in outputs}
    configs = load_dataset_configs(args.dataset_config)
    configs_by_session = {str(cfg["session"]): cfg for cfg in configs}

    all_metrics: list[UnitResult] = []
    all_map_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for session_number, session in enumerate(sessions, start=1):
        print(f"[{session_number}/{len(sessions)}] Loading grating metadata for {session}", flush=True)
        if session not in outputs_by_session:
            raise KeyError(f"Session {session} missing from {args.outputs_cache}")
        if session not in configs_by_session:
            raise KeyError(f"Session {session} missing from {args.dataset_config}")
        config = copy.deepcopy(configs_by_session[session])
        config["types"] = ["gratings"]
        # The currently materialized Logan grating files are already unit-
        # subsetted, whereas the legacy YAML stores the original cell labels.
        # Those labels cannot safely be reused as tensor column positions.  We
        # only need the deterministic validation time indices and grating
        # covariates here, so leave the materialized Logan unit columns intact.
        if session.startswith("Logan_"):
            config["cids"] = None
        # Stimulus pixels and behavior are unnecessary for this cache-backed check.
        # Keeping the original 32-bin stimulus embedding preserves the exact split.
        config["transforms"] = {}
        config["keys_lags"] = {
            "robs": 0,
            "stim": list(config["keys_lags"]["stim"]),
            "dfs": 0,
        }
        train_data, val_data, loaded_config = prepare_data(config, strict=True)
        gratings_inds = val_data.get_dataset_inds("gratings")
        dset_indices = np.unique(gratings_inds[:, 0].detach().cpu().numpy())
        if dset_indices.size != 1:
            raise ValueError(f"Expected one grating dataset for {session}; got {dset_indices}")
        dset_idx = int(dset_indices[0])
        local_inds = gratings_inds[:, 1]
        dset = val_data.dsets[dset_idx]
        sf = dset["sf"][local_inds].detach().cpu().numpy().astype(np.float64)
        ori = dset["ori"][local_inds].detach().cpu().numpy().astype(np.float64)
        dataset_robs = dset["robs"][local_inds].detach().cpu()
        cached_robs = outputs_by_session[session]["bps_results"]["gratings"]["robs"].detach().cpu()
        length_alignment = bool(dataset_robs.shape[0] == cached_robs.shape[0])
        exact_alignment = bool(
            dataset_robs.shape == cached_robs.shape
            and np.array_equal(dataset_robs.numpy(), cached_robs.numpy())
        )
        max_abs_difference = (
            float(np.max(np.abs(dataset_robs.numpy() - cached_robs.numpy())))
            if dataset_robs.shape == cached_robs.shape
            else float("nan")
        )
        alignment_rows.append(
            {
                "session": session,
                "dataset_shape": str(tuple(dataset_robs.shape)),
                "cache_shape": str(tuple(cached_robs.shape)),
                "validation_length_alignment": length_alignment,
                "exact_robs_alignment": exact_alignment,
                "alignment_basis": (
                    "sample_for_sample_exact_robs"
                    if exact_alignment
                    else "deterministic_validation_split_and_equal_time_length"
                ),
                "max_abs_robs_difference": max_abs_difference,
            }
        )
        if not length_alignment:
            raise AssertionError(
                f"Cached grating response length does not align with reloaded metadata for {session}"
            )

        sfs = np.unique(sf[np.isfinite(sf)])
        oris = np.unique(ori[np.isfinite(ori)])
        for sf_value in sfs:
            for ori_value in oris:
                support_rows.append(
                    {
                        "session": session,
                        "spatial_frequency_cpd": float(sf_value),
                        "orientation_deg": float(ori_value),
                        "n_validation_frames": int(np.sum((sf == sf_value) & (ori == ori_value))),
                    }
                )

        session_reps = [row for row in reps if str(row["selected_session"]) == session]
        dt = 1.0 / float(loaded_config["sampling"]["target_rate"])
        metrics, map_rows, records = analyze_session_units(
            session_reps,
            outputs_by_session[session],
            sf,
            ori,
            dt=dt,
            n_lags=int(args.n_lags),
            min_condition_samples=int(args.min_condition_samples),
        )
        all_metrics.extend(metrics)
        all_map_rows.extend(map_rows)
        all_records.extend(records)
        print(
            f"[{session_number}/{len(sessions)}] {session}: "
            f"{'exact robs' if exact_alignment else 'deterministic time-index'} alignment; "
            f"analyzed {len(metrics)} RR100 units",
            flush=True,
        )
        del train_data, val_data, dset, dataset_robs, cached_robs
        gc.collect()

    metrics_df = pd.DataFrame([asdict(row) for row in all_metrics]).sort_values("rr100_index")
    map_df = pd.DataFrame(all_map_rows).sort_values(
        ["rr100_index", "source", "spatial_frequency_cpd", "orientation_deg"]
    )
    support_df = pd.DataFrame(support_rows)
    alignment_df = pd.DataFrame(alignment_rows)
    metrics_df.to_csv(args.out_dir / "rr100_grating_tuning_metrics.csv", index=False)
    map_df.to_csv(args.out_dir / "rr100_grating_tuning_maps_long.csv", index=False)
    support_df.to_csv(args.out_dir / "rr100_grating_stimulus_support.csv", index=False)
    alignment_df.to_csv(args.out_dir / "rr100_cache_alignment.csv", index=False)

    records_sorted = sorted(all_records, key=lambda record: int(record["metric"].rr100_index))
    write_support_figure(support_df, args.out_dir / "rr100_grating_stimulus_support.png")
    write_atlas(records_sorted, args.out_dir / "rr100_recorded_twin_grating_tuning_atlas.pdf")
    examples = choose_examples(metrics_df)
    examples.to_csv(args.out_dir / "rr100_grating_example_unit_selection.csv", index=False)
    write_example_figure(records_sorted, examples, args.out_dir / "rr100_grating_example_unit_comparison.png")

    manifest = {
        "analysis": "rr100_recorded_twin_gratings_map_first_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "smoke" if args.max_sessions > 0 or args.max_units > 0 else "targeted_all_rr100",
        "population_version": args.population_version,
        "population_spec_json": file_identity(spec_json),
        "population_spec_npz": file_identity(spec_npz),
        "dataset_config": file_identity(args.dataset_config),
        "outputs_cache": file_identity(args.outputs_cache),
        "output_dir": str(args.out_dir.resolve()),
        "n_rr100_units": int(len(metrics_df)),
        "n_sessions": int(len(sessions)),
        "session_unit_counts": dict(Counter(row.session for row in all_metrics)),
        "n_lags": int(args.n_lags),
        "lag_selection": "per-unit recorded-map maximum across-condition standard deviation",
        "twin_comparison_lag": "same recorded-selected lag",
        "rate_units": "spikes_per_second (cached bin response divided by dt)",
        "min_condition_samples": int(args.min_condition_samples),
        "robs_alignment": (
            "sample-for-sample exact where current materialized unit columns match; "
            "otherwise deterministic validation split plus exact cached/reloaded time length"
        ),
        "population_inference_performed": False,
        "artifacts": {
            "unit_mapping": "rr100_unit_mapping.csv",
            "stimulus_support_table": "rr100_grating_stimulus_support.csv",
            "stimulus_support_figure": "rr100_grating_stimulus_support.png",
            "cache_alignment": "rr100_cache_alignment.csv",
            "unit_metrics": "rr100_grating_tuning_metrics.csv",
            "unit_maps_long": "rr100_grating_tuning_maps_long.csv",
            "unit_atlas": "rr100_recorded_twin_grating_tuning_atlas.pdf",
            "example_selection": "rr100_grating_example_unit_selection.csv",
            "example_figure": "rr100_grating_example_unit_comparison.png",
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "population_meta_summary": {
            "n_input_channels": int(population_meta["n_input_channels"]),
            "n_representatives": int(population_meta["n_representatives"]),
            "pooling_mode": str(population_meta["pooling_mode"]),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote RR100 grating checkpoint to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

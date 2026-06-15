"""Audit alternative twin stability metrics for BackImage edge-parallel result."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .posthoc_backimage_stability_wrong_direction_controls import (
        _fit_model,
        _load_merged,
        _permute_model,
        _session_bootstrap_model,
    )
    from .run_backimage_edge_parallel_stability_screen import _axis_vector, _tail_mean
    from .run_backimage_twin_drift_geometry import TwinScorer, _clip_patch
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.posthoc_backimage_stability_wrong_direction_controls import (
        _fit_model,
        _load_merged,
        _permute_model,
        _session_bootstrap_model,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_edge_parallel_stability_screen import _axis_vector, _tail_mean
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import TwinScorer, _clip_patch


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_STABILITY_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_twin_stability_metric_audit"
)


MODEL_SPECS = (
    ("unadjusted", []),
    ("coherence", ["image_orientation_coherence"]),
    ("full_low_level", ["image_orientation_coherence", "drift_anisotropy", "local_contrast", "edge_strength"]),
)


def _load_source_rows(input_path: Path, stability_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged, metadata = _load_merged(input_path, stability_dir)
    cfg = metadata.get("config", {})
    source = pd.read_csv(input_path)
    if "duration_s" not in source.columns:
        source["duration_s"] = source.get("epoch_duration_s", np.nan)
    margin = float(cfg.get("min_patch_image_margin_px", cfg.get("patch_size_px", 540) / 2.0))
    keep = (
        np.isfinite(source["drift_orientation_deg"].astype(float))
        & np.isfinite(source["image_edge_axis_deg"].astype(float))
        & (source["anisotropy"].astype(float) >= float(cfg.get("reliable_drift_anisotropy_min", 0.2)))
        & (source["image_orientation_coherence"].astype(float) >= float(cfg.get("reliable_image_coherence_min", 0.2)))
        & (source["duration_s"].astype(float) >= float(cfg.get("min_duration_s", 0.1)))
        & (source["image_patch_distance_to_image_border_px"].astype(float) >= margin)
    )
    work = source.loc[keep].copy()
    work["source_window_id"] = np.arange(work.shape[0], dtype=int)
    cols = [
        "source_window_id",
        "session",
        "trial_idx",
        "mean_x_deg",
        "mean_y_deg",
        "image_edge_axis_deg",
    ]
    source_subset = work[cols].copy()
    out = merged.merge(
        source_subset,
        left_on=["window_id", "session_id", "image_id"],
        right_on=["source_window_id", "session", "trial_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    return out, metadata


def _endpoint_tail_vectors(
    scorer: TwinScorer,
    patch: np.ndarray,
    *,
    edge_axis_deg: float,
    displacement_deg: float,
    hold_frames: int,
    tail_frames: int,
) -> dict[str, np.ndarray]:
    parallel = _axis_vector(edge_axis_deg, displacement_deg)
    orthogonal = _axis_vector(edge_axis_deg + 90.0, displacement_deg)
    endpoints = [
        np.asarray([0.0, 0.0], dtype=np.float32),
        parallel,
        -parallel,
        orthogonal,
        -orthogonal,
    ]
    trace = np.concatenate([np.repeat(endpoint[None, :], int(hold_frames), axis=0) for endpoint in endpoints], axis=0).astype(np.float32)
    response = scorer.response(patch, trace)
    return {
        "base": _tail_mean(response, block_idx=0, hold_frames=hold_frames, tail_frames=tail_frames),
        "parallel_plus": _tail_mean(response, block_idx=1, hold_frames=hold_frames, tail_frames=tail_frames),
        "parallel_minus": _tail_mean(response, block_idx=2, hold_frames=hold_frames, tail_frames=tail_frames),
        "orthogonal_plus": _tail_mean(response, block_idx=3, hold_frames=hold_frames, tail_frames=tail_frames),
        "orthogonal_minus": _tail_mean(response, block_idx=4, hold_frames=hold_frames, tail_frames=tail_frames),
    }


def _compute_or_load_endpoint_cache(source_rows: pd.DataFrame, metadata: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache_path = out_dir / "twin_endpoint_tail_vectors.npz"
    if cache_path.exists() and not bool(args.recompute):
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}
    cfg = metadata.get("config", {})
    scorer = TwinScorer(
        device=str(args.device or cfg.get("device", "auto")),
        population_n=int(cfg.get("twin_population_n", 256)),
        batch_size=int(cfg.get("twin_batch_size", 24)),
        seed=int(cfg.get("seed", 0)),
    )
    patch_size_px = int(cfg.get("patch_size_px", 540))
    displacement_deg = float(cfg.get("displacement_deg", 0.125))
    hold_frames = int(cfg.get("twin_hold_frames", 40))
    tail_frames = int(cfg.get("twin_tail_frames", 8))
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ["base", "parallel_plus", "parallel_minus", "orthogonal_plus", "orthogonal_minus"]}
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    for idx, row in source_rows.iterrows():
        canvas_key = (str(row["session_id"]), int(row["image_id"]))
        if canvas_key not in canvas_cache:
            canvas_cache[canvas_key] = _backimage_canvas(str(row["session_id"]), int(row["image_id"]))
        canvas, ppd, screen_shape = canvas_cache[canvas_key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), patch_size_px)
        vecs = _endpoint_tail_vectors(
            scorer,
            patch,
            edge_axis_deg=float(row["edge_axis_deg"]),
            displacement_deg=displacement_deg,
            hold_frames=hold_frames,
            tail_frames=tail_frames,
        )
        for key, value in vecs.items():
            arrays[key].append(np.asarray(value, dtype=np.float32))
        done = int(idx) + 1
        if done == 1 or done == source_rows.shape[0] or done % 16 == 0:
            print(f"[twin-stability-metric-audit] endpoint responses {done}/{source_rows.shape[0]}", flush=True)
    out = {key: np.vstack(values).astype(np.float32) for key, values in arrays.items()}
    out["window_row"] = source_rows["window_row"].to_numpy(dtype=np.int64)
    np.savez_compressed(cache_path, **out)
    return out


def _mean_sq(diffs: np.ndarray, unit_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(diffs, dtype=np.float64)
    if unit_mask is not None:
        arr = arr[..., unit_mask]
    return np.nanmean(arr * arr, axis=(1, 2))


def _metric_costs(endpoints: dict[str, np.ndarray]) -> pd.DataFrame:
    eps = 1e-8
    base = endpoints["base"].astype(np.float64)
    par = np.stack([endpoints["parallel_plus"] - endpoints["base"], endpoints["parallel_minus"] - endpoints["base"]], axis=1).astype(np.float64)
    orth = np.stack([endpoints["orthogonal_plus"] - endpoints["base"], endpoints["orthogonal_minus"] - endpoints["base"]], axis=1).astype(np.float64)
    all_diffs = np.concatenate([par.reshape(-1, par.shape[-1]), orth.reshape(-1, orth.shape[-1])], axis=0)
    unit_var = np.nanvar(all_diffs, axis=0)
    unit_var[~np.isfinite(unit_var) | (unit_var <= 1e-12)] = float(np.nanmedian(unit_var[unit_var > 1e-12])) if np.any(unit_var > 1e-12) else 1.0
    mean_base = np.nanmean(base, axis=0)
    high_rate = mean_base >= np.nanmedian(mean_base)
    unit_mod = np.nanmean(all_diffs * all_diffs, axis=0)
    top_mod = unit_mod >= np.nanquantile(unit_mod, 0.75)

    metrics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metrics["raw_mse"] = (_mean_sq(par), _mean_sq(orth))
    base_energy = np.nanmean(base * base, axis=1) + eps
    metrics["response_norm_mse"] = (metrics["raw_mse"][0] / base_energy, metrics["raw_mse"][1] / base_energy)
    abs_base = np.abs(base)
    finite_abs_base = abs_base[np.isfinite(abs_base)]
    rate_floor = float(np.nanpercentile(finite_abs_base, 5.0)) if finite_abs_base.size else 1.0
    if not np.isfinite(rate_floor) or rate_floor <= eps:
        rate_floor = eps
    base_rate = np.maximum(abs_base, rate_floor)
    metrics["per_rate_mse"] = (
        np.nanmean((par * par) / base_rate[:, None, :], axis=(1, 2)),
        np.nanmean((orth * orth) / base_rate[:, None, :], axis=(1, 2)),
    )
    metrics["fractional_rate_mse"] = (
        np.nanmean((par / base_rate[:, None, :]) ** 2, axis=(1, 2)),
        np.nanmean((orth / base_rate[:, None, :]) ** 2, axis=(1, 2)),
    )
    metrics["diag_whitened_mse"] = (
        np.nanmean((par * par) / unit_var[None, None, :], axis=(1, 2)),
        np.nanmean((orth * orth) / unit_var[None, None, :], axis=(1, 2)),
    )
    try:
        lw = LedoitWolf().fit(all_diffs)
        precision = lw.precision_

        def maha(diffs: np.ndarray) -> np.ndarray:
            return np.asarray([np.nanmean([float(d @ precision @ d) / d.size for d in row]) for row in diffs], dtype=np.float64)

        metrics["full_cov_whitened_mse"] = (maha(par), maha(orth))
    except Exception:
        metrics["full_cov_whitened_mse"] = (np.full(base.shape[0], np.nan), np.full(base.shape[0], np.nan))
    metrics["high_rate_units_raw_mse"] = (_mean_sq(par, high_rate), _mean_sq(orth, high_rate))
    metrics["low_rate_units_raw_mse"] = (_mean_sq(par, ~high_rate), _mean_sq(orth, ~high_rate))
    metrics["top_modulated_units_raw_mse"] = (_mean_sq(par, top_mod), _mean_sq(orth, top_mod))
    metrics["other_units_raw_mse"] = (_mean_sq(par, ~top_mod), _mean_sq(orth, ~top_mod))

    rows = []
    for name, (parallel, orthogonal) in metrics.items():
        advantage = orthogonal - parallel
        relative = advantage / (np.abs(orthogonal) + np.abs(parallel) + eps)
        for i in range(base.shape[0]):
            rows.append(
                {
                    "window_row": int(endpoints["window_row"][i]),
                    "metric": name,
                    "parallel_cost": float(parallel[i]),
                    "orthogonal_cost": float(orthogonal[i]),
                    "stability_advantage": float(advantage[i]),
                    "relative_advantage": float(relative[i]),
                    "base_mean_rate": float(np.nanmean(base[i])),
                    "base_response_energy": float(base_energy[i]),
                }
            )
    return pd.DataFrame(rows)


def _wide_metrics(metric_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_row, block in metric_long.groupby("window_row", sort=True):
        row: dict[str, Any] = {"window_row": int(window_row)}
        for rec in block.to_dict(orient="records"):
            metric = str(rec["metric"])
            row[f"{metric}_parallel_cost"] = float(rec["parallel_cost"])
            row[f"{metric}_orthogonal_cost"] = float(rec["orthogonal_cost"])
            row[f"{metric}_stability_advantage"] = float(rec["stability_advantage"])
            row[f"{metric}_relative_advantage"] = float(rec["relative_advantage"])
        rows.append(row)
    return pd.DataFrame(rows)


def _run_metric_models(df: pd.DataFrame, metric_names: list[str], *, rng: np.random.Generator, n_bootstrap: int, n_permutations: int) -> pd.DataFrame:
    rows = []
    for metric in metric_names:
        for predictor_kind, predictor in (
            ("relative_advantage", f"{metric}_relative_advantage"),
            ("signed_advantage", f"{metric}_stability_advantage"),
        ):
            if predictor not in df.columns:
                continue
            for model, controls in MODEL_SPECS:
                block = df.dropna(subset=["drift_edge_align_signed", predictor, "session_id", *controls]).copy()
                if block.shape[0] < len(controls) + 8 or block["session_id"].nunique() < 2:
                    continue
                fit = _fit_model(block, predictor, controls)
                boot = _session_bootstrap_model(block, predictor, controls, rng=rng, n_bootstrap=int(n_bootstrap))
                perm = _permute_model(block, predictor, controls, fit["coef_stability"], rng=rng, n_permutations=int(n_permutations))
                rows.append(
                    {
                        "metric": metric,
                        "predictor_kind": predictor_kind,
                        "predictor": predictor,
                        "model": model,
                        "controls": "+".join(controls) if controls else "none",
                        **fit,
                        "ci_low": boot["ci_low"],
                        "ci_high": boot["ci_high"],
                        **perm,
                        "n_bootstrap": int(n_bootstrap),
                        "n_permutations": int(n_permutations),
                    }
                )
    return pd.DataFrame(rows)


def _split_summaries(df: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    rows = []
    splits: list[tuple[str, str, pd.DataFrame]] = []
    q = df["image_orientation_coherence"].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    splits.extend(
        [
            ("coherence", "low", df[df["image_orientation_coherence"] <= q[0]].copy()),
            ("coherence", "middle", df[(df["image_orientation_coherence"] > q[0]) & (df["image_orientation_coherence"] <= q[1])].copy()),
            ("coherence", "high", df[df["image_orientation_coherence"] > q[1]].copy()),
        ]
    )
    cq = df["local_contrast"].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    splits.extend(
        [
            ("contrast", "low", df[df["local_contrast"] <= cq[0]].copy()),
            ("contrast", "middle", df[(df["local_contrast"] > cq[0]) & (df["local_contrast"] <= cq[1])].copy()),
            ("contrast", "high", df[df["local_contrast"] > cq[1]].copy()),
        ]
    )
    if "image_high_freq_power_fraction" in df.columns:
        hq = df["image_high_freq_power_fraction"].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        splits.extend(
            [
                ("high_freq_power", "low", df[df["image_high_freq_power_fraction"] <= hq[0]].copy()),
                ("high_freq_power", "middle", df[(df["image_high_freq_power_fraction"] > hq[0]) & (df["image_high_freq_power_fraction"] <= hq[1])].copy()),
                ("high_freq_power", "high", df[df["image_high_freq_power_fraction"] > hq[1]].copy()),
            ]
        )
    for phase, block in df.groupby("phase", sort=True):
        splits.append(("phase", str(phase), block.copy()))
    for split_family, split_level, block in splits:
        for metric in metric_names:
            for predictor_kind, predictor in (
                ("relative_advantage", f"{metric}_relative_advantage"),
                ("signed_advantage", f"{metric}_stability_advantage"),
            ):
                sub = block.dropna(subset=["drift_edge_align_signed", predictor, "session_id"]).copy()
                if sub.shape[0] < 8 or sub["session_id"].nunique() < 2:
                    continue
                fit = _fit_model(sub, predictor, [])
                rows.append(
                    {
                        "split_family": split_family,
                        "split_level": split_level,
                        "metric": metric,
                        "predictor_kind": predictor_kind,
                        "mean_alignment_session": float(sub.groupby("session_id")["drift_edge_align_signed"].mean().mean()),
                        "mean_advantage_session": float(sub.groupby("session_id")[predictor].mean().mean()),
                        "coef_stability_unadjusted": fit["coef_stability"],
                        "incremental_r2": fit["incremental_r2"],
                        "n_windows": int(sub.shape[0]),
                        "n_sessions": int(sub["session_id"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def _write_plots(out_dir: Path, model_summary: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    focus = model_summary[(model_summary["model"] == "full_low_level") & (model_summary["predictor_kind"] == "signed_advantage")].sort_values("coef_stability")
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=150)
    y = np.arange(focus.shape[0])
    ax.barh(y, focus["coef_stability"], color=["#c16622" if v < 0 else "#4878a8" for v in focus["coef_stability"]])
    ax.errorbar(
        focus["coef_stability"],
        y,
        xerr=[focus["coef_stability"] - focus["ci_low"], focus["ci_high"] - focus["coef_stability"]],
        fmt="none",
        color="black",
        linewidth=0.8,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(focus["metric"], fontsize=7)
    ax.set_xlabel("controlled slope: alignment ~ stability advantage")
    fig.tight_layout()
    fig.savefig(out_dir / "twin_metric_controlled_slopes.png", dpi=150)
    plt.close(fig)

    for split_family in ("coherence", "contrast", "phase"):
        block = split_summary[
            (split_summary["split_family"] == split_family)
            & (split_summary["predictor_kind"] == "signed_advantage")
            & split_summary["metric"].isin(["raw_mse", "response_norm_mse", "per_rate_mse", "full_cov_whitened_mse"])
        ].copy()
        if block.empty:
            continue
        levels = list(dict.fromkeys(block["split_level"].tolist()))
        metrics = list(dict.fromkeys(block["metric"].tolist()))
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        x = np.arange(len(levels))
        for metric in metrics:
            vals = [float(block[(block["split_level"] == level) & (block["metric"] == metric)]["coef_stability_unadjusted"].iloc[0]) if not block[(block["split_level"] == level) & (block["metric"] == metric)].empty else np.nan for level in levels]
            ax.plot(x, vals, marker="o", label=metric)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylabel("unadjusted within-session slope")
        ax.set_title(split_family, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"twin_metric_slopes_by_{split_family}.png", dpi=150)
        plt.close(fig)


def _write_decision(out_dir: Path, model_summary: pd.DataFrame, split_summary: pd.DataFrame, merged: pd.DataFrame) -> None:
    full = model_summary[(model_summary["model"] == "full_low_level") & (model_summary["predictor_kind"] == "signed_advantage")].copy()
    stable_positive = full[full["ci_low"] > 0]
    robust_negative = full[full["ci_high"] < 0]
    relative_full = model_summary[(model_summary["model"] == "full_low_level") & (model_summary["predictor_kind"] == "relative_advantage")].copy()
    lines = [
        "# BackImage Twin Stability Metric Audit",
        "",
        "## Summary",
        "",
        f"- Metrics tested: `{full.shape[0]}` twin disruption variants.",
        f"- Primary predictor: signed normalized advantage, `orthogonal_cost - parallel_cost`, standardized in the within-session model.",
        f"- Full-control positive signed-advantage metrics: `{', '.join(stable_positive['metric'].tolist()) if not stable_positive.empty else 'none'}`.",
        f"- Full-control negative signed-advantage metrics: `{', '.join(robust_negative['metric'].tolist()) if not robust_negative.empty else 'none'}`.",
        f"- Mean session drift-edge alignment: `{merged.groupby('session_id')['drift_edge_align_signed'].mean().mean():+.4f}`.",
        "",
        "## Full-Control Signed-Advantage Slopes",
        "",
    ]
    for _, row in full.sort_values("coef_stability").iterrows():
        lines.append(
            f"- `{row['metric']}`: coef `{row['coef_stability']:+.4f}` CI "
            f"`[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]`, "
            f"dR2 `{row['incremental_r2']:+.4f}`, p2 `{row['perm_p_two_sided']:.4f}`."
        )
    lines.extend(["", "## Full-Control Relative-Advantage Slopes", ""])
    for _, row in relative_full.sort_values("coef_stability").iterrows():
        lines.append(
            f"- `{row['metric']}`: coef `{row['coef_stability']:+.4f}` CI "
            f"`[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]`, "
            f"dR2 `{row['incremental_r2']:+.4f}`, p2 `{row['perm_p_two_sided']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Relative advantage is invariant to scalar per-window normalizations, so response-norm checks should be read primarily from the signed-advantage block. If the negative slope disappears under response-norm, per-rate, or covariance-normalized signed metrics, the original twin result is likely sensitive to metric scale. If it remains negative across normalized variants, the wrong-direction concern is more intrinsic to the sampled twin endpoint geometry.",
            "",
        ]
    )
    (out_dir / "decision_table.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    source_rows, metadata = _load_source_rows(Path(args.input), Path(args.stability_dir))
    endpoints = _compute_or_load_endpoint_cache(source_rows, metadata, out_dir, args)
    metric_long = _metric_costs(endpoints)
    metric_long.to_csv(out_dir / "twin_stability_metric_long.csv", index=False)
    metric_wide = _wide_metrics(metric_long)
    merged = source_rows.merge(metric_wide, on="window_row", how="left", validate="one_to_one")
    merged.to_csv(out_dir / "twin_stability_metric_by_window.csv", index=False)
    metric_names = sorted(metric_long["metric"].unique())
    model_summary = _run_metric_models(
        merged,
        metric_names,
        rng=rng,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
    )
    model_summary.to_csv(out_dir / "twin_stability_metric_model_summary.csv", index=False)
    split_summary = _split_summaries(merged, metric_names)
    split_summary.to_csv(out_dir / "twin_stability_metric_split_summary.csv", index=False)
    _write_plots(out_dir, model_summary, split_summary)
    _write_decision(out_dir, model_summary, split_summary, merged)
    (out_dir / "posthoc_metadata.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "stability_dir": str(args.stability_dir),
                "source_stability_metadata": metadata,
                "n_bootstrap": int(args.n_bootstrap),
                "n_permutations": int(args.n_permutations),
                "seed": int(args.seed),
                "notes": "Positive relative_advantage means orthogonal endpoint responses are more disruptive than parallel endpoint responses.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote BackImage twin stability metric audit to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

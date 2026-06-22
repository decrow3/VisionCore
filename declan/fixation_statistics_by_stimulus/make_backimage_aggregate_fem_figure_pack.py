"""Build figure-ready summaries for the n256 aggregate BackImage FEM result.

This script is cache-first: it reads the completed aggregate run and a
configured incremental posthoc folder, then writes a compact figure package for
collaborator review without rerunning the V1 twin or the decoders.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_RUN_DIR = (
    BASE
    / "backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched"
)
DEFAULT_INC_DIRNAME: str | Path = "incremental_static_plus_motion_relids"
DEFAULT_OUT_DIRNAME = "figure_pack_v1"

EXPECTED_SCALES = ["rel_0p25x", "rel_0p5x", "rel_1x", "rel_1p5x", "rel_2x"]
FAMILY_ORDER = ["empirical", "ou", "brownian", "rotated"]
CONTROL_ORDER = ["ou", "brownian", "rotated"]
PRIMARY_SUMMARY = "delta_mean"
PRIMARY_GAIN_ROWS = [
    ("gabor_local_field", 4),
    ("pyramid_local_field", 8),
]
PRIMARY_CONTRAST_LATENT = "gabor_local_field"
PRIMARY_CONTRAST_K = 4

COLORS = {
    "empirical": "#2c5c8a",
    "ou": "#c46a2b",
    "brownian": "#6f7378",
    "rotated": "#7b5fa8",
    "gabor_local_field": "#2c5c8a",
    "pyramid_local_field": "#2d8a68",
    "ratio": "#2c5c8a",
    "overlap": "#b24b3f",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scale_value(scale_id: str) -> float:
    if scale_id == "static":
        return 0.0
    return float(scale_id.replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(scale_id: str) -> str:
    if scale_id == "static":
        return "static"
    return f"{_scale_value(scale_id):g}x"


def _with_scale_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scale_value"] = out["scale_id"].map(_scale_value)
    out["scale_label"] = out["scale_id"].map(_scale_label)
    return out.sort_values(["scale_value"]).reset_index(drop=True)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9dde2", linewidth=0.6, alpha=0.8)


def _label_panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png", dpi=220)
    fig.savefig(figures_dir / f"{stem}.pdf")
    plt.close(fig)


def _parse_csv(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def _parse_primary_gain_rows(raw: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for item in _parse_csv(raw):
        if ":" not in item:
            raise ValueError(
                f"Invalid primary gain row {item!r}; expected LATENT:K, "
                "for example pyramid_local_field:16."
            )
        latent, k_text = item.split(":", 1)
        latent = latent.strip()
        if not latent:
            raise ValueError(f"Invalid primary gain row {item!r}; latent is empty.")
        rows.append((latent, int(k_text.strip())))
    return rows


def _configure_from_args(args: argparse.Namespace) -> None:
    global DEFAULT_INC_DIRNAME
    global EXPECTED_SCALES
    global PRIMARY_SUMMARY
    global PRIMARY_GAIN_ROWS
    global PRIMARY_CONTRAST_LATENT
    global PRIMARY_CONTRAST_K

    if args.incremental_dir is not None:
        DEFAULT_INC_DIRNAME = Path(args.incremental_dir)
    EXPECTED_SCALES = _parse_csv(args.expected_scales)
    PRIMARY_SUMMARY = str(args.primary_summary)
    PRIMARY_GAIN_ROWS = _parse_primary_gain_rows(args.primary_gain_rows)
    PRIMARY_CONTRAST_LATENT = str(args.primary_contrast_latent)
    PRIMARY_CONTRAST_K = int(args.primary_contrast_k)


def _primary_readout_detail() -> str:
    return f"{PRIMARY_CONTRAST_LATENT} k={PRIMARY_CONTRAST_K} {PRIMARY_SUMMARY}"


def _latent_color(latent: str) -> str:
    return COLORS.get(latent, "#2c5c8a")


def _incremental_label() -> str:
    return str(DEFAULT_INC_DIRNAME)


def _resolve_incremental_dir(run_dir: Path) -> Path:
    inc = Path(DEFAULT_INC_DIRNAME)
    if inc.is_absolute() or inc.parent != Path("."):
        return inc
    return run_dir / inc


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(col) for col in df.columns]
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _errbar_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    y_col: str,
    lo_col: str,
    hi_col: str,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    rows = _with_scale_columns(df)
    x = rows["scale_value"].to_numpy(dtype=float)
    y = rows[y_col].to_numpy(dtype=float)
    lo = rows[lo_col].to_numpy(dtype=float)
    hi = rows[hi_col].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color=color,
        marker=marker,
        linewidth=1.8,
        capsize=3,
        label=label,
    )


def _validate_incremental_folder(
    gain: pd.DataFrame,
    contrasts: pd.DataFrame,
    inc_decode: pd.DataFrame,
    inc_meta: dict[str, Any],
) -> None:
    for name, df in {
        "incremental_gain_vs_static.csv": gain,
        "incremental_gain_contrasts.csv": contrasts,
    }.items():
        scales = sorted(df["scale_id"].astype(str).unique(), key=_scale_value)
        if scales != EXPECTED_SCALES:
            raise ValueError(
                f"{name} has scale_id values {scales}; expected {EXPECTED_SCALES}. "
                "Use the configured repaired incremental posthoc folder."
            )
        if sorted(df["n_images"].dropna().astype(int).unique()) != [256]:
            raise ValueError(f"{name} must have n_images == 256 in every row.")
        if sorted(df["n_sessions"].dropna().astype(int).unique()) != [29]:
            raise ValueError(f"{name} must have n_sessions == 29 in every row.")
    decode_images = sorted(inc_decode["n_images"].dropna().astype(int).unique())
    if decode_images != [256]:
        raise ValueError(f"incremental_decode_summary.csv must have n_images == 256; got {decode_images}.")
    if inc_meta.get("ridge_alpha_mode") != "fixed":
        raise ValueError("Incremental relids metadata must report ridge_alpha_mode == fixed.")
    if float(inc_meta.get("fixed_ridge_alpha", float("nan"))) != 10.0:
        raise ValueError("Incremental relids metadata must report fixed_ridge_alpha == 10.0.")


def _validate_primary_tables(
    motion_qc: pd.DataFrame,
    gain_curve: pd.DataFrame,
    contrast_curve: pd.DataFrame,
    cov_panel: pd.DataFrame,
) -> None:
    expected = {
        "motion_qc_table": (motion_qc, len(FAMILY_ORDER) * len(EXPECTED_SCALES)),
        "empirical_gain_curve": (gain_curve, len(PRIMARY_GAIN_ROWS) * len(EXPECTED_SCALES)),
        "empirical_control_contrasts": (contrast_curve, len(CONTROL_ORDER) * len(EXPECTED_SCALES)),
        "covariance_panel_table": (cov_panel, len(FAMILY_ORDER) * len(EXPECTED_SCALES)),
    }
    for name, (df, n_rows) in expected.items():
        if int(df.shape[0]) != int(n_rows):
            raise ValueError(f"{name} has {df.shape[0]} rows; expected {n_rows}.")
        scale_ids = sorted(df["scale_id"].astype(str).unique(), key=_scale_value)
        if scale_ids != EXPECTED_SCALES:
            raise ValueError(f"{name} has scale_id values {scale_ids}; expected {EXPECTED_SCALES}.")


def _validate_fixed_alpha_tables(decode: pd.DataFrame, inc_decode: pd.DataFrame, inc_meta: dict[str, Any]) -> None:
    modes = sorted(decode["ridge_alpha_mode"].astype(str).unique())
    if modes != ["fixed"]:
        raise ValueError(f"decode_summary.csv ridge_alpha_mode must be ['fixed']; got {modes}.")
    fixed_alphas = sorted(decode["fixed_ridge_alpha"].dropna().astype(float).unique())
    if fixed_alphas != [10.0]:
        raise ValueError(f"decode_summary.csv fixed_ridge_alpha must be [10.0]; got {fixed_alphas}.")
    chosen_alphas = sorted(decode["chosen_alpha_median"].dropna().astype(float).unique())
    if chosen_alphas != [10.0]:
        raise ValueError(f"decode_summary.csv chosen_alpha_median must be [10.0]; got {chosen_alphas}.")

    if str(inc_meta.get("ridge_alpha_mode")) != "fixed":
        raise ValueError("incremental relids metadata ridge_alpha_mode must be fixed.")
    if float(inc_meta.get("fixed_ridge_alpha", float("nan"))) != 10.0:
        raise ValueError("incremental relids metadata fixed_ridge_alpha must be 10.0.")
    inc_chosen_alphas = sorted(inc_decode["chosen_alpha_median"].dropna().astype(float).unique())
    if inc_chosen_alphas != [10.0]:
        raise ValueError(
            f"incremental_decode_summary.csv chosen_alpha_median must be [10.0]; got {inc_chosen_alphas}."
        )


def _primary_gain_table(gain: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for latent, k in PRIMARY_GAIN_ROWS:
        block = gain[
            (gain["motion_summary"] == PRIMARY_SUMMARY)
            & (gain["family"] == "empirical")
            & (gain["latent"] == latent)
            & (gain["k"].astype(int) == int(k))
        ].copy()
        block["primary_label"] = f"{latent.replace('_local_field', '')} k={k}"
        block["primary_latent"] = latent
        block["primary_k"] = int(k)
        keep.append(block)
    out = pd.concat(keep, ignore_index=True)
    return _with_scale_columns(out)


def _primary_contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    out = contrasts[
        (contrasts["motion_summary"] == PRIMARY_SUMMARY)
        & (contrasts["lhs_family"] == "empirical")
        & (contrasts["rhs_family"].isin(CONTROL_ORDER))
        & (contrasts["latent"] == PRIMARY_CONTRAST_LATENT)
        & (contrasts["k"].astype(int) == PRIMARY_CONTRAST_K)
    ].copy()
    out["rhs_family"] = pd.Categorical(out["rhs_family"], CONTROL_ORDER, ordered=True)
    out = _with_scale_columns(out)
    return out.sort_values(["rhs_family", "scale_value"]).reset_index(drop=True)


def _covariance_panel_table(cov: pd.DataFrame) -> pd.DataFrame:
    out = cov[
        (cov["summary"] == PRIMARY_SUMMARY)
        & (cov["family"].isin(FAMILY_ORDER))
        & (cov["scale_id"].isin(EXPECTED_SCALES))
    ].copy()
    out["family"] = pd.Categorical(out["family"], FAMILY_ORDER, ordered=True)
    out = _with_scale_columns(out)
    return out.sort_values(["family", "scale_value"]).reset_index(drop=True)


def _caption_numbers(
    motion: pd.DataFrame,
    motion_meta: pd.DataFrame,
    gain: pd.DataFrame,
    contrasts: pd.DataFrame,
    trace_bank: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nonstatic_meta = motion_meta[motion_meta["family"].astype(str) != "static"]
    rows.append(
        {
            "section": "motion_qc",
            "metric": "accepted_drift_only_trace_sources",
            "value": int(nonstatic_meta["trace_source_row"].dropna().nunique()),
            "detail": f"unique non-static trace_source_row values out of {int(trace_bank.shape[0])} trace-bank rows",
        }
    )
    rows.append(
        {
            "section": "motion_qc",
            "metric": "trace_bank_rows",
            "value": int(trace_bank.shape[0]),
            "detail": "trace_bank_metadata rows",
        }
    )
    rows.append(
        {
            "section": "motion_qc",
            "metric": "rows_per_family_scale",
            "value": int(motion["n"].dropna().astype(int).min()),
            "detail": "minimum aggregate_motion_summary n",
        }
    )
    rows.append(
        {
            "section": "motion_qc",
            "metric": "max_clipped_fraction",
            "value": float(motion["clipped_fraction"].max()),
            "detail": "all families and scales",
        }
    )
    rows.append(
        {
            "section": "motion_qc",
            "metric": "median_effective_to_requested_rms_min",
            "value": float(motion["median_effective_to_requested_rms"].min()),
            "detail": "all families and scales",
        }
    )
    rows.append(
        {
            "section": "motion_qc",
            "metric": "median_effective_to_requested_rms_max",
            "value": float(motion["median_effective_to_requested_rms"].max()),
            "detail": "all families and scales",
        }
    )
    for _, row in gain.iterrows():
        rows.append(
            {
                "section": "empirical_gain",
                "metric": f"{row['primary_label']} {row['scale_label']}",
                "value": float(row["incremental_gain_neg_mse"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "detail": "empirical static-plus-motion minus static",
            }
        )
    for _, row in contrasts.iterrows():
        rows.append(
            {
                "section": "empirical_control_contrast",
                "metric": f"empirical_vs_{row['rhs_family']} {row['scale_label']}",
                "value": float(row["incremental_gain_delta_neg_mse"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "detail": _primary_readout_detail(),
            }
        )
    return pd.DataFrame(rows)


def _make_motion_qc_figure(motion_qc: pd.DataFrame, figures_dir: Path) -> None:
    motion = _with_scale_columns(motion_qc)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True)
    metrics = [
        ("median_effective_to_requested_rms", "effective / requested RMS"),
        ("clipped_fraction", "clipped fraction"),
        ("median_path_length_deg", "median path length (deg)"),
        ("median_speed_mean_deg_s", "mean speed (deg/s)"),
    ]
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for family in FAMILY_ORDER:
            block = motion[motion["family"] == family]
            ax.plot(
                block["scale_value"],
                block[metric],
                marker="o",
                linewidth=1.8,
                color=COLORS[family],
                label=family,
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
        _clean_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0, 0].set_title("RMS bookkeeping")
    axes[0, 1].set_title("No clipping")
    axes[1, 0].set_title("Path length")
    axes[1, 1].set_title("Speed")
    fig.suptitle("Aggregate motion quality control", x=0.02, ha="left", fontsize=13, fontweight="bold")
    _save(fig, figures_dir, "aggregate_motion_qc")


def _make_gain_figure(gain_curve: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for label, block in gain_curve.groupby("primary_label", sort=False):
        latent = str(block["primary_latent"].iloc[0])
        _errbar_line(
            ax,
            block,
            y_col="incremental_gain_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=label,
            color=_latent_color(latent),
        )
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("incremental gain in neg-MSE")
    ax.set_title("Empirical motion adds feature-decodable signal beyond static", loc="left")
    ax.legend(frameon=False)
    _clean_axis(ax)
    _save(fig, figures_dir, "empirical_static_plus_motion_gain")


def _make_contrast_figure(contrast_curve: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    markers = {"ou": "o", "brownian": "s", "rotated": "^"}
    for control in CONTROL_ORDER:
        block = contrast_curve[contrast_curve["rhs_family"].astype(str) == control]
        _errbar_line(
            ax,
            block,
            y_col="incremental_gain_delta_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=f"vs {control}",
            color=COLORS[control],
            marker=markers[control],
        )
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("empirical minus control gain")
    ax.set_title("Empirical advantage over controls narrows at larger scales", loc="left")
    ax.legend(frameon=False)
    _clean_axis(ax)
    _save(fig, figures_dir, "empirical_vs_controls_scale_curve")


def _make_covariance_figure(cov_panel: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True)
    for family in FAMILY_ORDER:
        block = cov_panel[cov_panel["family"].astype(str) == family]
        axes[0].plot(
            block["scale_value"],
            block["signal_motion_trace_ratio"],
            marker="o",
            linewidth=1.8,
            color=COLORS[family],
            label=family,
        )
        axes[1].plot(
            block["scale_value"],
            block["signal_motion_subspace_overlap"],
            marker="o",
            linewidth=1.8,
            color=COLORS[family],
            label=family,
        )
    for ax in axes:
        ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
        ax.set_xlabel("motion scale")
        _clean_axis(ax)
    axes[0].set_ylabel("signal / motion covariance trace")
    axes[1].set_ylabel("signal-motion subspace overlap")
    axes[0].set_title("Signal-motion trace ratio", loc="left")
    axes[1].set_title("Subspace overlap", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Response covariance context", x=0.02, ha="left", fontsize=13, fontweight="bold")
    _save(fig, figures_dir, "signal_motion_covariance_summary")


def _draw_motion_panel(ax: plt.Axes, motion_qc: pd.DataFrame) -> None:
    motion = _with_scale_columns(motion_qc)
    for family in FAMILY_ORDER:
        block = motion[motion["family"] == family]
        ax.plot(block["scale_value"], block["median_effective_to_requested_rms"], marker="o", linewidth=1.6, color=COLORS[family], label=family)
    ax.axhline(1.0, color="#222222", linewidth=0.8)
    ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
    ax.set_ylabel("effective / requested RMS")
    ax.set_title("Motion QC", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _clean_axis(ax)


def _draw_gain_panel(ax: plt.Axes, gain_curve: pd.DataFrame) -> None:
    for label, block in gain_curve.groupby("primary_label", sort=False):
        latent = str(block["primary_latent"].iloc[0])
        _errbar_line(
            ax,
            block,
            y_col="incremental_gain_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=label,
            color=_latent_color(latent),
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
    ax.set_ylabel("gain vs static")
    ax.set_title("Static-plus-motion gain", loc="left")
    ax.legend(frameon=False, fontsize=8)
    _clean_axis(ax)


def _draw_contrast_panel(ax: plt.Axes, contrast_curve: pd.DataFrame) -> None:
    for control in CONTROL_ORDER:
        block = contrast_curve[contrast_curve["rhs_family"].astype(str) == control]
        _errbar_line(
            ax,
            block,
            y_col="incremental_gain_delta_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=f"vs {control}",
            color=COLORS[control],
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
    ax.set_ylabel("empirical - control")
    ax.set_title("Empirical versus controls", loc="left")
    ax.legend(frameon=False, fontsize=8)
    _clean_axis(ax)


def _draw_cov_panel(ax: plt.Axes, cov_panel: pd.DataFrame) -> None:
    for family in FAMILY_ORDER:
        block = cov_panel[cov_panel["family"].astype(str) == family]
        ax.plot(block["scale_value"], block["signal_motion_trace_ratio"], marker="o", linewidth=1.5, color=COLORS[family], label=family)
    ax.set_xticks([_scale_value(s) for s in EXPECTED_SCALES], [_scale_label(s) for s in EXPECTED_SCALES])
    ax.set_ylabel("signal / motion covariance")
    ax.set_title("Covariance context", loc="left")
    _clean_axis(ax)


def _make_multipanel(
    motion_qc: pd.DataFrame,
    gain_curve: pd.DataFrame,
    contrast_curve: pd.DataFrame,
    cov_panel: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    draw_calls = [
        ("A", _draw_motion_panel, motion_qc),
        ("B", _draw_gain_panel, gain_curve),
        ("C", _draw_contrast_panel, contrast_curve),
        ("D", _draw_cov_panel, cov_panel),
    ]
    for ax, (label, fn, data) in zip(axes.flat, draw_calls):
        _label_panel(ax, label)
        fn(ax, data)
    fig.suptitle("Aggregate natural-image FEM information summary", x=0.02, ha="left", fontsize=14, fontweight="bold")
    _save(fig, figures_dir, "aggregate_fem_summary_multipanel")


def _fixed_alpha_summary(run_dir: Path, decode: pd.DataFrame, inc_decode: pd.DataFrame, run_meta: dict[str, Any], inc_meta: dict[str, Any]) -> pd.DataFrame:
    config = run_meta.get("config", {})
    alphas = config.get("ridge_alphas", [])
    main_fixed_alpha = config.get("fixed_ridge_alpha")
    inc_dir = _resolve_incremental_dir(run_dir)
    if main_fixed_alpha is None and alphas:
        main_fixed_alpha = float(alphas[len(alphas) // 2])

    rows = [
        {
            "check": "aggregate_decode_summary",
            "source_file": str(run_dir / "decode_summary.csv"),
            "ridge_alpha_mode": ",".join(sorted(decode["ridge_alpha_mode"].astype(str).unique())),
            "fixed_ridge_alpha": float(decode["fixed_ridge_alpha"].dropna().median()),
            "chosen_alpha_min": float(decode["chosen_alpha_median"].min()),
            "chosen_alpha_median": float(decode["chosen_alpha_median"].median()),
            "chosen_alpha_max": float(decode["chosen_alpha_median"].max()),
            "n_rows": int(decode.shape[0]),
            "conclusion": "main aggregate decode table used a fixed/shared ridge alpha",
        },
        {
            "check": "incremental_relids_decode_summary",
            "source_file": str(inc_dir / "incremental_decode_summary.csv"),
            "ridge_alpha_mode": str(inc_meta.get("ridge_alpha_mode")),
            "fixed_ridge_alpha": float(inc_meta.get("fixed_ridge_alpha", float("nan"))),
            "chosen_alpha_min": float(inc_decode["chosen_alpha_median"].min()),
            "chosen_alpha_median": float(inc_decode["chosen_alpha_median"].median()),
            "chosen_alpha_max": float(inc_decode["chosen_alpha_median"].max()),
            "n_rows": int(inc_decode.shape[0]),
            "conclusion": "incremental static-plus-motion posthoc used a fixed/shared ridge alpha",
            "metadata_fixed_alpha": float(inc_meta.get("fixed_ridge_alpha", float("nan"))),
        },
    ]
    if main_fixed_alpha is not None:
        rows[0]["metadata_fixed_alpha"] = float(main_fixed_alpha)
    return pd.DataFrame(rows)


def _resampling_summary(gain: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_gain_blocks = []
    for latent, k in PRIMARY_GAIN_ROWS:
        primary_gain_blocks.append(
            gain[
                (gain["motion_summary"] == PRIMARY_SUMMARY)
                & (gain["family"] == "empirical")
                & (gain["latent"] == latent)
                & (gain["k"].astype(int) == int(k))
            ].copy()
        )
    primary_gain = pd.concat(primary_gain_blocks, ignore_index=True)
    primary_contrasts = _primary_contrast_table(contrasts)
    for _, row in _with_scale_columns(primary_gain).iterrows():
        estimate = float(row["incremental_gain_neg_mse"])
        rows.append(
            {
                "check": "session_bootstrap_from_incremental_gain_vs_static",
                "motion_summary": row["motion_summary"],
                "latent": row["latent"],
                "k": int(row["k"]),
                "family_or_contrast": row["family"],
                "scale_id": row["scale_id"],
                "estimate": estimate,
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "n_images": int(row["n_images"]),
                "n_sessions": int(row["n_sessions"]),
                "sign_positive": bool(estimate > 0),
                "ci_excludes_zero": bool(float(row["ci95_low"]) > 0 or float(row["ci95_high"]) < 0),
                "sign_supported_by_ci": bool(estimate > 0 and float(row["ci95_low"]) > 0),
            }
        )
    for _, row in primary_contrasts.iterrows():
        estimate = float(row["incremental_gain_delta_neg_mse"])
        rows.append(
            {
                "check": "session_bootstrap_from_incremental_gain_contrasts",
                "motion_summary": row["motion_summary"],
                "latent": row["latent"],
                "k": int(row["k"]),
                "family_or_contrast": f"{row['lhs_family']}_minus_{row['rhs_family']}",
                "scale_id": row["scale_id"],
                "estimate": estimate,
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "n_images": int(row["n_images"]),
                "n_sessions": int(row["n_sessions"]),
                "sign_positive": bool(estimate > 0),
                "ci_excludes_zero": bool(float(row["ci95_low"]) > 0 or float(row["ci95_high"]) < 0),
                "sign_supported_by_ci": bool(estimate > 0 and float(row["ci95_low"]) > 0),
            }
        )
    return pd.DataFrame(rows)


def _panel_provenance(run_dir: Path, out_dir: Path) -> pd.DataFrame:
    inc = _resolve_incremental_dir(run_dir)
    rows = [
        {
            "panel": "A",
            "claim": "Motion bookkeeping is clean across families and scales.",
            "script": "declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py",
            "output_folder": str(out_dir),
            "source_files_read": "aggregate_motion_summary.csv; aggregate_motion_metadata.csv; trace_bank_metadata.csv",
            "main_metric": "effective/requested RMS, clipping fraction, path length, speed",
            "sample_size": "256 images; 4 trace samples per family/scale/image",
            "unit_space": "canonical 756-unit V1 twin",
            "known_caveats": "Brownian path length/speed differs by construction; this is a QC panel, not a biological optimality claim.",
        },
        {
            "panel": "B",
            "claim": "Empirical motion adds feature-decodable signal beyond static responses.",
            "script": "declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py",
            "output_folder": str(out_dir),
            "source_files_read": str(inc / "incremental_gain_vs_static.csv"),
            "main_metric": "incremental gain in negative MSE",
            "sample_size": "256 images; 29 sessions for bootstrap grouping",
            "unit_space": "canonical 756-unit V1 twin",
            "known_caveats": "Deterministic twin result; not recorded V1 evidence.",
        },
        {
            "panel": "C",
            "claim": "Empirical gain robustly beats OU and is strongest versus Brownian/rotated at small scales.",
            "script": "declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py",
            "output_folder": str(out_dir),
            "source_files_read": str(inc / "incremental_gain_contrasts.csv"),
            "main_metric": "empirical minus control incremental gain",
            "sample_size": "256 images; 29 sessions for bootstrap grouping",
            "unit_space": "canonical 756-unit V1 twin",
            "known_caveats": "Brownian and rotated advantages narrow around 1x-2x.",
        },
        {
            "panel": "D",
            "claim": "Decode result is contextualized by signal and motion covariance structure.",
            "script": "declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py",
            "output_folder": str(out_dir),
            "source_files_read": "covariance_summary.csv",
            "main_metric": "signal/motion covariance trace ratio and subspace overlap",
            "sample_size": "256 images; 4 trace samples per family/scale/image",
            "unit_space": "canonical 756-unit V1 twin",
            "known_caveats": "Covariance panel is descriptive and should not be framed as a biological temporal-code proof.",
        },
    ]
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    run_dir: Path,
    motion_qc: pd.DataFrame,
    gain_curve: pd.DataFrame,
    contrast_curve: pd.DataFrame,
    fixed_alpha: pd.DataFrame,
    resampling: pd.DataFrame,
) -> None:
    clipped_max = float(motion_qc["clipped_fraction"].max())
    rms_min = float(motion_qc["median_effective_to_requested_rms"].min())
    rms_max = float(motion_qc["median_effective_to_requested_rms"].max())
    motion_meta = _read_csv(run_dir / "aggregate_motion_metadata.csv")
    accepted_sources = int(
        motion_meta[motion_meta["family"].astype(str) != "static"]["trace_source_row"].dropna().nunique()
    )
    trace_bank_rows = int(_read_csv(run_dir / "trace_bank_metadata.csv").shape[0])
    gain_all_positive = bool((gain_curve["ci95_low"] > 0).all())
    ou_all_positive = bool(
        (
            contrast_curve[contrast_curve["rhs_family"].astype(str) == "ou"]["ci95_low"].astype(float)
            > 0
        ).all()
    )
    brownian_small_positive = bool(
        (
            contrast_curve[
                (contrast_curve["rhs_family"].astype(str) == "brownian")
                & (contrast_curve["scale_id"].isin(["rel_0p25x", "rel_0p5x"]))
            ]["ci95_low"].astype(float)
            > 0
        ).all()
    )
    report = [
        "# Aggregate FEM Figure Pack",
        "",
        f"Source run: `{run_dir}`",
        f"Output folder: `{out_dir}`",
        "",
        "## Guardrails",
        "",
        f"- Uses incremental posthoc folder: `{_incremental_label()}`.",
        f"- Validated `scale_id` values: `{', '.join(EXPECTED_SCALES)}`.",
        "- Validated `n_images == 256` and `n_sessions == 29` for incremental gain and contrast tables.",
        "- Claim boundary: empirical FEM statistics improve a V1-twin representation of natural-image structure under this readout.",
        "",
        "## Main Takeaways",
        "",
        f"- Motion QC is clean: {accepted_sources} / {trace_bank_rows} drift-only trace sources used, max clipping fraction {clipped_max:.3g}, and effective/requested RMS range {rms_min:.3g}-{rms_max:.3g}.",
        f"- Empirical static-plus-motion gains are positive with session-bootstrap CIs above zero for the primary curves: `{gain_all_positive}`.",
        f"- Empirical-minus-OU contrasts are positive with CIs above zero across scales: `{ou_all_positive}`.",
        f"- Empirical-minus-Brownian is supported at 0.25x and 0.5x, then narrows at larger scales: `{brownian_small_positive}`.",
        "",
        "## Figure Files",
        "",
        "- `figures/aggregate_motion_qc.png` and `.pdf`",
        "- `figures/empirical_static_plus_motion_gain.png` and `.pdf`",
        "- `figures/empirical_vs_controls_scale_curve.png` and `.pdf`",
        "- `figures/signal_motion_covariance_summary.png` and `.pdf`",
        "- `figures/aggregate_fem_summary_multipanel.png` and `.pdf`",
        "",
        "## Robustness",
        "",
        "- Fixed-alpha audit: the main aggregate and incremental decoder summaries use fixed/shared ridge alpha.",
        "- Resampling audit: this package summarizes the existing session-bootstrap CIs from the configured incremental tables; no new twin or decoder rerun was performed.",
        "",
        "## Key Source Tables",
        "",
        "- `figure_source_tables/motion_qc_table.csv`",
        "- `figure_source_tables/empirical_gain_curve.csv`",
        "- `figure_source_tables/empirical_control_contrasts.csv`",
        "- `figure_source_tables/covariance_panel_table.csv`",
        "- `figure_source_tables/key_numbers_for_caption.csv`",
        "- `panel_provenance.csv`",
    ]
    (out_dir / "figure_pack_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    fixed_alpha_lines = _markdown_table(fixed_alpha)
    resampling_supported = resampling.groupby("check")["sign_supported_by_ci"].agg(["sum", "count"]).reset_index()
    robustness_report = [
        "# Aggregate FEM Robustness Report",
        "",
        "## Fixed/Shared Alpha Sensitivity",
        "",
        "The existing decode outputs already use fixed/shared ridge regularization; no family-specific ridge retuning is present in these summaries.",
        "",
        fixed_alpha_lines,
        "",
        "## Session Bootstrap / Source Resampling Audit",
        "",
        "The configured incremental posthoc bootstrapped image-level score deltas by session. This report collects the primary rows and records whether each 95% CI supports a positive sign.",
        "",
        _markdown_table(resampling_supported),
        "",
        "No new broad forward run was launched.",
    ]
    (out_dir / "robustness" / "robustness_report.md").write_text("\n".join(robustness_report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--incremental-dir",
        type=Path,
        default=None,
        help=(
            "Incremental posthoc directory. A bare name is resolved under RUN_DIR; "
            "an absolute path or relative path with parent components is used as given. "
            "Defaults to RUN_DIR/incremental_static_plus_motion_relids."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--expected-scales", default=",".join(EXPECTED_SCALES))
    parser.add_argument("--primary-summary", default=PRIMARY_SUMMARY)
    parser.add_argument(
        "--primary-gain-rows",
        default=",".join(f"{latent}:{k}" for latent, k in PRIMARY_GAIN_ROWS),
        help="Comma-separated LATENT:K rows to plot in the static-plus-motion gain panel.",
    )
    parser.add_argument("--primary-contrast-latent", default=PRIMARY_CONTRAST_LATENT)
    parser.add_argument("--primary-contrast-k", type=int, default=PRIMARY_CONTRAST_K)
    return parser


def run(args: argparse.Namespace) -> Path:
    _configure_from_args(args)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / DEFAULT_OUT_DIRNAME
    source_dir = out_dir / "figure_source_tables"
    figures_dir = out_dir / "figures"
    robustness_dir = out_dir / "robustness"
    source_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    robustness_dir.mkdir(parents=True, exist_ok=True)

    inc_dir = _resolve_incremental_dir(run_dir)
    motion = _read_csv(run_dir / "aggregate_motion_summary.csv")
    motion_meta = _read_csv(run_dir / "aggregate_motion_metadata.csv")
    trace_bank = _read_csv(run_dir / "trace_bank_metadata.csv")
    decode = _read_csv(run_dir / "decode_summary.csv")
    cov = _read_csv(run_dir / "covariance_summary.csv")
    gain = _read_csv(inc_dir / "incremental_gain_vs_static.csv")
    contrasts = _read_csv(inc_dir / "incremental_gain_contrasts.csv")
    inc_decode = _read_csv(inc_dir / "incremental_decode_summary.csv")
    run_meta = _read_json(run_dir / "run_metadata.json")
    inc_meta = _read_json(inc_dir / "run_metadata.json")

    _validate_incremental_folder(gain, contrasts, inc_decode, inc_meta)

    motion_qc = motion[motion["family"].isin(FAMILY_ORDER) & motion["scale_id"].isin(EXPECTED_SCALES)].copy()
    motion_qc["family"] = pd.Categorical(motion_qc["family"], FAMILY_ORDER, ordered=True)
    motion_qc = _with_scale_columns(motion_qc).sort_values(["family", "scale_value"]).reset_index(drop=True)
    gain_curve = _primary_gain_table(gain)
    contrast_curve = _primary_contrast_table(contrasts)
    cov_panel = _covariance_panel_table(cov)
    _validate_primary_tables(motion_qc, gain_curve, contrast_curve, cov_panel)
    _validate_fixed_alpha_tables(decode, inc_decode, inc_meta)
    key_numbers = _caption_numbers(motion_qc, motion_meta, gain_curve, contrast_curve, trace_bank)
    fixed_alpha = _fixed_alpha_summary(run_dir, decode, inc_decode, run_meta, inc_meta)
    resampling = _resampling_summary(gain, contrasts)
    provenance = _panel_provenance(run_dir, out_dir)

    motion_qc.to_csv(source_dir / "motion_qc_table.csv", index=False)
    gain_curve.to_csv(source_dir / "empirical_gain_curve.csv", index=False)
    contrast_curve.to_csv(source_dir / "empirical_control_contrasts.csv", index=False)
    cov_panel.to_csv(source_dir / "covariance_panel_table.csv", index=False)
    key_numbers.to_csv(source_dir / "key_numbers_for_caption.csv", index=False)
    provenance.to_csv(out_dir / "panel_provenance.csv", index=False)
    fixed_alpha.to_csv(robustness_dir / "robustness_fixed_alpha_summary.csv", index=False)
    resampling.to_csv(robustness_dir / "robustness_resampling_summary.csv", index=False)

    _make_motion_qc_figure(motion_qc, figures_dir)
    _make_gain_figure(gain_curve, figures_dir)
    _make_contrast_figure(contrast_curve, figures_dir)
    _make_covariance_figure(cov_panel, figures_dir)
    _make_multipanel(motion_qc, gain_curve, contrast_curve, cov_panel, figures_dir)

    _write_report(out_dir, run_dir, motion_qc, gain_curve, contrast_curve, fixed_alpha, resampling)
    _write_json(
        out_dir / "figure_pack_metadata.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py",
            "source_run_dir": str(run_dir),
            "incremental_dir": str(inc_dir),
            "output_dir": str(out_dir),
            "primary_summary": PRIMARY_SUMMARY,
            "primary_gain_rows": [{"latent": latent, "k": k} for latent, k in PRIMARY_GAIN_ROWS],
            "primary_contrast": {
                "latent": PRIMARY_CONTRAST_LATENT,
                "k": PRIMARY_CONTRAST_K,
                "lhs_family": "empirical",
                "rhs_families": CONTROL_ORDER,
            },
            "expected_scales": EXPECTED_SCALES,
            "n_motion_metadata_rows": int(motion_meta.shape[0]),
            "n_trace_bank_rows": int(trace_bank.shape[0]),
            "n_images": 256,
            "n_sessions": 29,
            "ridge_alpha_mode": str(inc_meta.get("ridge_alpha_mode")),
            "fixed_ridge_alpha": float(inc_meta.get("fixed_ridge_alpha")),
        },
    )
    print(f"Wrote aggregate FEM figure pack to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

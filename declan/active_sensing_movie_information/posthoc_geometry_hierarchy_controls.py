"""Posthoc checks for the covariance-aware Fisher hierarchy.

This script is intentionally downstream of ``jake.twininfo.run_covariance_optimality``:
it reads the completed covariance-optimality cache/results and writes gap-closure
tables, figures, and focused specificity controls without re-rendering model rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VISIONCORE_ROOT = Path(__file__).resolve().parents[2]
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from jake.twininfo.covariance_optimality import (
    covariance_fisher_by_time,
    covariance_residual_after_subspace,
    top_eigenvectors,
)


GEOMETRY_MODE_NOTE = (
    "Current cov_geometry_aware_k is implemented as the residual after removing "
    "the top eigenspace of the movement covariance. Therefore cov_topPC_aware_k "
    "is identical to cov_geometry_aware_k in this run; it is not an independent "
    "translation-tangent control."
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _cluster_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "image_index",
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    vals = df[[cluster_col, value_col]].dropna()
    if vals.empty:
        return float("nan"), float("nan")
    clusters = np.asarray(sorted(vals[cluster_col].unique()))
    if clusters.size <= 1 or n_bootstrap <= 0:
        v = vals[value_col].to_numpy(dtype=float)
        return float(np.nanmean(v)), float(np.nanmean(v))
    rng = np.random.default_rng(seed)
    by_cluster = {c: vals.loc[vals[cluster_col] == c, value_col].to_numpy(dtype=float) for c in clusters}
    boot = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        chosen = rng.choice(clusters, size=clusters.size, replace=True)
        sample = np.concatenate([by_cluster[c] for c in chosen])
        boot[b] = np.nanmean(sample)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _gap_rows(row_metrics: pd.DataFrame, *, eps: float = 1e-9) -> pd.DataFrame:
    metric = "final_fisher_trace_per_spike"
    key_cols = ["row_id", "example_id", "kind", "image_index", "crop_rank", "family", "scale_D"]
    regimes = ["cov_pose_aware", "cov_pose_blind"] + sorted(
        r for r in row_metrics["regime"].unique() if str(r).startswith("cov_geometry_aware_k")
    )
    wide = (
        row_metrics.loc[row_metrics["regime"].isin(regimes), key_cols + ["regime", metric]]
        .pivot_table(index=key_cols, columns="regime", values=metric, aggfunc="mean")
        .reset_index()
    )
    out: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        aware = float(row.get("cov_pose_aware", np.nan))
        blind = float(row.get("cov_pose_blind", np.nan))
        pose_gap = aware - blind
        for regime in regimes:
            if not str(regime).startswith("cov_geometry_aware_k"):
                continue
            k = int(str(regime).rsplit("k", 1)[1])
            geom = float(row.get(regime, np.nan))
            geom_gain = geom - blind
            geom_residual = aware - geom
            valid_gap = bool(np.isfinite(pose_gap) and pose_gap > eps)
            closure = geom_gain / pose_gap if valid_gap else np.nan
            residual_frac = geom_residual / pose_gap if valid_gap else np.nan
            out.append({
                "row_id": int(row["row_id"]),
                "example_id": row["example_id"],
                "kind": row["kind"],
                "image_index": int(row["image_index"]),
                "crop_rank": int(row["crop_rank"]),
                "family": row["family"],
                "scale_D": float(row["scale_D"]),
                "k": k,
                "cov_pose_aware": aware,
                "cov_pose_blind": blind,
                "cov_geometry_aware_k": geom,
                "pose_gap": pose_gap,
                "geom_gain_k": geom_gain,
                "geom_residual_k": geom_residual,
                "closure_k": closure,
                "residual_frac_k": residual_frac,
                "valid_pose_gap": valid_gap,
            })
    return pd.DataFrame(out)


def _summarize_gap_rows(gaps: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["family", "kind", "scale_D", "k"]
    for key, grp in gaps.groupby(group_cols, sort=True):
        family, kind, scale, k = key
        row: dict[str, Any] = {
            "family": family,
            "kind": kind,
            "scale_D": float(scale),
            "k": int(k),
            "n_rows": int(len(grp)),
            "n_images": int(grp["image_index"].nunique()),
            "pose_gap_mean": float(np.nanmean(grp["pose_gap"])),
            "pose_gap_sem": _sem(grp["pose_gap"].to_numpy()),
            "closure_mean": float(np.nanmean(grp["closure_k"])),
            "closure_sem": _sem(grp["closure_k"].to_numpy()),
            "residual_frac_mean": float(np.nanmean(grp["residual_frac_k"])),
            "residual_frac_sem": _sem(grp["residual_frac_k"].to_numpy()),
            "closure_sign_positive": int(np.nansum(grp["closure_k"].to_numpy(dtype=float) > 0.0)),
            "closure_sign_negative": int(np.nansum(grp["closure_k"].to_numpy(dtype=float) < 0.0)),
            "pose_gap_positive": int(np.nansum(grp["pose_gap"].to_numpy(dtype=float) > 0.0)),
            "pose_gap_nonpositive": int(np.nansum(grp["pose_gap"].to_numpy(dtype=float) <= 0.0)),
        }
        row["pose_gap_ci_low"], row["pose_gap_ci_high"] = _cluster_bootstrap_ci(
            grp, "pose_gap", n_bootstrap=n_bootstrap, seed=seed
        )
        row["closure_ci_low"], row["closure_ci_high"] = _cluster_bootstrap_ci(
            grp, "closure_k", n_bootstrap=n_bootstrap, seed=seed + 1
        )
        row["residual_frac_ci_low"], row["residual_frac_ci_high"] = _cluster_bootstrap_ci(
            grp, "residual_frac_k", n_bootstrap=n_bootstrap, seed=seed + 2
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cov_key(family: str, kind: str, scale: float, estimator: str = "pooled_residual") -> str:
    scale_label = str(float(scale)).replace(".", "p").replace("-", "m")
    return f"{family}__{kind or 'all'}__D{scale_label}__{estimator}"


def _lookup_covariance(covs: dict[str, np.ndarray], family: str, kind: str, scale: float) -> np.ndarray:
    key = _cov_key(family, kind, scale)
    if key in covs:
        return covs[key]
    compact_label = f"{float(scale):.6g}".replace(".", "p").replace("-", "m")
    compact_key = f"{family}__{kind or 'all'}__D{compact_label}__pooled_residual"
    if compact_key in covs:
        return covs[compact_key]
    matches = [
        name for name in covs
        if name.startswith(f"{family}__{kind or 'all'}__")
        and name.endswith("__pooled_residual")
    ]
    raise KeyError(f"Could not find covariance for {family}/{kind}/D={scale}; candidates={matches[:8]}")


def _score_row(mu_tn: np.ndarray, j_tnd: np.ndarray, expected_t: np.ndarray, cov: np.ndarray | None) -> float:
    f = covariance_fisher_by_time(mu_tn, j_tnd, cov)
    trace = float(np.trace(np.sum(f, axis=0)))
    expected = float(np.sum(expected_t))
    return trace / max(expected, 1e-12)


def _sample_rows(group_rows: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    if max_rows <= 0 or len(group_rows) <= max_rows:
        return group_rows
    return group_rows.sample(n=int(max_rows), random_state=int(rng.integers(0, 2**31 - 1))).sort_values("row_id")


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _r = np.linalg.qr(rng.standard_normal((n_units, int(k))))
    return q[:, : int(k)]


def _control_pass(
    *,
    covopt_dir: Path,
    gaps: pd.DataFrame,
    control_scales: tuple[float, ...],
    n_random_subspaces: int,
    max_control_rows_per_group: int,
    seed: int,
) -> pd.DataFrame:
    if n_random_subspaces <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = _read(covopt_dir / "metadata" / "covopt_rate_records.csv")
    with np.load(covopt_dir / "cache" / "covopt_mu_j.npz") as z:
        mu_all = np.asarray(z["mu"], dtype=np.float32)
        j_all = np.asarray(z["J"], dtype=np.float32)
        expected_all = np.asarray(z["expected_spikes_t"], dtype=np.float32)
    with np.load(covopt_dir / "cache" / "covopt_covariances.npz") as z:
        covs = {name: np.asarray(z[name], dtype=np.float64) for name in z.files}

    rows: list[dict[str, Any]] = []
    control_records = records.loc[records["scale_D"].astype(float).isin([float(x) for x in control_scales])]
    group_items = list(control_records.groupby(["family", "kind", "scale_D"], sort=True))
    print(
        f"Running Fisher specificity controls for {len(group_items)} groups; "
        f"n_random_subspaces={n_random_subspaces}, max_rows_per_group={max_control_rows_per_group}",
        flush=True,
    )
    for group_idx, ((family, kind, scale), group) in enumerate(group_items, start=1):
        print(
            f"[controls {group_idx}/{len(group_items)}] family={family} kind={kind} D={float(scale):g}",
            flush=True,
        )
        sigma = _lookup_covariance(covs, str(family), str(kind), float(scale))
        n_units = sigma.shape[0]
        group = _sample_rows(group, max_control_rows_per_group, rng)
        base_gaps = gaps[
            (gaps["family"] == family)
            & (gaps["kind"] == kind)
            & np.isclose(gaps["scale_D"].astype(float), float(scale))
            & (gaps["row_id"].isin(group["row_id"]))
        ]
        for k in sorted(base_gaps["k"].unique()):
            kk = int(k)
            print(f"  k={kk}", flush=True)
            basis = top_eigenvectors(sigma, kk)
            shuffled_basis = basis[rng.permutation(n_units), :]
            control_bases: list[tuple[str, int, np.ndarray]] = [
                ("unitshuffled", 0, shuffled_basis),
            ]
            for draw in range(int(n_random_subspaces)):
                control_bases.append(("random", draw, _random_basis(n_units, kk, rng)))
            for control_name, draw, u in control_bases:
                _compact, residual = covariance_residual_after_subspace(sigma, u)
                score_by_row: dict[int, float] = {}
                for row_id in group["row_id"].astype(int):
                    score_by_row[int(row_id)] = _score_row(
                        mu_all[int(row_id)],
                        j_all[int(row_id)],
                        expected_all[int(row_id)],
                        residual,
                    )
                merged = base_gaps.loc[base_gaps["k"] == kk].copy()
                merged["control_score"] = merged["row_id"].map(score_by_row)
                merged["control_gain"] = merged["control_score"] - merged["cov_pose_blind"]
                merged["control_closure"] = np.where(
                    merged["valid_pose_gap"],
                    merged["control_gain"] / merged["pose_gap"],
                    np.nan,
                )
                rows.append({
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "k": kk,
                    "control": control_name,
                    "draw": int(draw),
                    "n_rows": int(len(merged)),
                    "n_random_subspaces_requested": int(n_random_subspaces),
                    "max_control_rows_per_group": int(max_control_rows_per_group),
                    "closure_control_mean": float(np.nanmean(merged["control_closure"])),
                    "closure_geometry_mean": float(np.nanmean(merged["closure_k"])),
                    "geometry_minus_control": float(
                        np.nanmean(merged["closure_k"] - merged["control_closure"])
                    ),
                })
            rows.append({
                "family": family,
                "kind": kind,
                "scale_D": float(scale),
                "k": kk,
                "control": "topPC",
                "draw": 0,
                "n_rows": int(len(base_gaps.loc[base_gaps["k"] == kk])),
                "n_random_subspaces_requested": int(n_random_subspaces),
                "max_control_rows_per_group": int(max_control_rows_per_group),
                "closure_control_mean": float(np.nanmean(base_gaps.loc[base_gaps["k"] == kk, "closure_k"])),
                "closure_geometry_mean": float(np.nanmean(base_gaps.loc[base_gaps["k"] == kk, "closure_k"])),
                "geometry_minus_control": 0.0,
                "note": GEOMETRY_MODE_NOTE,
            })
        print(
            f"[controls {group_idx}/{len(group_items)}] done family={family} kind={kind} D={float(scale):g}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _control_summary(control_rows: pd.DataFrame) -> pd.DataFrame:
    if control_rows.empty:
        return control_rows
    rows: list[dict[str, Any]] = []
    for key, grp in control_rows.groupby(["family", "kind", "scale_D", "k", "control"], sort=True):
        family, kind, scale, k, control = key
        rows.append({
            "family": family,
            "kind": kind,
            "scale_D": float(scale),
            "k": int(k),
            "control": control,
            "n_draws": int(grp["draw"].nunique()),
            "n_rows_mean": float(np.nanmean(grp["n_rows"])),
            "closure_geometry_mean": float(np.nanmean(grp["closure_geometry_mean"])),
            "closure_control_mean": float(np.nanmean(grp["closure_control_mean"])),
            "closure_control_sem": _sem(grp["closure_control_mean"].to_numpy()),
            "geometry_minus_control_mean": float(np.nanmean(grp["geometry_minus_control"])),
            "geometry_minus_control_sem": _sem(grp["geometry_minus_control"].to_numpy()),
            "note": GEOMETRY_MODE_NOTE if control == "topPC" else "",
        })
    return pd.DataFrame(rows)


def _signal_summary(geometry: pd.DataFrame) -> pd.DataFrame:
    out = geometry.copy()
    out["frac_signal_in_Uk"] = out["signal_variance_geometry"]
    out["frac_coding_in_Uk"] = out["coding_variance_geometry"]
    return out[[
        "family",
        "kind",
        "scale_D",
        "k",
        "sigma_trace",
        "nuisance_variance_removed_fraction",
        "nuisance_variance_remaining_fraction",
        "signal_trace",
        "frac_signal_in_Uk",
        "coding_trace",
        "frac_coding_in_Uk",
        "signal_reference_scale_D",
    ]]


def _decision_label(row: pd.Series) -> str:
    pose_gap = row.get("pose_gap_mean", row.get("pose_gap", np.nan))
    if not np.isfinite(pose_gap) or pose_gap <= 1e-6:
        return "gap_too_small_to_interpret"
    closure = row.get("closure_geometry", np.nan)
    if not np.isfinite(closure) or closure <= 0.05:
        return "no_geometry_rescue"
    top = row.get("geometry_minus_topPC", 0.0)
    if np.isfinite(top) and top < -0.05:
        return "generic_topPC_better"
    rand = row.get("geometry_minus_random", np.nan)
    shuf = row.get("geometry_minus_unitshuffled", np.nan)
    if np.isfinite(rand) and np.isfinite(shuf) and rand > 0.05 and shuf > 0.05:
        return "strong_geometry_specific"
    return "low_rank_not_geometry_specific"


def _decision_table(gap_summary: pd.DataFrame, control_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    d1 = gap_summary[np.isclose(gap_summary["scale_D"].astype(float), 1.0)]
    for _, g in d1.iterrows():
        row = {
            "trajectory_family": g["family"],
            "regime": g["kind"],
            "D": float(g["scale_D"]),
            "k": int(g["k"]),
            "pose_gap": float(g["pose_gap_mean"]),
            "closure_geometry": float(g["closure_mean"]),
            "residual_to_pose_aware": float(g["residual_frac_mean"]),
            "closure_random_mean": np.nan,
            "closure_unitshuffled": np.nan,
            "closure_topPC": float(g["closure_mean"]),
            "geometry_minus_random": np.nan,
            "geometry_minus_unitshuffled": np.nan,
            "geometry_minus_topPC": 0.0,
            "topPC_note": GEOMETRY_MODE_NOTE,
        }
        if not control_summary.empty:
            c = control_summary[
                (control_summary["family"] == g["family"])
                & (control_summary["kind"] == g["kind"])
                & np.isclose(control_summary["scale_D"].astype(float), float(g["scale_D"]))
                & (control_summary["k"].astype(int) == int(g["k"]))
            ]
            for control, col in [
                ("random", "closure_random_mean"),
                ("unitshuffled", "closure_unitshuffled"),
                ("topPC", "closure_topPC"),
            ]:
                cc = c[c["control"] == control]
                if not cc.empty:
                    row[col] = float(cc["closure_control_mean"].iloc[0])
            row["geometry_minus_random"] = row["closure_geometry"] - row["closure_random_mean"]
            row["geometry_minus_unitshuffled"] = row["closure_geometry"] - row["closure_unitshuffled"]
            row["geometry_minus_topPC"] = row["closure_geometry"] - row["closure_topPC"]
        row["decision_label"] = _decision_label(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_score_curves(row_metrics: pd.DataFrame, out: Path) -> None:
    regimes = [
        "cov_pose_aware",
        "cov_geometry_aware_k2",
        "cov_geometry_aware_k5",
        "cov_geometry_aware_k10",
        "cov_geometry_aware_k20",
        "cov_pose_blind",
    ]
    df = row_metrics[row_metrics["regime"].isin(regimes)]
    summary = (
        df.groupby(["family", "kind", "scale_D", "regime"], as_index=False)["final_fisher_trace_per_spike"]
        .mean()
    )
    families = sorted(summary["family"].unique())
    kinds = sorted(summary["kind"].unique())
    fig, axes = plt.subplots(len(families), len(kinds), figsize=(12, 12), sharex=True, sharey=False)
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j] if axes.ndim == 2 else axes[max(i, j)]
            sub = summary[(summary["family"] == family) & (summary["kind"] == kind)]
            for regime in regimes:
                rr = sub[sub["regime"] == regime].sort_values("scale_D")
                if rr.empty:
                    continue
                ax.plot(rr["scale_D"], rr["final_fisher_trace_per_spike"], marker="o", linewidth=1.5, label=regime)
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.axvline(1.0, color="0.6", linewidth=1, linestyle=":")
            ax.set_xlabel("D")
            ax.set_ylabel("Fisher trace/spike")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_gap_figures(gap_summary: pd.DataFrame, out_dir: Path) -> None:
    families = sorted(gap_summary["family"].unique())
    kinds = sorted(gap_summary["kind"].unique())
    d1 = gap_summary[np.isclose(gap_summary["scale_D"].astype(float), 1.0)]

    fig, axes = plt.subplots(len(families), len(kinds), figsize=(10, 11), sharex=True, sharey=True)
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = d1[(d1["family"] == family) & (d1["kind"] == kind)].sort_values("k")
            ax.plot(sub["k"], sub["closure_mean"], marker="o")
            ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1)
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel("Gap closure")
    fig.tight_layout()
    fig.savefig(out_dir / "closure_vs_k_at_D1.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(families), len(kinds), figsize=(10, 11), sharex=True, sharey=True)
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = gap_summary[(gap_summary["family"] == family) & (gap_summary["kind"] == kind)]
            pivot = sub.pivot_table(index="k", columns="scale_D", values="closure_mean")
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
            ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65, label="Gap closure")
    fig.savefig(out_dir / "closure_heatmap_k_by_D.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(families), len(kinds), figsize=(10, 11), sharex=True, sharey=False)
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = gap_summary[(gap_summary["family"] == family) & (gap_summary["kind"] == kind)]
            pose = sub.groupby("scale_D", as_index=False)["pose_gap_mean"].mean().sort_values("scale_D")
            ax.plot(pose["scale_D"], pose["pose_gap_mean"], marker="o")
            ax.axvline(1.0, color="0.6", linewidth=1, linestyle=":")
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.set_xlabel("D")
            ax.set_ylabel("Pose gap")
    fig.tight_layout()
    fig.savefig(out_dir / "pose_gap_vs_D.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(families), len(kinds), figsize=(10, 11), sharex=True, sharey=True)
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = d1[(d1["family"] == family) & (d1["kind"] == kind)].sort_values("k")
            ax.plot(sub["k"], sub["residual_frac_mean"], marker="o")
            ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel("Residual fraction")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_to_pose_aware_vs_k.png", dpi=180)
    plt.close(fig)


def _write_decision_md(path: Path, decision: pd.DataFrame, control_summary: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4g") -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(str(c) for c in cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    vals.append("nan" if not np.isfinite(value) else format(float(value), floatfmt))
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Geometry Hierarchy Decision Table",
        "",
        "## Scope",
        "",
        f"- {GEOMETRY_MODE_NOTE}",
        "- Gap-closure metrics are computed for all D scales.",
        "- Specificity controls are reported only for the configured control scales and row/random scope.",
        "",
        "## Decision Counts",
        "",
    ]
    if decision.empty:
        lines.append("No decision rows.")
    else:
        counts = decision["decision_label"].value_counts().sort_index()
        for label, count in counts.items():
            lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## D=1 Decisions", ""])
    if not decision.empty:
        cols = [
            "trajectory_family",
            "regime",
            "k",
            "pose_gap",
            "closure_geometry",
            "closure_random_mean",
            "closure_unitshuffled",
            "closure_topPC",
            "geometry_minus_random",
            "geometry_minus_unitshuffled",
            "geometry_minus_topPC",
            "decision_label",
        ]
        lines.append(markdown_table(decision[cols]))
    if not control_summary.empty:
        lines.extend(["", "## Control Scope", ""])
        scope_cols = ["control", "n_draws", "n_rows_mean"]
        lines.append(markdown_table(control_summary[scope_cols].drop_duplicates()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--covopt-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-fisher-controls", action="store_true")
    p.add_argument("--control-scales", default="1")
    p.add_argument("--n-random-subspaces", type=int, default=50)
    p.add_argument(
        "--max-control-rows-per-group",
        type=int,
        default=0,
        help="0 uses all rows. Positive values subsample rows per family/kind/D for faster pilot controls.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    covopt_dir = Path(args.covopt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    row_metrics = _read(covopt_dir / "results" / "covopt_row_metrics.csv")
    geometry = _read(covopt_dir / "results" / "covopt_geometry_diagnostics.csv")
    gaps = _gap_rows(row_metrics)
    gap_summary = _summarize_gap_rows(gaps, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    signal = _signal_summary(geometry)

    control_rows = pd.DataFrame()
    control_summary = pd.DataFrame()
    if args.run_fisher_controls:
        control_scales = tuple(float(x) for x in str(args.control_scales).split(",") if x.strip())
        control_rows = _control_pass(
            covopt_dir=covopt_dir,
            gaps=gaps,
            control_scales=control_scales,
            n_random_subspaces=int(args.n_random_subspaces),
            max_control_rows_per_group=int(args.max_control_rows_per_group),
            seed=int(args.seed),
        )
        control_summary = _control_summary(control_rows)
    decision = _decision_table(gap_summary, control_summary)

    gaps.to_csv(out_dir / "geometry_hierarchy_gap_closure_rows.csv", index=False)
    gap_summary.to_csv(out_dir / "geometry_hierarchy_gap_closure_summary.csv", index=False)
    signal.to_csv(out_dir / "geometry_hierarchy_signal_preservation_summary.csv", index=False)
    control_rows.to_csv(out_dir / "geometry_hierarchy_control_rows.csv", index=False)
    control_summary.to_csv(out_dir / "geometry_hierarchy_control_summary.csv", index=False)
    decision.to_csv(out_dir / "geometry_hierarchy_decision_table.csv", index=False)
    _write_decision_md(out_dir / "geometry_hierarchy_decision_table.md", decision, control_summary)

    _plot_score_curves(row_metrics, fig_dir / "cov_score_vs_D_by_observer.png")
    _plot_gap_figures(gap_summary, fig_dir)
    metadata = {
        "covopt_dir": str(covopt_dir),
        "out_dir": str(out_dir),
        "n_gap_rows": int(len(gaps)),
        "n_gap_summary_rows": int(len(gap_summary)),
        "n_signal_rows": int(len(signal)),
        "n_control_rows": int(len(control_rows)),
        "n_control_summary_rows": int(len(control_summary)),
        "n_decision_rows": int(len(decision)),
        "geometry_mode_note": GEOMETRY_MODE_NOTE,
        "run_fisher_controls": bool(args.run_fisher_controls),
        "n_random_subspaces": int(args.n_random_subspaces),
        "max_control_rows_per_group": int(args.max_control_rows_per_group),
        "outputs": {
            "gap_rows": str(out_dir / "geometry_hierarchy_gap_closure_rows.csv"),
            "gap_summary": str(out_dir / "geometry_hierarchy_gap_closure_summary.csv"),
            "signal_summary": str(out_dir / "geometry_hierarchy_signal_preservation_summary.csv"),
            "control_rows": str(out_dir / "geometry_hierarchy_control_rows.csv"),
            "control_summary": str(out_dir / "geometry_hierarchy_control_summary.csv"),
            "decision_table_csv": str(out_dir / "geometry_hierarchy_decision_table.csv"),
            "decision_table_md": str(out_dir / "geometry_hierarchy_decision_table.md"),
            "figures": [
                str(fig_dir / "cov_score_vs_D_by_observer.png"),
                str(fig_dir / "closure_vs_k_at_D1.png"),
                str(fig_dir / "closure_heatmap_k_by_D.png"),
                str(fig_dir / "pose_gap_vs_D.png"),
                str(fig_dir / "residual_to_pose_aware_vs_k.png"),
            ],
        },
    }
    (out_dir / "geometry_hierarchy_posthoc_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

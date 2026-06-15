#!/usr/bin/env python3
"""Diagnostic atlas and pair-composition audit for chart-swap alignment runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from declan.compact_retinal_translation_geometry.run_correct_chart_swap_alignment import write_csv
from declan.compact_retinal_translation_geometry.summarize_correct_chart_swap_alignment import (
    _load_json,
    read_csv_rows,
)
from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    bootstrap_mean_ci,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_ROOT = Path("outputs") / "compact_retinal_translation_geometry" / "chart_swap_diagnostics"


def _parse_str_list(spec: str) -> list[str]:
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _sanitize_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def _quantile_bin_labels(values: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    vals = _finite(values)
    if vals.size == 0:
        return np.full(values.shape, "missing", dtype=object), ["missing"], np.asarray([], dtype=np.float64)
    if np.allclose(vals, vals[0]):
        label = f"all:{vals[0]:.3g}"
        out = np.full(values.shape, label, dtype=object)
        out[~np.isfinite(values)] = "missing"
        return out, ["missing", label] if np.any(~np.isfinite(values)) else [label], np.asarray([vals[0]])
    qs = np.nanpercentile(vals, [25, 50, 75])
    edges = np.unique(qs[np.isfinite(qs)])
    if edges.size == 0:
        label = "all"
        out = np.full(values.shape, label, dtype=object)
        out[~np.isfinite(values)] = "missing"
        return out, ["missing", label] if np.any(~np.isfinite(values)) else [label], edges
    bins = np.digitize(np.nan_to_num(values, nan=-np.inf), edges, right=False)
    labels: list[str] = []
    out = np.empty(values.shape, dtype=object)
    for idx, val in enumerate(values):
        if not np.isfinite(val):
            out[idx] = "missing"
            continue
        b = int(bins[idx])
        if b == 0:
            lo, hi = float(np.min(vals)), float(edges[0])
        elif b >= edges.size:
            lo, hi = float(edges[-1]), float(np.max(vals))
        else:
            lo, hi = float(edges[b - 1]), float(edges[b])
        label = f"q{b + 1}:{lo:.3g}-{hi:.3g}"
        out[idx] = label
        labels.append(label)
    uniq = []
    seen = set()
    if np.any(~np.isfinite(values)):
        uniq.append("missing")
        seen.add("missing")
    for label in labels:
        if label not in seen:
            uniq.append(label)
            seen.add(label)
    return out, uniq, edges


def _atlas_rows_for_root(
    *,
    label: str,
    root: Path,
    projection_control: str,
    basis_k: int,
    chart_space: str,
    unit_score_subsets: list[str],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = read_csv_rows(root / "chart_alignment_pair_metrics.csv")
    inventory_rows = read_csv_rows(root / "session_inventory.csv")
    leakage_rows = read_csv_rows(root / "fold_leakage_audit.csv")
    manifest = _load_json(root / "manifest.json")
    audit = _load_json(root / "audit.json")

    inventory_by_session = {str(row.get("session")): row for row in inventory_rows}
    session_fold_rows: dict[str, list[dict[str, Any]]] = {}
    for row in leakage_rows:
        session_fold_rows.setdefault(str(row.get("session")), []).append(row)

    atlas_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    split_mode = str(manifest.get("config", {}).get("split_mode", audit.get("split_mode", "")))
    n_folds_cfg = _safe_int(manifest.get("config", {}).get("n_folds", 0))
    for unit_subset in unit_score_subsets:
        session_blocks: list[dict[str, Any]] = []
        for session in sorted({str(row.get("session")) for row in pair_rows}):
            rows = [
                row
                for row in pair_rows
                if str(row.get("session")) == session
                and str(row.get("projection_control")) == projection_control
                and _safe_int(row.get("basis_k", -1)) == int(basis_k)
                and str(row.get("chart_space")) == chart_space
                and str(row.get("unit_score_subset", "all_units")) == unit_subset
            ]
            vals = _finite([_safe_float(row.get(metric, float("nan"))) for row in rows])
            if vals.size == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, rng=rng, n_bootstrap=int(n_bootstrap))
            fold_rows = session_fold_rows.get(session, [])
            inv = inventory_by_session.get(session, {})
            atlas_row = {
                "run_label": label,
                "root": str(root),
                "split_mode": split_mode,
                "configured_n_folds": int(n_folds_cfg),
                "session": session,
                "projection_control": projection_control,
                "basis_k": int(basis_k),
                "chart_space": chart_space,
                "unit_score_subset": unit_subset,
                "metric": metric,
                "n_pairs": int(vals.size),
                "n_folds_scored": int(len({int(_safe_int(row.get("fold", -1))) for row in rows})),
                "n_folds_audited": int(len(fold_rows)),
                "session_mean": mean,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "bootstrap_ci_width": (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else float("nan"),
                "pair_positive_fraction": float(np.mean(vals > 0.0)),
                "pair_abs_p95": float(np.nanpercentile(np.abs(vals), 95)),
                "delta_eye_norm_mean": float(np.nanmean([_safe_float(row.get("delta_eye_norm")) for row in rows])),
                "prediction_norm_true_mean": float(np.nanmean([_safe_float(row.get("prediction_norm_true")) for row in rows])),
                "image_structure_score_mean": float(np.nanmean([_safe_float(row.get("image_structure_score")) for row in rows])),
                "local_image_structure_score_mean": float(np.nanmean([_safe_float(row.get("local_image_structure_score")) for row in rows])),
                "n_unique_images": int(len({row.get("image_id") for row in rows if row.get("image_id", "") != ""})),
                "n_unique_time_contexts": int(len({row.get("time_context") for row in rows if row.get("time_context", "") != ""})),
                "inventory_status": str(inv.get("status", "")),
                "inventory_scored_rows": _safe_int(inv.get("n_scored_rows", 0)),
                "inventory_total_pairs": _safe_int(inv.get("n_pairs", 0)),
                "inventory_drift_pairs": _safe_int(inv.get("n_drift_pairs", 0)),
                "median_train_pairs": float(np.nanmedian([_safe_float(row.get("n_train_pairs")) for row in fold_rows])) if fold_rows else float("nan"),
                "median_test_pairs": float(np.nanmedian([_safe_float(row.get("n_test_pairs")) for row in fold_rows])) if fold_rows else float("nan"),
            }
            atlas_rows.append(atlas_row)
            session_blocks.append(atlas_row)
        means = _finite([_safe_float(row.get("session_mean")) for row in session_blocks])
        run_rows.append(
            {
                "run_label": label,
                "root": str(root),
                "split_mode": split_mode,
                "configured_n_folds": int(n_folds_cfg),
                "projection_control": projection_control,
                "basis_k": int(basis_k),
                "chart_space": chart_space,
                "unit_score_subset": unit_subset,
                "metric": metric,
                "n_sessions": int(means.size),
                "session_mean": float(np.mean(means)) if means.size else float("nan"),
                "bootstrap_ci_low": float(np.nanpercentile(means, 2.5)) if means.size else float("nan"),
                "bootstrap_ci_high": float(np.nanpercentile(means, 97.5)) if means.size else float("nan"),
                "n_positive_sessions": int(np.sum(means > 0.0)) if means.size else 0,
                "total_pairs": int(sum(_safe_int(row.get("n_pairs", 0)) for row in session_blocks)),
                "median_pairs_per_session": float(np.nanmedian([_safe_float(row.get("n_pairs")) for row in session_blocks])) if session_blocks else float("nan"),
                "median_folds_scored": float(np.nanmedian([_safe_float(row.get("n_folds_scored")) for row in session_blocks])) if session_blocks else float("nan"),
                "median_ci_width": float(np.nanmedian([_safe_float(row.get("bootstrap_ci_width")) for row in session_blocks])) if session_blocks else float("nan"),
            }
        )
    return atlas_rows, run_rows


def _top_group_rows(
    *,
    label: str,
    root: Path,
    pair_rows: list[dict[str, Any]],
    group_name: str,
    group_values: list[str],
    projection_control: str,
    basis_k: int,
    chart_space: str,
    unit_score_subset: str,
    metric: str,
    top_n: int,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row, group_value in zip(pair_rows, group_values, strict=False)
        if str(row.get("projection_control")) == projection_control
        and _safe_int(row.get("basis_k", -1)) == int(basis_k)
        and str(row.get("chart_space")) == chart_space
        and str(row.get("unit_score_subset", "all_units")) == unit_score_subset
        and group_value != "missing"
    ]
    filtered_groups = [group_value for row, group_value in zip(pair_rows, group_values, strict=False)
                       if str(row.get("projection_control")) == projection_control
                       and _safe_int(row.get("basis_k", -1)) == int(basis_k)
                       and str(row.get("chart_space")) == chart_space
                       and str(row.get("unit_score_subset", "all_units")) == unit_score_subset
                       and group_value != "missing"]
    out: list[dict[str, Any]] = []
    sessions = sorted({str(row.get("session")) for row in filtered})
    for session in sessions:
        session_rows = [row for row in filtered if str(row.get("session")) == session]
        session_groups = [group for row, group in zip(filtered, filtered_groups, strict=False) if str(row.get("session")) == session]
        total = max(1, len(session_rows))
        counts: dict[str, list[dict[str, Any]]] = {}
        for row, group in zip(session_rows, session_groups, strict=False):
            counts.setdefault(group, []).append(row)
        ranked = sorted(counts.items(), key=lambda item: len(item[1]), reverse=True)
        for rank, (group, rows) in enumerate(ranked[: int(top_n)], start=1):
            vals = _finite([_safe_float(row.get(metric, float("nan"))) for row in rows])
            out.append(
                {
                    "run_label": label,
                    "root": str(root),
                    "session": session,
                    "projection_control": projection_control,
                    "basis_k": int(basis_k),
                    "chart_space": chart_space,
                    "unit_score_subset": unit_score_subset,
                    "metric": metric,
                    "group_name": group_name,
                    "group_value": group,
                    "rank_by_count": int(rank),
                    "n_pairs": int(len(rows)),
                    "pair_fraction": float(len(rows) / total),
                    "mean_metric": float(np.mean(vals)) if vals.size else float("nan"),
                    "positive_fraction": float(np.mean(vals > 0.0)) if vals.size else float("nan"),
                    "delta_eye_norm_mean": float(np.nanmean([_safe_float(row.get("delta_eye_norm")) for row in rows])),
                    "image_structure_score_mean": float(np.nanmean([_safe_float(row.get("image_structure_score")) for row in rows])),
                    "local_image_structure_score_mean": float(np.nanmean([_safe_float(row.get("local_image_structure_score")) for row in rows])),
                    "prediction_norm_true_mean": float(np.nanmean([_safe_float(row.get("prediction_norm_true")) for row in rows])),
                }
            )
    return out


def _composition_rows_for_root(
    *,
    label: str,
    root: Path,
    projection_control: str,
    basis_k: int,
    chart_space: str,
    unit_score_subset: str,
    metric: str,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = read_csv_rows(root / "chart_alignment_pair_metrics.csv")
    filtered = [
        row
        for row in pair_rows
        if str(row.get("projection_control")) == projection_control
        and _safe_int(row.get("basis_k", -1)) == int(basis_k)
        and str(row.get("chart_space")) == chart_space
        and str(row.get("unit_score_subset", "all_units")) == unit_score_subset
    ]
    delta_bins, _, _ = _quantile_bin_labels(np.asarray([_safe_float(row.get("delta_eye_norm")) for row in filtered], dtype=np.float64))
    q_bins, _, _ = _quantile_bin_labels(np.asarray([_safe_float(row.get("prediction_norm_true")) for row in filtered], dtype=np.float64))
    img_bins, _, _ = _quantile_bin_labels(np.asarray([_safe_float(row.get("image_structure_score")) for row in filtered], dtype=np.float64))
    local_bins, _, _ = _quantile_bin_labels(np.asarray([_safe_float(row.get("local_image_structure_score")) for row in filtered], dtype=np.float64))

    group_specs = {
        "image_id": [str(row.get("image_id", "missing")) for row in filtered],
        "wrong_image_id": [str(row.get("wrong_image_id", "missing")) for row in filtered],
        "time_context": [str(row.get("time_context", "missing")) for row in filtered],
        "wrong_time_context": [str(row.get("wrong_time_context", "missing")) for row in filtered],
        "fold": [str(row.get("fold", "missing")) for row in filtered],
        "drift_mask": [str(bool(row.get("drift_mask", False))) for row in filtered],
        "delta_eye_norm_bin": [str(v) for v in delta_bins],
        "prediction_norm_true_bin": [str(v) for v in q_bins],
        "image_structure_bin": [str(v) for v in img_bins],
        "local_image_structure_bin": [str(v) for v in local_bins],
    }

    top_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    for group_name, group_values in group_specs.items():
        top_rows.extend(
            _top_group_rows(
                label=label,
                root=root,
                pair_rows=filtered,
                group_name=group_name,
                group_values=group_values,
                projection_control=projection_control,
                basis_k=basis_k,
                chart_space=chart_space,
                unit_score_subset=unit_score_subset,
                metric=metric,
                top_n=top_n,
            )
        )
        sessions = sorted({str(row.get("session")) for row in filtered})
        for session in sessions:
            counts: dict[str, int] = {}
            session_groups = [group for row, group in zip(filtered, group_values, strict=False) if str(row.get("session")) == session and group != "missing"]
            total = len(session_groups)
            if total == 0:
                continue
            for group in session_groups:
                counts[group] = counts.get(group, 0) + 1
            fracs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
            concentration_rows.append(
                {
                    "run_label": label,
                    "root": str(root),
                    "session": session,
                    "projection_control": projection_control,
                    "basis_k": int(basis_k),
                    "chart_space": chart_space,
                    "unit_score_subset": unit_score_subset,
                    "metric": metric,
                    "group_name": group_name,
                    "n_groups": int(len(counts)),
                    "top_group_fraction": float(np.max(fracs)),
                    "top2_group_fraction": float(np.sum(np.sort(fracs)[-2:])),
                    "herfindahl_index": float(np.sum(fracs**2)),
                }
            )
    return top_rows, concentration_rows


def _write_atlas_figure(out: Path, atlas_rows: list[dict[str, Any]], metric: str) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted({str(row.get("run_label")) for row in atlas_rows})
    subsets = ["all_units", "gain_bottom50"]
    colors = {"all_units": "#4c78a8", "gain_bottom50": "#e15759"}
    for run_label in runs:
        rows = [row for row in atlas_rows if str(row.get("run_label")) == run_label]
        sessions = sorted({str(row.get("session")) for row in rows})
        if not sessions:
            continue
        fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(sessions)), 4.8))
        base_x = np.arange(len(sessions), dtype=np.float64)
        offsets = {"all_units": -0.15, "gain_bottom50": 0.15}
        for subset in subsets:
            block = [row for row in rows if str(row.get("unit_score_subset")) == subset]
            by_session = {str(row.get("session")): row for row in block}
            xs, ys, yerr_lo, yerr_hi, labels = [], [], [], [], []
            for idx, session in enumerate(sessions):
                row = by_session.get(session)
                if row is None:
                    continue
                xs.append(base_x[idx] + offsets[subset])
                ys.append(_safe_float(row.get("session_mean")))
                lo = _safe_float(row.get("bootstrap_ci_low"))
                hi = _safe_float(row.get("bootstrap_ci_high"))
                yerr_lo.append(max(0.0, ys[-1] - lo) if np.isfinite(lo) else 0.0)
                yerr_hi.append(max(0.0, hi - ys[-1]) if np.isfinite(hi) else 0.0)
                labels.append(f"n={_safe_int(row.get('n_pairs'))}, f={_safe_int(row.get('n_folds_scored'))}")
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="o",
                    capsize=3.0,
                    color=colors[subset],
                    label=subset,
                )
                for x, y, text in zip(xs, ys, labels, strict=False):
                    ax.text(x, y, text, fontsize=7, ha="center", va="bottom")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(base_x)
        ax.set_xticklabels(sessions, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{run_label}: per-session effect atlas")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        safe = _sanitize_label(run_label)
        fig.savefig(fig_dir / f"{safe}_session_effect_atlas.png", dpi=220)
        fig.savefig(fig_dir / f"{safe}_session_effect_atlas.pdf")
        plt.close(fig)


def _write_composition_figure(out: Path, concentration_rows: list[dict[str, Any]]) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in concentration_rows if str(row.get("group_name")) in {"image_id", "time_context", "fold"}]
    if not rows:
        return
    sessions = sorted({str(row.get("session")) for row in rows})
    groups = ["image_id", "time_context", "fold"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(max(7.5, 1.0 * len(sessions)), 7.5), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups, strict=False):
        block = [row for row in rows if str(row.get("group_name")) == group]
        vals = []
        for session in sessions:
            row = next((r for r in block if str(r.get("session")) == session), None)
            vals.append(_safe_float(row.get("top_group_fraction")) if row is not None else float("nan"))
        ax.bar(np.arange(len(sessions)), vals, color="#76b7b2", alpha=0.9)
        ax.set_ylabel(f"{group}\n top frac")
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(np.arange(len(sessions)))
    axes[-1].set_xticklabels(sessions, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pair_composition_concentration.png", dpi=220)
    fig.savefig(fig_dir / "pair_composition_concentration.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build chart-swap diagnostic atlas and pair-composition audits.")
    p.add_argument("--roots", type=str, required=True, help="Comma-separated chart-swap output roots.")
    p.add_argument("--labels", type=str, default="", help="Optional comma-separated labels matching --roots.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--projection-control", type=str, default="global_rate")
    p.add_argument("--basis-k", type=int, default=10)
    p.add_argument("--chart-space", type=str, default="compact")
    p.add_argument("--unit-score-subsets", type=str, default="all_units,gain_bottom50")
    p.add_argument("--metric", type=str, default="true_minus_wrong")
    p.add_argument("--composition-root-label", type=str, default="", help="Run label to use for pair composition audit; defaults to the first label.")
    p.add_argument("--composition-unit-score-subset", type=str, default="gain_bottom50")
    p.add_argument("--top-n-groups", type=int, default=8)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(part) for part in _parse_str_list(args.roots)]
    labels = _parse_str_list(args.labels)
    if labels and len(labels) != len(roots):
        raise SystemExit("--labels must match --roots length when provided")
    if not labels:
        labels = [root.name for root in roots]
    subsets = _parse_str_list(args.unit_score_subsets)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    atlas_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for idx, (label, root) in enumerate(zip(labels, roots, strict=False)):
        root_atlas, root_runs = _atlas_rows_for_root(
            label=label,
            root=root,
            projection_control=str(args.projection_control),
            basis_k=int(args.basis_k),
            chart_space=str(args.chart_space),
            unit_score_subsets=subsets,
            metric=str(args.metric),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + idx * 101,
        )
        atlas_rows.extend(root_atlas)
        run_rows.extend(root_runs)
    write_csv(out / "session_effect_atlas.csv", atlas_rows)
    write_csv(out / "run_effect_summary.csv", run_rows)
    _write_atlas_figure(out, atlas_rows, str(args.metric))

    comp_label = str(args.composition_root_label) if str(args.composition_root_label).strip() else labels[0]
    try:
        comp_root = roots[labels.index(comp_label)]
    except ValueError as exc:
        raise SystemExit(f"Unknown composition label: {comp_label}") from exc
    top_rows, concentration_rows = _composition_rows_for_root(
        label=comp_label,
        root=comp_root,
        projection_control=str(args.projection_control),
        basis_k=int(args.basis_k),
        chart_space=str(args.chart_space),
        unit_score_subset=str(args.composition_unit_score_subset),
        metric=str(args.metric),
        top_n=int(args.top_n_groups),
    )
    write_csv(out / "pair_composition_top_groups.csv", top_rows)
    write_csv(out / "pair_composition_concentration.csv", concentration_rows)
    _write_composition_figure(out, concentration_rows)


if __name__ == "__main__":
    main()

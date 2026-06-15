#!/usr/bin/env python3
"""Summarize FEM scale tradeoffs from completed non-circular analyses.

This is a cache-first synthesis.  It does not rerender retinal movies and does
not run the digital twin.  It combines already computed scale curves from:

- input whitening;
- covariance-aware optimality;
- Vernier active-sensing scale sweeps;
- simple analytic displacement/fixation-window cost proxies.

The goal is not to fit a biological objective.  The goal is to ask whether
adding explicit costs to the whitening benefit can move the preferred scale
away from the upper whitening boundary and toward biological ``D_scale = 1``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from non_circular_fem_common import (
    DEFAULT_STACK_OUT_DIR,
    read_csv_rows,
    write_csv_rows,
    write_json,
)


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "fem_scale_tradeoff"
DEFAULT_WHITENING_DIR = DEFAULT_STACK_OUT_DIR / "input_whitening"
DEFAULT_COVOPT_DIR = DEFAULT_STACK_OUT_DIR / "covariance_optimality" / "covopt_full_gpu1"
DEFAULT_VERNIER_DIR = Path("outputs") / "vernier_active_sensing_component_scale"
PRIMARY_PASSBAND = "spatial_4_40cpd__temporal_1_30hz"
BIOLOGICAL_D = 1.0


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def minmax_by_component(rows: list[dict[str, Any]], value_key: str = "utility_value") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["component"])].append(row)
    out: list[dict[str, Any]] = []
    for component, items in groups.items():
        vals = np.asarray([fnum(row, value_key) for row in items], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        lo = float(np.min(finite)) if finite.size else float("nan")
        hi = float(np.max(finite)) if finite.size else float("nan")
        for row in items:
            value = fnum(row, value_key)
            if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi):
                norm = float("nan")
            elif abs(hi - lo) <= 1e-12:
                norm = 0.5
            else:
                norm = (value - lo) / (hi - lo)
            out.append({**row, "normalization": "minmax_within_component", "norm_value": float(norm), "norm_min": lo, "norm_max": hi})
    return out


def component_map(rows: list[dict[str, Any]]) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = defaultdict(dict)
    for row in rows:
        out[str(row["component"])][fnum(row, "scale_D")] = fnum(row, "norm_value")
    return out


def interpolate_component(points: dict[float, float], scales: Iterable[float], *, extrapolate: bool = False) -> dict[float, float]:
    xs = np.asarray(sorted(points), dtype=np.float64)
    ys = np.asarray([points[float(x)] for x in xs], dtype=np.float64)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[keep]
    ys = ys[keep]
    out: dict[float, float] = {}
    if xs.size == 0:
        return {float(scale): float("nan") for scale in scales}
    for scale in scales:
        s = float(scale)
        if not extrapolate and (s < float(np.min(xs)) or s > float(np.max(xs))):
            out[s] = float("nan")
        else:
            out[s] = float(np.interp(s, xs, ys))
    return out


def summarize_whitening(whitening_dir: Path, *, passband: str, family: str) -> list[dict[str, Any]]:
    rows = read_csv_rows(whitening_dir / "whitening_scale_summary.csv")
    components: list[dict[str, Any]] = []
    metric_specs = {
        "spectral_entropy": ("whitening_entropy", True),
        "spectral_flatness": ("whitening_flatness", True),
        "abs_loglog_temporal_psd_slope": ("whitening_abs_slope", False),
    }
    for row in rows:
        if row.get("motion_family") != family or row.get("passband_label") != passband:
            continue
        metric = str(row.get("metric", ""))
        if metric not in metric_specs:
            continue
        component, higher_is_better = metric_specs[metric]
        raw = fnum(row, "mean")
        utility = raw if higher_is_better else -raw
        components.append(
            {
                "component": component,
                "source": "input_whitening",
                "source_family": family,
                "kind": "fixation",
                "scale_D": fnum(row, "D_scale"),
                "metric": metric,
                "raw_value": raw,
                "utility_value": utility,
                "higher_is_better": bool(higher_is_better),
                "sem": fnum(row, "sem"),
                "n": fnum(row, "n"),
                "notes": f"primary passband {passband}",
            }
        )
    return components


def summarize_covariance(covopt_dir: Path, *, family: str, kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_csv_rows(covopt_dir / "covopt_paired_contrasts.csv"):
        if row.get("family") == family and row.get("kind") == kind and row.get("contrast") == "pose_gap":
            raw = fnum(row, "mean")
            out.append(
                {
                    "component": "pose_blind_covariance_cost",
                    "source": "covariance_optimality",
                    "source_family": family,
                    "kind": kind,
                    "scale_D": fnum(row, "scale_D"),
                    "metric": "cov_pose_aware_minus_cov_pose_blind",
                    "raw_value": raw,
                    "utility_value": raw,
                    "higher_is_better": False,
                    "sem": fnum(row, "sem"),
                    "n": fnum(row, "n"),
                    "notes": "larger means more information recovered when pose is known; used as pose-blind cost",
                }
            )
    for row in read_csv_rows(covopt_dir / "covopt_scale_summary.csv"):
        if row.get("family") == family and row.get("kind") == kind and row.get("regime") == "cov_pose_aware":
            raw = fnum(row, "mean")
            out.append(
                {
                    "component": "pose_aware_information",
                    "source": "covariance_optimality",
                    "source_family": family,
                    "kind": kind,
                    "scale_D": fnum(row, "scale_D"),
                    "metric": "final_fisher_trace_per_spike",
                    "raw_value": raw,
                    "utility_value": raw,
                    "higher_is_better": True,
                    "sem": fnum(row, "sem"),
                    "n": fnum(row, "n"),
                    "notes": "model-side pose-aware covariance Fisher efficiency",
                }
            )
    return out


def _vernier_condition_scale(condition: str) -> tuple[str, float] | None:
    if condition == "static_center":
        return "static_center", 0.0
    if condition == "real_fem":
        return "real_fem", 1.0
    prefixes = ("drift_only_scaled_", "drift_scaled_", "microsaccade_only_scaled_", "microsaccade_scaled_")
    for prefix in prefixes:
        if condition.startswith(prefix):
            return prefix.rstrip("_"), float(condition.removeprefix(prefix))
    return None


def summarize_vernier(vernier_dir: Path, *, fd_step_arcmin: float, family: str) -> list[dict[str, Any]]:
    rows = read_csv_rows(vernier_dir / "information_summary.csv")
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("readout") != "pose_aware_diagonal_poisson":
            continue
        if not np.isclose(fnum(row, "fd_step_arcmin"), fd_step_arcmin):
            continue
        parsed = _vernier_condition_scale(str(row.get("condition", "")))
        if parsed is None:
            continue
        cond_family, scale = parsed
        if cond_family != family:
            continue
        trace_index = str(row.get("trace_index", ""))
        if trace_index == "all":
            continue
        grouped[float(scale)].append(fnum(row, "final_fisher"))
    out: list[dict[str, Any]] = []
    for scale, vals in sorted(grouped.items()):
        raw = mean(vals)
        out.append(
            {
                "component": "vernier_acuity",
                "source": "vernier_active_sensing",
                "source_family": family,
                "kind": "fixation",
                "scale_D": scale,
                "metric": f"pose_aware_final_fisher_fd{fd_step_arcmin:g}arcmin",
                "raw_value": raw,
                "utility_value": raw,
                "higher_is_better": True,
                "sem": sem(vals),
                "n": len([v for v in vals if np.isfinite(v)]),
                "notes": "Vernier component-scale run; common scale support is limited",
            }
        )
    return out


def summarize_motion_costs(scales: Iterable[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scale in sorted({float(s) for s in scales}):
        for component, raw, notes in (
            ("motion_cost_linear_radius_proxy", abs(scale), "proxy for RMS radius under scaled measured displacement"),
            ("motion_cost_diffusion_proxy", scale * scale, "proxy for diffusion/variance cost under displacement scaling"),
            ("motion_cost_above_biological_window", max(0.0, scale - BIOLOGICAL_D) ** 2, "one-sided penalty for exceeding biological fixation-window scale"),
        ):
            out.append(
                {
                    "component": component,
                    "source": "analytic_scale_proxy",
                    "source_family": "scale_proxy",
                    "kind": "fixation",
                    "scale_D": scale,
                    "metric": component,
                    "raw_value": raw,
                    "utility_value": raw,
                    "higher_is_better": False,
                    "sem": 0.0,
                    "n": 0,
                    "notes": notes,
                }
            )
    return out


def add_composite_whitening(norm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components = component_map(norm_rows)
    required = ["whitening_entropy", "whitening_flatness", "whitening_abs_slope"]
    scales = sorted(set().union(*(components.get(name, {}).keys() for name in required)))
    out: list[dict[str, Any]] = []
    for scale in scales:
        vals = [components[name].get(scale, float("nan")) for name in required]
        finite = [v for v in vals if np.isfinite(v)]
        if len(finite) != len(required):
            continue
        raw = float(np.mean(finite))
        out.append(
            {
                "component": "whitening_composite",
                "source": "input_whitening",
                "source_family": "scaled_measured_drift_D",
                "kind": "fixation",
                "scale_D": scale,
                "metric": "mean_minmax_entropy_flatness_negative_abs_slope",
                "raw_value": raw,
                "utility_value": raw,
                "higher_is_better": True,
                "sem": float("nan"),
                "n": len(required),
                "notes": "mean of normalized primary-passband whitening components",
                "normalization": "already_normalized_component_mean",
                "norm_value": raw,
                "norm_min": 0.0,
                "norm_max": 1.0,
            }
        )
    return out


def tradeoff_rows(norm_rows: list[dict[str, Any]], *, scales: list[float], weight_values: list[float]) -> list[dict[str, Any]]:
    comps = component_map(norm_rows)
    component_points = {
        name: interpolate_component(vals, scales, extrapolate=False)
        for name, vals in comps.items()
    }
    out: list[dict[str, Any]] = []

    def emit(test: str, weights: dict[str, float], score_components: dict[str, float], allowed_scales: list[float]) -> None:
        score_by_scale: dict[float, float] = {}
        for scale in allowed_scales:
            total = 0.0
            ok = True
            for component, weight in score_components.items():
                value = component_points.get(component, {}).get(scale, float("nan"))
                if not np.isfinite(value):
                    ok = False
                    break
                total += float(weight) * value
            if ok:
                score_by_scale[float(scale)] = total
        if not score_by_scale:
            return
        scale_arr = np.asarray(sorted(score_by_scale), dtype=np.float64)
        value_arr = np.asarray([score_by_scale[float(scale)] for scale in scale_arr], dtype=np.float64)
        opt_i = int(np.nanargmax(value_arr))
        bio_i = int(np.nanargmin(np.abs(scale_arr - BIOLOGICAL_D)))
        opt_d = float(scale_arr[opt_i])
        opt_v = float(value_arr[opt_i])
        bio_d = float(scale_arr[bio_i])
        bio_v = float(value_arr[bio_i])
        row = {
            "test": test,
            **weights,
            "D_opt": opt_d,
            "score_at_opt": opt_v,
            "nearest_biological_D": bio_d,
            "score_at_nearest_biological": bio_v,
            "biological_score_gap_from_opt": opt_v - bio_v,
            "n_scales": int(scale_arr.size),
            "scale_min": float(np.min(scale_arr)),
            "scale_max": float(np.max(scale_arr)),
            "score_components": ";".join(f"{k}:{v:g}" for k, v in score_components.items()),
        }
        out.append(row)

    all_scales = sorted(scales)
    for lam in weight_values:
        emit(
            "whitening_minus_pose_blind_covariance_cost",
            {"lambda_covariance": lam, "alpha_acuity": 0.0, "gamma_motion": 0.0},
            {"whitening_composite": 1.0, "pose_blind_covariance_cost": -lam},
            all_scales,
        )
        emit(
            "whitening_minus_diffusion_cost",
            {"lambda_covariance": 0.0, "alpha_acuity": 0.0, "gamma_motion": lam},
            {"whitening_composite": 1.0, "motion_cost_diffusion_proxy": -lam},
            all_scales,
        )
        emit(
            "whitening_minus_above_biological_window_cost",
            {"lambda_covariance": 0.0, "alpha_acuity": 0.0, "gamma_motion": lam},
            {"whitening_composite": 1.0, "motion_cost_above_biological_window": -lam},
            all_scales,
        )
        emit(
            "whitening_plus_vernier_acuity",
            {"lambda_covariance": 0.0, "alpha_acuity": lam, "gamma_motion": 0.0},
            {"whitening_composite": 1.0, "vernier_acuity": lam},
            all_scales,
        )

    compact_weights = [0.0, 0.5, 1.0, 2.0]
    for lam in compact_weights:
        for alpha in compact_weights:
            for gamma in compact_weights:
                if lam == alpha == gamma == 0.0:
                    continue
                emit(
                    "combined_whitening_covariance_acuity_motion",
                    {"lambda_covariance": lam, "alpha_acuity": alpha, "gamma_motion": gamma},
                    {
                        "whitening_composite": 1.0,
                        "pose_blind_covariance_cost": -lam,
                        "vernier_acuity": alpha,
                        "motion_cost_diffusion_proxy": -gamma,
                    },
                    all_scales,
                )
    return out


def decision_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["test"])].append(row)
    out: list[dict[str, Any]] = []
    for test, items in sorted(grouped.items()):
        d_opts = [fnum(row, "D_opt") for row in items]
        finite = [d for d in d_opts if np.isfinite(d)]
        if not finite:
            continue
        near_bio = [d for d in finite if abs(d - BIOLOGICAL_D) <= 0.5]
        at_upper = [d for d in finite if d >= 3.0 - 1e-9]
        out.append(
            {
                "test": test,
                "n_weight_settings": len(finite),
                "n_opt_near_biological_pm0p5": len(near_bio),
                "fraction_opt_near_biological_pm0p5": len(near_bio) / len(finite),
                "n_opt_at_upper_D3": len(at_upper),
                "fraction_opt_at_upper_D3": len(at_upper) / len(finite),
                "median_D_opt": float(np.median(finite)),
                "min_D_opt": float(np.min(finite)),
                "max_D_opt": float(np.max(finite)),
            }
        )
    return out


def write_figures(out_dir: Path, norm_rows: list[dict[str, Any]], tradeoffs: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    comps = component_map(norm_rows)
    key_components = [
        "whitening_composite",
        "pose_blind_covariance_cost",
        "vernier_acuity",
        "motion_cost_diffusion_proxy",
        "motion_cost_above_biological_window",
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for component in key_components:
        vals = comps.get(component, {})
        if not vals:
            continue
        xs = np.asarray(sorted(vals), dtype=np.float64)
        ys = np.asarray([vals[float(x)] for x in xs], dtype=np.float64)
        ax.plot(xs, ys, marker="o", label=component)
    ax.axvline(BIOLOGICAL_D, color="black", linestyle="--", linewidth=1, label="biological D=1")
    ax.set_xlabel("D scale")
    ax.set_ylabel("normalized component value")
    ax.set_title("FEM scale tradeoff components")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "tradeoff_component_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    for test in ("whitening_minus_pose_blind_covariance_cost", "whitening_plus_vernier_acuity", "whitening_minus_diffusion_cost"):
        rows = [row for row in tradeoffs if row.get("test") == test]
        if not rows:
            continue
        xs = np.asarray([fnum(row, "lambda_covariance") + fnum(row, "alpha_acuity") + fnum(row, "gamma_motion") for row in rows], dtype=np.float64)
        ys = np.asarray([fnum(row, "D_opt") for row in rows], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(xs, ys, marker="o")
        ax.axhline(BIOLOGICAL_D, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("cost/benefit weight")
        ax.set_ylabel("D_opt")
        ax.set_title(test.replace("_", " "))
        fig.tight_layout()
        fig.savefig(fig_dir / f"{test}.pdf", bbox_inches="tight")
        plt.close(fig)


def write_summary(out_dir: Path, decisions: list[dict[str, Any]], tradeoffs: list[dict[str, Any]]) -> None:
    by_test = {str(row["test"]): row for row in decisions}
    lines = [
        "# FEM Scale Tradeoff Summary",
        "",
        "This is a summary-only synthesis over completed caches. It does not rerun retinal movies or the digital twin.",
        "",
        "Biological scale is defined as `D_scale = 1`.",
        "",
        "Main conclusion: the ingredients point in sensible opposing directions, but a generic scalar tradeoff does not yet explain biological scale.",
        "",
        "## Decision Overview",
        "",
    ]
    for test in sorted(by_test):
        row = by_test[test]
        lines.append(
            f"- `{test}`: median D_opt={fnum(row, 'median_D_opt'):.3g}, "
            f"near-biological fraction={fnum(row, 'fraction_opt_near_biological_pm0p5'):.3g}, "
            f"upper-bound fraction={fnum(row, 'fraction_opt_at_upper_D3'):.3g} "
            f"over {int(fnum(row, 'n_weight_settings', 0))} weight settings."
        )
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "Input whitening pushes toward larger motion. Vernier acuity, pose-blind covariance, and generic diffusion costs push toward smaller motion. The one-sided above-biological window penalty can recover `D_scale = 1`, but partly by construction because it explicitly penalizes scales above the biological range.",
            "",
            "The most defensible qualitative statement is: component objectives exert opposing pressures, but simple generic costs do not recover biological scale without a biologically specific constraint.",
            "",
            "## Interpretation Guardrails",
            "",
            "- Weight sweeps are diagnostic, not fitted biological objectives.",
            "- Independent min-max normalization makes optima sensitive to scale, sign, and grid density.",
            "- The covariance term uses the pose-aware minus pose-blind Fisher gap as a cost proxy.",
            "- The Vernier term uses the completed component-scale cache and only has limited scale support.",
            "- Motion costs are analytic scale proxies, not measured fixation-window costs.",
            "- The above-biological window cost is not explanatory as written; it is a placeholder for measurable fixation-window, stability, blur, pose-precision, or motor constraints.",
            "- A shift of D_opt toward 1 should be read as evidence that added constraints can counter the whitening boundary, not as proof of optimization.",
            "",
            "## Next Test",
            "",
            "The next non-circular test should be V1 temporal-sensitivity-weighted whitening: replace raw temporal flatness with a metric that weights retinal temporal power by the temporal frequencies foveal V1 can actually encode. This may reveal whether large drift keeps flattening raw input while pushing power into less usable temporal bands.",
            "",
        ]
    )
    best_near = [
        row for row in tradeoffs
        if np.isfinite(fnum(row, "D_opt")) and abs(fnum(row, "D_opt") - BIOLOGICAL_D) <= 0.5
    ]
    if best_near:
        lines.extend(["## Example Near-Biological Settings", ""])
        for row in best_near[:12]:
            lines.append(
                f"- `{row['test']}` with lambda={fnum(row, 'lambda_covariance'):.3g}, "
                f"alpha={fnum(row, 'alpha_acuity'):.3g}, gamma={fnum(row, 'gamma_motion'):.3g}: "
                f"D_opt={fnum(row, 'D_opt'):.3g}, biological gap={fnum(row, 'biological_score_gap_from_opt'):.3g}."
            )
        lines.append("")
    (out_dir / "fem_scale_tradeoff_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_csv_floats(args.scales)
    weights = parse_csv_floats(args.weights)

    component_rows: list[dict[str, Any]] = []
    component_rows.extend(summarize_whitening(Path(args.whitening_dir), passband=args.passband, family=args.whitening_family))
    component_rows.extend(summarize_covariance(Path(args.covopt_dir), family=args.covariance_family, kind=args.covariance_kind))
    component_rows.extend(summarize_vernier(Path(args.vernier_dir), fd_step_arcmin=float(args.vernier_fd_step_arcmin), family=args.vernier_family))
    component_rows.extend(summarize_motion_costs(scales))

    norm_rows = minmax_by_component(component_rows)
    norm_rows.extend(add_composite_whitening(norm_rows))
    tradeoffs = tradeoff_rows(norm_rows, scales=scales, weight_values=weights)
    decisions = decision_summary(tradeoffs)

    write_csv_rows(out_dir / "component_scale_curves.csv", component_rows)
    write_csv_rows(out_dir / "normalized_component_curves.csv", norm_rows)
    write_csv_rows(out_dir / "tradeoff_weight_sweep.csv", tradeoffs)
    write_csv_rows(out_dir / "tradeoff_decision_table.csv", decisions)
    write_figures(out_dir, norm_rows, tradeoffs)
    write_summary(out_dir, decisions, tradeoffs)
    write_json(
        out_dir / "fem_scale_tradeoff_manifest.json",
        {
            "analysis": "fem_scale_tradeoff_summary",
            "out_dir": out_dir,
            "whitening_dir": Path(args.whitening_dir),
            "covopt_dir": Path(args.covopt_dir),
            "vernier_dir": Path(args.vernier_dir),
            "passband": args.passband,
            "whitening_family": args.whitening_family,
            "covariance_family": args.covariance_family,
            "covariance_kind": args.covariance_kind,
            "vernier_family": args.vernier_family,
            "vernier_fd_step_arcmin": float(args.vernier_fd_step_arcmin),
            "scales": scales,
            "weights": weights,
            "n_component_rows": len(component_rows),
            "n_normalized_rows": len(norm_rows),
            "n_tradeoff_rows": len(tradeoffs),
            "n_decision_rows": len(decisions),
            "biological_D_scale": BIOLOGICAL_D,
            "claim_boundary": (
                "Diagnostic weight sweep only. Weight settings are not fitted, "
                "motion costs are analytic proxies, and Vernier scale support is limited."
            ),
        },
    )
    print(f"Wrote FEM scale tradeoff outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whitening-dir", type=Path, default=DEFAULT_WHITENING_DIR)
    parser.add_argument("--covopt-dir", type=Path, default=DEFAULT_COVOPT_DIR)
    parser.add_argument("--vernier-dir", type=Path, default=DEFAULT_VERNIER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--passband", default=PRIMARY_PASSBAND)
    parser.add_argument("--whitening-family", default="scaled_measured_drift_D")
    parser.add_argument("--covariance-family", default="scaled_real")
    parser.add_argument("--covariance-kind", default="fixation")
    parser.add_argument("--vernier-family", default="drift_only_scaled")
    parser.add_argument("--vernier-fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--scales", default="0,0.125,0.25,0.5,0.75,1,1.5,2,3")
    parser.add_argument("--weights", default="0,0.25,0.5,1,2,4")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

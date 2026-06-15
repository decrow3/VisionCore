"""Summarize conditional-fixation BackImage objective residuals and confidence.

This is a cache-first follow-up to ``run_backimage_twin_drift_geometry.py``.
It asks whether the conditional objective family explains anything beyond the
raw edge axis: residual drift direction, alignment strength, or drift
anisotropy.  It does not run the digital twin.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_conditional_fixation_objectives_twin_axis_only_n256"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "conditional_residual_summary"
KEY_OBJECTIVES = (
    "optimized_pixel_isophote",
    "optimized_response_stability",
    "optimized_response_refresh_lambda_0.25",
    "optimized_response_refresh_lambda_0.5",
    "optimized_PA",
    "optimized_PB",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _axis_delta_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def _cos2(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return np.cos(2.0 * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))


def _mean_session_stat(df: pd.DataFrame, value_col: str) -> float:
    if df.empty:
        return float("nan")
    return float(df.groupby("session")[value_col].mean().mean())


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    means = np.mean(draws, axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if frame.shape[0] < 3:
        return float("nan"), int(frame.shape[0])
    return float(frame["x"].corr(frame["y"], method=method)), int(frame.shape[0])


def _window_key_cols() -> list[str]:
    return ["window_row", "window_id", "session", "trial_idx", "phase"]


def landscape_confidence(candidate_df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("pixel_isophote", "pixel_instability_cost", "min"),
        ("response_stability", "response_stability_cost", "min"),
        ("pose_aware", "pose_aware_score", "max"),
    ]
    rows: list[dict[str, Any]] = []
    group_cols = _window_key_cols()
    for key, block in candidate_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key, strict=True))
        first = block.iloc[0]
        base.update(
            {
                "real_drift_axis_deg": float(first["real_drift_axis_deg"]),
                "image_edge_axis_deg": float(first["image_edge_axis_deg"]),
                "image_gradient_axis_deg": float(first["image_gradient_axis_deg"]),
                "image_spectrum_axis_deg": float(first["image_spectrum_axis_deg"]),
                "edge_cos2_alignment": float(_cos2(float(first["real_drift_axis_deg"]), float(first["image_edge_axis_deg"]))),
                "edge_abs_delta_deg": abs(float(_axis_delta_deg(float(first["real_drift_axis_deg"]), float(first["image_edge_axis_deg"])))),
            }
        )
        for label, value_col, direction in specs:
            axis_values = block.groupby("candidate_axis_deg")[value_col].mean().sort_index()
            vals = axis_values.to_numpy(dtype=np.float64)
            axes = axis_values.index.to_numpy(dtype=np.float64)
            keep = np.isfinite(vals)
            vals = vals[keep]
            axes = axes[keep]
            if vals.size < 2:
                continue
            order = np.argsort(vals)
            if direction == "max":
                order = order[::-1]
                margin = vals[order[0]] - vals[order[1]]
            else:
                margin = vals[order[1]] - vals[order[0]]
            sd = float(np.nanstd(vals))
            if not np.isfinite(sd) or sd <= 1e-12:
                sd = 1.0
            best_axis = float(axes[order[0]])
            rows.append(
                {
                    **base,
                    "landscape": label,
                    "score_col": value_col,
                    "direction": direction,
                    "best_axis_deg": best_axis,
                    "best_value": float(vals[order[0]]),
                    "second_value": float(vals[order[1]]),
                    "margin": float(margin),
                    "margin_z": float(margin / sd),
                    "n_axes": int(vals.size),
                    "best_cos2_alignment": float(_cos2(float(first["real_drift_axis_deg"]), best_axis)),
                    "best_delta_vs_edge_cos2": float(_cos2(float(first["real_drift_axis_deg"]), best_axis))
                    - float(_cos2(float(first["real_drift_axis_deg"]), float(first["image_edge_axis_deg"]))),
                    "best_abs_delta_deg": abs(float(_axis_delta_deg(float(first["real_drift_axis_deg"]), best_axis))),
                    "best_abs_delta_improvement_vs_edge_deg": abs(
                        float(_axis_delta_deg(float(first["real_drift_axis_deg"]), float(first["image_edge_axis_deg"])))
                    )
                    - abs(float(_axis_delta_deg(float(first["real_drift_axis_deg"]), best_axis))),
                }
            )
    return pd.DataFrame(rows)


def objective_delta_table(alignment_df: pd.DataFrame, objectives: tuple[str, ...]) -> pd.DataFrame:
    raw = alignment_df[alignment_df["objective"] == "raw_edge_axis"].copy()
    raw_cols = _window_key_cols() + ["cos2_alignment", "axis_delta_deg", "predicted_axis_deg"]
    raw = raw[raw_cols].rename(
        columns={
            "cos2_alignment": "raw_edge_cos2",
            "axis_delta_deg": "raw_edge_axis_delta_deg",
            "predicted_axis_deg": "raw_edge_axis_deg",
        }
    )
    blocks = []
    for obj in objectives:
        block = alignment_df[alignment_df["objective"] == obj].copy()
        if block.empty:
            continue
        merged = block.merge(raw, on=_window_key_cols(), how="inner")
        merged["delta_cos2_vs_raw_edge"] = merged["cos2_alignment"] - merged["raw_edge_cos2"]
        merged["abs_delta_deg"] = np.abs(merged["axis_delta_deg"].astype(float))
        merged["raw_edge_abs_delta_deg"] = np.abs(merged["raw_edge_axis_delta_deg"].astype(float))
        merged["abs_delta_improvement_vs_raw_edge_deg"] = merged["raw_edge_abs_delta_deg"] - merged["abs_delta_deg"]
        blocks.append(merged)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def confidence_correlation_summary(conf_df: pd.DataFrame, delta_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if conf_df.empty:
        return rows
    for landscape, block in conf_df.groupby("landscape"):
        for target in ("edge_cos2_alignment", "edge_abs_delta_deg", "best_delta_vs_edge_cos2", "best_abs_delta_improvement_vs_edge_deg"):
            for method in ("pearson", "spearman"):
                val, n = _corr(block["margin_z"], block[target], method)
                rows.append(
                    {
                        "analysis": "landscape_confidence",
                        "landscape": landscape,
                        "objective": "",
                        "predictor": "margin_z",
                        "target": target,
                        "method": method,
                        "correlation": val,
                        "n": n,
                    }
                )
    if delta_df.empty:
        return rows
    keep_cols = _window_key_cols()
    for landscape, conf_block in conf_df.groupby("landscape"):
        conf_small = conf_block[keep_cols + ["margin_z"]].copy()
        for objective, obj_block in delta_df.groupby("objective"):
            merged = obj_block.merge(conf_small, on=keep_cols, how="inner")
            for target in ("delta_cos2_vs_raw_edge", "abs_delta_improvement_vs_raw_edge_deg", "cos2_alignment", "drift_anisotropy"):
                if target not in merged.columns:
                    continue
                for method in ("pearson", "spearman"):
                    val, n = _corr(merged["margin_z"], merged[target], method)
                    rows.append(
                        {
                            "analysis": "objective_delta_by_confidence",
                            "landscape": landscape,
                            "objective": objective,
                            "predictor": "margin_z",
                            "target": target,
                            "method": method,
                            "correlation": val,
                            "n": n,
                        }
                    )
    return rows


def confidence_strata_summary(
    conf_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if conf_df.empty or delta_df.empty:
        return rows
    keep_cols = _window_key_cols()
    for landscape, conf_block in conf_df.groupby("landscape"):
        threshold = float(conf_block["margin_z"].median())
        labels = conf_block[keep_cols + ["margin_z"]].copy()
        labels["confidence_stratum"] = np.where(labels["margin_z"] >= threshold, "high", "low")
        merged_all = delta_df.merge(labels, on=keep_cols, how="inner")
        for (objective, stratum), block in merged_all.groupby(["objective", "confidence_stratum"]):
            session_delta = block.groupby("session")["delta_cos2_vs_raw_edge"].mean().to_numpy(dtype=np.float64)
            ci_lo, ci_hi = _bootstrap_ci(session_delta, rng, n_bootstrap)
            rows.append(
                {
                    "landscape": landscape,
                    "objective": objective,
                    "confidence_stratum": stratum,
                    "threshold_margin_z": threshold,
                    "n_windows": int(block.shape[0]),
                    "n_sessions": int(block["session"].nunique()),
                    "mean_cos2_window": float(block["cos2_alignment"].mean()),
                    "mean_cos2_session_mean": _mean_session_stat(block, "cos2_alignment"),
                    "raw_edge_mean_cos2_session_mean": _mean_session_stat(block, "raw_edge_cos2"),
                    "mean_delta_cos2_vs_raw_edge_window": float(block["delta_cos2_vs_raw_edge"].mean()),
                    "mean_delta_cos2_vs_raw_edge_session": float(np.mean(session_delta)) if session_delta.size else float("nan"),
                    "delta_ci95_low": ci_lo,
                    "delta_ci95_high": ci_hi,
                    "n_sessions_delta_positive": int(np.count_nonzero(session_delta > 0)),
                    "mean_abs_delta_improvement_deg": float(block["abs_delta_improvement_vs_raw_edge_deg"].mean()),
                    "mean_drift_anisotropy": float(block["drift_anisotropy"].mean()) if "drift_anisotropy" in block else float("nan"),
                }
            )
    return rows


def write_figures(out_dir: Path, conf_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not conf_df.empty:
        fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
        for landscape, block in conf_df.groupby("landscape"):
            ax.scatter(block["margin_z"], block["edge_cos2_alignment"], s=12, alpha=0.55, label=landscape)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("landscape margin z")
        ax.set_ylabel("raw edge cos2 alignment")
        ax.set_title("Does objective confidence predict edge alignment?", loc="left", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "confidence_vs_raw_edge_alignment.png", dpi=150)
        plt.close(fig)
    if not delta_df.empty:
        key = delta_df[delta_df["objective"].isin(KEY_OBJECTIVES)].copy()
        if not key.empty:
            fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=150)
            order = key.groupby("objective")["delta_cos2_vs_raw_edge"].mean().sort_values(ascending=False).index
            data = [key.loc[key["objective"] == obj, "delta_cos2_vs_raw_edge"].to_numpy(dtype=float) for obj in order]
            ax.boxplot(data, tick_labels=order, showfliers=False)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("window delta cos2 vs raw edge")
            ax.set_title("Objective improvement over raw edge", loc="left", fontsize=10)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / "objective_delta_vs_raw_edge.png", dpi=150)
            plt.close(fig)


def write_summary(
    out_dir: Path,
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
) -> None:
    def find_stratum(landscape: str, objective: str, stratum: str) -> dict[str, Any] | None:
        for row in strata_rows:
            if row["landscape"] == landscape and row["objective"] == objective and row["confidence_stratum"] == stratum:
                return row
        return None

    lines = [
        "# Conditional Fixation Residual Summary",
        "",
        f"Source run: `{run_dir}`",
        "",
        "This cache-first summary tests whether objective confidence or residual predictions explain BackImage drift beyond the raw edge axis.",
        "",
        "## Main Readout",
        "",
    ]
    key = [row for row in summary_rows if row.get("objective") in KEY_OBJECTIVES]
    if key:
        for row in sorted(key, key=lambda r: float(r.get("mean_delta_cos2_session", np.nan)), reverse=True):
            lines.append(
                f"- `{row['objective']}`: session delta vs raw edge "
                f"{float(row['mean_delta_cos2_session']):+.4f}, CI "
                f"[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}], "
                f"{int(row['n_sessions_delta_positive'])}/{int(row['n_sessions'])} sessions positive."
            )
    else:
        lines.append("- No key objective rows found.")
    lines.extend(["", "## Confidence Strata", ""])
    for landscape, objective in (
        ("response_stability", "optimized_response_stability"),
        ("response_stability", "optimized_response_refresh_lambda_0.25"),
        ("pixel_isophote", "optimized_pixel_isophote"),
    ):
        hi = find_stratum(landscape, objective, "high")
        lo = find_stratum(landscape, objective, "low")
        if hi and lo:
            lines.append(
                f"- `{objective}` by `{landscape}` confidence: high-confidence session delta "
                f"{float(hi['mean_delta_cos2_vs_raw_edge_session']):+.4f}; low-confidence "
                f"{float(lo['mean_delta_cos2_vs_raw_edge_session']):+.4f}."
            )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "A positive result would show that high-confidence V1 stability landscapes predict stronger alignment, "
        "better residual axes, or positive deltas over raw edge. If these effects are absent, the current "
        "twin objective should remain demoted relative to simple local image geometry."
    )
    lines.append("")
    (out_dir / "conditional_residual_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = run_dir / "candidate_trajectory_scores.csv"
    alignment_path = run_dir / "real_vs_predicted_axis_alignment.csv"
    paired_path = run_dir / "paired_session_deltas_vs_raw_edge.csv"
    if not candidate_path.exists() or not alignment_path.exists():
        raise FileNotFoundError(f"Missing candidate/alignment CSVs in {run_dir}")
    candidate_df = pd.read_csv(candidate_path)
    alignment_df = pd.read_csv(alignment_path)
    paired_df = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    objectives = tuple(part.strip() for part in str(args.objectives).split(",") if part.strip())
    conf_df = landscape_confidence(candidate_df)
    delta_df = objective_delta_table(alignment_df, objectives)
    rng = np.random.default_rng(int(args.seed))
    corr_rows = confidence_correlation_summary(conf_df, delta_df)
    strata_rows = confidence_strata_summary(conf_df, delta_df, rng=rng, n_bootstrap=int(args.n_bootstrap))
    _write_csv(out_dir / "landscape_confidence_by_window.csv", conf_df.to_dict("records"))
    _write_csv(out_dir / "objective_delta_vs_raw_edge_by_window.csv", delta_df.to_dict("records"))
    _write_csv(out_dir / "confidence_correlation_summary.csv", corr_rows)
    _write_csv(out_dir / "confidence_stratified_objective_summary.csv", strata_rows)
    if not paired_df.empty:
        paired_key = paired_df[paired_df["objective"].isin(objectives)].to_dict("records")
    else:
        paired_key = []
    _write_csv(out_dir / "key_paired_deltas_vs_raw_edge.csv", paired_key)
    write_figures(out_dir, conf_df, delta_df)
    write_summary(out_dir, run_dir, paired_key, corr_rows, strata_rows)
    _write_json(
        out_dir / "conditional_residual_manifest.json",
        {
            "analysis": "backimage_conditional_fixation_residual_summary",
            "run_dir": run_dir,
            "out_dir": out_dir,
            "objectives": objectives,
            "n_candidate_rows": int(candidate_df.shape[0]),
            "n_alignment_rows": int(alignment_df.shape[0]),
            "n_confidence_rows": int(conf_df.shape[0]),
            "n_delta_rows": int(delta_df.shape[0]),
            "n_correlation_rows": len(corr_rows),
            "n_strata_rows": len(strata_rows),
            "claim_boundary": "Cache-first residual/confidence diagnostic; no new model responses are computed.",
        },
    )
    print(f"Wrote conditional fixation residual summary to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--objectives", default=",".join(KEY_OBJECTIVES))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

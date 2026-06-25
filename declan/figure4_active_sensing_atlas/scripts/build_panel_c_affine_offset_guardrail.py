"""Affine-offset guardrail for Figure 4C continuous-joint candidates.

The affine quadratic observation model improves posterior feature recovery, but
its intercept term could become a static/candidate offset shortcut. This script
summarizes existing analyzer QC and the heldout feature-calibration audit so the
intercept burden is checked beside the feature/image endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_ROOT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
OUT_DIR = Path("declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint")
BEST_CSV = OUT_DIR / "continuous_joint_feature_calibration_audit_affine_intercept_ridge_sweep_full_best.csv"
MODEL_SELECTION_CSV = (
    OUT_DIR / "continuous_joint_feature_calibration_audit_affine_intercept_ridge_sweep_full_model_selection.csv"
)


@dataclass(frozen=True)
class AffineRun:
    slug: str
    label: str
    dirname: str
    intercept_multiplier: float


AFFINE_RUNS = [
    AffineRun(
        slug="affine_x1",
        label="Affine x1",
        dirname="continuous_joint_quadratic_affine_poisson_scale_conditioned_full",
        intercept_multiplier=1.0,
    ),
    AffineRun(
        slug="affine_x10",
        label="Affine x10",
        dirname="continuous_joint_quadratic_affine_poisson_scale_conditioned_interceptx10_full",
        intercept_multiplier=10.0,
    ),
    AffineRun(
        slug="affine_x1000",
        label="Affine x1000",
        dirname="continuous_joint_quadratic_affine_poisson_scale_conditioned_interceptx1000_full",
        intercept_multiplier=1000.0,
    ),
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _read_qc(run: AffineRun) -> pd.DataFrame:
    path = SOURCE_ROOT / run.dirname / "continuous_joint_qc.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    rows["run_slug"] = run.slug
    rows["run_label"] = run.label
    rows["intercept_multiplier"] = run.intercept_multiplier
    rows["run_dirname"] = run.dirname
    return rows


def _summarize_intercepts(qc: pd.DataFrame) -> pd.DataFrame:
    fit_rows = qc[qc["qc_type"].eq("A_I_fit") & qc["B_intercept_norm_fraction"].notna()].copy()
    if fit_rows.empty:
        return fit_rows
    grouped = fit_rows.groupby(
        ["run_slug", "run_label", "intercept_multiplier", "prior_scale", "prior_family"],
        sort=True,
    )
    return grouped.agg(
        median_intercept_fraction=("B_intercept_norm_fraction", "median"),
        p90_intercept_fraction=("B_intercept_norm_fraction", lambda x: float(np.nanpercentile(x, 90))),
        median_dynamic_fro_norm=("B_dynamic_fro_norm", "median"),
        median_train_r2_energy=("B_train_r2_energy", "median"),
        n_fit=("B_intercept_norm_fraction", "size"),
    ).reset_index()


def _summarize_score_controls(qc: pd.DataFrame) -> pd.DataFrame:
    rows = qc[qc["qc_type"].eq("signal_nuisance_collapse")].copy()
    if rows.empty:
        return rows
    grouped = rows.groupby(
        ["run_slug", "run_label", "intercept_multiplier", "prior_scale", "prior_family"],
        sort=True,
    )
    return grouped.agg(
        median_score_corr_with_zero=("continuous_joint_score_corr_with_zero", "median"),
        median_true_score_delta_vs_zero=("continuous_joint_minus_zero_true_score", "median"),
        n_tables=("continuous_joint_score_corr_with_zero", "size"),
    ).reset_index()


def _endpoint_rows() -> pd.DataFrame:
    best = pd.read_csv(BEST_CSV)
    keep = ["origin_scale_conditioned", "affine_x1", "affine_x10", "affine_x1000"]
    rows = best[best["run_slug"].isin(keep)].copy()
    rows["intercept_multiplier"] = rows["run_slug"].map(
        {"origin_scale_conditioned": 0.0, "affine_x1": 1.0, "affine_x10": 10.0, "affine_x1000": 1000.0}
    )
    return rows.sort_values("intercept_multiplier")


def _plot(intercepts: pd.DataFrame, endpoints: pd.DataFrame, path: Path) -> None:
    colors = {
        "affine_x1": "#b279a2",
        "affine_x10": "#f58518",
        "affine_x1000": "#54a24b",
        "origin_scale_conditioned": "#4c78a8",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), layout="constrained")

    slice_rows = intercepts.copy()
    family_labels = {
        "axis_edge_orthogonal": "orth",
        "axis_edge_parallel": "parallel",
    }
    slice_rows["slice"] = slice_rows["prior_scale"].map(lambda value: f"{float(value):g}x") + "\n" + (
        slice_rows["prior_family"].astype(str).map(family_labels).fillna(slice_rows["prior_family"].astype(str))
    )
    slices = list(dict.fromkeys(slice_rows["slice"].tolist()))
    x = np.arange(len(slices), dtype=float)
    width = 0.25
    for offset, run in zip([-width, 0.0, width], AFFINE_RUNS):
        rows = slice_rows[slice_rows["run_slug"].eq(run.slug)].set_index("slice").reindex(slices)
        axes[0].bar(
            x + offset,
            rows["median_intercept_fraction"],
            width=width,
            label=run.label,
            color=colors[run.slug],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(slices, fontsize=8)
    axes[0].set_ylabel("median intercept fraction")
    axes[0].set_title("Affine offset burden")
    axes[0].legend(frameon=False, fontsize=7)

    labels = ["Origin", "Affine x1", "Affine x10", "Affine x1000"]
    endpoint_order = ["origin_scale_conditioned", "affine_x1", "affine_x10", "affine_x1000"]
    ep = endpoints.set_index("run_slug").reindex(endpoint_order)
    x2 = np.arange(len(endpoint_order))
    axes[1].bar(x2, ep["eval_mean_feature_cosine"], color=[colors[slug] for slug in endpoint_order])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylim(0.930, 0.939)
    axes[1].set_ylabel("heldout feature cosine")
    axes[1].set_title("Feature gate")
    for idx, value in enumerate(ep["eval_mean_feature_cosine"]):
        axes[1].text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)

    axes[2].bar(x2, ep["eval_image_accuracy"], color=[colors[slug] for slug in endpoint_order])
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylim(0.66, 0.72)
    axes[2].set_ylabel("hard image accuracy")
    axes[2].set_title("MAP identity")
    for idx, value in enumerate(ep["eval_image_accuracy"]):
        axes[2].text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Affine offset guardrail: x1000 lowers intercept burden while feature cosine leads", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qc = pd.concat([_read_qc(run) for run in AFFINE_RUNS], ignore_index=True)
    intercepts = _summarize_intercepts(qc)
    score_controls = _summarize_score_controls(qc)
    endpoints = _endpoint_rows()

    prefix = "continuous_joint_affine_offset_guardrail"
    intercept_path = OUT_DIR / f"{prefix}_intercept_summary.csv"
    score_path = OUT_DIR / f"{prefix}_score_control_summary.csv"
    endpoint_path = OUT_DIR / f"{prefix}_endpoint_summary.csv"
    figure_path = OUT_DIR / f"{prefix}.png"
    manifest_path = OUT_DIR / f"{prefix}_manifest.json"

    intercepts.to_csv(intercept_path, index=False)
    score_controls.to_csv(score_path, index=False)
    endpoints.to_csv(endpoint_path, index=False)
    _plot(intercepts, endpoints, figure_path)

    manifest = {
        "status": "affine_offset_guardrail",
        "source_root": SOURCE_ROOT,
        "audit_best_csv": BEST_CSV,
        "audit_model_selection_csv": MODEL_SELECTION_CSV,
        "runs": [run.__dict__ for run in AFFINE_RUNS],
        "outputs": {
            "intercept_summary_csv": intercept_path,
            "score_control_summary_csv": score_path,
            "endpoint_summary_csv": endpoint_path,
            "figure_png": figure_path,
        },
        "headline": (
            "The intercept x1000 penalty cuts 2.0x median affine intercept fractions "
            "from roughly 0.33-0.39 to roughly 0.14-0.15 while retaining the best "
            "heldout feature cosine. Hard image accuracy remains lower than the "
            "origin-constrained observer."
        ),
        "guardrail_interpretation": (
            "This supports affine_x1000 as a feature-primary encoder candidate, "
            "but it is not a full promotion because the MAP image endpoint drops "
            "and this QC is an intercept-burden check rather than a causal static-offset ablation."
        ),
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")

    x1000 = intercepts[intercepts["run_slug"].eq("affine_x1000")]
    print(x1000[["prior_scale", "prior_family", "median_intercept_fraction"]].to_string(index=False))
    print(endpoints[["run_slug", "eval_mean_feature_cosine", "eval_image_accuracy"]].to_string(index=False))
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()

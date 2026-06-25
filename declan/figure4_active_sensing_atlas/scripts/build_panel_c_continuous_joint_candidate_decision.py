"""Build the Figure 4C continuous-joint candidate decision table.

This consolidates the current feature-primary encoder/trajectory-prior
candidates into one provenance artifact. It separates full-cache promotion
gates from smoke screens so diagnostic leads are not accidentally promoted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint")

KNOWNSTART_FULL = OUT_DIR / "continuous_joint_feature_calibration_audit_knownstart_vs_inferred_full_best.csv"
AFFINE_FULL = OUT_DIR / "continuous_joint_feature_calibration_audit_affine_intercept_ablation_full_best.csv"
PRIOR_MEAN_SMOKE = OUT_DIR / "continuous_joint_feature_calibration_audit_prior_mean_centered_smoke64_best.csv"
KNOWNSTART_BROWNIAN_FULL = OUT_DIR / "continuous_joint_feature_calibration_audit_knownstart_brownian8_full_best.csv"
STRICT_SCALE_PRIOR_FULL = OUT_DIR / "continuous_joint_feature_calibration_audit_strict_scale_prior_predeclared_full_best.csv"
KNOWNSTART_SCALE_PRIOR_HYBRID_FULL = (
    OUT_DIR / "continuous_joint_feature_calibration_audit_knownstart_scale_prior_hybrid_predeclared_full_best.csv"
)
KNOWNSTART_MANIFEST = OUT_DIR / "continuous_joint_knownstart_observer_manifest.json"
PROMOTED_MANIFEST = OUT_DIR / "continuous_joint_promoted_observer_manifest.json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return _json_ready(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _row_from_best(
    best: pd.DataFrame,
    *,
    run_slug: str,
    decision_slug: str,
    decision_label: str,
    status: str,
    scope: str,
    gate_type: str,
    baseline_feature: float,
    interpretation: str,
    next_action: str,
    source_csv: Path,
) -> dict[str, Any]:
    row = best[best["run_slug"].eq(run_slug)]
    if row.empty:
        raise ValueError(f"missing run_slug {run_slug!r} in {source_csv}")
    item = row.iloc[0]
    feature = float(item["eval_mean_feature_cosine"])
    return {
        "decision_slug": decision_slug,
        "decision_label": decision_label,
        "run_slug": str(item["run_slug"]),
        "run_label": str(item["run_label"]),
        "status": status,
        "scope": scope,
        "gate_type": gate_type,
        "n_eval": int(item["n_eval"]),
        "heldout_feature_cosine": feature,
        "delta_vs_scope_baseline": feature - float(baseline_feature),
        "image_accuracy": float(item["eval_image_accuracy"]),
        "mean_true_mass": float(item["eval_mean_true_mass"]),
        "median_neff_fraction": float(item["eval_median_N_eff_fraction"]),
        "interpretation": interpretation,
        "next_action": next_action,
        "source_csv": source_csv,
    }


def build_decision_table() -> pd.DataFrame:
    known = pd.read_csv(KNOWNSTART_FULL)
    affine = pd.read_csv(AFFINE_FULL)
    smoke = pd.read_csv(PRIOR_MEAN_SMOKE)
    brownian = pd.read_csv(KNOWNSTART_BROWNIAN_FULL)
    strict_scale_prior = pd.read_csv(STRICT_SCALE_PRIOR_FULL)
    hybrid = pd.read_csv(KNOWNSTART_SCALE_PRIOR_HYBRID_FULL)

    full_baseline = float(known[known["run_slug"].eq("origin_inferred_full")]["eval_mean_feature_cosine"].iloc[0])
    smoke_baseline = float(smoke[smoke["run_slug"].eq("origin_smoke64")]["eval_mean_feature_cosine"].iloc[0])

    rows = [
        _row_from_best(
            strict_scale_prior,
            run_slug="strict_scale_prior_predeclared_full",
            decision_slug="strict_scale_prior_promoted",
            decision_label="Strict scale-prior",
            status="promoted_strict_endpoint",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation=(
                "Predeclared no-start scale-specific trajectory prior: AR(1) at 0.5x/1.0x "
                "and matched-Brownian scale 8 at 2.0x."
            ),
            next_action="Use as promoted strict no-start feature-recovery endpoint.",
            source_csv=STRICT_SCALE_PRIOR_FULL,
        ),
        _row_from_best(
            known,
            run_slug="origin_inferred_full",
            decision_slug="strict_inferred_start_previous",
            decision_label="Strict inferred-start",
            status="superseded_strict_endpoint",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation="Previous strict no-start feature-recovery endpoint.",
            next_action="Keep as baseline for strict endpoint improvement.",
            source_csv=KNOWNSTART_FULL,
        ),
        _row_from_best(
            known,
            run_slug="origin_knownstart_full",
            decision_slug="known_start_candidate",
            decision_label="Known-start candidate",
            status="candidate_less_strict",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation=(
                "Improves feature recovery and trajectory correlation by adding a soft prior "
                "on the first measured eye-position sample."
            ),
            next_action="Treat as clean feature-primary candidate, with first-sample caveat.",
            source_csv=KNOWNSTART_FULL,
        ),
        _row_from_best(
            brownian,
            run_slug="knownstart_brownian8_full",
            decision_slug="known_start_brownian8_full",
            decision_label="Known-start Brownian8",
            status="candidate_blocked_full_cache",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation=(
                "Smoke64 lead from a heldout matched-Brownian covariance prior does not beat "
                "known-start AR(1) on the full-cache feature gate."
            ),
            next_action="Keep as prior diagnostic; do not promote over known-start AR(1).",
            source_csv=KNOWNSTART_BROWNIAN_FULL,
        ),
        _row_from_best(
            hybrid,
            run_slug="knownstart_scale_prior_hybrid_predeclared_full",
            decision_slug="known_start_scale_prior_hybrid",
            decision_label="Scale-prior hybrid",
            status="candidate_predeclared_feature_lead",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation=(
                "Combines known-start AR(1) at 0.5x/1.0x with matched-Brownian scale-8 "
                "at 2.0x; predeclared source rerun improves full-cache feature recovery "
                "without image loss."
            ),
            next_action=(
                "Treat as the leading less-strict feature-primary candidate; keep strict "
                "inferred-start as the no-start endpoint."
            ),
            source_csv=KNOWNSTART_SCALE_PRIOR_HYBRID_FULL,
        ),
        _row_from_best(
            affine,
            run_slug="affine_x1000",
            decision_slug="affine_x1000_diagnostic",
            decision_label="Affine x1000",
            status="diagnostic_blocked",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation="Feature lead is intercept-dependent and image accuracy drops.",
            next_action="Do not promote without a principled interpretation of the intercept.",
            source_csv=AFFINE_FULL,
        ),
        _row_from_best(
            affine,
            run_slug="affine_x1000_intercept0",
            decision_slug="affine_intercept0_control",
            decision_label="Affine x1000 intercept=0",
            status="failed_control",
            scope="full_cache",
            gate_type="trial_split_scale_specific",
            baseline_feature=full_baseline,
            interpretation="Zeroing the affine intercept removes the feature lead.",
            next_action="Use as guardrail against affine promotion.",
            source_csv=AFFINE_FULL,
        ),
        _row_from_best(
            smoke,
            run_slug="origin_smoke64",
            decision_slug="smoke_origin_baseline",
            decision_label="Smoke origin",
            status="smoke_baseline",
            scope="smoke64",
            gate_type="trial_split_scale_specific",
            baseline_feature=smoke_baseline,
            interpretation="Matched smoke baseline for centered-prior screen.",
            next_action="Reference only within smoke64 comparisons.",
            source_csv=PRIOR_MEAN_SMOKE,
        ),
        _row_from_best(
            smoke,
            run_slug="prior_mean_centered_smoke64",
            decision_slug="prior_mean_centered_screen",
            decision_label="Prior-mean centered",
            status="screened_negative",
            scope="smoke64",
            gate_type="trial_split_scale_specific",
            baseline_feature=smoke_baseline,
            interpretation="More principled than a free intercept but below origin on the smoke screen.",
            next_action="Do not run full-cache unless a narrower subset predicts reversal.",
            source_csv=PRIOR_MEAN_SMOKE,
        ),
    ]
    return pd.DataFrame(rows)


def _plot(decisions: pd.DataFrame, path: Path) -> None:
    colors = {
        "promoted_strict_endpoint": "#4c78a8",
        "superseded_strict_endpoint": "#9ecae9",
        "candidate_less_strict": "#54a24b",
        "candidate_blocked_full_cache": "#72b7b2",
        "candidate_predeclared_feature_lead": "#59a14f",
        "diagnostic_blocked": "#f58518",
        "failed_control": "#b279a2",
        "smoke_baseline": "#4c78a8",
        "screened_negative": "#72b7b2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), layout="constrained")

    full = decisions[decisions["scope"].eq("full_cache")].copy()
    smoke = decisions[decisions["scope"].eq("smoke64")].copy()
    panels = [
        (axes[0], full, "Full-cache gates", (0.915, 0.940)),
        (axes[1], smoke, "Smoke64 screen", (0.950, 0.966)),
    ]
    for ax, rows, title, ylim in panels:
        x = np.arange(rows.shape[0])
        ax.bar(
            x,
            rows["heldout_feature_cosine"],
            color=[colors[str(status)] for status in rows["status"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rows["decision_label"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("heldout feature cosine")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        for patch, value in zip(ax.patches, rows["heldout_feature_cosine"]):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                float(value),
                f"{float(value):.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Figure 4C continuous-joint candidate decision", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions = build_decision_table()
    csv_path = OUT_DIR / "continuous_joint_candidate_decision_table.csv"
    fig_path = OUT_DIR / "continuous_joint_candidate_decision_table.png"
    manifest_path = OUT_DIR / "continuous_joint_candidate_decision_manifest.json"
    decisions.to_csv(csv_path, index=False)
    _plot(decisions, fig_path)
    manifest = {
        "status": "continuous_joint_candidate_decision",
        "decision_csv": csv_path,
        "figure_png": fig_path,
        "source_csvs": {
            "knownstart_full": KNOWNSTART_FULL,
            "knownstart_brownian_full": KNOWNSTART_BROWNIAN_FULL,
            "strict_scale_prior_full": STRICT_SCALE_PRIOR_FULL,
            "knownstart_scale_prior_hybrid_full": KNOWNSTART_SCALE_PRIOR_HYBRID_FULL,
            "affine_full": AFFINE_FULL,
            "prior_mean_smoke": PRIOR_MEAN_SMOKE,
        },
        "manifests": {
            "promoted": PROMOTED_MANIFEST,
            "knownstart_candidate": KNOWNSTART_MANIFEST,
        },
        "headline": (
            "Promote the predeclared strict scale-prior endpoint: it improves the no-start "
            "feature gate without changing image accuracy. Known-start remains a less-strict "
            "candidate with first-sample caveat. Matched Brownian improves the smoke screen "
            "but does not beat known-start AR(1) as a global prior on full cache. Affine x1000 remains "
            "diagnostic because its feature lead is intercept-dependent. Prior-mean "
            "centering is screened negative on smoke64."
        ),
        "decisions": decisions,
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(decisions[["decision_slug", "scope", "status", "heldout_feature_cosine", "image_accuracy"]].to_string(index=False))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

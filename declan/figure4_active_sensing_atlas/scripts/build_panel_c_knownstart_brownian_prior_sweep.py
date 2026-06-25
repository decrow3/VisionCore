"""Plot the Figure 4C known-start matched-Brownian prior smoke sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint")
BEST_CSV = OUT_DIR / "continuous_joint_feature_calibration_audit_knownstart_brownian_sweep_smoke64_best.csv"
MODEL_SELECTION_CSV = (
    OUT_DIR / "continuous_joint_feature_calibration_audit_knownstart_brownian_sweep_smoke64_model_selection.csv"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return _json_ready(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _brownian_scale(run_slug: str) -> float | None:
    marker = "knownstart_brownian_"
    suffix = "_smoke64"
    if not run_slug.startswith(marker) or not run_slug.endswith(suffix):
        return None
    token = run_slug[len(marker) : -len(suffix)]
    return float(token.replace("p", "."))


def build_summary() -> pd.DataFrame:
    best = pd.read_csv(BEST_CSV)
    rows = []
    for item in best.to_dict(orient="records"):
        scale = _brownian_scale(str(item["run_slug"]))
        prior_family = "matched_brownian" if scale is not None else "reference"
        rows.append(
            {
                "run_slug": str(item["run_slug"]),
                "run_label": str(item["run_label"]),
                "prior_family": prior_family,
                "brownian_cov_scale": scale,
                "heldout_feature_cosine": float(item["eval_mean_feature_cosine"]),
                "default_feature_cosine": float(item["default_mean_feature_cosine"]),
                "image_accuracy": float(item["eval_image_accuracy"]),
                "mean_true_mass": float(item["eval_mean_true_mass"]),
                "median_neff_fraction": float(item["eval_median_N_eff_fraction"]),
                "selected_temperature_by_split": str(item["selected_temperature_by_split"]),
            }
        )
    out = pd.DataFrame(rows)
    out["plot_order"] = out["brownian_cov_scale"].fillna(-1.0)
    return out.sort_values(["prior_family", "plot_order", "run_slug"]).drop(columns=["plot_order"])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    brownian = summary[summary["prior_family"].eq("matched_brownian")].sort_values("brownian_cov_scale")
    refs = summary[summary["prior_family"].eq("reference")].copy()
    ar1 = refs[refs["run_slug"].eq("knownstart_ar1_smoke64")]
    inferred = refs[refs["run_slug"].eq("origin_smoke64")]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), layout="constrained")
    ax = axes[0]
    ax.plot(
        brownian["brownian_cov_scale"],
        brownian["heldout_feature_cosine"],
        marker="o",
        color="#4c78a8",
        label="known-start Brownian",
    )
    if not ar1.empty:
        ax.axhline(float(ar1["heldout_feature_cosine"].iloc[0]), color="#54a24b", linestyle="--", label="known-start AR(1)")
    if not inferred.empty:
        ax.axhline(float(inferred["heldout_feature_cosine"].iloc[0]), color="#777777", linestyle=":", label="inferred-start")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Brownian covariance scale")
    ax.set_ylabel("heldout feature cosine")
    ax.set_title("Feature recovery")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(
        brownian["brownian_cov_scale"],
        brownian["image_accuracy"],
        marker="o",
        color="#f58518",
        label="known-start Brownian",
    )
    if not ar1.empty:
        ax.axhline(float(ar1["image_accuracy"].iloc[0]), color="#54a24b", linestyle="--", label="known-start AR(1)")
    if not inferred.empty:
        ax.axhline(float(inferred["image_accuracy"].iloc[0]), color="#777777", linestyle=":", label="inferred-start")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Brownian covariance scale")
    ax.set_ylabel("image accuracy")
    ax.set_title("Hard identity")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Known-start matched-Brownian prior sweep (smoke64)", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary_path = OUT_DIR / "continuous_joint_knownstart_brownian_prior_sweep_smoke64.csv"
    fig_path = OUT_DIR / "continuous_joint_knownstart_brownian_prior_sweep_smoke64.png"
    manifest_path = OUT_DIR / "continuous_joint_knownstart_brownian_prior_sweep_smoke64_manifest.json"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, fig_path)
    best = summary.sort_values("heldout_feature_cosine", ascending=False).iloc[0]
    model_selection = pd.read_csv(MODEL_SELECTION_CSV)
    manifest = {
        "status": "knownstart_brownian_prior_sweep_smoke64",
        "summary_csv": summary_path,
        "figure_png": fig_path,
        "source_best_csv": BEST_CSV,
        "source_model_selection_csv": MODEL_SELECTION_CSV,
        "best_run_slug": str(best["run_slug"]),
        "best_heldout_feature_cosine": float(best["heldout_feature_cosine"]),
        "best_image_accuracy": float(best["image_accuracy"]),
        "model_selection": model_selection,
        "interpretation": (
            "After fixing quadratic matched-Brownian covariance use in the final "
            "profile objective, the 64-table smoke sweep favors a loose matched "
            "Brownian known-start prior. Treat as a full-cache candidate, not a "
            "promotion, until the same heldout feature gate is run on the full cache."
        ),
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(summary[["run_slug", "heldout_feature_cosine", "image_accuracy"]].to_string(index=False))
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()

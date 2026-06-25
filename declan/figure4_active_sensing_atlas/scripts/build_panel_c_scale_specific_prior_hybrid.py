"""Build a scale-specific Figure 4C prior-hybrid posterior artifact.

The hybrid is deliberately narrow: use the known-start AR(1) quadratic observer
for 0.5x/1.0x and the matched-Brownian scale-8 observer for the hard 2.0x
slice. This tests whether the Brownian prior is useful as a scale-specific
trajectory prior without changing the observation model or using anchors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_ROOT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
AR1_DIR = SOURCE_ROOT / "continuous_joint_quadratic_poisson_scale_conditioned_knownstart_full"
BROWNIAN8_DIR = SOURCE_ROOT / "continuous_joint_quadratic_poisson_scale_conditioned_knownstart_matched_brownian_scale8_full"
OUT_DIR = SOURCE_ROOT / "continuous_joint_quadratic_poisson_scale_conditioned_knownstart_scale_specific_prior_hybrid_full"


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


def _read_posterior(path: Path) -> pd.DataFrame:
    csv_path = path / "continuous_joint_feature_posterior.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = pd.read_csv(csv_path)
    if "prior_scale" not in rows.columns:
        raise ValueError(f"posterior lacks prior_scale: {csv_path}")
    return rows


def build_hybrid() -> pd.DataFrame:
    ar1 = _read_posterior(AR1_DIR)
    brownian = _read_posterior(BROWNIAN8_DIR)
    key_cols = ["table_index", "candidate_index", "observer_mode"]
    ar1_keys = ar1[key_cols].astype(str).agg("|".join, axis=1)
    brownian_keys = brownian[key_cols].astype(str).agg("|".join, axis=1)
    if set(ar1_keys) != set(brownian_keys):
        raise ValueError("AR(1) and Brownian posterior tables do not contain matching candidate keys")

    use_brownian = brownian["prior_scale"].astype(float).eq(2.0)
    hybrid = pd.concat(
        [
            ar1[~ar1["prior_scale"].astype(float).eq(2.0)],
            brownian[use_brownian],
        ],
        ignore_index=True,
    )
    hybrid["prior_hybrid_source"] = np.where(
        hybrid["prior_scale"].astype(float).eq(2.0),
        "knownstart_brownian8",
        "knownstart_ar1",
    )
    return hybrid.sort_values(["table_index", "observer_mode", "candidate_index"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hybrid = build_hybrid()
    posterior_path = OUT_DIR / "continuous_joint_feature_posterior.csv"
    manifest_path = OUT_DIR / "scale_specific_prior_hybrid_manifest.json"
    hybrid.to_csv(posterior_path, index=False)

    summary = (
        hybrid[["table_index", "prior_scale", "prior_hybrid_source"]]
        .drop_duplicates()
        .groupby(["prior_scale", "prior_hybrid_source"], as_index=False)
        .agg(n_tables=("table_index", "nunique"))
        .sort_values(["prior_scale", "prior_hybrid_source"])
    )
    manifest = {
        "status": "scale_specific_prior_hybrid_posterior",
        "posterior_csv": posterior_path,
        "source_dirs": {
            "knownstart_ar1": AR1_DIR,
            "knownstart_brownian8": BROWNIAN8_DIR,
        },
        "rule": "Use known-start AR(1) posterior rows for 0.5x and 1.0x; use known-start matched-Brownian scale-8 rows for 2.0x.",
        "summary": summary,
        "interpretation": (
            "Diagnostic candidate for a scale-specific trajectory prior. It is not "
            "an anchor and does not alter the compact quadratic observation model."
        ),
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"wrote {posterior_path}")


if __name__ == "__main__":
    main()

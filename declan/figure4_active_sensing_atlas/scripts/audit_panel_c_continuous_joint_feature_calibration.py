"""Heldout feature-calibration audit for Figure 4C continuous-joint runs.

This is a reusable gate for proposed no-anchor encoder changes. It scores a
run's continuous-joint candidate posteriors against the same feature-recovery
metric used by the Panel C diagnostics, then selects posterior temperatures on
one split and evaluates them on the heldout split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_joint_feature_recovery import (
    OUT_DIR,
    PRIMARY_LATENT,
    TEMPERATURES,
    _compute_temperature_cv,
    _load_feature_tables,
    _summarize_temperature_cv,
    _vectorized_mode_rows,
)
from declan.figure4_active_sensing_atlas.scripts.run_panel_c_promoted_continuous_joint_observer import (
    DEFAULT_OUT_DIR,
)


@dataclass(frozen=True)
class AuditRun:
    slug: str
    label: str
    run_dir: Path
    family: str = "candidate"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return [_json_ready(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _parse_run_spec(text: str) -> AuditRun:
    parts = text.split("|")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "--run must be formatted as 'slug|label|run_dir' or 'slug|label|family|run_dir'"
        )
    if len(parts) == 3:
        slug, label, run_dir = parts
        family = "candidate"
    else:
        slug, label, family, run_dir = parts
    if not slug:
        raise argparse.ArgumentTypeError("run slug must be non-empty")
    if not label:
        raise argparse.ArgumentTypeError("run label must be non-empty")
    return AuditRun(slug=slug, label=label, family=family or "candidate", run_dir=Path(run_dir))


def _parse_temperatures(text: str) -> np.ndarray:
    values = np.asarray([float(part.strip()) for part in text.split(",") if part.strip()], dtype=np.float64)
    if values.size == 0:
        raise argparse.ArgumentTypeError("at least one posterior temperature is required")
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise argparse.ArgumentTypeError("posterior temperatures must be positive finite values")
    return np.unique(values)


def _read_posterior(run: AuditRun) -> pd.DataFrame:
    path = run.run_dir / "continuous_joint_feature_posterior.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing posterior CSV for {run.slug}: {path}")
    rows = pd.read_csv(path)
    if "likelihood_scale" in rows.columns:
        rows = rows[rows["likelihood_scale"].astype(float).eq(1.0)].copy()
    rows["run_slug"] = run.slug
    rows["run_label"] = run.label
    rows["run_family"] = run.family
    rows["run_dirname"] = run.run_dir.name
    return rows


def _compute_temperature_sweep(
    runs: list[AuditRun],
    *,
    latent: str,
    temperatures: np.ndarray,
) -> pd.DataFrame:
    feature_tables = _load_feature_tables()
    if latent not in feature_tables:
        raise ValueError(f"latent {latent!r} not found in feature table")
    feature_table = feature_tables[latent]

    out_frames: list[pd.DataFrame] = []
    for run in runs:
        rows = _read_posterior(run)
        mode_rows = rows[rows["observer_mode"].eq("continuous_joint")].copy()
        if mode_rows.empty:
            raise ValueError(f"{run.slug} has no continuous_joint posterior rows")
        required = {"candidate_id", "candidate_index", "candidate_score", "is_true_candidate"}
        missing = required.difference(mode_rows.columns)
        if missing:
            raise ValueError(f"{run.slug} posterior rows missing required columns: {sorted(missing)}")
        score_column = "candidate_score_raw" if "candidate_score_raw" in mode_rows.columns else "candidate_score"
        for temp in temperatures:
            out_frames.append(
                _vectorized_mode_rows(
                    rows=mode_rows,
                    latent=latent,
                    feature_table=feature_table,
                    posterior_temperature=float(temp),
                    score_column=score_column,
                )
            )
    return pd.concat(out_frames, ignore_index=True)


def _summary_for_mode(cv_summary: pd.DataFrame, split_key: str, calibration_mode: str) -> pd.DataFrame:
    return cv_summary[
        cv_summary["split_key"].eq(split_key)
        & cv_summary["calibration_mode"].eq(calibration_mode)
    ].copy()


def _best_rows(cv_summary: pd.DataFrame, *, calibration_mode: str) -> pd.DataFrame:
    if cv_summary.empty:
        return cv_summary
    rows = cv_summary.copy()
    preferred_split = "trial_id" if rows["split_key"].eq("trial_id").any() else "table_index"
    rows = rows[rows["split_key"].eq(preferred_split)].copy()
    if calibration_mode != "best":
        rows = rows[rows["calibration_mode"].eq(calibration_mode)].copy()
    if rows.empty:
        return rows
    return rows.sort_values(
        ["eval_mean_feature_cosine", "delta_vs_default", "run_slug"],
        ascending=[False, False, True],
    ).groupby("run_slug", as_index=False).head(1)


def _model_selection_rows(cv_rows: pd.DataFrame, *, calibration_mode: str) -> pd.DataFrame:
    """Split-swapped run selection using heldout feature cosine as the score."""
    if cv_rows.empty:
        return pd.DataFrame()
    rows = cv_rows.copy()
    split_key = "trial_id" if rows["split_key"].eq("trial_id").any() else "table_index"
    rows = rows[
        rows["split_key"].eq(split_key)
        & rows["calibration_mode"].eq(calibration_mode)
        & rows["prior_scale"].astype(str).eq("all")
        & rows["prior_family"].astype(str).eq("all")
    ].copy()
    if rows.empty:
        return rows

    eval_splits = sorted(int(value) for value in rows["eval_split"].dropna().unique())
    selected_rows: list[dict[str, Any]] = []
    for eval_split in eval_splits:
        train_splits = [value for value in eval_splits if value != eval_split]
        if not train_splits:
            continue
        train_split = train_splits[0]
        train_rows = rows[rows["eval_split"].astype(int).eq(train_split)].copy()
        eval_rows = rows[rows["eval_split"].astype(int).eq(eval_split)].set_index("run_slug")
        if train_rows.empty or eval_rows.empty:
            continue
        winner = train_rows.sort_values(
            ["eval_mean_feature_cosine", "delta_vs_default", "run_slug"],
            ascending=[False, False, True],
        ).iloc[0]
        run_slug = str(winner["run_slug"])
        if run_slug not in eval_rows.index:
            continue
        evaluated = eval_rows.loc[run_slug]
        if isinstance(evaluated, pd.DataFrame):
            evaluated = evaluated.iloc[0]
        selected_rows.append(
            {
                "row_type": "split",
                "split_key": split_key,
                "calibration_mode": calibration_mode,
                "prior_scale": "all",
                "prior_family": "all",
                "eval_split": eval_split,
                "selected_on_split": train_split,
                "selected_run_slug": run_slug,
                "selected_run_label": str(winner["run_label"]),
                "train_feature_cosine": float(winner["eval_mean_feature_cosine"]),
                "eval_feature_cosine": float(evaluated["eval_mean_feature_cosine"]),
                "eval_image_accuracy": float(evaluated["eval_image_accuracy"]),
                "eval_delta_vs_default": float(evaluated["delta_vs_default"]),
                "selected_temperature": str(evaluated["selected_temperature"]),
                "n_eval": int(evaluated["n_eval"]),
            }
        )

    out = pd.DataFrame(selected_rows)
    if out.empty:
        return out
    n_eval = out["n_eval"].astype(float)
    aggregate = {
        "row_type": "aggregate",
        "split_key": split_key,
        "calibration_mode": calibration_mode,
        "prior_scale": "all",
        "prior_family": "all",
        "eval_split": -1,
        "selected_on_split": -1,
        "selected_run_slug": ";".join(out["selected_run_slug"].astype(str)),
        "selected_run_label": ";".join(out["selected_run_label"].astype(str)),
        "train_feature_cosine": float(np.average(out["train_feature_cosine"], weights=n_eval)),
        "eval_feature_cosine": float(np.average(out["eval_feature_cosine"], weights=n_eval)),
        "eval_image_accuracy": float(np.average(out["eval_image_accuracy"], weights=n_eval)),
        "eval_delta_vs_default": float(np.average(out["eval_delta_vs_default"], weights=n_eval)),
        "selected_temperature": ";".join(out["selected_temperature"].astype(str)),
        "n_eval": int(out["n_eval"].sum()),
    }
    return pd.concat([out, pd.DataFrame([aggregate])], ignore_index=True)


def audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = list(args.run) if args.run else [
        AuditRun(
            slug="promoted_scale_calibrated",
            label="Promoted scale-calibrated",
            family="promoted",
            run_dir=DEFAULT_OUT_DIR,
        )
    ]
    temperatures = _parse_temperatures(str(args.temperatures))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)

    sweep = _compute_temperature_sweep(runs, latent=str(args.latent), temperatures=temperatures)
    cv = _compute_temperature_cv(sweep)
    cv_summary = _summarize_temperature_cv(cv)
    best = _best_rows(cv_summary, calibration_mode=str(args.promotion_calibration_mode))
    model_selection = _model_selection_rows(
        cv,
        calibration_mode=str(args.promotion_calibration_mode),
    )

    sweep_path = out_dir / f"{prefix}_trials.csv"
    cv_path = out_dir / f"{prefix}_cv.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    best_path = out_dir / f"{prefix}_best.csv"
    model_selection_path = out_dir / f"{prefix}_model_selection.csv"
    manifest_path = out_dir / f"{prefix}_manifest.json"
    sweep.to_csv(sweep_path, index=False)
    cv.to_csv(cv_path, index=False)
    cv_summary.to_csv(summary_path, index=False)
    best.to_csv(best_path, index=False)
    model_selection.to_csv(model_selection_path, index=False)

    manifest = {
        "status": "heldout_feature_calibration_audit",
        "latent": str(args.latent),
        "temperatures": temperatures.tolist(),
        "promotion_calibration_mode": str(args.promotion_calibration_mode),
        "runs": [
            {
                "slug": run.slug,
                "label": run.label,
                "family": run.family,
                "run_dir": run.run_dir,
            }
            for run in runs
        ],
        "outputs": {
            "trials_csv": sweep_path,
            "cv_csv": cv_path,
            "summary_csv": summary_path,
            "best_csv": best_path,
            "model_selection_csv": model_selection_path,
        },
        "best_by_run": best,
        "model_selection": model_selection,
        "interpretation": (
            "Use heldout posterior-weighted feature cosine as the development gate. "
            "Image accuracy remains the hard MAP identity endpoint and should not be "
            "used alone to promote continuous no-anchor encoder changes. "
            "When multiple run classes are supplied, model_selection_csv reports the "
            "split-swapped run choice under the predeclared promotion calibration mode."
        ),
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")

    print(best.to_string(index=False))
    if not model_selection.empty:
        print(model_selection.to_string(index=False))
    print(f"wrote {summary_path}")
    return cv_summary, best, model_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_spec,
        default=[],
        help="Run to audit as 'slug|label|run_dir' or 'slug|label|family|run_dir'. Defaults to the promoted run.",
    )
    parser.add_argument("--latent", default=PRIMARY_LATENT)
    parser.add_argument(
        "--temperatures",
        default=",".join(f"{value:g}" for value in TEMPERATURES),
        help="Comma-separated positive posterior temperatures to evaluate.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--output-prefix", default="continuous_joint_feature_calibration_audit")
    parser.add_argument(
        "--promotion-calibration-mode",
        choices=["scale_specific", "global", "scale_family_specific", "best"],
        default="scale_specific",
        help=(
            "Calibration mode used for the best-by-run gate. The default is predeclared "
            "scale-specific calibration; 'best' is exploratory and chooses among modes post hoc."
        ),
    )
    return parser


def main() -> None:
    audit(build_parser().parse_args())


if __name__ == "__main__":
    main()

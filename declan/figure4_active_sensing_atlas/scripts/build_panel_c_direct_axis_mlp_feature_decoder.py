"""Direct along/across response-movie MLP feature decoder for Figure 4C/4D.

This is the direct version of the final axis question.  It does not hold the
observed movie fixed and only change the trajectory prior.  Instead, for each
axis-conditioned response table, the true candidate's sampled along- or
across-axis response movie is treated as the observation, and the candidate-free
MLP decoder maps that compact response movie directly to ``phi(image)``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    COMPACT_BASIS,
    FEATURE_NPZ,
    MLPConfig,
    _assign_source_folds,
    _bootstrap_mean,
    _fit_feature_transform,
    _fit_predict_mlp,
    _json_ready,
    _load_basis,
    _load_feature_table,
    _load_npz,
    _metrics,
    _parse_scales,
    _parse_str_list,
    _source_row_from_candidate_id,
    _transform_feature_sources,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_tau_mlp_feature_decoder import (
    PROMOTED_MANIFEST,
    _base_mode,
    _compact_matrix,
    _fit_predict_residual_mlp,
    _float_key_map,
    _input_vector,
    _load_json,
    _load_sidecar,
    _parse_float_list,
    _sidecar_path,
    _trajectory_from_table_or_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "direct_axis_mlp_feature_decoder"
)
AXIS_FAMILIES = {"axis_edge_parallel", "axis_edge_orthogonal"}
DEFAULT_INPUT_MODES = (
    "augmented_observed_compact",
    "augmented_true_tau",
    "augmented_true_tau_residual",
    "augmented_zero_static",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_rows(trials: pd.DataFrame, *, scales: set[float], max_tables: int) -> pd.DataFrame:
    rows = trials.copy()
    if "prior_scale" not in rows.columns and "scale" in rows.columns:
        rows["prior_scale"] = rows["scale"].astype(float)
    if "manifest_table_index" not in rows.columns:
        rows["manifest_table_index"] = np.arange(rows.shape[0], dtype=int)
    rows = rows[rows["response_cache_path"].astype(str).str.len() > 0].copy()
    rows = rows[rows["prior_family"].astype(str).isin(AXIS_FAMILIES)].copy()
    if scales:
        rows = rows[rows["prior_scale"].astype(float).isin(scales)].copy()
    rows = rows.sort_values(["trial_id", "prior_scale", "prior_family"]).reset_index(drop=True)
    if int(max_tables) > 0:
        rows = rows.iloc[: int(max_tables)].copy()
    if rows.empty:
        raise ValueError("No axis-conditioned response rows selected")
    return rows


def _is_residual_mode(mode: str) -> bool:
    return str(mode) == "augmented_true_tau_residual"


def _is_prior_augmented_mode(mode: str) -> bool:
    return str(mode) in {"augmented_observed_compact", "augmented_true_tau", "augmented_true_tau_residual"}


def _mode_needs_tau(mode: str) -> bool:
    return _base_mode(mode, for_test=True) in {
        "observed_plus_continuous_tau",
        "observed_plus_continuous_tau_interactions",
        "observed_plus_true_tau",
        "observed_plus_true_tau_interactions",
    } or _base_mode(mode, for_test=False) in {
        "observed_plus_continuous_tau",
        "observed_plus_continuous_tau_interactions",
        "observed_plus_true_tau",
        "observed_plus_true_tau_interactions",
    }


def _load_trajectory_if_needed(
    *,
    table: dict[str, np.ndarray],
    sidecar: dict[str, np.ndarray] | None,
    metadata: dict[str, Any],
    response_cache_path: str,
    needs_tau: bool,
) -> np.ndarray | None:
    if not needs_tau:
        return None
    try:
        return _trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=sidecar,
            trajectory_key=str(metadata.get("trajectory_key", "prior_trajectory_xy")),
        )
    except Exception as exc:
        raise ValueError(
            f"Input modes require trajectory coordinates, but none were available for {response_cache_path}"
        ) from exc


def _build_dataset(
    *,
    rows: pd.DataFrame,
    response_run_dir: Path,
    metadata: dict[str, Any],
    projection_basis: np.ndarray,
    input_modes: list[str],
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    sidecar_dir = Path(metadata["trajectory_sidecar_dir"]) if metadata.get("trajectory_sidecar_dir") else None
    needs_tau = any(_mode_needs_tau(mode) for mode in input_modes)
    parts = {mode: [] for mode in input_modes}
    train_parts = {mode: [] for mode in input_modes if str(mode).startswith("augmented_")}
    train_sources = {mode: [] for mode in train_parts}
    table_rows: list[dict[str, Any]] = []

    for row_index, row in rows.iterrows():
        if progress_every > 0 and (row_index + 1) % int(progress_every) == 0:
            print(f"loaded direct-axis rows {row_index + 1} / {rows.shape[0]}", flush=True)
        response_cache_path = str(row["response_cache_path"])
        table = _load_npz(response_run_dir / response_cache_path)
        sidecar = _load_sidecar(_sidecar_path(sidecar_dir, response_cache_path))
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        candidate_ids = [str(value) for value in np.asarray(table["candidate_ids"]).tolist()]
        source_rows = [_source_row_from_candidate_id(candidate_id) for candidate_id in candidate_ids]
        true_idx = int(np.asarray(table["true_candidate_index"]).reshape(-1)[0])
        true_source = int(source_rows[true_idx])
        trajectory_xy = _load_trajectory_if_needed(
            table=table,
            sidecar=sidecar,
            metadata=metadata,
            response_cache_path=response_cache_path,
            needs_tau=needs_tau,
        )
        zero_compact = _compact_matrix(zero[true_idx], projection_basis)

        for trajectory_index in range(int(prior.shape[1])):
            observed_compact = _compact_matrix(prior[true_idx, trajectory_index], projection_basis)
            true_tau = (
                np.asarray(trajectory_xy, dtype=np.float64)[true_idx, trajectory_index]
                if trajectory_xy is not None
                else np.zeros((observed_compact.shape[0], 2), dtype=np.float64)
            )
            for mode in input_modes:
                parts[mode].append(
                    _input_vector(
                        _base_mode(mode, for_test=True),
                        observed_compact=observed_compact,
                        zero_compact=zero_compact,
                        known_compact=observed_compact,
                        tau_hat=true_tau,
                        true_tau=true_tau,
                    )
                )
            table_rows.append(
                {
                    "row_index": int(row_index),
                    "table_index": int(row.get("table_index", row_index)),
                    "manifest_table_index": int(row.get("manifest_table_index", row_index)),
                    "trial_id": int(row["trial_id"]),
                    "trajectory_index": int(trajectory_index),
                    "response_cache_path": response_cache_path,
                    "candidate_set_mode": str(row.get("candidate_set_mode", "")),
                    "observation_family": str(row.get("observation_family", "")),
                    "prior_family": str(row.get("prior_family", "")),
                    "prior_scale": float(row["prior_scale"]),
                    "axis_catalog_mode": str(row.get("axis_catalog_mode", "")),
                    "true_candidate_index": int(true_idx),
                    "true_candidate_id": candidate_ids[true_idx],
                    "true_source_row": int(true_source),
                    "n_candidates": int(prior.shape[0]),
                    "n_trajectories": int(prior.shape[1]),
                    "n_timebins": int(prior.shape[2]),
                    "n_units": int(prior.shape[3]),
                    "projection_basis_dim": int(projection_basis.shape[1]),
                }
            )

        for mode in train_parts:
            train_base_mode = _base_mode(mode, for_test=False)
            if _is_prior_augmented_mode(mode):
                for candidate_index, source_row in enumerate(source_rows):
                    for trajectory_index in range(int(prior.shape[1])):
                        train_compact = _compact_matrix(prior[candidate_index, trajectory_index], projection_basis)
                        train_tau = (
                            np.asarray(trajectory_xy, dtype=np.float64)[candidate_index, trajectory_index]
                            if trajectory_xy is not None
                            else np.zeros((train_compact.shape[0], 2), dtype=np.float64)
                        )
                        train_parts[mode].append(
                            _input_vector(
                                train_base_mode,
                                observed_compact=train_compact,
                                zero_compact=train_compact,
                                known_compact=train_compact,
                                tau_hat=train_tau,
                                true_tau=train_tau,
                            )
                        )
                        train_sources[mode].append(int(source_row))
            elif mode == "augmented_zero_static":
                for candidate_index, source_row in enumerate(source_rows):
                    train_compact = _compact_matrix(zero[candidate_index], projection_basis)
                    train_tau = np.zeros((train_compact.shape[0], 2), dtype=np.float64)
                    train_parts[mode].append(
                        _input_vector(
                            train_base_mode,
                            observed_compact=train_compact,
                            zero_compact=train_compact,
                            known_compact=train_compact,
                            tau_hat=train_tau,
                            true_tau=train_tau,
                        )
                    )
                    train_sources[mode].append(int(source_row))
            else:
                raise ValueError(f"No training bank defined for {mode!r}")

    arrays = {mode: np.stack(values, axis=0).astype(np.float32) for mode, values in parts.items()}
    train_arrays = {mode: np.stack(values, axis=0).astype(np.float32) for mode, values in train_parts.items()}
    train_source_arrays = {mode: np.asarray(values, dtype=np.int64) for mode, values in train_sources.items()}
    meta = {
        "input_modes": input_modes,
        "n_axis_tables": int(rows.shape[0]),
        "n_test_rows": int(len(table_rows)),
        "input_dims": {mode: int(arr.shape[1]) for mode, arr in arrays.items()},
        "augmented_train_rows": {mode: int(train_arrays[mode].shape[0]) for mode in train_arrays},
    }
    return arrays, train_arrays, train_source_arrays, pd.DataFrame(table_rows), meta


def _fit_modes(
    *,
    arrays: dict[str, np.ndarray],
    train_arrays: dict[str, np.ndarray],
    train_source_rows_by_mode: dict[str, np.ndarray],
    table_rows: pd.DataFrame,
    feature_table: Any,
    feature_space_mode: str,
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    mlp_config: MLPConfig,
    residual_alpha_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = table_rows["true_source_row"].to_numpy(dtype=int)
    fold_by_source = _assign_source_folds(source_rows, n_folds=n_folds, seed=fold_seed)
    row_folds = np.asarray([fold_by_source[int(source)] for source in source_rows], dtype=int)
    response_input_dim = int(arrays["augmented_observed_compact"].shape[1])
    trial_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for fold in sorted(set(row_folds.tolist())):
        heldout_sources = {int(source) for source, source_fold in fold_by_source.items() if int(source_fold) == int(fold)}
        fit_sources = np.asarray(
            [int(source) for source in feature_table.source_rows.tolist() if int(source) not in heldout_sources],
            dtype=int,
        )
        transform = _fit_feature_transform(
            feature_table,
            fit_sources=fit_sources,
            feature_dim=int(feature_dim),
            feature_space_mode=str(feature_space_mode),
            feature_weights=None,
        )
        z_all = _transform_feature_sources(transform, feature_table, source_rows)
        test_mask = row_folds == int(fold)
        z_true = z_all[test_mask]
        test_meta = table_rows.loc[test_mask].reset_index(drop=True)
        for mode, x_all in arrays.items():
            fit_x_all = train_arrays[mode]
            fit_source_rows = train_source_rows_by_mode[mode]
            fit_z_all = _transform_feature_sources(transform, feature_table, fit_source_rows)
            train_mask = np.asarray([fold_by_source.get(int(source), -1) != int(fold) for source in fit_source_rows])
            if _is_residual_mode(mode):
                z_hat, stats = _fit_predict_residual_mlp(
                    base_x_all=fit_x_all[:, :response_input_dim],
                    residual_x_all=fit_x_all,
                    z_all=fit_z_all,
                    source_rows=fit_source_rows,
                    train_mask=train_mask,
                    base_x_test=x_all[test_mask, :response_input_dim],
                    residual_x_test=x_all[test_mask],
                    config=mlp_config,
                    fold=int(fold),
                    spec_slug=mode,
                    feature_space_mode=transform.feature_space_mode,
                    alpha_grid=residual_alpha_grid,
                )
                decoder_mode = "mlp_residual"
            else:
                z_hat, stats = _fit_predict_mlp(
                    x_all=fit_x_all,
                    z_all=fit_z_all,
                    source_rows=fit_source_rows,
                    train_mask=train_mask,
                    x_test=x_all[test_mask],
                    config=mlp_config,
                    fold=int(fold),
                    spec_slug=mode,
                    feature_space_mode=transform.feature_space_mode,
                )
                decoder_mode = "mlp"
            model_row = {
                "input_mode": mode,
                "decoder_mode": decoder_mode,
                "fold": int(fold),
                "n_train_rows": int(np.sum(train_mask)),
                "n_test_rows": int(np.sum(test_mask)),
                "input_dim": int(x_all.shape[1]),
                "latent": transform.latent,
                "feature_space_mode": transform.feature_space_mode,
                "feature_dim": int(transform.feature_dim),
                "raw_feature_dim": int(transform.raw_feature_dim),
                "feature_fit_scope": transform.fit_scope,
                "feature_preprocessing": transform.preprocessing,
                "feature_whitened": bool(transform.whitened),
                "feature_variance_fraction": float(transform.explained_variance_sum),
            }
            model_row.update(stats)
            model_rows.append(model_row)
            for local_index, meta in enumerate(test_meta.to_dict(orient="records")):
                out = dict(meta)
                out.update(
                    {
                        "input_mode": mode,
                        "decoder_mode": decoder_mode,
                        "fold": int(fold),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "feature_dim": int(transform.feature_dim),
                        "feature_variance_fraction": float(transform.explained_variance_sum),
                        "input_dim": int(x_all.shape[1]),
                        "n_train_rows": int(np.sum(train_mask)),
                    }
                )
                out.update(_metrics(z_hat[local_index], z_true[local_index]))
                trial_rows.append(out)
    return pd.DataFrame(trial_rows), pd.DataFrame(model_rows)


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["input_mode", "decoder_mode", "prior_scale", "prior_family"]
    scale = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_rmse=("feature_rmse", "median"),
        )
        .sort_values(["input_mode", "prior_scale", "prior_family"])
    )
    overall = (
        trials.groupby(["input_mode", "decoder_mode", "prior_family"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_rmse=("feature_rmse", "median"),
        )
        .sort_values(["input_mode", "prior_family"])
    )
    overall["prior_scale"] = "all"
    return pd.concat([scale, overall[scale.columns]], ignore_index=True)


def _contrasts(trials: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    pivot = trials.pivot_table(
        index=["trial_id", "trajectory_index", "prior_scale", "true_source_row", "input_mode", "decoder_mode"],
        columns="prior_family",
        values="feature_cosine",
        aggfunc="first",
    )
    if not AXIS_FAMILIES.issubset(set(pivot.columns)):
        return pd.DataFrame()
    paired = pivot.dropna(subset=sorted(AXIS_FAMILIES)).reset_index()
    paired["delta"] = paired["axis_edge_parallel"] - paired["axis_edge_orthogonal"]
    rows: list[dict[str, Any]] = []
    for (mode, decoder), mode_rows in paired.groupby(["input_mode", "decoder_mode"], sort=True):
        for scale_value, scale_rows in mode_rows.groupby("prior_scale", sort=True):
            row_values = scale_rows["delta"].to_numpy(dtype=float)
            cluster_values = scale_rows.groupby("trial_id")["delta"].mean().to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(cluster_values, rng, int(n_bootstrap))
            row_mean, row_lo, row_hi = _bootstrap_mean(row_values, rng, int(n_bootstrap))
            rows.append(
                {
                    "input_mode": str(mode),
                    "decoder_mode": str(decoder),
                    "prior_scale": float(scale_value),
                    "mean_parallel_minus_orthogonal": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "uncertainty_unit": "trial_cluster_mean",
                    "fraction_positive": float(np.mean(cluster_values > 0.0)),
                    "n_pairs": int(row_values.size),
                    "n_clusters": int(cluster_values.size),
                    "row_mean_parallel_minus_orthogonal": row_mean,
                    "row_ci_low": row_lo,
                    "row_ci_high": row_hi,
                    "row_fraction_positive": float(np.mean(row_values > 0.0)),
                }
            )
        row_values = mode_rows["delta"].to_numpy(dtype=float)
        cluster_values = mode_rows.groupby(["trial_id", "prior_scale"])["delta"].mean().to_numpy(dtype=float)
        mean, lo, hi = _bootstrap_mean(cluster_values, rng, int(n_bootstrap))
        row_mean, row_lo, row_hi = _bootstrap_mean(row_values, rng, int(n_bootstrap))
        rows.append(
            {
                "input_mode": str(mode),
                "decoder_mode": str(decoder),
                "prior_scale": "all",
                "mean_parallel_minus_orthogonal": mean,
                "ci_low": lo,
                "ci_high": hi,
                "uncertainty_unit": "trial_scale_cluster_mean",
                "fraction_positive": float(np.mean(cluster_values > 0.0)),
                "n_pairs": int(row_values.size),
                "n_clusters": int(cluster_values.size),
                "row_mean_parallel_minus_orthogonal": row_mean,
                "row_ci_low": row_lo,
                "row_ci_high": row_hi,
                "row_fraction_positive": float(np.mean(row_values > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _write_readme(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    all_rows = contrasts[contrasts["prior_scale"].astype(str).eq("all")].set_index("input_mode")

    def value(mode: str) -> tuple[float, float, float] | None:
        if mode not in all_rows.index:
            return None
        row = all_rows.loc[mode]
        return (
            float(row["mean_parallel_minus_orthogonal"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
        )

    mode_labels = {
        "augmented_observed_compact": "compact response-only",
        "augmented_true_tau_residual": "true-tau residual",
        "augmented_true_tau": "raw true-tau concat",
        "augmented_zero_static": "0x static control",
    }
    contrast_lines = []
    for mode, label in mode_labels.items():
        vals = value(mode)
        if vals is None:
            continue
        mean, lo, hi = vals
        contrast_lines.append(f"{label + ':':27s} {mean:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")
    lines = [
        "# Direct Axis MLP Feature Decoder",
        "",
        "Direct along/across response-movie test for the candidate-free MLP decoder.",
        "For every axis-conditioned table and trajectory sample, the true",
        "candidate's sampled response movie is treated as the observation.",
        "",
        "All-scale feature-cosine contrast, `axis_edge_parallel - axis_edge_orthogonal`:",
        "",
        "```text",
        *contrast_lines,
        "```",
        "",
        "This is the direct test of whether along-axis response movies produce",
        "better recovered image features than across-axis response movies under",
        "the same source-disjoint nonlinear decoder.",
        "",
        "This README reports one source-fold seed only; use seed-stability checks",
        "for the final along/across interpretation.",
        "",
        "Outputs:",
        "",
        "- `direct_axis_mlp_feature_decoder_trials.csv`",
        "- `direct_axis_mlp_feature_decoder_summary.csv`",
        "- `direct_axis_mlp_feature_decoder_contrasts.csv`",
        "- `direct_axis_mlp_feature_decoder_models.csv`",
        "- `direct_axis_mlp_feature_decoder_manifest.json`",
    ]
    (out_dir / "direct_axis_mlp_feature_decoder_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-manifest", type=Path, default=PROMOTED_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--response-manifest", type=Path, default=None)
    parser.add_argument("--trajectory-sidecar-dir", type=Path, default=None)
    parser.add_argument("--trajectory-key", default="prior_trajectory_xy")
    parser.add_argument("--feature-npz", type=Path, default=FEATURE_NPZ)
    parser.add_argument("--latent", default="pyramid_local_field")
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--feature-space-mode", default="fold_zscore_whitened_pca")
    parser.add_argument("--projection-basis-path", type=Path, default=COMPACT_BASIS)
    parser.add_argument("--projection-basis-key", default="basis")
    parser.add_argument("--projection-basis-dim", type=int, default=20)
    parser.add_argument("--input-modes", default=",".join(DEFAULT_INPUT_MODES))
    parser.add_argument("--scales", default="0.5,1.0,2.0")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260624)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=4)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-patience", type=int, default=40)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.2)
    parser.add_argument("--mlp-max-train-samples", type=int, default=0)
    parser.add_argument("--mlp-device", default="auto")
    parser.add_argument("--residual-alpha-grid", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is not None:
        promoted = None
        continuous_run_dir = None
        response_run_dir = Path(args.run_dir)
        manifest_path = Path(args.response_manifest) if args.response_manifest else response_run_dir / "response_cache_manifest.csv"
        metadata = {
            "run_dir": str(response_run_dir),
            "trajectory_sidecar_dir": str(args.trajectory_sidecar_dir) if args.trajectory_sidecar_dir else "",
            "trajectory_key": str(args.trajectory_key),
        }
        response_rows = pd.read_csv(manifest_path)
    else:
        promoted = _load_json(Path(args.promoted_manifest))
        artifact = dict(promoted["artifact"])
        continuous_run_dir = Path(artifact["run_dir"])
        metadata = _load_json(Path(artifact["metadata_json"]))
        response_run_dir = Path(metadata["run_dir"])
        manifest_path = continuous_run_dir / "continuous_joint_trials.csv"
        response_rows = pd.read_csv(manifest_path)
    selected = _selected_rows(response_rows, scales=_parse_scales(args.scales), max_tables=int(args.max_tables))
    first_table = _load_npz(response_run_dir / str(selected.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first_table["y_obs_counts"]).shape[1])
    projection_basis, projection_meta = _load_basis(
        Path(args.projection_basis_path),
        n_units=n_units,
        basis_key=str(args.projection_basis_key),
        max_dim=int(args.projection_basis_dim),
    )
    input_modes = _parse_str_list(args.input_modes)
    arrays, train_arrays, train_source_arrays, table_rows, dataset_meta = _build_dataset(
        rows=selected,
        response_run_dir=response_run_dir,
        metadata=metadata,
        projection_basis=projection_basis,
        input_modes=input_modes,
        progress_every=int(args.progress_every),
    )
    feature_table, feature_meta = _load_feature_table(Path(args.feature_npz), latent=str(args.latent))
    mlp_config = MLPConfig(
        hidden_dim=int(args.mlp_hidden_dim),
        layers=int(args.mlp_layers),
        dropout=float(args.mlp_dropout),
        learning_rate=float(args.mlp_learning_rate),
        weight_decay=float(args.mlp_weight_decay),
        batch_size=int(args.mlp_batch_size),
        epochs=int(args.mlp_epochs),
        patience=int(args.mlp_patience),
        validation_fraction=float(args.mlp_validation_fraction),
        max_train_samples=int(args.mlp_max_train_samples),
        device=str(args.mlp_device),
        seed=int(args.fold_seed),
    )
    trials, models = _fit_modes(
        arrays=arrays,
        train_arrays=train_arrays,
        train_source_rows_by_mode=train_source_arrays,
        table_rows=table_rows,
        feature_table=feature_table,
        feature_space_mode=str(args.feature_space_mode),
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        mlp_config=mlp_config,
        residual_alpha_grid=_parse_float_list(args.residual_alpha_grid),
    )
    summary = _summarize(trials)
    contrasts = _contrasts(trials, n_bootstrap=int(args.n_bootstrap), seed=int(args.fold_seed) + 23)
    trials_path = out_dir / "direct_axis_mlp_feature_decoder_trials.csv"
    summary_path = out_dir / "direct_axis_mlp_feature_decoder_summary.csv"
    contrasts_path = out_dir / "direct_axis_mlp_feature_decoder_contrasts.csv"
    models_path = out_dir / "direct_axis_mlp_feature_decoder_models.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    models.to_csv(models_path, index=False)
    manifest = {
        "analysis": "direct_axis_mlp_feature_decoder",
        "promoted_manifest": Path(args.promoted_manifest),
        "response_manifest": manifest_path,
        "continuous_run_dir": continuous_run_dir,
        "response_run_dir": response_run_dir,
        "n_axis_tables": int(selected.shape[0]),
        "n_test_rows": int(table_rows.shape[0]),
        "feature": {**feature_meta, "feature_dim_requested": int(args.feature_dim), "feature_space_mode": str(args.feature_space_mode)},
        "projection_basis": projection_meta,
        "input_modes": input_modes,
        "dataset": dataset_meta,
        "mlp": mlp_config.__dict__,
        "outputs": {
            "trials": trials_path,
            "summary": summary_path,
            "contrasts": contrasts_path,
            "models": models_path,
        },
    }
    _write_json(out_dir / "direct_axis_mlp_feature_decoder_manifest.json", manifest)
    _write_readme(out_dir, summary, contrasts)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Run a predeclared recorded-tau coordinate audit for direct feature decoding.

This audit keeps the feature target, source-disjoint folds, train-fold
normalization, and pooled R2_cv score fixed.  It only changes the trajectory
coordinates supplied to a direct known-tau decoder, so that sign, axis, lag, or
scale convention errors can be detected before interpreting a failed true-tau
ceiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.feature_recovery_scores import R2_CV_METHOD, pooled_multioutput_r2_from_sse_sst
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer import (
    DEFAULT_FEATURE_SPACE_MODES,
    FEATURE_NPZ,
    OUT_DIR,
    SOURCE_ROOT,
    _assign_source_folds,
    _build_sample_banks,
    _canonical_feature_modes,
    _feature_vector,
    _fit_feature_transform,
    _fit_forward_posterior,
    _fit_transform_for_fold,
    _json_ready,
    _load_basis,
    _load_feature_table,
    _load_feature_weights,
    _load_npz,
    _load_table_with_sidecar,
    _metrics,
    _parse_csv_values,
    _parse_scales,
    _parse_str_list,
    _predict_z,
    _read_manifest,
    _selected_specs,
    _summarize,
    _transform_feature_sources,
    _write_json,
    build_parser as _observer_build_parser,
)


DEFAULT_OUT_DIR = OUT_DIR.parent / "recorded_tau_alignment_audit"
DEFAULT_VARIANTS = (
    "identity",
    "neg_xy",
    "neg_x",
    "neg_y",
    "swap_xy",
    "swap_neg_xy",
    "lag_m2",
    "lag_m1",
    "lag_p1",
    "lag_p2",
    "scale_0p5",
    "scale_2p0",
)


def _parse_args() -> argparse.Namespace:
    parser = _observer_build_parser()
    parser.description = __doc__
    parser.set_defaults(
        out_dir=DEFAULT_OUT_DIR,
        observer_modes="response_only,zero_static",
        n_bootstrap=0,
    )
    parser.add_argument("--tau-variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--known-feature-vector-mode",
        choices=["response_tau_linear", "response_tau_interactions"],
        default="response_tau_interactions",
    )
    return parser.parse_args()


def _shift_tau(tau: np.ndarray, lag: int) -> np.ndarray:
    arr = np.asarray(tau, dtype=np.float64)
    out = np.empty_like(arr)
    if int(lag) == 0:
        return arr.copy()
    if int(lag) > 0:
        out[:lag] = arr[:1]
        out[lag:] = arr[:-lag]
    else:
        k = -int(lag)
        out[:-k] = arr[k:]
        out[-k:] = arr[-1:]
    return out


def _tau_variant(tau: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(tau, dtype=np.float64)
    if name == "identity":
        return arr.copy()
    if name == "neg_xy":
        return -arr
    if name == "neg_x":
        out = arr.copy()
        out[:, 0] *= -1.0
        return out
    if name == "neg_y":
        out = arr.copy()
        out[:, 1] *= -1.0
        return out
    if name == "swap_xy":
        return arr[:, [1, 0]].copy()
    if name == "swap_neg_xy":
        return -arr[:, [1, 0]].copy()
    if name == "lag_m2":
        return _shift_tau(arr, -2)
    if name == "lag_m1":
        return _shift_tau(arr, -1)
    if name == "lag_p1":
        return _shift_tau(arr, 1)
    if name == "lag_p2":
        return _shift_tau(arr, 2)
    if name == "scale_0p5":
        return 0.5 * arr
    if name == "scale_2p0":
        return 2.0 * arr
    raise ValueError(f"Unknown tau variant {name!r}")


def _global_transforms_for_modes(
    *,
    modes: list[str],
    feature_table: Any,
    feature_dim: int,
    feature_weights: np.ndarray | None,
) -> dict[str, Any]:
    from declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer import (
        _feature_space_config,
    )

    global_transforms: dict[str, Any] = {}
    for mode in modes:
        if _feature_space_config(mode)["fit_scope"] == "global":
            global_transforms[mode] = _fit_feature_transform(
                feature_table,
                fit_sources=feature_table.source_rows,
                feature_dim=int(feature_dim),
                feature_space_mode=mode,
                feature_weights=feature_weights,
            )
    return global_transforms


def _make_variant_features(tests: Any, variant: str, vector_mode: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index in range(tests.observed_compact.shape[0]):
        tau = np.asarray(tests.tau_true[index], dtype=np.float64)
        compact = np.asarray(tests.observed_compact[index], dtype=np.float64)
        if not np.isfinite(tau).all() or not np.isfinite(compact).all():
            rows.append(np.full(1, np.nan, dtype=np.float32))
            continue
        rows.append(_feature_vector(compact, _tau_variant(tau, variant), mode=vector_mode))
    max_dim = max(row.size for row in rows)
    out = np.full((len(rows), max_dim), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        if row.size == max_dim:
            out[index] = row
    return out


def _pooled_score(trials: pd.DataFrame, observer_mode: str) -> dict[str, float | int]:
    subset = trials[trials["observer_mode"].astype(str).eq(observer_mode)]
    if subset.empty:
        return {"R2_cv": float("nan"), "sse": float("nan"), "sst": float("nan"), "n": 0}
    score = pooled_multioutput_r2_from_sse_sst(
        subset["feature_sse"].to_numpy(dtype=np.float64),
        subset["feature_sst_train_baseline"].to_numpy(dtype=np.float64),
    )
    return {"R2_cv": score.r2, "sse": score.sse, "sst": score.sst, "n": score.n_samples}


def _alignment_table(summary: pd.DataFrame) -> pd.DataFrame:
    all_rows = summary[summary["observation_scale"].astype(str).eq("all")].copy()
    if "tau_variant" not in all_rows.columns:
        all_rows["tau_variant"] = all_rows["observer_mode"].astype(str).str.replace(
            "recorded_tau_",
            "",
            regex=False,
        )
        all_rows.loc[all_rows["observer_mode"].astype(str).isin(["response_only", "zero_static"]), "tau_variant"] = (
            all_rows.loc[
                all_rows["observer_mode"].astype(str).isin(["response_only", "zero_static"]),
                "observer_mode",
            ].astype(str)
        )
    keep = [
        "observer_mode",
        "tau_variant",
        "R2_cv",
        "median_feature_cosine",
        "median_feature_pred_norm",
        "median_feature_true_norm",
        "n",
    ]
    keep = [col for col in keep if col in all_rows.columns]
    return all_rows[keep].sort_values("R2_cv", ascending=False).reset_index(drop=True)


def _run_alignment(
    *,
    tests: Any,
    feature_table: Any,
    feature_weights: np.ndarray | None,
    feature_space_modes: list[str],
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    ridge: float,
    noise_floor: float,
    tau_variants: list[str],
    vector_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_by_source = _assign_source_folds(
        tests.rows["true_source_row"].to_numpy(dtype=int),
        n_folds=int(n_folds),
        seed=int(fold_seed),
    )
    test_folds = tests.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    canonical_modes = _canonical_feature_modes(feature_space_modes)
    global_transforms = _global_transforms_for_modes(
        modes=canonical_modes,
        feature_table=feature_table,
        feature_dim=int(feature_dim),
        feature_weights=feature_weights,
    )
    variant_x = {
        variant: _make_variant_features(tests, variant, vector_mode)
        for variant in tau_variants
    }
    observer_inputs = {
        "response_only": ("response_only", tests.x_by_mode["observed_response_only"]),
        "zero_static": ("zero_static", tests.x_by_mode["zero_static_response_only"]),
    }
    for variant, x in variant_x.items():
        observer_inputs[f"recorded_tau_{variant}"] = (variant, x)

    trial_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for mode in canonical_modes:
        for fold in sorted(set(test_folds.tolist())):
            base_test_mask = test_folds == int(fold)
            if int(np.sum(base_test_mask)) == 0:
                continue
            transform = _fit_transform_for_fold(
                mode=mode,
                fold=int(fold),
                fold_by_source=fold_by_source,
                feature_table=feature_table,
                feature_dim=int(feature_dim),
                feature_weights=feature_weights,
                global_transforms=global_transforms,
            )
            baseline_sources = np.asarray(
                [
                    int(source)
                    for source, source_fold in fold_by_source.items()
                    if int(source_fold) != int(fold)
                ],
                dtype=int,
            )
            z_train_baseline = _transform_feature_sources(transform, feature_table, baseline_sources)
            z_train_mean = np.mean(z_train_baseline, axis=0)
            z_true_all = _transform_feature_sources(
                transform,
                feature_table,
                tests.rows["true_source_row"].to_numpy(dtype=int),
            )
            for observer_mode, (tau_variant, x_all) in observer_inputs.items():
                valid = np.isfinite(x_all).all(axis=1)
                train_mask = np.asarray(
                    [fold_by_source.get(int(source), -1) != int(fold) for source in tests.rows["true_source_row"]],
                    dtype=bool,
                )
                train_mask &= valid
                test_mask = base_test_mask & valid
                if int(np.sum(train_mask)) <= int(transform.feature_dim) or int(np.sum(test_mask)) == 0:
                    continue
                model = _fit_forward_posterior(
                    z_train=z_true_all[train_mask],
                    x_train=x_all[train_mask],
                    ridge=float(ridge),
                    noise_floor=float(noise_floor),
                )
                z_hat = _predict_z(model, x_all[test_mask])
                z_true = z_true_all[test_mask]
                model_rows.append(
                    {
                        "decoder_mode": "recorded_tau_alignment_direct",
                        "observer_mode": observer_mode,
                        "tau_variant": tau_variant,
                        "known_feature_vector_mode": vector_mode,
                        "fold": int(fold),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "n_train_samples": int(model.n_train),
                        "ridge": float(model.ridge),
                        "noise_variance": float(model.noise_variance),
                    }
                )
                test_meta = tests.rows.loc[test_mask].reset_index(drop=True)
                for row_index, meta in enumerate(test_meta.to_dict(orient="records")):
                    row = dict(meta)
                    row.update(
                        {
                            "decoder_mode": "recorded_tau_alignment_direct",
                            "observer_mode": observer_mode,
                            "observer_label": observer_mode.replace("_", " "),
                            "tau_variant": tau_variant,
                            "known_feature_vector_mode": vector_mode,
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "feature_fit_scope": transform.fit_scope,
                            "feature_preprocessing": transform.preprocessing,
                            "feature_whitened": bool(transform.whitened),
                            "feature_weighted": bool(transform.weighted),
                            "feature_variance_fraction": float(transform.explained_variance_sum),
                            "r2_cv_train_baseline": "source_fold_train_feature_mean",
                            "fold": int(fold),
                            "n_train_samples": int(model.n_train),
                            "n_fit_sources": int(transform.n_fit_sources),
                        }
                    )
                    row.update(_metrics(z_hat[row_index], z_true[row_index], train_mean=z_train_mean))
                    trial_rows.append(row)
    if not trial_rows:
        raise ValueError("No alignment audit trial rows produced")
    return pd.DataFrame(trial_rows), pd.DataFrame(model_rows)


def _write_readme(out_dir: Path, ranking: pd.DataFrame) -> None:
    best = ranking.iloc[0]
    identity = ranking[ranking["observer_mode"].astype(str).eq("recorded_tau_identity")]
    identity_r2 = float(identity["R2_cv"].iloc[0]) if not identity.empty else float("nan")
    lines = [
        "# Recorded-tau alignment audit",
        "",
        "This is a direct feature-decoder coordinate audit, not an open-ended model rescue.",
        "The target, folds, train-fold normalization, and pooled `R2_cv` score are held fixed.",
        "",
        f"Best observer: `{best['observer_mode']}`",
        f"Best R2_cv: {float(best['R2_cv']):.6g}",
        f"Identity recorded-tau R2_cv: {identity_r2:.6g}",
        "",
        "Use this audit only to identify sign, axis, lag, or scale convention problems before interpreting a failed known-tau ceiling.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.response_manifest) if args.response_manifest else run_dir / "response_cache_manifest.csv"
    manifest = _read_manifest(
        manifest_path,
        scales=_parse_scales(str(args.scales)),
        prior_family_filter=_parse_csv_values(str(args.prior_family_filter)),
        skip_tables=int(args.skip_tables),
        max_tables=int(args.max_tables),
    )
    sidecar_dir = (
        Path(args.trajectory_sidecar_dir)
        if args.trajectory_sidecar_dir is not None
        else run_dir / "continuous_joint_trajectory_sidecars"
    )
    if not sidecar_dir.exists():
        sidecar_dir = None
    trajectory_npz = _load_npz(Path(args.trajectory_npz)) if args.trajectory_npz is not None else None
    first_table = _load_table_with_sidecar(
        run_dir=run_dir,
        response_cache_path=str(manifest.iloc[0]["response_cache_path"]),
        trajectory_sidecar_dir=sidecar_dir,
    )
    n_units = int(np.asarray(first_table["y_obs_counts"]).shape[1])
    basis, basis_meta = _load_basis(
        Path(args.compact_basis_path),
        n_units=n_units,
        basis_key=str(args.basis_key),
        max_dim=int(args.basis_max_dim),
    )
    feature_table, feature_meta = _load_feature_table(Path(args.feature_npz), latent=str(args.latent))
    feature_weights, feature_weight_meta = _load_feature_weights(
        Path(args.feature_weights_npz) if args.feature_weights_npz is not None else None,
        latent=str(args.latent),
        raw_feature_dim=int(feature_table.features.shape[1]),
    )
    feature_space_modes = _parse_str_list(args.feature_space_modes)
    if not feature_space_modes:
        feature_space_modes = [DEFAULT_FEATURE_SPACE_MODES[0]]
    _selected_specs(_parse_str_list(args.observer_modes))
    _banks, tests, fit_rows = _build_sample_banks(
        run_dir=run_dir,
        manifest=manifest,
        basis=basis,
        feature_sources=set(int(value) for value in feature_table.source_rows.tolist()),
        trajectory_npz=trajectory_npz,
        trajectory_key=str(args.trajectory_key),
        observed_trajectory_key=str(args.observed_trajectory_key),
        trajectory_sidecar_dir=sidecar_dir,
        continuous_args=args,
        compute_pooled_tau_hat=False,
        progress_every=int(args.progress_every),
    )
    tau_variants = _parse_str_list(args.tau_variants)
    trials, models = _run_alignment(
        tests=tests,
        feature_table=feature_table,
        feature_weights=feature_weights,
        feature_space_modes=feature_space_modes,
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        ridge=float(args.linear_ridge),
        noise_floor=float(args.noise_floor),
        tau_variants=tau_variants,
        vector_mode=str(args.known_feature_vector_mode),
    )
    summary = _summarize(trials)
    ranking = _alignment_table(summary)
    trials.to_csv(out_dir / "recorded_tau_alignment_trials.csv", index=False)
    summary.to_csv(out_dir / "recorded_tau_alignment_summary.csv", index=False)
    ranking.to_csv(out_dir / "recorded_tau_alignment_ranking.csv", index=False)
    models.to_csv(out_dir / "recorded_tau_alignment_models.csv", index=False)
    fit_rows.to_csv(out_dir / "recorded_tau_alignment_raw_fit_rows.csv", index=False)
    manifest_payload = {
        "analysis": "recorded_tau_alignment_audit",
        "run_dir": run_dir,
        "response_manifest": manifest_path,
        "trajectory_sidecar_dir": sidecar_dir,
        "n_response_tables": int(manifest.shape[0]),
        "tau_variants": tau_variants,
        "known_feature_vector_mode": str(args.known_feature_vector_mode),
        "score": {
            "name": "R2_cv",
            "method": R2_CV_METHOD,
            "space": "locked_train_normalized_feature_space",
        },
        "feature": {
            **feature_meta,
            "feature_dim_requested": int(args.feature_dim),
            "feature_space_modes_requested": feature_space_modes,
            "feature_weights": feature_weight_meta,
        },
        "basis": basis_meta,
        "crossfit": {"n_folds": int(args.n_folds), "fold_seed": int(args.fold_seed)},
        "outputs": {
            "trials": "recorded_tau_alignment_trials.csv",
            "summary": "recorded_tau_alignment_summary.csv",
            "ranking": "recorded_tau_alignment_ranking.csv",
        },
    }
    _write_json(out_dir / "recorded_tau_alignment_manifest.json", _json_ready(manifest_payload))
    _write_readme(out_dir, ranking)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(_parse_args())


if __name__ == "__main__":
    main()

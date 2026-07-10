"""Run a synthetic self-consistency control for the candidate-free Figure 4C observer.

The control fits the same source-disjoint compact observation model used by the
candidate-free observer, generates synthetic compact responses from that fitted
model with the true feature vector and recorded trajectory, and then reruns the
forward-model observers on those synthetic responses.

Expected sanity pattern, before using the branch for biological interpretation:

    known_tau_forward_model > hidden_joint_forward_model > zero_tau_forward_model

under the same locked, train-normalized pooled R2_cv feature-recovery score.
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
    DEFAULT_OBSERVER_MODES,
    FEATURE_NPZ,
    OUT_DIR,
    PRIMARY_LATENT,
    SOURCE_ROOT,
    TestSet,
    _assign_source_folds,
    _build_sample_banks,
    _canonical_feature_modes,
    _clean_axis,
    _configure_matplotlib,
    _continuous_args_for_scale,
    _feature_space_config,
    _fit_feature_conditioned_baseline,
    _fit_feature_conditioned_quadratic_observation_map,
    _fit_feature_transform,
    _fit_forward_posterior,
    _fit_transform_for_fold,
    _forward_model_observation_var,
    _json_ready,
    _joint_z_tau_map,
    _load_basis,
    _load_feature_table,
    _load_feature_weights,
    _load_npz,
    _load_table_with_sidecar,
    _metrics,
    _parse_csv_values,
    _parse_scales,
    _parse_str_list,
    _predict_compact_from_z_tau,
    _predict_z,
    _read_manifest,
    _response_vector,
    _selected_specs,
    _solve_z_given_tau,
    _summarize,
    _tau_metrics,
    _transform_feature_sources,
    _write_json,
    build_parser as _observer_build_parser,
)


DEFAULT_OUT_DIR = OUT_DIR.parent / "candidate_free_self_consistency_control"


def build_parser() -> argparse.ArgumentParser:
    parser = _observer_build_parser()
    parser.description = __doc__
    parser.set_defaults(
        out_dir=DEFAULT_OUT_DIR,
        observer_modes="response_only,pose_known_forward_model,hidden_joint_forward_model,zero_tau_forward_model",
        n_bootstrap=0,
    )
    parser.add_argument(
        "--self-consistency-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Scale of Gaussian compact-response noise relative to the fitted forward-model observation "
            "standard deviation. Use 0 for the deterministic exact-model control."
        ),
    )
    parser.add_argument("--self-consistency-seed", type=int, default=20260706)
    return parser


def _parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()


def _global_transforms_for_modes(
    *,
    modes: list[str],
    feature_table: Any,
    feature_dim: int,
    feature_weights: np.ndarray | None,
) -> dict[str, Any]:
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


def _pooled_score(trials: pd.DataFrame, observer_mode: str) -> dict[str, float | int]:
    subset = trials[trials["observer_mode"].astype(str).eq(observer_mode)]
    if subset.empty:
        return {"R2_cv": float("nan"), "sse": float("nan"), "sst": float("nan"), "n": 0}
    score = pooled_multioutput_r2_from_sse_sst(
        subset["feature_sse"].to_numpy(dtype=np.float64),
        subset["feature_sst_train_baseline"].to_numpy(dtype=np.float64),
    )
    return {"R2_cv": score.r2, "sse": score.sse, "sst": score.sst, "n": score.n_samples}


def _gate_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        col
        for col in [
            "decoder_mode",
            "latent",
            "feature_space_mode",
            "observation_scale",
            "prior_family",
        ]
        if col in trials.columns
    ]
    groups: list[tuple[dict[str, Any], pd.DataFrame, str]] = [({}, trials.copy(), "all")]
    for values, group in trials.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        groups.append((dict(zip(group_cols, values)), group.copy(), "group"))

    for group_values, group, group_kind in groups:
        known = _pooled_score(group, "pose_known_forward_model")
        joint = _pooled_score(group, "hidden_joint_forward_model")
        zero = _pooled_score(group, "zero_tau_forward_model")
        response = _pooled_score(group, "response_only")
        s_known = float(known["R2_cv"])
        s_joint = float(joint["R2_cv"])
        s_zero = float(zero["R2_cv"])
        s_response = float(response["R2_cv"])
        rows.append(
            {
                **group_values,
                "group_kind": group_kind,
                "score": "R2_cv",
                "score_method": R2_CV_METHOD,
                "score_space": "locked_train_normalized_feature_space",
                "score_aggregation": "pooled_multioutput_out_of_fold_sse_sst",
                "known_mode": "pose_known_forward_model",
                "joint_mode": "hidden_joint_forward_model",
                "zero_mode": "zero_tau_forward_model",
                "response_mode": "response_only",
                "S_known": s_known,
                "S_joint": s_joint,
                "S_zero": s_zero,
                "S_response": s_response,
                "known_minus_zero": s_known - s_zero,
                "joint_minus_zero": s_joint - s_zero,
                "known_minus_joint": s_known - s_joint,
                "joint_minus_response": s_joint - s_response,
                "known_gt_joint_gt_zero": bool(
                    np.isfinite(s_known)
                    and np.isfinite(s_joint)
                    and np.isfinite(s_zero)
                    and s_known > s_joint
                    and s_joint > s_zero
                ),
                "known_gt_zero": bool(np.isfinite(s_known) and np.isfinite(s_zero) and s_known > s_zero),
                "joint_gt_zero": bool(np.isfinite(s_joint) and np.isfinite(s_zero) and s_joint > s_zero),
                "known_n": int(known["n"]),
                "joint_n": int(joint["n"]),
                "zero_n": int(zero["n"]),
                "response_n": int(response["n"]),
            }
        )
    return pd.DataFrame(rows)


def _run_self_consistency(
    *,
    tests: TestSet,
    feature_table: Any,
    feature_weights: np.ndarray | None,
    feature_space_modes: list[str],
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    ridge: float,
    noise_floor: float,
    continuous_args: argparse.Namespace,
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
    rng = np.random.default_rng(int(continuous_args.self_consistency_seed))
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
            heldout_sources = {
                int(source)
                for source, source_fold in fold_by_source.items()
                if int(source_fold) == int(fold)
            }
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
            include_intercept = str(continuous_args.continuous_score_mode) in {
                "quadratic_affine_poisson_profile",
                "quadratic_prior_mean_poisson_profile",
            }
            baseline_coef, baseline_residual_var, baseline_fit_row = _fit_feature_conditioned_baseline(
                geometry_tables=tests.geometry_tables,
                transform=transform,
                feature_table=feature_table,
                heldout_sources=heldout_sources,
                ridge=float(continuous_args.continuous_ridge),
            )
            observation_coef, observation_residual_var, observation_fit_row = (
                _fit_feature_conditioned_quadratic_observation_map(
                    geometry_tables=tests.geometry_tables,
                    transform=transform,
                    feature_table=feature_table,
                    heldout_sources=heldout_sources,
                    ridge=float(continuous_args.continuous_ridge),
                    include_intercept=include_intercept,
                    intercept_ridge_multiplier=float(continuous_args.quadratic_intercept_ridge_multiplier),
                )
            )
            synthetic_compact = np.full_like(tests.observed_compact, np.nan, dtype=np.float64)
            synthetic_observation_var = np.full(tests.observed_compact.shape[0], np.nan, dtype=np.float64)
            for row_index, meta in tests.rows.reset_index(drop=True).iterrows():
                tau_true = np.asarray(tests.tau_true[int(row_index)], dtype=np.float64)
                if not np.isfinite(tau_true).all():
                    continue
                table_args = _continuous_args_for_scale(continuous_args, float(meta["observation_scale"]))
                compact = _predict_compact_from_z_tau(
                    z=z_true_all[int(row_index)],
                    tau=tau_true,
                    observation_scale=float(meta["observation_scale"]),
                    baseline_coef=baseline_coef,
                    observation_coef=observation_coef,
                    include_intercept=include_intercept,
                )
                obs_var = _forward_model_observation_var(
                    baseline_residual_var=float(baseline_residual_var),
                    observation_residual_var=float(observation_residual_var),
                    continuous_args=table_args,
                )
                synthetic_observation_var[int(row_index)] = obs_var
                noise_scale = float(continuous_args.self_consistency_noise_scale)
                if noise_scale > 0.0:
                    compact = compact + rng.normal(0.0, noise_scale * np.sqrt(obs_var), size=compact.shape)
                synthetic_compact[int(row_index)] = compact

            x_response = np.stack(
                [_response_vector(synthetic_compact[index]) for index in range(synthetic_compact.shape[0])],
                axis=0,
            )
            train_mask = np.asarray(
                [fold_by_source.get(int(source), -1) != int(fold) for source in tests.rows["true_source_row"]],
                dtype=bool,
            )
            train_mask &= np.isfinite(x_response).all(axis=1)
            if int(np.sum(train_mask)) <= int(transform.feature_dim):
                raise ValueError("too few synthetic train rows for self-consistency response-only decoder")
            response_model = _fit_forward_posterior(
                z_train=z_true_all[train_mask],
                x_train=x_response[train_mask],
                ridge=float(ridge),
                noise_floor=float(noise_floor),
            )
            z0_all = _predict_z(response_model, x_response)
            model_rows.extend(
                [
                    {
                        **dict(baseline_fit_row),
                        "fold": int(fold),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "synthetic_control": "self_consistency_exact_forward_model",
                    },
                    {
                        **dict(observation_fit_row),
                        "fold": int(fold),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "synthetic_control": "self_consistency_exact_forward_model",
                    },
                    {
                        "qc_type": "self_consistency_response_only_decoder",
                        "fold": int(fold),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "n_train_samples": int(response_model.n_train),
                        "ridge": float(response_model.ridge),
                        "noise_variance": float(response_model.noise_variance),
                        "synthetic_control": "self_consistency_exact_forward_model",
                    },
                ]
            )

            for global_index in np.flatnonzero(base_test_mask):
                compact = synthetic_compact[int(global_index)]
                tau_true = np.asarray(tests.tau_true[int(global_index)], dtype=np.float64)
                if not np.isfinite(compact).all() or not np.isfinite(tau_true).all():
                    continue
                meta = tests.rows.iloc[int(global_index)]
                table_args = _continuous_args_for_scale(continuous_args, float(meta["observation_scale"]))
                common = {
                    "observed_compact": compact,
                    "geometry_table": tests.geometry_tables[int(global_index)],
                    "baseline_coef": baseline_coef,
                    "baseline_residual_var": float(baseline_residual_var),
                    "observation_coef": observation_coef,
                    "observation_residual_var": float(observation_residual_var),
                    "include_intercept": include_intercept,
                    "continuous_args": table_args,
                }
                observer_outputs: list[tuple[str, str, np.ndarray, np.ndarray | None, dict[str, Any]]] = []
                observer_outputs.append(
                    (
                        "response_only",
                        "synthetic response-only decoder",
                        z0_all[int(global_index)],
                        None,
                        {"feature_update_mode": "linear_response_only_decoder_on_synthetic_F_response"},
                    )
                )
                z_known, _pred_known, known_meta = _solve_z_given_tau(tau=tau_true, **common)
                observer_outputs.append(
                    (
                        "pose_known_forward_model",
                        "known tau compact forward inversion",
                        z_known,
                        tau_true,
                        known_meta,
                    )
                )
                zero_tau = np.zeros((compact.shape[0], 2), dtype=np.float64)
                z_zero, _pred_zero, zero_meta = _solve_z_given_tau(tau=zero_tau, **common)
                observer_outputs.append(
                    (
                        "zero_tau_forward_model",
                        "zero tau compact forward inversion",
                        z_zero,
                        zero_tau,
                        zero_meta,
                    )
                )
                z_joint, tau_joint, _pred_joint, joint_meta = _joint_z_tau_map(
                    observed_compact=compact,
                    initial_z=z0_all[int(global_index)],
                    geometry_table=tests.geometry_tables[int(global_index)],
                    baseline_coef=baseline_coef,
                    baseline_residual_var=float(baseline_residual_var),
                    observation_coef=observation_coef,
                    observation_residual_var=float(observation_residual_var),
                    include_intercept=include_intercept,
                    continuous_args=table_args,
                    table_index=int(meta["table_index"]),
                )
                observer_outputs.append(
                    (
                        "hidden_joint_forward_model",
                        "hidden joint compact forward inversion",
                        z_joint,
                        tau_joint,
                        joint_meta,
                    )
                )
                z_true = z_true_all[int(global_index)]
                for observer_mode, observer_label, z_hat, tau_hat, observer_meta in observer_outputs:
                    row = dict(meta)
                    row.update(
                        {
                            "decoder_mode": "candidate_free_self_consistency",
                            "observer_mode": observer_mode,
                            "observer_label": observer_label,
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "feature_fit_scope": transform.fit_scope,
                            "feature_preprocessing": transform.preprocessing,
                            "feature_whitened": bool(transform.whitened),
                            "feature_weighted": bool(transform.weighted),
                            "feature_variance_fraction": float(transform.explained_variance_sum),
                            "fold": int(fold),
                            "n_fit_sources": int(transform.n_fit_sources),
                            "n_train_samples": int(response_model.n_train),
                            "r2_cv_train_baseline": "source_fold_train_feature_mean",
                            "synthetic_control": "self_consistency_exact_forward_model",
                            "synthetic_response_source": "F(z_true, tau_true)",
                            "synthetic_noise_scale": float(continuous_args.self_consistency_noise_scale),
                            "synthetic_observation_variance": float(synthetic_observation_var[int(global_index)]),
                            "continuous_score_mode": str(table_args.continuous_score_mode),
                            "trajectory_process_model": str(table_args.trajectory_process_model),
                        }
                    )
                    row.update(_metrics(z_hat, z_true, train_mean=z_train_mean))
                    if tau_hat is None:
                        row.update(
                            {
                                "trajectory_rmse": float("nan"),
                                "trajectory_corr_x": float("nan"),
                                "trajectory_corr_y": float("nan"),
                                "trajectory_corr_mean": float("nan"),
                                "trajectory_r2": float("nan"),
                            }
                        )
                    else:
                        row.update(_tau_metrics(tau_hat, tau_true))
                    row.update({key: value for key, value in observer_meta.items() if np.isscalar(value)})
                    trial_rows.append(row)
    if not trial_rows:
        raise ValueError("No self-consistency trial rows produced")
    return pd.DataFrame(trial_rows), pd.DataFrame(model_rows)


def _plot_gate(gate: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    all_row = gate[gate["group_kind"].astype(str).eq("all")].iloc[0]
    values = pd.DataFrame(
        [
            {"observer": "known", "R2_cv": float(all_row["S_known"])},
            {"observer": "joint", "R2_cv": float(all_row["S_joint"])},
            {"observer": "zero", "R2_cv": float(all_row["S_zero"])},
            {"observer": "response", "R2_cv": float(all_row["S_response"])},
        ]
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colors = ["#047857", "#0ea5e9", "#92400e", "#4b5563"]
    ax.barh(values["observer"], values["R2_cv"], color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("pooled held-out R2_cv")
    ax.set_title("Candidate-free self-consistency control")
    _clean_axis(ax)
    fig.tight_layout()
    png = out_dir / "candidate_free_self_consistency_gate.png"
    pdf = out_dir / "candidate_free_self_consistency_gate.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _write_readme(out_dir: Path, gate: pd.DataFrame) -> None:
    all_row = gate[gate["group_kind"].astype(str).eq("all")].iloc[0]
    passed = bool(all_row["known_gt_joint_gt_zero"])
    lines = [
        "# Candidate-free self-consistency control",
        "",
        "Synthetic response source: `F(z_true, tau_true)` from the fitted source-disjoint compact observation model.",
        "",
        f"Overall pass pattern `known > joint > zero`: `{passed}`",
        "",
        "Overall scores:",
        f"- known: {float(all_row['S_known']):.6g}",
        f"- joint: {float(all_row['S_joint']):.6g}",
        f"- zero: {float(all_row['S_zero']):.6g}",
        f"- response-only: {float(all_row['S_response']):.6g}",
        "",
        "If this control fails, the candidate-free branch has an implementation, scoring, inversion, or prior issue before biological interpretation.",
        "If this control passes, the next recovery rung is the direct known-tau feature decoder and recorded-tau alignment audit on biological responses.",
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
    _selected_specs(_parse_str_list(args.observer_modes) or list(DEFAULT_OBSERVER_MODES))
    _banks, tests, raw_fit_rows = _build_sample_banks(
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
    trials, model_rows = _run_self_consistency(
        tests=tests,
        feature_table=feature_table,
        feature_weights=feature_weights,
        feature_space_modes=feature_space_modes,
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        ridge=float(args.linear_ridge),
        noise_floor=float(args.noise_floor),
        continuous_args=args,
    )
    summary = _summarize(trials)
    gate = _gate_table(trials)
    trials.to_csv(out_dir / "candidate_free_self_consistency_trials.csv", index=False)
    summary.to_csv(out_dir / "candidate_free_self_consistency_summary.csv", index=False)
    gate.to_csv(out_dir / "candidate_free_self_consistency_gate_table.csv", index=False)
    model_rows.to_csv(out_dir / "candidate_free_self_consistency_model_rows.csv", index=False)
    raw_fit_rows.to_csv(out_dir / "candidate_free_self_consistency_raw_fit_rows.csv", index=False)
    png, pdf = _plot_gate(gate, out_dir)
    manifest_payload = {
        "analysis": "candidate_free_self_consistency_control",
        "run_dir": run_dir,
        "response_manifest": manifest_path,
        "trajectory_sidecar_dir": sidecar_dir,
        "n_response_tables": int(manifest.shape[0]),
        "synthetic_response_source": "F(z_true, tau_true)",
        "self_consistency_noise_scale": float(args.self_consistency_noise_scale),
        "observer_modes": [
            "response_only",
            "pose_known_forward_model",
            "hidden_joint_forward_model",
            "zero_tau_forward_model",
        ],
        "feature": {
            **feature_meta,
            "feature_dim_requested": int(args.feature_dim),
            "feature_space_modes_requested": feature_space_modes,
            "feature_weights": feature_weight_meta,
        },
        "basis": basis_meta,
        "crossfit": {"n_folds": int(args.n_folds), "fold_seed": int(args.fold_seed)},
        "continuous_scorer": {
            "continuous_score_mode": str(args.continuous_score_mode),
            "observation_var": args.observation_var,
            "observation_var_floor": float(args.observation_var_floor),
            "continuous_ridge": float(args.continuous_ridge),
            "forward_model_z_prior_precision": float(args.forward_model_z_prior_precision),
            "forward_model_joint_iterations": int(args.forward_model_joint_iterations),
            "quadratic_optimizer_max_iter": int(args.quadratic_optimizer_max_iter),
        },
        "outputs": {
            "trials": "candidate_free_self_consistency_trials.csv",
            "summary": "candidate_free_self_consistency_summary.csv",
            "gate": "candidate_free_self_consistency_gate_table.csv",
            "plot_png": png.name,
            "plot_pdf": pdf.name,
        },
    }
    _write_json(out_dir / "candidate_free_self_consistency_manifest.json", _json_ready(manifest_payload))
    _write_readme(out_dir, gate)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(_parse_args())


if __name__ == "__main__":
    main()

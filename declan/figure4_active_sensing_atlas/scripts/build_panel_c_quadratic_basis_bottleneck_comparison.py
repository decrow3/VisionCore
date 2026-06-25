"""Summarize quadratic encoder basis/ridge checks for the 2x parallel bottleneck."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_joint_feature_recovery import (
    OUT_DIR,
    PRIMARY_LATENT,
    SOURCE_ROOT,
    TEMPERATURES,
    _load_feature_tables,
    _vectorized_mode_rows,
)


SUBSET_RUNS = [
    ("k=10, ridge 0.01", "scale2_parallel_k10_64"),
    ("k=20, ridge 0.01", "scale2_parallel_k20_64"),
    ("k=20, ridge 0.1", "scale2_parallel_k20_ridge0p1_64"),
    ("k=30, ridge 0.01", "scale2_parallel_k30_64"),
]

FULL_RUNS = [
    ("k=10, ridge 0.01", "full", True),
    ("k=20, ridge 0.1", "scale2_parallel_k20_ridge0p1_full", False),
]

ALL_SCALE_RUNS = [
    ("k=10 all scales", "full"),
    ("k=20/ridge0.1 all scales", "full_k20_ridge0p1"),
]

ENCODER_SELECTION_RUNS = [
    ("k10_ridge0p01", "k=10, ridge 0.01", "full"),
    ("k20_ridge0p1", "k=20, ridge 0.1", "full_k20_ridge0p1"),
]


def _overall_row(suffix: str, *, from_full_slice: bool) -> dict[str, float]:
    path = OUT_DIR / f"continuous_joint_quadratic_feature_trials_{suffix}.csv"
    rows = pd.read_csv(path)
    if from_full_slice:
        rows = rows[
            rows["prior_scale"].astype(float).eq(2.0)
            & rows["prior_family"].astype(str).eq("axis_edge_parallel")
        ].copy()
    quad = rows[rows["observer_mode"].eq("quadratic_poisson")]
    joint = rows[rows["observer_mode"].eq("joint")]
    known = rows[rows["observer_mode"].eq("known")]
    return {
        "n_tables": int(quad.shape[0]),
        "image_accuracy": float(quad["image_correct"].mean()),
        "mean_feature_cosine": float(quad["feature_cosine"].mean()),
        "mean_map_feature_cosine": float(quad["map_feature_cosine"].mean()),
        "mean_true_mass": float(quad["candidate_posterior_true_mass"].mean()),
        "median_N_eff_fraction": float(quad["candidate_posterior_N_eff_fraction"].median()),
        "gap_to_finite_joint": float(joint["feature_cosine"].mean() - quad["feature_cosine"].mean()),
        "gap_to_known_eye": float(known["feature_cosine"].mean() - quad["feature_cosine"].mean()),
    }


def _qc_row(suffix: str, *, from_full_slice: bool) -> dict[str, float]:
    path = OUT_DIR / f"continuous_joint_quadratic_feature_qc_{suffix}.csv"
    rows = pd.read_csv(path)
    if from_full_slice:
        rows = rows[
            rows["prior_scale"].astype(float).eq(2.0)
            & rows["response_cache_path"].astype(str).str.contains("axis_edge_parallel", regex=False)
        ].copy()
    return {
        "optimizer_success": float(rows["optimizer_success"].mean()),
        "median_optimizer_iterations": float(rows["optimizer_iterations"].median()),
        "mean_quadratic_train_r2": float(rows["quadratic_train_r2"].mean()),
        "mean_quadratic_residual_var": float(rows["quadratic_residual_var"].mean()),
    }


def _best_temperature_row(suffix: str, *, from_full_slice: bool) -> dict[str, float]:
    path = OUT_DIR / f"continuous_joint_quadratic_feature_posterior_{suffix}.csv"
    posterior = pd.read_csv(path)
    posterior = posterior[posterior["observer_mode"].eq("quadratic_poisson")].copy()
    if from_full_slice:
        posterior = posterior[
            posterior["prior_scale"].astype(float).eq(2.0)
            & posterior["prior_family"].astype(str).eq("axis_edge_parallel")
        ].copy()
    feature_table = _load_feature_tables()[PRIMARY_LATENT]
    rows = []
    for temperature in TEMPERATURES:
        scored = _vectorized_mode_rows(
            rows=posterior,
            latent=PRIMARY_LATENT,
            feature_table=feature_table,
            posterior_temperature=float(temperature),
        )
        rows.append(
            {
                "best_temperature": float(temperature),
                "best_temperature_feature_cosine": float(scored["feature_cosine"].mean()),
                "best_temperature_true_mass": float(scored["candidate_posterior_true_mass"].mean()),
                "best_temperature_N_eff_fraction": float(scored["candidate_posterior_N_eff_fraction"].median()),
            }
        )
    return max(rows, key=lambda row: row["best_temperature_feature_cosine"])


def _hybrid_trials() -> pd.DataFrame:
    k10 = pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_trials_full.csv")
    k20 = pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_trials_full_k20_ridge0p1.csv")
    return pd.concat(
        [
            k10[k10["prior_scale"].astype(float).eq(0.5)],
            k20[k20["prior_scale"].astype(float).isin([1.0, 2.0])],
        ],
        ignore_index=True,
    )


def _hybrid_posterior() -> pd.DataFrame:
    k10 = pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_posterior_full.csv")
    k20 = pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_posterior_full_k20_ridge0p1.csv")
    return pd.concat(
        [
            k10[k10["prior_scale"].astype(float).eq(0.5)],
            k20[k20["prior_scale"].astype(float).isin([1.0, 2.0])],
        ],
        ignore_index=True,
    )


def _hybrid_qc() -> pd.DataFrame:
    k10 = pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_qc_full.csv")
    k20 = pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_qc_full_k20_ridge0p1.csv")
    return pd.concat(
        [
            k10[k10["prior_scale"].astype(float).eq(0.5)],
            k20[k20["prior_scale"].astype(float).isin([1.0, 2.0])],
        ],
        ignore_index=True,
    )


def _native_scale_conditioned_trials() -> pd.DataFrame:
    rows = pd.read_csv(OUT_DIR / "continuous_joint_feature_recovery_trials.csv")
    return rows[rows["run_slug"].eq("noanchor_quadratic_scale_conditioned")].copy()


def _native_scale_conditioned_posterior() -> pd.DataFrame:
    return pd.read_csv(
        SOURCE_ROOT
        / "continuous_joint_quadratic_poisson_scale_conditioned_full"
        / "continuous_joint_feature_posterior.csv"
    )


def _native_scale_conditioned_qc() -> pd.DataFrame:
    rows = pd.read_csv(
        SOURCE_ROOT
        / "continuous_joint_quadratic_poisson_scale_conditioned_full"
        / "continuous_joint_qc.csv"
    )
    return rows[rows["qc_type"].eq("quadratic_profile_optimizer")].copy()


def _metrics_from_trials(rows: pd.DataFrame) -> dict[str, float]:
    if "quadratic_poisson" in set(rows["observer_mode"].astype(str)):
        continuous_mode = "quadratic_poisson"
    else:
        continuous_mode = "continuous_joint"
    quad = rows[rows["observer_mode"].eq(continuous_mode)]
    joint = rows[rows["observer_mode"].eq("joint")]
    known = rows[rows["observer_mode"].eq("known")]
    return {
        "n_tables": int(quad.shape[0]),
        "image_accuracy": float(quad["image_correct"].mean()),
        "mean_feature_cosine": float(quad["feature_cosine"].mean()),
        "mean_map_feature_cosine": float(quad["map_feature_cosine"].mean()),
        "mean_true_mass": float(quad["candidate_posterior_true_mass"].mean()),
        "median_N_eff_fraction": float(quad["candidate_posterior_N_eff_fraction"].median()),
        "gap_to_finite_joint": float(joint["feature_cosine"].mean() - quad["feature_cosine"].mean()),
        "gap_to_known_eye": float(known["feature_cosine"].mean() - quad["feature_cosine"].mean()),
    }


def _qc_from_rows(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "optimizer_success": float(rows["optimizer_success"].mean()),
        "median_optimizer_iterations": float(rows["optimizer_iterations"].median()),
        "mean_quadratic_train_r2": float(rows["quadratic_train_r2"].mean()),
        "mean_quadratic_residual_var": float(rows["quadratic_residual_var"].mean()),
    }


def _native_qc_from_rows(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "optimizer_success": float(rows["optimizer_success"].mean()),
        "median_optimizer_iterations": float(rows["optimizer_iterations"].median()),
        "mean_quadratic_train_r2": float(rows["B_train_r2_energy"].mean()),
        "mean_quadratic_residual_var": float(rows["residual_variance"].mean()),
    }


def _best_temperature_from_posterior(posterior: pd.DataFrame) -> dict[str, float]:
    if "quadratic_poisson" in set(posterior["observer_mode"].astype(str)):
        mode = "quadratic_poisson"
    else:
        mode = "continuous_joint"
    posterior = posterior[posterior["observer_mode"].eq(mode)].copy()
    feature_table = _load_feature_tables()[PRIMARY_LATENT]
    rows = []
    for temperature in TEMPERATURES:
        scored = _vectorized_mode_rows(
            rows=posterior,
            latent=PRIMARY_LATENT,
            feature_table=feature_table,
            posterior_temperature=float(temperature),
        )
        rows.append(
            {
                "best_temperature": float(temperature),
                "best_temperature_feature_cosine": float(scored["feature_cosine"].mean()),
                "best_temperature_true_mass": float(scored["candidate_posterior_true_mass"].mean()),
                "best_temperature_N_eff_fraction": float(scored["candidate_posterior_N_eff_fraction"].median()),
            }
        )
    return max(rows, key=lambda row: row["best_temperature_feature_cosine"])


def _scale_specific_best_from_posterior(posterior: pd.DataFrame) -> dict[str, float]:
    if "quadratic_poisson" in set(posterior["observer_mode"].astype(str)):
        mode = "quadratic_poisson"
    else:
        mode = "continuous_joint"
    posterior = posterior[posterior["observer_mode"].eq(mode)].copy()
    feature_table = _load_feature_tables()[PRIMARY_LATENT]
    values = []
    selected = []
    for scale, scale_rows in posterior.groupby("prior_scale", sort=True):
        best = None
        for temperature in TEMPERATURES:
            scored = _vectorized_mode_rows(
                rows=scale_rows,
                latent=PRIMARY_LATENT,
                feature_table=feature_table,
                posterior_temperature=float(temperature),
            )
            value = float(scored["feature_cosine"].mean())
            if best is None or value > best[1]:
                best = (float(temperature), value)
        if best is not None:
            selected.append(f"{float(scale):g}:{best[0]:g}")
            values.append(best[1])
    return {
        "scale_specific_best_temperature_by_scale": ",".join(selected),
        "scale_specific_best_feature_cosine": float(np.mean(values)) if values else float("nan"),
    }


def _encoder_temperature_scores() -> pd.DataFrame:
    feature_table = _load_feature_tables()[PRIMARY_LATENT]
    out = []
    for run_slug, run_label, suffix in ENCODER_SELECTION_RUNS:
        posterior = pd.read_csv(OUT_DIR / f"continuous_joint_quadratic_feature_posterior_{suffix}.csv")
        posterior = posterior[posterior["observer_mode"].eq("quadratic_poisson")].copy()
        for temperature in TEMPERATURES:
            scored = _vectorized_mode_rows(
                rows=posterior,
                latent=PRIMARY_LATENT,
                feature_table=feature_table,
                posterior_temperature=float(temperature),
            )
            scored["encoder_slug"] = run_slug
            scored["encoder_label"] = run_label
            scored["posterior_temperature"] = float(temperature)
            out.append(scored)
    return pd.concat(out, ignore_index=True)


def _mean_metrics(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {
            "n_eval": 0,
            "image_accuracy": float("nan"),
            "mean_feature_cosine": float("nan"),
            "mean_true_mass": float("nan"),
            "median_N_eff_fraction": float("nan"),
        }
    return {
        "n_eval": int(rows.shape[0]),
        "image_accuracy": float(rows["image_correct"].mean()),
        "mean_feature_cosine": float(rows["feature_cosine"].mean()),
        "mean_true_mass": float(rows["candidate_posterior_true_mass"].mean()),
        "median_N_eff_fraction": float(rows["candidate_posterior_N_eff_fraction"].median()),
    }


def _best_encoder_temperature(rows: pd.DataFrame, *, allow_temperature: bool) -> pd.Series:
    train = rows.copy()
    if not allow_temperature:
        train = train[np.isclose(train["posterior_temperature"].astype(float), 1.0)].copy()
    grouped = (
        train.groupby(["encoder_slug", "encoder_label", "posterior_temperature"], as_index=False)
        .agg(mean_feature_cosine=("feature_cosine", "mean"), n=("feature_cosine", "size"))
        .sort_values(["mean_feature_cosine", "n", "posterior_temperature"], ascending=[False, False, True])
    )
    if grouped.empty:
        raise ValueError("Cannot select encoder from empty training rows")
    return grouped.iloc[0]


def _compute_encoder_selection_cv() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _encoder_temperature_scores()
    scores["split"] = scores["trial_id"].astype(int) % 2
    rows = []
    for calibration_mode, allow_temperature in [
        ("encoder_only_default_temp", False),
        ("encoder_and_temperature", True),
    ]:
        for eval_split in [0, 1]:
            train = scores[scores["split"].ne(eval_split)].copy()
            eval_rows = scores[scores["split"].eq(eval_split)].copy()
            selected_rows = []
            for scale, scale_eval in eval_rows.groupby("prior_scale", sort=True):
                scale_train = train[train["prior_scale"].eq(scale)]
                selected = _best_encoder_temperature(scale_train, allow_temperature=allow_temperature)
                chosen = scale_eval[
                    scale_eval["encoder_slug"].eq(selected["encoder_slug"])
                    & np.isclose(
                        scale_eval["posterior_temperature"].astype(float),
                        float(selected["posterior_temperature"]),
                    )
                ].copy()
                selected_rows.append(chosen)
                metrics = _mean_metrics(chosen)
                rows.append(
                    {
                        "calibration_mode": calibration_mode,
                        "eval_split": int(eval_split),
                        "prior_scale": float(scale),
                        "selected_encoder_slug": str(selected["encoder_slug"]),
                        "selected_encoder_label": str(selected["encoder_label"]),
                        "selected_temperature": float(selected["posterior_temperature"]),
                        "train_mean_feature_cosine": float(selected["mean_feature_cosine"]),
                        **metrics,
                    }
                )
            if selected_rows:
                combined = pd.concat(selected_rows, ignore_index=True)
                metrics = _mean_metrics(combined)
                selected_text = ",".join(
                    f"{row['prior_scale']:g}:{row['selected_encoder_slug']}@{row['selected_temperature']:g}"
                    for row in rows
                    if row["calibration_mode"] == calibration_mode
                    and row["eval_split"] == int(eval_split)
                    and row["prior_scale"] != "all"
                )
                rows.append(
                    {
                        "calibration_mode": calibration_mode,
                        "eval_split": int(eval_split),
                        "prior_scale": "all",
                        "selected_encoder_slug": selected_text,
                        "selected_encoder_label": selected_text,
                        "selected_temperature": float("nan"),
                        "train_mean_feature_cosine": float("nan"),
                        **metrics,
                    }
                )
    cv_rows = pd.DataFrame(rows)
    overall = cv_rows[cv_rows["prior_scale"].astype(str).eq("all")].copy()
    summary_rows = []
    for mode, group in overall.groupby("calibration_mode", sort=False):
        weights = group["n_eval"].astype(float).to_numpy()
        summary_rows.append(
            {
                "calibration_mode": str(mode),
                "n_eval": int(group["n_eval"].sum()),
                "selected_by_split": ";".join(group["selected_encoder_slug"].astype(str).tolist()),
                "image_accuracy": float(np.average(group["image_accuracy"], weights=weights)),
                "mean_feature_cosine": float(np.average(group["mean_feature_cosine"], weights=weights)),
                "mean_true_mass": float(np.average(group["mean_true_mass"], weights=weights)),
                "median_N_eff_fraction": float(group["median_N_eff_fraction"].median()),
            }
        )
    return cv_rows, pd.DataFrame(summary_rows)


def _plot_encoder_selection_cv(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    labels = {
        "encoder_only_default_temp": "encoder only",
        "encoder_and_temperature": "encoder + temp",
    }
    rows = summary.copy()
    x = np.arange(rows.shape[0], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.bar(x, rows["mean_feature_cosine"], color="#8a5ca8")
    ax.set_xticks(x, [labels.get(str(mode), str(mode)) for mode in rows["calibration_mode"]])
    ax.set_ylim(0.90, 0.94)
    ax.set_ylabel("heldout mean feature cosine")
    ax.set_title("Trial-disjoint encoder selection")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_encoder_selection_cv.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_encoder_selection_cv.pdf")
    plt.close(fig)


def _compute_encoder_axis_selection_cv() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _encoder_temperature_scores()
    scores["split"] = scores["trial_id"].astype(int) % 2
    rows = []
    for calibration_mode, allow_temperature in [
        ("axis_encoder_only_default_temp", False),
        ("axis_encoder_and_temperature", True),
    ]:
        for eval_split in [0, 1]:
            train = scores[scores["split"].ne(eval_split)].copy()
            eval_rows = scores[scores["split"].eq(eval_split)].copy()
            selected_rows = []
            selected_texts = []
            for (scale, prior_family), slice_eval in eval_rows.groupby(["prior_scale", "prior_family"], sort=True):
                slice_train = train[
                    train["prior_scale"].eq(scale)
                    & train["prior_family"].astype(str).eq(str(prior_family))
                ].copy()
                selected = _best_encoder_temperature(slice_train, allow_temperature=allow_temperature)
                chosen = slice_eval[
                    slice_eval["encoder_slug"].eq(selected["encoder_slug"])
                    & np.isclose(
                        slice_eval["posterior_temperature"].astype(float),
                        float(selected["posterior_temperature"]),
                    )
                ].copy()
                selected_rows.append(chosen)
                selected_text = (
                    f"{float(scale):g}:{prior_family}:"
                    f"{selected['encoder_slug']}@{float(selected['posterior_temperature']):g}"
                )
                selected_texts.append(selected_text)
                rows.append(
                    {
                        "calibration_mode": calibration_mode,
                        "eval_split": int(eval_split),
                        "prior_scale": float(scale),
                        "prior_family": str(prior_family),
                        "selected_encoder_slug": str(selected["encoder_slug"]),
                        "selected_encoder_label": str(selected["encoder_label"]),
                        "selected_temperature": float(selected["posterior_temperature"]),
                        "train_mean_feature_cosine": float(selected["mean_feature_cosine"]),
                        **_mean_metrics(chosen),
                    }
                )
            if selected_rows:
                combined = pd.concat(selected_rows, ignore_index=True)
                rows.append(
                    {
                        "calibration_mode": calibration_mode,
                        "eval_split": int(eval_split),
                        "prior_scale": "all",
                        "prior_family": "all",
                        "selected_encoder_slug": ",".join(selected_texts),
                        "selected_encoder_label": ",".join(selected_texts),
                        "selected_temperature": float("nan"),
                        "train_mean_feature_cosine": float("nan"),
                        **_mean_metrics(combined),
                    }
                )
    cv_rows = pd.DataFrame(rows)
    overall = cv_rows[
        cv_rows["prior_scale"].astype(str).eq("all")
        & cv_rows["prior_family"].astype(str).eq("all")
    ].copy()
    summary_rows = []
    for mode, group in overall.groupby("calibration_mode", sort=False):
        weights = group["n_eval"].astype(float).to_numpy()
        summary_rows.append(
            {
                "calibration_mode": str(mode),
                "n_eval": int(group["n_eval"].sum()),
                "selected_by_split": ";".join(group["selected_encoder_slug"].astype(str).tolist()),
                "image_accuracy": float(np.average(group["image_accuracy"], weights=weights)),
                "mean_feature_cosine": float(np.average(group["mean_feature_cosine"], weights=weights)),
                "mean_true_mass": float(np.average(group["mean_true_mass"], weights=weights)),
                "median_N_eff_fraction": float(group["median_N_eff_fraction"].median()),
            }
        )
    return cv_rows, pd.DataFrame(summary_rows)


def _plot_encoder_axis_selection_cv(scale_summary: pd.DataFrame, axis_summary: pd.DataFrame) -> None:
    if scale_summary.empty or axis_summary.empty:
        return
    rows = pd.concat(
        [
            scale_summary.assign(selection_granularity="scale"),
            axis_summary.assign(selection_granularity="scale + axis"),
        ],
        ignore_index=True,
    )
    order = [
        ("scale", "encoder_only_default_temp", "scale\nencoder"),
        ("scale + axis", "axis_encoder_only_default_temp", "scale+axis\nencoder"),
        ("scale", "encoder_and_temperature", "scale\nencoder+temp"),
        ("scale + axis", "axis_encoder_and_temperature", "scale+axis\nencoder+temp"),
    ]
    plot_rows = []
    for granularity, mode, label in order:
        block = rows[
            rows["selection_granularity"].eq(granularity)
            & rows["calibration_mode"].eq(mode)
        ]
        if not block.empty:
            row = block.iloc[0].copy()
            row["plot_label"] = label
            plot_rows.append(row)
    if not plot_rows:
        return
    plot_df = pd.DataFrame(plot_rows)
    x = np.arange(plot_df.shape[0], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    ax.bar(x, plot_df["mean_feature_cosine"], color=["#6b7280", "#8a5ca8", "#235789", "#2f8f6a"])
    ax.set_xticks(x, plot_df["plot_label"])
    ax.set_ylim(0.90, 0.94)
    ax.set_ylabel("heldout mean feature cosine")
    ax.set_title("Trial-disjoint encoder selection granularity")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_encoder_axis_selection_cv.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_encoder_axis_selection_cv.pdf")
    plt.close(fig)


def _as_bool_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    return text.isin(["true", "1", "1.0", "yes"])


def _native_optimizer_table_status() -> pd.DataFrame:
    opt = _native_scale_conditioned_qc().copy()
    opt["optimizer_success_bool"] = _as_bool_series(opt["optimizer_success"])
    true_candidate = opt["candidate_index"].astype(float).eq(opt["true_candidate_index"].astype(float))
    true_rows = opt[true_candidate].copy()
    by_table = (
        opt.groupby("table_index", as_index=False)
        .agg(
            optimizer_success_fraction=("optimizer_success_bool", "mean"),
            all_candidate_optimizer_success=("optimizer_success_bool", "all"),
            n_candidate_optimizer_rows=("optimizer_success_bool", "size"),
        )
    )
    true_status = true_rows[["table_index", "optimizer_success_bool"]].rename(
        columns={"optimizer_success_bool": "true_candidate_optimizer_success"}
    )
    return by_table.merge(true_status, on="table_index", how="left")


def _compute_native_optimizer_feature_attribution() -> pd.DataFrame:
    trials = _native_scale_conditioned_trials()
    trials = trials[trials["observer_mode"].eq("continuous_joint")].copy()
    status = _native_optimizer_table_status()
    rows = trials.merge(status, on="table_index", how="left")
    out = []
    for grouping, col in [
        ("all_candidates", "all_candidate_optimizer_success"),
        ("true_candidate", "true_candidate_optimizer_success"),
    ]:
        for prior_scale, scale_rows in rows.groupby("prior_scale", sort=True):
            for success_value, group in scale_rows.groupby(col, dropna=False):
                out.append(
                    {
                        "grouping": grouping,
                        "prior_scale": float(prior_scale),
                        "optimizer_success": bool(success_value) if pd.notna(success_value) else False,
                        "n_tables": int(group.shape[0]),
                        "image_accuracy": float(group["image_correct"].mean()),
                        "mean_feature_cosine": float(group["feature_cosine"].mean()),
                        "mean_true_mass": float(group["candidate_posterior_true_mass"].mean()),
                        "median_N_eff_fraction": float(group["candidate_posterior_N_eff_fraction"].median()),
                        "mean_optimizer_success_fraction": float(group["optimizer_success_fraction"].mean()),
                    }
                )
        for success_value, group in rows.groupby(col, dropna=False):
            out.append(
                {
                    "grouping": grouping,
                    "prior_scale": "all",
                    "optimizer_success": bool(success_value) if pd.notna(success_value) else False,
                    "n_tables": int(group.shape[0]),
                    "image_accuracy": float(group["image_correct"].mean()),
                    "mean_feature_cosine": float(group["feature_cosine"].mean()),
                    "mean_true_mass": float(group["candidate_posterior_true_mass"].mean()),
                    "median_N_eff_fraction": float(group["candidate_posterior_N_eff_fraction"].median()),
                    "mean_optimizer_success_fraction": float(group["optimizer_success_fraction"].mean()),
                }
            )
    return pd.DataFrame(out)


def _plot_native_optimizer_feature_attribution(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    rows = summary[
        summary["grouping"].eq("true_candidate")
        & ~summary["prior_scale"].astype(str).eq("all")
    ].copy()
    if rows.empty:
        return
    rows["status"] = np.where(rows["optimizer_success"].astype(bool), "true optimizer success", "true optimizer failed")
    pivot = rows.pivot_table(index="prior_scale", columns="status", values="mean_feature_cosine", aggfunc="first")
    scales = [0.5, 1.0, 2.0]
    x = np.arange(len(scales), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    success = pivot.reindex(scales).get("true optimizer success", pd.Series(index=scales, dtype=float))
    failed = pivot.reindex(scales).get("true optimizer failed", pd.Series(index=scales, dtype=float))
    ax.bar(x - width / 2, success, width=width, color="#235789", label="success")
    ax.bar(x + width / 2, failed, width=width, color="#8a5ca8", label="failed")
    ax.set_xticks(x, [f"{scale:g}x" for scale in scales])
    ax.set_ylim(0.86, 0.94)
    ax.set_ylabel("mean feature cosine")
    ax.set_title("Feature recovery by true-candidate optimizer status")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_optimizer_feature_attribution.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_optimizer_feature_attribution.pdf")
    plt.close(fig)


def _true_candidate_quadratic_geometry() -> pd.DataFrame:
    fit = pd.read_csv(
        SOURCE_ROOT
        / "continuous_joint_quadratic_poisson_scale_conditioned_full"
        / "continuous_joint_qc.csv"
    )
    fit = fit[
        fit["qc_type"].eq("A_I_fit")
        & fit["observation_model"].eq("quadratic_time_constant")
        & fit["candidate_index"].astype(float).eq(fit["true_candidate_index"].astype(float))
    ].copy()
    fit["B_singular2_fraction"] = fit["B_singular2"].astype(float) / np.maximum(
        fit["B_singular1"].astype(float),
        1e-12,
    )
    fit["B_singular3_fraction"] = fit["B_singular3"].astype(float) / np.maximum(
        fit["B_singular1"].astype(float),
        1e-12,
    )
    return fit[
        [
            "table_index",
            "prior_scale",
            "prior_family",
            "basis_dim",
            "ridge",
            "residual_variance",
            "B_train_r2_energy",
            "B_singular1",
            "B_singular2",
            "B_singular3",
            "B_singular2_fraction",
            "B_singular3_fraction",
        ]
    ].copy()


def _candidate_blind_quadratic_geometry() -> pd.DataFrame:
    fit = pd.read_csv(
        SOURCE_ROOT
        / "continuous_joint_quadratic_poisson_scale_conditioned_full"
        / "continuous_joint_qc.csv"
    )
    fit = fit[
        fit["qc_type"].eq("A_I_fit")
        & fit["observation_model"].eq("quadratic_time_constant")
    ].copy()
    grouped = (
        fit.groupby(["table_index", "prior_scale", "prior_family"], as_index=False)
        .agg(
            candidate_mean_residual_variance=("residual_variance", "mean"),
            candidate_min_residual_variance=("residual_variance", "min"),
            candidate_median_residual_variance=("residual_variance", "median"),
            candidate_mean_B_train_r2_energy=("B_train_r2_energy", "mean"),
            n_geometry_candidates=("candidate_index", "nunique"),
        )
    )
    return grouped


def _compute_native_geometry_feature_attribution() -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = _native_scale_conditioned_trials()
    trials = trials[trials["observer_mode"].eq("continuous_joint")].copy()
    rows = trials.merge(_true_candidate_quadratic_geometry(), on=["table_index", "prior_scale", "prior_family"], how="left")
    metric_cols = [
        "residual_variance",
        "B_train_r2_energy",
        "B_singular1",
        "B_singular2_fraction",
        "B_singular3_fraction",
    ]
    corr_rows = []
    for prior_scale, group in rows.groupby("prior_scale", sort=True):
        for metric in metric_cols:
            corr_rows.append(
                {
                    "prior_scale": float(prior_scale),
                    "metric": metric,
                    "pearson_feature_cosine": float(group[metric].corr(group["feature_cosine"], method="pearson")),
                    "spearman_feature_cosine": float(group[metric].corr(group["feature_cosine"], method="spearman")),
                    "pearson_image_correct": float(group[metric].corr(group["image_correct"].astype(float), method="pearson")),
                    "spearman_image_correct": float(group[metric].corr(group["image_correct"].astype(float), method="spearman")),
                    "n_tables": int(group.shape[0]),
                }
            )
    for metric in metric_cols:
        corr_rows.append(
            {
                "prior_scale": "all",
                "metric": metric,
                "pearson_feature_cosine": float(rows[metric].corr(rows["feature_cosine"], method="pearson")),
                "spearman_feature_cosine": float(rows[metric].corr(rows["feature_cosine"], method="spearman")),
                "pearson_image_correct": float(rows[metric].corr(rows["image_correct"].astype(float), method="pearson")),
                "spearman_image_correct": float(rows[metric].corr(rows["image_correct"].astype(float), method="spearman")),
                "n_tables": int(rows.shape[0]),
            }
        )

    bin_rows = []
    for metric in ["B_train_r2_energy", "residual_variance", "B_singular2_fraction"]:
        valid = rows[np.isfinite(rows[metric].astype(float))].copy()
        valid["quartile"] = pd.qcut(valid[metric], 4, labels=False, duplicates="drop")
        for quartile, group in valid.groupby("quartile", sort=True):
            bin_rows.append(
                {
                    "metric": metric,
                    "quartile": int(quartile),
                    "metric_min": float(group[metric].min()),
                    "metric_max": float(group[metric].max()),
                    "n_tables": int(group.shape[0]),
                    "image_accuracy": float(group["image_correct"].mean()),
                    "mean_feature_cosine": float(group["feature_cosine"].mean()),
                    "mean_true_mass": float(group["candidate_posterior_true_mass"].mean()),
                }
            )
    details = rows[
        [
            "table_index",
            "trial_id",
            "prior_scale",
            "prior_family",
            "image_correct",
            "feature_cosine",
            "candidate_posterior_true_mass",
            *metric_cols,
        ]
    ].copy()
    return details, pd.concat(
        [
            pd.DataFrame(corr_rows).assign(summary_type="correlation"),
            pd.DataFrame(bin_rows).assign(summary_type="quartile"),
        ],
        ignore_index=True,
        sort=False,
    )


def _plot_native_geometry_feature_attribution(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    rows = summary[
        summary["summary_type"].eq("quartile")
        & summary["metric"].isin(["B_train_r2_energy", "residual_variance"])
    ].copy()
    if rows.empty:
        return
    labels = {
        "B_train_r2_energy": "quadratic fit R2",
        "residual_variance": "quadratic residual var",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    for ax, metric in zip(axes, ["B_train_r2_energy", "residual_variance"]):
        block = rows[rows["metric"].eq(metric)].sort_values("quartile")
        ax.plot(block["quartile"], block["mean_feature_cosine"], marker="o", color="#8a5ca8", lw=1.8)
        ax.set_xticks(block["quartile"], [f"Q{int(q) + 1}" for q in block["quartile"]])
        ax.set_ylim(0.88, 0.93)
        ax.set_title(labels[metric])
        ax.set_xlabel("true-candidate geometry quartile")
        ax.set_ylabel("mean feature cosine")
        _clean_axis(ax)
    fig.suptitle("Native scale-conditioned feature recovery by observation geometry")
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_geometry_feature_attribution.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_geometry_feature_attribution.pdf")
    plt.close(fig)


def _native_scale_conditioned_temperature_scores() -> pd.DataFrame:
    posterior = _native_scale_conditioned_posterior()
    if "continuous_joint" in set(posterior["observer_mode"].astype(str)):
        posterior = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    else:
        posterior = posterior[posterior["observer_mode"].eq("quadratic_poisson")].copy()
    feature_table = _load_feature_tables()[PRIMARY_LATENT]
    scored = []
    for temperature in TEMPERATURES:
        scored.append(
            _vectorized_mode_rows(
                rows=posterior,
                latent=PRIMARY_LATENT,
                feature_table=feature_table,
                posterior_temperature=float(temperature),
            )
        )
    rows = pd.concat(scored, ignore_index=True)
    geometry = _candidate_blind_quadratic_geometry()
    return rows.merge(geometry, on=["table_index", "prior_scale", "prior_family"], how="left")


def _temperature_rows(rows: pd.DataFrame, temperature: float) -> pd.DataFrame:
    return rows[np.isclose(rows["posterior_temperature"].astype(float), float(temperature))].copy()


def _best_temperature_for_feature(rows: pd.DataFrame) -> float:
    grouped = (
        rows.groupby("posterior_temperature", as_index=False)
        .agg(mean_feature_cosine=("feature_cosine", "mean"), n=("feature_cosine", "size"))
        .sort_values(["mean_feature_cosine", "n", "posterior_temperature"], ascending=[False, False, True])
    )
    if grouped.empty:
        return 1.0
    return float(grouped.iloc[0]["posterior_temperature"])


def _residual_bin_assignments(
    train_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    *,
    metric: str = "candidate_mean_residual_variance",
    n_bins: int = 4,
) -> pd.Series:
    train_values = train_rows[metric].astype(float).to_numpy()
    train_values = train_values[np.isfinite(train_values)]
    if train_values.size < n_bins:
        return pd.Series(np.zeros(target_rows.shape[0], dtype=int), index=target_rows.index)
    edges = np.unique(np.nanquantile(train_values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size <= 2:
        return pd.Series(np.zeros(target_rows.shape[0], dtype=int), index=target_rows.index)
    bins = np.digitize(target_rows[metric].astype(float).to_numpy(), edges[1:-1], right=True)
    return pd.Series(np.clip(bins, 0, edges.size - 2).astype(int), index=target_rows.index)


def _append_calibration_metrics(
    out: list[dict[str, object]],
    *,
    calibration_mode: str,
    eval_split: int,
    prior_scale: object,
    residual_bin: object,
    selected_temperature: str,
    rows: pd.DataFrame,
) -> None:
    metrics = _mean_metrics(rows)
    out.append(
        {
            "calibration_mode": calibration_mode,
            "eval_split": int(eval_split),
            "prior_scale": prior_scale,
            "residual_bin": residual_bin,
            "selected_temperature": selected_temperature,
            **metrics,
        }
    )


def _select_temperature_by_scale(train: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    selected = []
    temp_text = []
    for prior_scale, scale_target in target.groupby("prior_scale", sort=True):
        scale_train = train[train["prior_scale"].eq(prior_scale)].copy()
        scale_temp = _best_temperature_for_feature(scale_train)
        selected.append(_temperature_rows(scale_target, scale_temp))
        temp_text.append(f"{float(prior_scale):g}:{scale_temp:g}")
    if not selected:
        return pd.DataFrame(), ""
    return pd.concat(selected, ignore_index=True), ",".join(temp_text)


def _select_temperature_by_scale_and_geometry(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    metric: str,
    n_bins: int,
) -> tuple[pd.DataFrame, str]:
    selected = []
    temp_text = []
    for prior_scale, scale_target in target.groupby("prior_scale", sort=True):
        scale_train = train[train["prior_scale"].eq(prior_scale)].copy()
        train_bins = _residual_bin_assignments(scale_train, scale_train, metric=metric, n_bins=n_bins)
        target_bins = _residual_bin_assignments(scale_train, scale_target, metric=metric, n_bins=n_bins)
        scale_train = scale_train.assign(residual_bin=train_bins)
        scale_target = scale_target.assign(residual_bin=target_bins)
        for residual_bin, bin_target in scale_target.groupby("residual_bin", sort=True):
            bin_train = scale_train[scale_train["residual_bin"].eq(residual_bin)]
            selected_temp = _best_temperature_for_feature(bin_train)
            selected.append(_temperature_rows(bin_target, selected_temp))
            temp_text.append(f"{float(prior_scale):g}:Q{int(residual_bin) + 1}:{selected_temp:g}")
    if not selected:
        return pd.DataFrame(), ""
    return pd.concat(selected, ignore_index=True), ",".join(temp_text)


def _scheme_mean_feature(rows: pd.DataFrame) -> float:
    return float(rows["feature_cosine"].mean()) if not rows.empty else float("-inf")


def _geometry_calibration_scheme_candidates() -> list[dict[str, object]]:
    metric_labels = {
        "candidate_mean_residual_variance": "mean residual",
        "candidate_min_residual_variance": "min residual",
        "candidate_median_residual_variance": "median residual",
        "candidate_mean_B_train_r2_energy": "mean B R2",
    }
    candidates: list[dict[str, object]] = [
        {
            "scheme": "scale_temp",
            "scheme_label": "scale temp",
            "metric": "",
            "n_bins": 0,
        }
    ]
    for metric, label in metric_labels.items():
        for n_bins in [2, 3, 4]:
            candidates.append(
                {
                    "scheme": "scale_geometry_temp",
                    "scheme_label": f"scale + {label} Q{n_bins}",
                    "metric": metric,
                    "n_bins": int(n_bins),
                }
            )
    return candidates


def _evaluate_calibration_scheme(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    scheme: dict[str, object],
) -> tuple[pd.DataFrame, str]:
    if scheme["scheme"] == "scale_temp":
        return _select_temperature_by_scale(train, target)
    return _select_temperature_by_scale_and_geometry(
        train,
        target,
        metric=str(scheme["metric"]),
        n_bins=int(scheme["n_bins"]),
    )


def _select_geometry_calibration_scheme(train: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    candidates = _geometry_calibration_scheme_candidates()
    train = train.copy()
    train["inner_split"] = train["table_index"].astype(int) % 2
    rows = []
    for candidate_index, scheme in enumerate(candidates):
        eval_blocks = []
        for inner_eval in [0, 1]:
            inner_train = train[train["inner_split"].ne(inner_eval)].copy()
            inner_eval_rows = train[train["inner_split"].eq(inner_eval)].copy()
            scored, _ = _evaluate_calibration_scheme(inner_train, inner_eval_rows, scheme=scheme)
            if not scored.empty:
                eval_blocks.append(scored)
        combined = pd.concat(eval_blocks, ignore_index=True) if eval_blocks else pd.DataFrame()
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "scheme": str(scheme["scheme"]),
                "scheme_label": str(scheme["scheme_label"]),
                "metric": str(scheme["metric"]),
                "n_bins": int(scheme["n_bins"]),
                "inner_mean_feature_cosine": _scheme_mean_feature(combined),
                "inner_image_accuracy": float(combined["image_correct"].mean()) if not combined.empty else float("nan"),
                "inner_n_eval": int(combined.shape[0]),
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["inner_mean_feature_cosine", "inner_n_eval", "n_bins", "candidate_index"],
        ascending=[False, False, True, True],
    )
    selected = selection.iloc[0].to_dict()
    return selected, selection


def _compute_native_geometry_temperature_cv() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _native_scale_conditioned_temperature_scores()
    split_col = "trial_id" if "trial_id" in scores.columns else "table_index"
    scores["split"] = scores[split_col].astype(int) % 2
    rows = []
    selection_rows = []
    for eval_split in [0, 1]:
        train = scores[scores["split"].ne(eval_split)].copy()
        eval_rows = scores[scores["split"].eq(eval_split)].copy()

        default_rows = _temperature_rows(eval_rows, 1.0)
        _append_calibration_metrics(
            rows,
            calibration_mode="default_temp",
            eval_split=eval_split,
            prior_scale="all",
            residual_bin="all",
            selected_temperature="1",
            rows=default_rows,
        )

        scale_selected = []
        geometry_selected = []
        scale_temp_text = []
        geometry_temp_text = []
        for prior_scale, scale_eval in eval_rows.groupby("prior_scale", sort=True):
            scale_train = train[train["prior_scale"].eq(prior_scale)].copy()
            scale_temp = _best_temperature_for_feature(scale_train)
            scale_block = _temperature_rows(scale_eval, scale_temp)
            scale_selected.append(scale_block)
            scale_temp_text.append(f"{float(prior_scale):g}:{scale_temp:g}")
            _append_calibration_metrics(
                rows,
                calibration_mode="scale_temp",
                eval_split=eval_split,
                prior_scale=float(prior_scale),
                residual_bin="all",
                selected_temperature=f"{scale_temp:g}",
                rows=scale_block,
            )

            train_bins = _residual_bin_assignments(scale_train, scale_train)
            eval_bins = _residual_bin_assignments(scale_train, scale_eval)
            scale_train = scale_train.assign(residual_bin=train_bins)
            scale_eval = scale_eval.assign(residual_bin=eval_bins)
            for residual_bin, bin_eval in scale_eval.groupby("residual_bin", sort=True):
                bin_train = scale_train[scale_train["residual_bin"].eq(residual_bin)]
                selected_temp = _best_temperature_for_feature(bin_train)
                bin_block = _temperature_rows(bin_eval, selected_temp)
                geometry_selected.append(bin_block)
                geometry_temp_text.append(f"{float(prior_scale):g}:Q{int(residual_bin) + 1}:{selected_temp:g}")
                _append_calibration_metrics(
                    rows,
                    calibration_mode="scale_residual_quartile_temp",
                    eval_split=eval_split,
                    prior_scale=float(prior_scale),
                    residual_bin=int(residual_bin),
                    selected_temperature=f"{selected_temp:g}",
                    rows=bin_block,
                )

        if scale_selected:
            _append_calibration_metrics(
                rows,
                calibration_mode="scale_temp",
                eval_split=eval_split,
                prior_scale="all",
                residual_bin="all",
                selected_temperature=",".join(scale_temp_text),
                rows=pd.concat(scale_selected, ignore_index=True),
            )
        if geometry_selected:
            _append_calibration_metrics(
                rows,
                calibration_mode="scale_residual_quartile_temp",
                eval_split=eval_split,
                prior_scale="all",
                residual_bin="all",
                selected_temperature=",".join(geometry_temp_text),
                rows=pd.concat(geometry_selected, ignore_index=True),
            )

        selected_scheme, scheme_selection = _select_geometry_calibration_scheme(train)
        scheme_selection["outer_eval_split"] = int(eval_split)
        scheme_selection["selected_for_outer_eval"] = scheme_selection["candidate_index"].astype(int).eq(
            int(selected_scheme["candidate_index"])
        )
        selection_rows.append(scheme_selection)
        selected_rows, selected_text = _evaluate_calibration_scheme(train, eval_rows, scheme=selected_scheme)
        _append_calibration_metrics(
            rows,
            calibration_mode="selected_geometry_temp",
            eval_split=eval_split,
            prior_scale="all",
            residual_bin="all",
            selected_temperature=(
                f"{selected_scheme['scheme_label']}|{selected_text}"
                if selected_text
                else str(selected_scheme["scheme_label"])
            ),
            rows=selected_rows,
        )

    cv = pd.DataFrame(rows)
    if selection_rows:
        selection = pd.concat(selection_rows, ignore_index=True)
        selection.to_csv(OUT_DIR / "continuous_joint_quadratic_geometry_temperature_rule_selection.csv", index=False)
    overall = cv[cv["prior_scale"].astype(str).eq("all") & cv["residual_bin"].astype(str).eq("all")].copy()
    summary = []
    for mode, group in overall.groupby("calibration_mode", sort=False):
        weights = group["n_eval"].astype(float).to_numpy()
        summary.append(
            {
                "calibration_mode": str(mode),
                "n_eval": int(group["n_eval"].sum()),
                "selected_temperature_by_split": ";".join(group["selected_temperature"].astype(str).tolist()),
                "image_accuracy": float(np.average(group["image_accuracy"], weights=weights)),
                "mean_feature_cosine": float(np.average(group["mean_feature_cosine"], weights=weights)),
                "mean_true_mass": float(np.average(group["mean_true_mass"], weights=weights)),
                "median_N_eff_fraction": float(group["median_N_eff_fraction"].median()),
            }
        )
    return cv, pd.DataFrame(summary)


def _plot_native_geometry_temperature_cv(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    labels = {
        "default_temp": "default",
        "scale_temp": "scale temp",
        "scale_residual_quartile_temp": "scale + geom temp",
        "selected_geometry_temp": "selected geom temp",
    }
    rows = summary.copy()
    x = np.arange(rows.shape[0], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), constrained_layout=True)
    colors = ["#6b7280", "#235789", "#8a5ca8", "#2f8f6a"]
    ax.bar(x, rows["mean_feature_cosine"], color=colors[: rows.shape[0]])
    ax.set_xticks(
        x,
        [labels.get(str(mode), str(mode)) for mode in rows["calibration_mode"]],
        rotation=12,
        ha="right",
    )
    ax.set_ylim(0.90, 0.94)
    ax.set_ylabel("heldout mean feature cosine")
    ax.set_title("Trial-disjoint candidate-blind geometry calibration")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_geometry_temperature_cv.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_geometry_temperature_cv.pdf")
    plt.close(fig)


def _default_encoder_rows(label: str, slug: str, rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    block = rows[rows["observer_mode"].eq(mode)].copy()
    block = block[
        [
            "table_index",
            "trial_id",
            "prior_scale",
            "prior_family",
            "feature_cosine",
            "image_correct",
            "candidate_posterior_true_mass",
            "candidate_posterior_N_eff_fraction",
        ]
    ].copy()
    block["encoder_label"] = label
    block["encoder_slug"] = slug
    block["image_correct_float"] = block["image_correct"].astype(float)
    return block


def _encoder_default_rows() -> pd.DataFrame:
    return pd.concat(
        [
            _default_encoder_rows(
                "k=10, ridge 0.01",
                "k10_ridge0p01",
                pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_trials_full.csv"),
                "quadratic_poisson",
            ),
            _default_encoder_rows(
                "k=20, ridge 0.1",
                "k20_ridge0p1",
                pd.read_csv(OUT_DIR / "continuous_joint_quadratic_feature_trials_full_k20_ridge0p1.csv"),
                "quadratic_poisson",
            ),
            _default_encoder_rows(
                "native scale-conditioned",
                "scale_conditioned",
                _native_scale_conditioned_trials(),
                "continuous_joint",
            ),
        ],
        ignore_index=True,
    )


def _bootstrap_mean_ci(values: np.ndarray, *, n_bootstrap: int = 5000, seed: int = 20260623) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    boot = values[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def _paired_encoder_delta(
    wide: pd.DataFrame,
    *,
    comparison: str,
    candidate_slug: str,
    baseline_slug: str,
    group_cols: list[str] | None = None,
) -> list[dict[str, object]]:
    if group_cols is None:
        group_cols = []
    rows = []
    grouped = [(("all",), wide)] if not group_cols else wide.groupby(group_cols, sort=True, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        feature_delta = (
            group[f"feature_cosine__{candidate_slug}"].astype(float)
            - group[f"feature_cosine__{baseline_slug}"].astype(float)
        )
        image_delta = (
            group[f"image_correct_float__{candidate_slug}"].astype(float)
            - group[f"image_correct_float__{baseline_slug}"].astype(float)
        )
        feature_lo, feature_hi = _bootstrap_mean_ci(feature_delta.to_numpy())
        image_lo, image_hi = _bootstrap_mean_ci(image_delta.to_numpy())
        row = {
            "comparison": comparison,
            "candidate_slug": candidate_slug,
            "baseline_slug": baseline_slug,
            "n_tables": int(group.shape[0]),
            "candidate_feature_cosine": float(group[f"feature_cosine__{candidate_slug}"].mean()),
            "baseline_feature_cosine": float(group[f"feature_cosine__{baseline_slug}"].mean()),
            "feature_cosine_delta": float(feature_delta.mean()),
            "feature_cosine_delta_ci_low": feature_lo,
            "feature_cosine_delta_ci_high": feature_hi,
            "candidate_image_accuracy": float(group[f"image_correct_float__{candidate_slug}"].mean()),
            "baseline_image_accuracy": float(group[f"image_correct_float__{baseline_slug}"].mean()),
            "image_accuracy_delta": float(image_delta.mean()),
            "image_accuracy_delta_ci_low": image_lo,
            "image_accuracy_delta_ci_high": image_hi,
            "fraction_feature_delta_positive": float((feature_delta > 0.0).mean()),
        }
        for col, value in zip(group_cols, group_key):
            row[col] = value
        rows.append(row)
    return rows


def _compute_encoder_default_stability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = _encoder_default_rows()
    index_cols = ["table_index", "trial_id", "prior_scale", "prior_family"]
    wide = rows.pivot_table(
        index=index_cols,
        columns="encoder_slug",
        values=["feature_cosine", "image_correct_float"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{slug}" for metric, slug in wide.columns]
    wide = wide.reset_index().dropna()
    comparisons = [
        ("k20_vs_k10", "k20_ridge0p1", "k10_ridge0p01"),
        ("scale_conditioned_vs_k10", "scale_conditioned", "k10_ridge0p01"),
        ("scale_conditioned_vs_k20", "scale_conditioned", "k20_ridge0p1"),
    ]
    overall_rows = []
    slice_rows = []
    for comparison, candidate_slug, baseline_slug in comparisons:
        overall_rows.extend(
            _paired_encoder_delta(
                wide,
                comparison=comparison,
                candidate_slug=candidate_slug,
                baseline_slug=baseline_slug,
            )
        )
        slice_rows.extend(
            _paired_encoder_delta(
                wide,
                comparison=comparison,
                candidate_slug=candidate_slug,
                baseline_slug=baseline_slug,
                group_cols=["prior_scale"],
            )
        )
        slice_rows.extend(
            _paired_encoder_delta(
                wide,
                comparison=comparison,
                candidate_slug=candidate_slug,
                baseline_slug=baseline_slug,
                group_cols=["prior_scale", "prior_family"],
            )
        )
    return rows, pd.DataFrame(overall_rows), pd.DataFrame(slice_rows)


def _plot_encoder_default_stability(overall: pd.DataFrame, by_slice: pd.DataFrame) -> None:
    if overall.empty or by_slice.empty:
        return
    labels = {
        "k20_vs_k10": "k20 - k10",
        "scale_conditioned_vs_k10": "scale-cond - k10",
        "scale_conditioned_vs_k20": "scale-cond - k20",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), constrained_layout=True)

    rows = overall.copy()
    x = np.arange(rows.shape[0], dtype=float)
    y = rows["feature_cosine_delta"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - rows["feature_cosine_delta_ci_low"].to_numpy(dtype=float),
            rows["feature_cosine_delta_ci_high"].to_numpy(dtype=float) - y,
        ]
    )
    axes[0].bar(x, y, color=["#6b7280", "#8a5ca8", "#235789"][: rows.shape[0]])
    axes[0].errorbar(x, y, yerr=yerr, fmt="none", ecolor="#111827", elinewidth=1.0, capsize=3)
    axes[0].axhline(0.0, color="#111827", lw=0.8)
    axes[0].set_xticks(x, [labels.get(str(comp), str(comp)) for comp in rows["comparison"]], rotation=15, ha="right")
    axes[0].set_ylabel("paired feature cosine delta")
    axes[0].set_title("Default encoder paired deltas")
    _clean_axis(axes[0])

    scale_rows = by_slice[
        by_slice["comparison"].isin(["scale_conditioned_vs_k10", "scale_conditioned_vs_k20"])
        & by_slice["prior_scale"].notna()
        & ~by_slice["prior_family"].notna()
    ].copy()
    scales = [0.5, 1.0, 2.0]
    x = np.arange(len(scales), dtype=float)
    width = 0.34
    for offset, comparison in [(-width / 2, "scale_conditioned_vs_k10"), (width / 2, "scale_conditioned_vs_k20")]:
        block = scale_rows[scale_rows["comparison"].eq(comparison)].set_index("prior_scale").reindex(scales)
        axes[1].bar(
            x + offset,
            block["feature_cosine_delta"],
            width=width,
            label=labels[comparison],
            color="#8a5ca8" if comparison.endswith("k10") else "#235789",
        )
    axes[1].axhline(0.0, color="#111827", lw=0.8)
    axes[1].set_xticks(x, [f"{scale:g}x" for scale in scales])
    axes[1].set_ylabel("paired feature cosine delta")
    axes[1].set_title("Scale-conditioned gain by prior scale")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _clean_axis(axes[1])

    fig.suptitle("Quadratic encoder default-score stability")
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_encoder_default_stability.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_encoder_default_stability.pdf")
    plt.close(fig)


def _score_native_analyzer_posterior(dirname: str, *, run_slug: str, run_label: str) -> pd.DataFrame:
    path = SOURCE_ROOT / dirname / "continuous_joint_feature_posterior.csv"
    if not path.exists():
        return pd.DataFrame()
    posterior = pd.read_csv(path)
    posterior = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    if posterior.empty:
        return pd.DataFrame()
    feature_table = _load_feature_tables()[PRIMARY_LATENT]
    scored = _vectorized_mode_rows(
        rows=posterior,
        latent=PRIMARY_LATENT,
        feature_table=feature_table,
        posterior_temperature=1.0,
    )
    scored["run_slug"] = run_slug
    scored["run_label"] = run_label
    return scored


def _compute_scale0p5_ridge_probe() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _score_native_analyzer_posterior(
        "continuous_joint_quadratic_poisson_scale0p5_k10_ridge0p1_full",
        run_slug="scale0p5_k10_ridge0p1",
        run_label="0.5x k=10, ridge 0.1",
    )
    if candidate.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = _native_scale_conditioned_trials()
    baseline = baseline[
        baseline["observer_mode"].eq("continuous_joint")
        & baseline["prior_scale"].astype(float).eq(0.5)
    ].copy()
    baseline["run_slug"] = "scale0p5_k10_ridge0p01"
    baseline["run_label"] = "0.5x k=10, ridge 0.01"
    common = set(candidate["response_cache_path"].astype(str)) & set(baseline["response_cache_path"].astype(str))
    rows = pd.concat(
        [
            baseline[baseline["response_cache_path"].astype(str).isin(common)],
            candidate[candidate["response_cache_path"].astype(str).isin(common)],
        ],
        ignore_index=True,
    )
    summary_rows = []
    for (run_slug, run_label), group in rows.groupby(["run_slug", "run_label"], sort=False):
        metrics = _mean_metrics(group)
        summary_rows.append(
            {
                "row_type": "run",
                "run_slug": str(run_slug),
                "run_label": str(run_label),
                **metrics,
            }
        )
    wide = rows.pivot_table(
        index="response_cache_path",
        columns="run_slug",
        values=["feature_cosine", "image_correct"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{slug}" for metric, slug in wide.columns]
    wide = wide.reset_index().dropna()
    if not wide.empty:
        feature_delta = (
            wide["feature_cosine__scale0p5_k10_ridge0p1"].astype(float)
            - wide["feature_cosine__scale0p5_k10_ridge0p01"].astype(float)
        )
        image_delta = (
            wide["image_correct__scale0p5_k10_ridge0p1"].astype(float)
            - wide["image_correct__scale0p5_k10_ridge0p01"].astype(float)
        )
        feature_lo, feature_hi = _bootstrap_mean_ci(feature_delta.to_numpy())
        image_lo, image_hi = _bootstrap_mean_ci(image_delta.to_numpy())
        summary_rows.append(
            {
                "row_type": "paired_delta",
                "run_slug": "scale0p5_k10_ridge0p1_minus_ridge0p01",
                "run_label": "0.5x k=10 ridge 0.1 - ridge 0.01",
                "n_eval": int(wide.shape[0]),
                "image_accuracy": float(image_delta.mean()),
                "image_accuracy_ci_low": image_lo,
                "image_accuracy_ci_high": image_hi,
                "mean_feature_cosine": float(feature_delta.mean()),
                "mean_feature_cosine_ci_low": feature_lo,
                "mean_feature_cosine_ci_high": feature_hi,
                "mean_true_mass": float("nan"),
                "median_N_eff_fraction": float("nan"),
                "fraction_feature_delta_positive": float((feature_delta > 0.0).mean()),
            }
        )
    return rows, pd.DataFrame(summary_rows)


def _plot_scale0p5_ridge_probe(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    run_rows = summary[summary["row_type"].eq("run")].copy()
    if run_rows.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), constrained_layout=True)
    x = np.arange(run_rows.shape[0], dtype=float)
    labels = run_rows["run_label"].str.replace(", ", "\n", regex=False).tolist()
    axes[0].bar(x, run_rows["mean_feature_cosine"], color=["#8a5ca8", "#235789"][: run_rows.shape[0]])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.91, 0.922)
    axes[0].set_ylabel("mean feature cosine")
    axes[0].set_title("0.5x feature recovery")
    _clean_axis(axes[0])

    axes[1].bar(x, run_rows["image_accuracy"], color=["#8a5ca8", "#235789"][: run_rows.shape[0]])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.76, 0.84)
    axes[1].set_ylabel("image accuracy")
    axes[1].set_title("0.5x hard-negative ID")
    _clean_axis(axes[1])

    fig.suptitle("0.5x k=10 ridge probe")
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_scale0p5_ridge_probe.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_scale0p5_ridge_probe.pdf")
    plt.close(fig)


def _compute_scale2_parallel_k15_probe() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _score_native_analyzer_posterior(
        "continuous_joint_quadratic_poisson_scale2_parallel_k15_ridge0p1_full",
        run_slug="scale2_parallel_k15_ridge0p1",
        run_label="2.0x parallel k=15, ridge 0.1",
    )
    if candidate.empty:
        return pd.DataFrame(), pd.DataFrame()
    refs = []
    for run_slug, run_label, suffix, from_full_slice in [
        ("scale2_parallel_k10_ridge0p01", "2.0x parallel k=10, ridge 0.01", "full", True),
        (
            "scale2_parallel_k20_ridge0p1",
            "2.0x parallel k=20, ridge 0.1",
            "scale2_parallel_k20_ridge0p1_full",
            False,
        ),
    ]:
        rows = pd.read_csv(OUT_DIR / f"continuous_joint_quadratic_feature_trials_{suffix}.csv")
        rows = rows[rows["observer_mode"].eq("quadratic_poisson")].copy()
        if from_full_slice:
            rows = rows[
                rows["prior_scale"].astype(float).eq(2.0)
                & rows["prior_family"].astype(str).eq("axis_edge_parallel")
            ].copy()
        rows["run_slug"] = run_slug
        rows["run_label"] = run_label
        refs.append(rows)
    common = set(candidate["response_cache_path"].astype(str))
    for ref in refs:
        common &= set(ref["response_cache_path"].astype(str))
    rows = pd.concat(
        [candidate[candidate["response_cache_path"].astype(str).isin(common)]]
        + [ref[ref["response_cache_path"].astype(str).isin(common)] for ref in refs],
        ignore_index=True,
    )
    summary_rows = []
    for (run_slug, run_label), group in rows.groupby(["run_slug", "run_label"], sort=False):
        summary_rows.append(
            {
                "row_type": "run",
                "run_slug": str(run_slug),
                "run_label": str(run_label),
                **_mean_metrics(group),
            }
        )
    wide = rows.pivot_table(
        index="response_cache_path",
        columns="run_slug",
        values=["feature_cosine", "image_correct"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{slug}" for metric, slug in wide.columns]
    wide = wide.reset_index().dropna()
    if not wide.empty:
        for baseline_slug in ["scale2_parallel_k20_ridge0p1", "scale2_parallel_k10_ridge0p01"]:
            feature_delta = (
                wide["feature_cosine__scale2_parallel_k15_ridge0p1"].astype(float)
                - wide[f"feature_cosine__{baseline_slug}"].astype(float)
            )
            image_delta = (
                wide["image_correct__scale2_parallel_k15_ridge0p1"].astype(float)
                - wide[f"image_correct__{baseline_slug}"].astype(float)
            )
            feature_lo, feature_hi = _bootstrap_mean_ci(feature_delta.to_numpy())
            image_lo, image_hi = _bootstrap_mean_ci(image_delta.to_numpy())
            summary_rows.append(
                {
                    "row_type": "paired_delta",
                    "run_slug": f"scale2_parallel_k15_ridge0p1_minus_{baseline_slug}",
                    "run_label": f"2.0x parallel k15/r0.1 - {baseline_slug}",
                    "baseline_slug": baseline_slug,
                    "n_eval": int(wide.shape[0]),
                    "image_accuracy": float(image_delta.mean()),
                    "image_accuracy_ci_low": image_lo,
                    "image_accuracy_ci_high": image_hi,
                    "mean_feature_cosine": float(feature_delta.mean()),
                    "mean_feature_cosine_ci_low": feature_lo,
                    "mean_feature_cosine_ci_high": feature_hi,
                    "mean_true_mass": float("nan"),
                    "median_N_eff_fraction": float("nan"),
                    "fraction_feature_delta_positive": float((feature_delta > 0.0).mean()),
                }
            )
    return rows, pd.DataFrame(summary_rows)


def _plot_scale2_parallel_k15_probe(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    run_rows = summary[summary["row_type"].eq("run")].copy()
    if run_rows.empty:
        return
    order = [
        "scale2_parallel_k10_ridge0p01",
        "scale2_parallel_k15_ridge0p1",
        "scale2_parallel_k20_ridge0p1",
    ]
    run_rows = run_rows.set_index("run_slug").reindex(order).dropna(subset=["run_label"]).reset_index()
    labels = ["k10\nr0.01", "k15\nr0.1", "k20\nr0.1"]
    x = np.arange(run_rows.shape[0], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), constrained_layout=True)
    axes[0].bar(x, run_rows["mean_feature_cosine"], color=["#6b7280", "#8a5ca8", "#235789"][: run_rows.shape[0]])
    axes[0].set_xticks(x, labels[: run_rows.shape[0]])
    axes[0].set_ylim(0.895, 0.908)
    axes[0].set_ylabel("mean feature cosine")
    axes[0].set_title("2.0x parallel feature recovery")
    _clean_axis(axes[0])

    axes[1].bar(x, run_rows["image_accuracy"], color=["#6b7280", "#8a5ca8", "#235789"][: run_rows.shape[0]])
    axes[1].set_xticks(x, labels[: run_rows.shape[0]])
    axes[1].set_ylim(0.54, 0.64)
    axes[1].set_ylabel("image accuracy")
    axes[1].set_title("2.0x parallel hard-negative ID")
    _clean_axis(axes[1])

    fig.suptitle("2.0x parallel k15 validation")
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_scale2_parallel_k15_probe.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_scale2_parallel_k15_probe.pdf")
    plt.close(fig)


def _compute_scale1_k15_probe() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _score_native_analyzer_posterior(
        "continuous_joint_quadratic_poisson_scale1_k15_ridge0p1_full",
        run_slug="scale1_k15_ridge0p1",
        run_label="1.0x k=15, ridge 0.1",
    )
    if candidate.empty:
        return pd.DataFrame(), pd.DataFrame()
    refs = []
    for run_slug, run_label, suffix in [
        ("scale1_k10_ridge0p01", "1.0x k=10, ridge 0.01", "full"),
        ("scale1_k20_ridge0p1", "1.0x k=20, ridge 0.1", "full_k20_ridge0p1"),
    ]:
        rows = pd.read_csv(OUT_DIR / f"continuous_joint_quadratic_feature_trials_{suffix}.csv")
        rows = rows[
            rows["observer_mode"].eq("quadratic_poisson")
            & rows["prior_scale"].astype(float).eq(1.0)
        ].copy()
        rows["run_slug"] = run_slug
        rows["run_label"] = run_label
        refs.append(rows)
    common = set(candidate["response_cache_path"].astype(str))
    for ref in refs:
        common &= set(ref["response_cache_path"].astype(str))
    rows = pd.concat(
        [candidate[candidate["response_cache_path"].astype(str).isin(common)]]
        + [ref[ref["response_cache_path"].astype(str).isin(common)] for ref in refs],
        ignore_index=True,
    )
    summary_rows = []
    for (run_slug, run_label), group in rows.groupby(["run_slug", "run_label"], sort=False):
        summary_rows.append(
            {
                "row_type": "run",
                "run_slug": str(run_slug),
                "run_label": str(run_label),
                **_mean_metrics(group),
            }
        )
    wide = rows.pivot_table(
        index="response_cache_path",
        columns="run_slug",
        values=["feature_cosine", "image_correct"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{slug}" for metric, slug in wide.columns]
    wide = wide.reset_index().dropna()
    if not wide.empty:
        for baseline_slug in ["scale1_k20_ridge0p1", "scale1_k10_ridge0p01"]:
            feature_delta = (
                wide["feature_cosine__scale1_k15_ridge0p1"].astype(float)
                - wide[f"feature_cosine__{baseline_slug}"].astype(float)
            )
            image_delta = (
                wide["image_correct__scale1_k15_ridge0p1"].astype(float)
                - wide[f"image_correct__{baseline_slug}"].astype(float)
            )
            feature_lo, feature_hi = _bootstrap_mean_ci(feature_delta.to_numpy())
            image_lo, image_hi = _bootstrap_mean_ci(image_delta.to_numpy())
            summary_rows.append(
                {
                    "row_type": "paired_delta",
                    "run_slug": f"scale1_k15_ridge0p1_minus_{baseline_slug}",
                    "run_label": f"1.0x k15/r0.1 - {baseline_slug}",
                    "baseline_slug": baseline_slug,
                    "n_eval": int(wide.shape[0]),
                    "image_accuracy": float(image_delta.mean()),
                    "image_accuracy_ci_low": image_lo,
                    "image_accuracy_ci_high": image_hi,
                    "mean_feature_cosine": float(feature_delta.mean()),
                    "mean_feature_cosine_ci_low": feature_lo,
                    "mean_feature_cosine_ci_high": feature_hi,
                    "mean_true_mass": float("nan"),
                    "median_N_eff_fraction": float("nan"),
                    "fraction_feature_delta_positive": float((feature_delta > 0.0).mean()),
                }
            )
    return rows, pd.DataFrame(summary_rows)


def _plot_scale1_k15_probe(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    run_rows = summary[summary["row_type"].eq("run")].copy()
    if run_rows.empty:
        return
    order = ["scale1_k10_ridge0p01", "scale1_k15_ridge0p1", "scale1_k20_ridge0p1"]
    run_rows = run_rows.set_index("run_slug").reindex(order).dropna(subset=["run_label"]).reset_index()
    labels = ["k10\nr0.01", "k15\nr0.1", "k20\nr0.1"]
    x = np.arange(run_rows.shape[0], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), constrained_layout=True)
    axes[0].bar(x, run_rows["mean_feature_cosine"], color=["#6b7280", "#8a5ca8", "#235789"][: run_rows.shape[0]])
    axes[0].set_xticks(x, labels[: run_rows.shape[0]])
    axes[0].set_ylim(0.906, 0.913)
    axes[0].set_ylabel("mean feature cosine")
    axes[0].set_title("1.0x feature recovery")
    _clean_axis(axes[0])

    axes[1].bar(x, run_rows["image_accuracy"], color=["#6b7280", "#8a5ca8", "#235789"][: run_rows.shape[0]])
    axes[1].set_xticks(x, labels[: run_rows.shape[0]])
    axes[1].set_ylim(0.64, 0.72)
    axes[1].set_ylabel("image accuracy")
    axes[1].set_title("1.0x hard-negative ID")
    _clean_axis(axes[1])

    fig.suptitle("1.0x k15 validation")
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_scale1_k15_probe.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_scale1_k15_probe.pdf")
    plt.close(fig)


def _compute_axis_interleaved_basis_smoke() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _score_native_analyzer_posterior(
        "continuous_joint_quadratic_poisson_scale_conditioned_axis_interleaved_smoke96",
        run_slug="axis_interleaved_basis",
        run_label="axis-interleaved basis",
    )
    if candidate.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = _native_scale_conditioned_trials()
    baseline = baseline[baseline["observer_mode"].eq("continuous_joint")].copy()
    baseline["run_slug"] = "standard_basis"
    baseline["run_label"] = "standard compact basis"
    common = set(candidate["response_cache_path"].astype(str)) & set(baseline["response_cache_path"].astype(str))
    rows = pd.concat(
        [
            baseline[baseline["response_cache_path"].astype(str).isin(common)],
            candidate[candidate["response_cache_path"].astype(str).isin(common)],
        ],
        ignore_index=True,
    )
    summary_rows = []
    for (run_slug, run_label), group in rows.groupby(["run_slug", "run_label"], sort=False):
        summary_rows.append(
            {
                "row_type": "run",
                "prior_scale": "all",
                "run_slug": str(run_slug),
                "run_label": str(run_label),
                **_mean_metrics(group),
            }
        )
        for prior_scale, scale_group in group.groupby("prior_scale", sort=True):
            summary_rows.append(
                {
                    "row_type": "run",
                    "prior_scale": float(prior_scale),
                    "run_slug": str(run_slug),
                    "run_label": str(run_label),
                    **_mean_metrics(scale_group),
                }
            )
    wide = rows.pivot_table(
        index="response_cache_path",
        columns="run_slug",
        values=["feature_cosine", "image_correct"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{slug}" for metric, slug in wide.columns]
    wide = wide.reset_index().dropna()
    if not wide.empty:
        for label, group in [("all", wide)]:
            feature_delta = (
                group["feature_cosine__axis_interleaved_basis"].astype(float)
                - group["feature_cosine__standard_basis"].astype(float)
            )
            image_delta = (
                group["image_correct__axis_interleaved_basis"].astype(float)
                - group["image_correct__standard_basis"].astype(float)
            )
            feature_lo, feature_hi = _bootstrap_mean_ci(feature_delta.to_numpy())
            image_lo, image_hi = _bootstrap_mean_ci(image_delta.to_numpy())
            summary_rows.append(
                {
                    "row_type": "paired_delta",
                    "prior_scale": label,
                    "run_slug": "axis_interleaved_minus_standard",
                    "run_label": "axis-interleaved basis - standard basis",
                    "n_eval": int(group.shape[0]),
                    "image_accuracy": float(image_delta.mean()),
                    "image_accuracy_ci_low": image_lo,
                    "image_accuracy_ci_high": image_hi,
                    "mean_feature_cosine": float(feature_delta.mean()),
                    "mean_feature_cosine_ci_low": feature_lo,
                    "mean_feature_cosine_ci_high": feature_hi,
                    "mean_true_mass": float("nan"),
                    "median_N_eff_fraction": float("nan"),
                    "fraction_feature_delta_positive": float((feature_delta > 0.0).mean()),
                }
            )
    return rows, pd.DataFrame(summary_rows)


def _plot_axis_interleaved_basis_smoke(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    rows = summary[summary["row_type"].eq("run")].copy()
    rows = rows[rows["prior_scale"].astype(str).isin(["0.5", "1.0", "2.0", "all"])]
    if rows.empty:
        return
    labels = ["0.5", "1.0", "2.0", "all"]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    for offset, run_slug, color, label in [
        (-width / 2, "standard_basis", "#235789", "standard"),
        (width / 2, "axis_interleaved_basis", "#8a5ca8", "axis-interleaved"),
    ]:
        block = rows[rows["run_slug"].eq(run_slug)].copy()
        block["scale_key"] = block["prior_scale"].astype(str)
        values = block.set_index("scale_key").reindex(labels)["mean_feature_cosine"]
        ax.bar(x + offset, values, width=width, color=color, label=label)
    ax.set_xticks(x, ["0.5x", "1x", "2x", "all"])
    ax.set_ylim(0.93, 0.945)
    ax.set_ylabel("mean feature cosine")
    ax.set_title("Scale-conditioned encoder basis smoke")
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_axis_interleaved_basis_smoke.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_axis_interleaved_basis_smoke.pdf")
    plt.close(fig)


def _build_summary() -> pd.DataFrame:
    out = []
    for label, suffix in SUBSET_RUNS:
        row = {
            "comparison": "basis_ridge_subset64",
            "run_label": label,
            "suffix": suffix,
            **_overall_row(suffix, from_full_slice=False),
            **_qc_row(suffix, from_full_slice=False),
        }
        out.append(row)
    for label, suffix, from_full_slice in FULL_RUNS:
        row = {
            "comparison": "full_2x_parallel",
            "run_label": label,
            "suffix": suffix,
            **_overall_row(suffix, from_full_slice=from_full_slice),
            **_qc_row(suffix, from_full_slice=from_full_slice),
            **_best_temperature_row(suffix, from_full_slice=from_full_slice),
        }
        out.append(row)
    for label, suffix in ALL_SCALE_RUNS:
        posterior = pd.read_csv(OUT_DIR / f"continuous_joint_quadratic_feature_posterior_{suffix}.csv")
        row = {
            "comparison": "all_scale",
            "run_label": label,
            "suffix": suffix,
            **_overall_row(suffix, from_full_slice=False),
            **_qc_row(suffix, from_full_slice=False),
            **_best_temperature_from_posterior(posterior),
            **_scale_specific_best_from_posterior(posterior),
        }
        out.append(row)
    native_posterior = _native_scale_conditioned_posterior()
    out.append(
        {
            "comparison": "all_scale",
            "run_label": "native scale-conditioned",
            "suffix": "continuous_joint_quadratic_poisson_scale_conditioned_full",
            **_metrics_from_trials(_native_scale_conditioned_trials()),
            **_native_qc_from_rows(_native_scale_conditioned_qc()),
            **_best_temperature_from_posterior(native_posterior),
            **_scale_specific_best_from_posterior(native_posterior),
        }
    )
    return pd.DataFrame(out)


def _build_native_hybrid_consistency() -> pd.DataFrame:
    native = {
        "source": "native_scale_conditioned",
        **_metrics_from_trials(_native_scale_conditioned_trials()),
        **_best_temperature_from_posterior(_native_scale_conditioned_posterior()),
        **_scale_specific_best_from_posterior(_native_scale_conditioned_posterior()),
    }
    hybrid = {
        "source": "standalone_hybrid",
        **_metrics_from_trials(_hybrid_trials()),
        **_best_temperature_from_posterior(_hybrid_posterior()),
        **_scale_specific_best_from_posterior(_hybrid_posterior()),
    }
    rows = pd.DataFrame([native, hybrid])
    delta = {"source": "native_minus_standalone_hybrid"}
    for col in [
        "image_accuracy",
        "mean_feature_cosine",
        "mean_map_feature_cosine",
        "mean_true_mass",
        "best_temperature_feature_cosine",
        "scale_specific_best_feature_cosine",
    ]:
        delta[col] = float(rows.loc[0, col] - rows.loc[1, col])
    return pd.concat([rows, pd.DataFrame([delta])], ignore_index=True)


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), constrained_layout=True)
    subset = summary[summary["comparison"].eq("basis_ridge_subset64")].copy()
    x = np.arange(subset.shape[0], dtype=float)
    axes[0].bar(x, subset["mean_feature_cosine"], color="#8a5ca8", label="feature cosine")
    axes[0].plot(x, subset["optimizer_success"], color="#235789", marker="o", lw=1.6, label="optimizer success")
    axes[0].set_xticks(x, subset["run_label"], rotation=20, ha="right")
    axes[0].set_ylim(0.50, 0.94)
    axes[0].set_ylabel("mean over 64 tables")
    axes[0].set_title("2x parallel basis/ridge smoke")
    axes[0].legend(frameon=False, loc="upper right")
    _clean_axis(axes[0])

    full = summary[summary["comparison"].eq("full_2x_parallel")].copy()
    x = np.arange(full.shape[0], dtype=float)
    width = 0.28
    axes[1].bar(x - width, full["mean_feature_cosine"], width=width, color="#6b7280", label="default")
    axes[1].bar(
        x,
        full["best_temperature_feature_cosine"],
        width=width,
        color="#8a5ca8",
        label="best temp",
    )
    axes[1].bar(x + width, full["optimizer_success"], width=width, color="#235789", label="optimizer")
    axes[1].set_xticks(x, full["run_label"], rotation=20, ha="right")
    axes[1].set_ylim(0.84, 0.93)
    axes[1].set_ylabel("full 2x parallel slice")
    axes[1].set_title("Focused full-slice check")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    _clean_axis(axes[1])

    all_scale = summary[summary["comparison"].eq("all_scale")].copy()
    x = np.arange(all_scale.shape[0], dtype=float)
    width = 0.28
    axes[2].bar(x - width, all_scale["mean_feature_cosine"], width=width, color="#6b7280", label="default")
    axes[2].bar(
        x,
        all_scale["best_temperature_feature_cosine"],
        width=width,
        color="#235789",
        label="global temp",
    )
    axes[2].bar(
        x + width,
        all_scale["scale_specific_best_feature_cosine"],
        width=width,
        color="#8a5ca8",
        label="scale temp",
    )
    axes[2].set_xticks(x, all_scale["run_label"], rotation=20, ha="right")
    axes[2].set_ylim(0.90, 0.94)
    axes[2].set_title("All-scale encoder candidate")
    axes[2].set_ylabel("mean feature cosine")
    axes[2].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    _clean_axis(axes[2])

    fig.suptitle("Quadratic no-anchor encoder bottleneck comparison")
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_basis_bottleneck_comparison.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_basis_bottleneck_comparison.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _build_summary()
    summary.to_csv(OUT_DIR / "continuous_joint_quadratic_basis_bottleneck_comparison.csv", index=False)
    consistency = _build_native_hybrid_consistency()
    consistency.to_csv(OUT_DIR / "continuous_joint_quadratic_native_hybrid_consistency.csv", index=False)
    _plot(summary)
    encoder_cv, encoder_cv_summary = _compute_encoder_selection_cv()
    encoder_cv.to_csv(OUT_DIR / "continuous_joint_quadratic_encoder_selection_cv.csv", index=False)
    encoder_cv_summary.to_csv(OUT_DIR / "continuous_joint_quadratic_encoder_selection_cv_summary.csv", index=False)
    _plot_encoder_selection_cv(encoder_cv_summary)
    encoder_axis_cv, encoder_axis_cv_summary = _compute_encoder_axis_selection_cv()
    encoder_axis_cv.to_csv(OUT_DIR / "continuous_joint_quadratic_encoder_axis_selection_cv.csv", index=False)
    encoder_axis_cv_summary.to_csv(
        OUT_DIR / "continuous_joint_quadratic_encoder_axis_selection_cv_summary.csv",
        index=False,
    )
    _plot_encoder_axis_selection_cv(encoder_cv_summary, encoder_axis_cv_summary)
    optimizer_feature = _compute_native_optimizer_feature_attribution()
    optimizer_feature.to_csv(OUT_DIR / "continuous_joint_quadratic_optimizer_feature_attribution.csv", index=False)
    _plot_native_optimizer_feature_attribution(optimizer_feature)
    geometry_details, geometry_summary = _compute_native_geometry_feature_attribution()
    geometry_details.to_csv(OUT_DIR / "continuous_joint_quadratic_geometry_feature_attribution_trials.csv", index=False)
    geometry_summary.to_csv(OUT_DIR / "continuous_joint_quadratic_geometry_feature_attribution_summary.csv", index=False)
    _plot_native_geometry_feature_attribution(geometry_summary)
    geometry_temp_cv, geometry_temp_summary = _compute_native_geometry_temperature_cv()
    geometry_temp_cv.to_csv(OUT_DIR / "continuous_joint_quadratic_geometry_temperature_cv.csv", index=False)
    geometry_temp_summary.to_csv(
        OUT_DIR / "continuous_joint_quadratic_geometry_temperature_cv_summary.csv",
        index=False,
    )
    _plot_native_geometry_temperature_cv(geometry_temp_summary)
    encoder_default_rows, encoder_default_summary, encoder_default_by_slice = _compute_encoder_default_stability()
    encoder_default_rows.to_csv(OUT_DIR / "continuous_joint_quadratic_encoder_default_rows.csv", index=False)
    encoder_default_summary.to_csv(
        OUT_DIR / "continuous_joint_quadratic_encoder_default_stability_summary.csv",
        index=False,
    )
    encoder_default_by_slice.to_csv(
        OUT_DIR / "continuous_joint_quadratic_encoder_default_stability_by_slice.csv",
        index=False,
    )
    _plot_encoder_default_stability(encoder_default_summary, encoder_default_by_slice)
    scale0p5_ridge_trials, scale0p5_ridge_summary = _compute_scale0p5_ridge_probe()
    if not scale0p5_ridge_summary.empty:
        scale0p5_ridge_trials.to_csv(OUT_DIR / "continuous_joint_quadratic_scale0p5_ridge_probe_trials.csv", index=False)
        scale0p5_ridge_summary.to_csv(
            OUT_DIR / "continuous_joint_quadratic_scale0p5_ridge_probe_summary.csv",
            index=False,
        )
        _plot_scale0p5_ridge_probe(scale0p5_ridge_summary)
    scale2_k15_trials, scale2_k15_summary = _compute_scale2_parallel_k15_probe()
    if not scale2_k15_summary.empty:
        scale2_k15_trials.to_csv(
            OUT_DIR / "continuous_joint_quadratic_scale2_parallel_k15_probe_trials.csv",
            index=False,
        )
        scale2_k15_summary.to_csv(
            OUT_DIR / "continuous_joint_quadratic_scale2_parallel_k15_probe_summary.csv",
            index=False,
        )
        _plot_scale2_parallel_k15_probe(scale2_k15_summary)
    scale1_k15_trials, scale1_k15_summary = _compute_scale1_k15_probe()
    if not scale1_k15_summary.empty:
        scale1_k15_trials.to_csv(
            OUT_DIR / "continuous_joint_quadratic_scale1_k15_probe_trials.csv",
            index=False,
        )
        scale1_k15_summary.to_csv(
            OUT_DIR / "continuous_joint_quadratic_scale1_k15_probe_summary.csv",
            index=False,
        )
        _plot_scale1_k15_probe(scale1_k15_summary)
    axis_basis_trials, axis_basis_summary = _compute_axis_interleaved_basis_smoke()
    if not axis_basis_summary.empty:
        axis_basis_trials.to_csv(
            OUT_DIR / "continuous_joint_quadratic_axis_interleaved_basis_smoke_trials.csv",
            index=False,
        )
        axis_basis_summary.to_csv(
            OUT_DIR / "continuous_joint_quadratic_axis_interleaved_basis_smoke_summary.csv",
            index=False,
        )
        _plot_axis_interleaved_basis_smoke(axis_basis_summary)


if __name__ == "__main__":
    main()

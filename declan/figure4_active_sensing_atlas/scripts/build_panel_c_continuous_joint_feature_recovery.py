"""Feature-recovery diagnostics for Figure 4C continuous-joint runs.

The continuous-joint experiments currently report image-identity accuracy,
which is a useful but very quantized endpoint. This script reuses the existing
feature-posterior metric: for each posterior over candidate images, recover a
local image-feature vector and compare it to the true candidate by cosine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.likelihood import effective_count, entropy, rank_desc, true_margin
from declan.backimage_trajectory_observer.observer import feature_recovery_metrics, posterior_weighted_feature


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
FEATURE_NPZ = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1"
    / "feature_latent_arrays.npz"
)
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
)

PRIMARY_LATENT = "pyramid_local_field"
LATENTS = [PRIMARY_LATENT]
OBSERVER_ORDER = ["zero", "joint", "best_single_tau", "continuous_joint", "known"]
OBSERVER_LABELS = {
    "zero": "zero eye",
    "joint": "finite joint",
    "best_single_tau": "best catalog",
    "continuous_joint": "continuous",
    "known": "known eye",
}
COLORS = {
    "zero": "#6b7280",
    "joint": "#235789",
    "best_single_tau": "#b35c2e",
    "continuous_joint": "#2f8f6a",
    "known": "#111827",
}
TEMPERATURES = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
TEMPERATURE_SWEEP_SLUGS = [
    "noanchor_ar1",
    "noanchor_residual_ctf",
    "noanchor_brownian_ctf",
    "noanchor_quadratic_poisson",
    "noanchor_quadratic_scale_conditioned",
    "noanchor_quadratic_scale_conditioned_iter160",
    "catalog_residual_top2_shrink",
]
PLOT_RUN_SLUGS = [
    "poisson_k10",
    "poisson_k10_timevary_smooth",
    "poisson_k20_timevary",
    "noanchor_ar1",
    "noanchor_residual_ctf",
    "noanchor_brownian_ctf",
    "noanchor_quadratic_poisson",
    "noanchor_quadratic_scale_conditioned",
    "noanchor_quadratic_scale_conditioned_calibrated",
    "noanchor_quadratic_scale_conditioned_iter160",
    "catalog_residual_all",
    "catalog_residual_top2_shrink",
    "catalog_residual_smooth6",
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    slug: str
    dirname: str
    family: str


RUNS = [
    RunSpec(
        label="Poisson k=5",
        slug="poisson_k5",
        dirname="continuous_joint_linear_poisson_profile_k5_v1",
        family="pure_continuous",
    ),
    RunSpec(
        label="Poisson k=10",
        slug="poisson_k10",
        dirname="continuous_joint_linear_poisson_profile_k10_v1",
        family="pure_continuous",
    ),
    RunSpec(
        label="Poisson k=10 A(t)",
        slug="poisson_k10_timevary",
        dirname="continuous_joint_linear_poisson_compact_k10_timevary_full",
        family="pure_continuous",
    ),
    RunSpec(
        label="Poisson k=10 smooth A(t)",
        slug="poisson_k10_timevary_smooth",
        dirname="continuous_joint_linear_poisson_compact_k10_smoothAt_a0p70_q1em02_full",
        family="pure_continuous",
    ),
    RunSpec(
        label="Poisson k=10 Brownian",
        slug="poisson_k10_matched_brownian",
        dirname="continuous_joint_linear_poisson_compact_k10_matched_brownian_scale1_full",
        family="pure_continuous",
    ),
    RunSpec(
        label="Poisson k=20",
        slug="poisson_k20",
        dirname="continuous_joint_linear_poisson_profile_k20_v1",
        family="pure_continuous",
    ),
    RunSpec(
        label="Poisson k=20 A(t)",
        slug="poisson_k20_timevary",
        dirname="continuous_joint_linear_poisson_compact_k20_timevary_full",
        family="pure_continuous",
    ),
    RunSpec(
        label="No-anchor AR(1)",
        slug="noanchor_ar1",
        dirname="continuous_joint_noanchor_knownstart_linear_ar1_var0p01_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor axis basis",
        slug="noanchor_axis_interleaved",
        dirname="continuous_joint_noanchor_knownstart_linear_ar1_axis_interleaved_basis_k10_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor residual CTF",
        slug="noanchor_residual_ctf",
        dirname="continuous_joint_noanchor_residual_c2f_knownstart_dct_ar1_var0p01_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor DCT CTF",
        slug="noanchor_dct_ctf",
        dirname="continuous_joint_noanchor_dct_c2f_k10_ar1_basis2_var0p05_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor Brownian CTF",
        slug="noanchor_brownian_ctf",
        dirname="continuous_joint_noanchor_dct_c2f_k10_matched_brownian_basis12_var0p05_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor quadratic",
        slug="noanchor_quadratic_poisson",
        dirname="continuous_joint_quadratic_poisson_profile_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor quadratic scale-conditioned",
        slug="noanchor_quadratic_scale_conditioned",
        dirname="continuous_joint_quadratic_poisson_scale_conditioned_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor quadratic scale-calibrated",
        slug="noanchor_quadratic_scale_conditioned_calibrated",
        dirname="continuous_joint_quadratic_poisson_scale_conditioned_calibrated_full",
        family="no_anchor",
    ),
    RunSpec(
        label="No-anchor quadratic scale-conditioned iter160",
        slug="noanchor_quadratic_scale_conditioned_iter160",
        dirname="continuous_joint_quadratic_poisson_scale_conditioned_iter160_full",
        family="no_anchor",
    ),
    RunSpec(
        label="Catalog residual all anchors",
        slug="catalog_residual_all",
        dirname="continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_full",
        family="catalog_residual",
    ),
    RunSpec(
        label="Catalog residual top-2 shrink",
        slug="catalog_residual_top2_shrink",
        dirname="continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_shrink0p19_full",
        family="catalog_residual",
    ),
    RunSpec(
        label="Catalog residual CTF keep 8",
        slug="catalog_residual_ctf_keep8",
        dirname="continuous_joint_catalog_residual_c2f_k10_sched6-0_keep8_topk2_shrink0p19_full",
        family="catalog_residual",
    ),
    RunSpec(
        label="Catalog residual smoothed anchor",
        slug="catalog_residual_smooth6",
        dirname="continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth6_full",
        family="catalog_residual",
    ),
]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _source_row_from_candidate_id(candidate_id: object) -> int:
    text = str(candidate_id)
    prefix = "source_row:"
    if text.startswith(prefix):
        return int(text[len(prefix) :])
    return int(text)


def _load_feature_tables() -> dict[str, dict[int, np.ndarray]]:
    if not FEATURE_NPZ.exists():
        raise FileNotFoundError(FEATURE_NPZ)
    tables: dict[str, dict[int, np.ndarray]] = {}
    with np.load(FEATURE_NPZ) as data:
        source_rows = data["source_row"].astype(int)
        for latent in LATENTS:
            if latent not in data.files:
                continue
            features = np.asarray(data[latent], dtype=np.float64)
            tables[latent] = {int(source): features[idx] for idx, source in enumerate(source_rows)}
    if PRIMARY_LATENT not in tables:
        raise ValueError(f"{FEATURE_NPZ} does not contain {PRIMARY_LATENT}")
    return tables


def _read_feature_posterior(spec: RunSpec) -> pd.DataFrame:
    path = SOURCE_ROOT / spec.dirname / "continuous_joint_feature_posterior.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    if "likelihood_scale" in rows.columns:
        rows = rows[rows["likelihood_scale"].astype(float).eq(1.0)].copy()
    rows["run_label"] = spec.label
    rows["run_slug"] = spec.slug
    rows["run_family"] = spec.family
    rows["run_dirname"] = spec.dirname
    return rows


def _group_columns(rows: pd.DataFrame) -> list[str]:
    preferred = [
        "run_label",
        "run_slug",
        "run_family",
        "run_dirname",
        "table_index",
        "manifest_table_index",
        "trial_id",
        "response_cache_path",
        "candidate_set_mode",
        "observation_family",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "likelihood_scale",
        "n_candidates",
        "n_timebins",
        "n_units",
        "observer_mode",
    ]
    return [col for col in preferred if col in rows.columns]


def _score_group_columns(rows: pd.DataFrame) -> list[str]:
    preferred = ["run_slug", "table_index", "observer_mode"]
    cols = [col for col in preferred if col in rows.columns]
    if len(cols) != len(preferred):
        raise ValueError(f"Missing score group columns: {sorted(set(preferred).difference(cols))}")
    return cols


def _feature_matrix(
    latent: str,
    group: pd.DataFrame,
    feature_table: dict[int, np.ndarray],
    cache: dict[tuple[str, tuple[str, ...]], np.ndarray],
) -> np.ndarray:
    candidate_key = tuple(str(candidate_id) for candidate_id in group["candidate_id"].tolist())
    key = (latent, candidate_key)
    if key in cache:
        return cache[key]
    features = []
    for candidate_id in candidate_key:
        source_row = _source_row_from_candidate_id(candidate_id)
        try:
            features.append(feature_table[source_row])
        except KeyError as exc:
            raise KeyError(f"No feature vector for candidate {candidate_id}") from exc
    matrix = np.vstack(features)
    cache[key] = matrix
    return matrix


def _score_group(
    group: pd.DataFrame,
    feature_tables: dict[str, dict[int, np.ndarray]],
    feature_cache: dict[tuple[str, tuple[str, ...]], np.ndarray],
) -> list[dict[str, object]]:
    ordered = group.sort_values("candidate_index").reset_index(drop=True)
    scores = ordered["candidate_score"].to_numpy(dtype=np.float64)
    true_mask = ordered["is_true_candidate"].astype(bool).to_numpy()
    if true_mask.sum() != 1:
        raise ValueError("Expected exactly one true candidate per observer group")
    true_pos = int(np.flatnonzero(true_mask)[0])
    pred_pos = int(np.nanargmax(scores)) if scores.size else -1
    true_candidate_id = str(ordered.loc[true_pos, "candidate_id"])
    pred_candidate_id = str(ordered.loc[pred_pos, "candidate_id"]) if pred_pos >= 0 else ""
    base = {col: ordered[col].iloc[0] for col in _group_columns(ordered)}
    base.update(
        {
            "true_candidate_index_local": true_pos,
            "pred_candidate_index_local": pred_pos,
            "true_candidate_id": true_candidate_id,
            "pred_candidate_id": pred_candidate_id,
            "true_source_row": _source_row_from_candidate_id(true_candidate_id),
            "pred_source_row": _source_row_from_candidate_id(pred_candidate_id) if pred_candidate_id else -1,
            "image_correct": bool(pred_pos == true_pos),
            "true_rank": rank_desc(scores, true_pos),
            "true_margin": true_margin(scores, true_pos),
        }
    )
    out = []
    for latent, table in feature_tables.items():
        features = _feature_matrix(latent, ordered, table, feature_cache)
        z_true = features[true_pos]
        z_hat, posterior = posterior_weighted_feature(scores, features)
        posterior_metrics = feature_recovery_metrics(z_hat, z_true)
        map_metrics = (
            feature_recovery_metrics(features[pred_pos], z_true)
            if pred_pos >= 0
            else {key: float("nan") for key in posterior_metrics}
        )
        row = dict(base)
        row["latent"] = latent
        row["candidate_posterior_true_mass"] = float(posterior[true_pos]) if posterior.size else float("nan")
        row["candidate_posterior_entropy"] = entropy(posterior)
        row["candidate_posterior_N_eff"] = effective_count(posterior)
        row["candidate_posterior_N_eff_fraction"] = (
            float(effective_count(posterior) / posterior.size) if posterior.size else float("nan")
        )
        row.update(posterior_metrics)
        row["map_feature_cosine"] = map_metrics["feature_cosine"]
        row["map_feature_mse"] = map_metrics["feature_mse"]
        row["map_feature_rmse"] = map_metrics["feature_rmse"]
        out.append(row)
    return out


def _features_for_candidate_ids(candidate_ids: pd.Series, feature_table: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack(
        [feature_table[_source_row_from_candidate_id(candidate_id)] for candidate_id in candidate_ids.tolist()],
        axis=0,
    )


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denom, out=np.full_like(numerator, np.nan, dtype=np.float64), where=denom > 1e-12)


def _format_unique_floats(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna().unique()
    if len(vals) == 0:
        return ""
    return ",".join(f"{float(value):g}" for value in sorted(vals))


def _vectorized_mode_rows(
    *,
    rows: pd.DataFrame,
    latent: str,
    feature_table: dict[int, np.ndarray],
    posterior_temperature: float = 1.0,
    score_column: str = "candidate_score",
) -> pd.DataFrame:
    temp = float(posterior_temperature)
    if temp <= 0.0 or not np.isfinite(temp):
        raise ValueError("posterior_temperature must be positive and finite")
    if score_column not in rows.columns:
        raise ValueError(f"Missing score column {score_column!r}")
    n_candidates = int(rows["n_candidates"].iloc[0]) if "n_candidates" in rows.columns else 4
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    sort_cols = ["table_index"]
    if "response_cache_path" in rows.columns:
        sort_cols.append("response_cache_path")
    sort_cols.append("candidate_index")
    ordered = rows.sort_values(sort_cols).reset_index(drop=True)
    if ordered.shape[0] % n_candidates != 0:
        raise ValueError(f"Rows are not divisible by n_candidates={n_candidates}")
    n_groups = ordered.shape[0] // n_candidates

    table_values = ordered["table_index"].to_numpy().reshape(n_groups, n_candidates)
    if not np.all(table_values == table_values[:, :1]):
        raise ValueError("table_index is not constant within candidate blocks")
    candidate_values = ordered["candidate_index"].to_numpy().reshape(n_groups, n_candidates)
    if not np.all(candidate_values == np.sort(candidate_values, axis=1)):
        raise ValueError("candidate_index is not sorted within candidate blocks")

    scores = ordered[score_column].to_numpy(dtype=np.float64).reshape(n_groups, n_candidates)
    true_mask = ordered["is_true_candidate"].astype(bool).to_numpy().reshape(n_groups, n_candidates)
    if not np.all(true_mask.sum(axis=1) == 1):
        raise ValueError("Expected exactly one true candidate per table/mode block")
    candidate_ids = ordered["candidate_id"].astype(str).to_numpy().reshape(n_groups, n_candidates)
    features = _features_for_candidate_ids(ordered["candidate_id"], feature_table).reshape(n_groups, n_candidates, -1)

    true_pos = np.argmax(true_mask, axis=1)
    pred_pos = np.argmax(scores, axis=1)
    group_index = np.arange(n_groups)
    calibrated_scores = scores / temp
    shifted = calibrated_scores - np.max(calibrated_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    posterior = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    true_features = features[group_index, true_pos, :]
    pred_features = features[group_index, pred_pos, :]
    z_hat = np.einsum("gc,gcd->gd", posterior, features)
    diff = z_hat - true_features
    mse = np.mean(diff * diff, axis=1)
    map_diff = pred_features - true_features
    map_mse = np.mean(map_diff * map_diff, axis=1)

    true_scores = scores[group_index, true_pos]
    competitor_scores = scores.copy()
    competitor_scores[group_index, true_pos] = -np.inf
    best_competitor = np.max(competitor_scores, axis=1)

    if "posterior_temperature" in ordered.columns:
        analyzer_temperature = ordered["posterior_temperature"].to_numpy(dtype=np.float64).reshape(n_groups, n_candidates)
        if not np.allclose(analyzer_temperature, analyzer_temperature[:, :1], rtol=0.0, atol=1e-12):
            raise ValueError("posterior_temperature is not constant within candidate blocks")
        analyzer_temperature_by_group = analyzer_temperature[:, 0]
    else:
        analyzer_temperature_by_group = np.ones(n_groups, dtype=np.float64)

    base = ordered.iloc[::n_candidates].reset_index(drop=True)[_group_columns(ordered)].copy()
    base["latent"] = latent
    base["posterior_temperature"] = temp
    base["analyzer_posterior_temperature"] = analyzer_temperature_by_group
    base["score_column"] = str(score_column)
    base["true_candidate_index_local"] = true_pos
    base["pred_candidate_index_local"] = pred_pos
    true_candidate_ids = candidate_ids[group_index, true_pos]
    pred_candidate_ids = candidate_ids[group_index, pred_pos]
    base["true_candidate_id"] = true_candidate_ids
    base["pred_candidate_id"] = pred_candidate_ids
    base["true_source_row"] = [_source_row_from_candidate_id(value) for value in true_candidate_ids]
    base["pred_source_row"] = [_source_row_from_candidate_id(value) for value in pred_candidate_ids]
    base["image_correct"] = pred_pos == true_pos
    base["true_rank"] = 1 + np.sum(scores > true_scores[:, None], axis=1)
    base["true_margin"] = true_scores - best_competitor
    base["candidate_posterior_true_mass"] = posterior[group_index, true_pos]
    positive = posterior > 0
    base["candidate_posterior_entropy"] = -np.sum(np.where(positive, posterior * np.log(posterior), 0.0), axis=1)
    base["candidate_posterior_N_eff"] = 1.0 / np.sum(posterior * posterior, axis=1)
    base["candidate_posterior_N_eff_fraction"] = base["candidate_posterior_N_eff"] / float(n_candidates)
    base["feature_mse"] = mse
    base["feature_neg_mse"] = -mse
    base["feature_rmse"] = np.sqrt(mse)
    base["feature_l2_error"] = np.linalg.norm(diff, axis=1)
    base["feature_cosine"] = _cosine_matrix(z_hat, true_features)
    base["feature_true_norm"] = np.linalg.norm(true_features, axis=1)
    base["feature_pred_norm"] = np.linalg.norm(z_hat, axis=1)
    base["map_feature_cosine"] = _cosine_matrix(pred_features, true_features)
    base["map_feature_mse"] = map_mse
    base["map_feature_rmse"] = np.sqrt(map_mse)
    return base


def _compute_rows(feature_tables: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    needed = {"candidate_id", "candidate_index", "candidate_score", "is_true_candidate", "observer_mode"}
    out_frames = []
    for spec in RUNS:
        run_rows = _read_feature_posterior(spec)
        missing = needed.difference(run_rows.columns)
        if missing:
            raise ValueError(f"{spec.dirname} missing required columns: {sorted(missing)}")
        for latent, feature_table in feature_tables.items():
            for _, mode_rows in run_rows.groupby("observer_mode", sort=False, dropna=False):
                out_frames.append(_vectorized_mode_rows(rows=mode_rows, latent=latent, feature_table=feature_table))
    return pd.concat(out_frames, ignore_index=True)


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["latent", "run_family", "run_slug", "run_label", "observer_mode", "prior_scale"]
    if "posterior_temperature" in trials.columns:
        group_cols.append("posterior_temperature")
    agg_kwargs = {
        "n": ("feature_cosine", "size"),
        "image_accuracy": ("image_correct", "mean"),
        "mean_feature_cosine": ("feature_cosine", "mean"),
        "median_feature_cosine": ("feature_cosine", "median"),
        "mean_map_feature_cosine": ("map_feature_cosine", "mean"),
        "median_map_feature_cosine": ("map_feature_cosine", "median"),
        "mean_feature_rmse": ("feature_rmse", "mean"),
        "mean_true_mass": ("candidate_posterior_true_mass", "mean"),
        "median_N_eff_fraction": ("candidate_posterior_N_eff_fraction", "median"),
        "median_true_margin": ("true_margin", "median"),
    }
    if "analyzer_posterior_temperature" in trials.columns:
        agg_kwargs["analyzer_posterior_temperature"] = (
            "analyzer_posterior_temperature",
            _format_unique_floats,
        )
    summary = (
        trials.groupby(group_cols, as_index=False, dropna=False)
        .agg(**agg_kwargs)
        .sort_values(group_cols)
    )
    weights = summary["n"].astype(float)
    summary["weighted_feature_cosine"] = summary["mean_feature_cosine"] * weights
    overall_agg = {
        "n": ("n", "sum"),
        "image_accuracy": ("image_accuracy", "mean"),
        "mean_feature_cosine_weighted_sum": ("weighted_feature_cosine", "sum"),
        "mean_map_feature_cosine": ("mean_map_feature_cosine", "mean"),
        "mean_true_mass": ("mean_true_mass", "mean"),
    }
    if "analyzer_posterior_temperature" in summary.columns:
        overall_agg["analyzer_posterior_temperature"] = (
            "analyzer_posterior_temperature",
            lambda values: ",".join(sorted({part for value in values.astype(str) for part in value.split(",") if part})),
        )
    overall = (
        summary.groupby(
            [
                col
                for col in ["latent", "run_family", "run_slug", "run_label", "observer_mode", "posterior_temperature"]
                if col in summary.columns
            ],
            as_index=False,
        )
        .agg(**overall_agg)
    )
    overall["mean_feature_cosine"] = overall["mean_feature_cosine_weighted_sum"] / overall["n"].astype(float)
    overall = overall.drop(columns=["mean_feature_cosine_weighted_sum"])
    overall["prior_scale"] = "all"
    return pd.concat([summary.drop(columns=["weighted_feature_cosine"]), overall], ignore_index=True)


def _plot_summary(summary: pd.DataFrame) -> None:
    primary = summary[(summary["latent"].eq(PRIMARY_LATENT)) & (summary["prior_scale"].astype(str).eq("all"))].copy()
    if "posterior_temperature" in primary.columns:
        primary = primary[primary["posterior_temperature"].astype(float).eq(1.0)]
    run_order = PLOT_RUN_SLUGS
    spec_by_slug = {spec.slug: spec for spec in RUNS}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.1), constrained_layout=True)

    x = np.arange(len(run_order), dtype=float)
    width = 0.16
    for idx, observer in enumerate(["zero", "joint", "best_single_tau", "continuous_joint", "known"]):
        block = primary[primary["observer_mode"].eq(observer)].set_index("run_slug").reindex(run_order)
        axes[0].bar(
            x + (idx - 2.0) * width,
            block["mean_feature_cosine"].to_numpy(dtype=float),
            width=width,
            color=COLORS[observer],
            label=OBSERVER_LABELS[observer],
        )
    axes[0].set_title("Posterior feature recovery")
    axes[0].set_ylabel("mean feature cosine")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [spec_by_slug[slug].label.replace(" ", "\n", 2) for slug in run_order],
        rotation=18,
        ha="right",
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.27))
    _clean_axis(axes[0])

    chosen_slugs = [
        "noanchor_ar1",
        "noanchor_residual_ctf",
        "noanchor_quadratic_poisson",
        "noanchor_quadratic_scale_conditioned",
        "noanchor_quadratic_scale_conditioned_calibrated",
        "noanchor_quadratic_scale_conditioned_iter160",
        "catalog_residual_top2_shrink",
        "catalog_residual_smooth6",
    ]
    line_styles = {
        "noanchor_ar1": ("#235789", "-"),
        "noanchor_residual_ctf": ("#b35c2e", "--"),
        "noanchor_quadratic_poisson": ("#8a5ca8", "-"),
        "noanchor_quadratic_scale_conditioned": ("#6f4ea1", "--"),
        "noanchor_quadratic_scale_conditioned_calibrated": ("#c44e8a", "-"),
        "noanchor_quadratic_scale_conditioned_iter160": ("#4f3b78", "-."),
        "catalog_residual_top2_shrink": ("#2f8f6a", "-"),
        "catalog_residual_smooth6": ("#d62728", "--"),
    }
    scales = [0.5, 1.0, 2.0]
    for slug in chosen_slugs:
        block = summary[
            summary["latent"].eq(PRIMARY_LATENT)
            & summary["run_slug"].eq(slug)
            & summary["observer_mode"].eq("continuous_joint")
            & summary["prior_scale"].isin(scales)
        ].sort_values("prior_scale")
        if "posterior_temperature" in block.columns:
            block = block[block["posterior_temperature"].astype(float).eq(1.0)]
        if block.empty:
            continue
        label = str(block["run_label"].iloc[0])
        color, linestyle = line_styles[slug]
        axes[1].plot(
            block["prior_scale"].astype(float),
            block["mean_feature_cosine"],
            marker="o",
            color=color,
            linestyle=linestyle,
            lw=1.7,
            label=label,
        )
    axes[1].set_title("Continuous feature cosine by scale")
    axes[1].set_xlabel("response/prior scale")
    axes[1].set_ylabel("mean feature cosine")
    axes[1].set_xticks(scales, ["0.5x", "1x", "2x"])
    axes[1].set_ylim(0.82, 0.97)
    axes[1].legend(frameon=False)
    _clean_axis(axes[1])

    fig.suptitle("Continuous-joint feature recovery diagnostics")
    fig.savefig(OUT_DIR / "continuous_joint_feature_recovery.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_feature_recovery.pdf")
    plt.close(fig)


def _compute_temperature_sweep(feature_tables: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    feature_table = feature_tables[PRIMARY_LATENT]
    specs = [spec for spec in RUNS if spec.slug in TEMPERATURE_SWEEP_SLUGS]
    out_frames = []
    for spec in specs:
        run_rows = _read_feature_posterior(spec)
        mode_rows = run_rows[run_rows["observer_mode"].eq("continuous_joint")].copy()
        if mode_rows.empty:
            continue
        score_column = "candidate_score_raw" if "candidate_score_raw" in mode_rows.columns else "candidate_score"
        for temp in TEMPERATURES:
            out_frames.append(
                _vectorized_mode_rows(
                    rows=mode_rows,
                    latent=PRIMARY_LATENT,
                    feature_table=feature_table,
                    posterior_temperature=float(temp),
                    score_column=score_column,
                )
            )
    return pd.concat(out_frames, ignore_index=True)


def _summarize_temperature_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    return (
        sweep.groupby(
            [
                "run_family",
                "run_slug",
                "run_label",
                "observer_mode",
                "latent",
                "prior_scale",
                "posterior_temperature",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n=("feature_cosine", "size"),
            image_accuracy=("image_correct", "mean"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_true_mass=("candidate_posterior_true_mass", "mean"),
            median_N_eff_fraction=("candidate_posterior_N_eff_fraction", "median"),
        )
        .sort_values(["run_slug", "prior_scale", "posterior_temperature"])
    )


def _plot_temperature_sweep(summary: pd.DataFrame) -> None:
    all_scale = (
        summary[summary["latent"].eq(PRIMARY_LATENT)]
        .assign(weight=lambda df: df["n"].astype(float))
        .groupby(["run_slug", "run_label", "posterior_temperature"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mean_feature_cosine": np.average(g["mean_feature_cosine"], weights=g["weight"]),
                    "mean_true_mass": np.average(g["mean_true_mass"], weights=g["weight"]),
                    "median_N_eff_fraction": float(g["median_N_eff_fraction"].median()),
                    "n": int(g["n"].sum()),
                }
            ),
            include_groups=False,
        )
    )
    best = all_scale.loc[all_scale.groupby("run_slug")["mean_feature_cosine"].idxmax()].copy()
    best.to_csv(OUT_DIR / "continuous_joint_feature_temperature_best.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    line_styles = {
        "noanchor_ar1": ("#235789", "-", "o"),
        "noanchor_residual_ctf": ("#b35c2e", "--", "s"),
        "noanchor_brownian_ctf": ("#8a5ca8", ":", "^"),
        "noanchor_quadratic_poisson": ("#8a5ca8", "-", "D"),
        "noanchor_quadratic_scale_conditioned": ("#6f4ea1", "--", "D"),
        "noanchor_quadratic_scale_conditioned_iter160": ("#4f3b78", "-.", "D"),
        "catalog_residual_top2_shrink": ("#2f8f6a", "-", "o"),
    }
    for slug in TEMPERATURE_SWEEP_SLUGS:
        block = all_scale[all_scale["run_slug"].eq(slug)].sort_values("posterior_temperature")
        if block.empty:
            continue
        color, linestyle, marker = line_styles[slug]
        axes[0].plot(
            block["posterior_temperature"],
            block["mean_feature_cosine"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            lw=1.6,
            ms=4.0,
            label=str(block["run_label"].iloc[0]),
        )
        axes[1].plot(
            block["posterior_temperature"],
            block["mean_true_mass"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            lw=1.6,
            ms=4.0,
            label=str(block["run_label"].iloc[0]),
        )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(TEMPERATURES, [f"{v:g}" for v in TEMPERATURES])
        ax.set_xlabel("posterior temperature")
        _clean_axis(ax)
    axes[0].set_ylabel("mean feature cosine")
    axes[0].set_title("Feature recovery calibration")
    axes[0].set_ylim(0.82, 0.98)
    axes[1].set_ylabel("mean true-candidate posterior mass")
    axes[1].set_title("Posterior confidence")
    axes[1].set_ylim(0.20, 0.75)
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Continuous-joint posterior temperature sweep")
    fig.savefig(OUT_DIR / "continuous_joint_feature_temperature_sweep.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_feature_temperature_sweep.pdf")
    plt.close(fig)


def _mean_at_temperature(rows: pd.DataFrame, temp: float) -> dict[str, float]:
    block = rows[np.isclose(rows["posterior_temperature"].astype(float), float(temp))].copy()
    if block.empty:
        return {
            "n": 0,
            "image_accuracy": float("nan"),
            "mean_feature_cosine": float("nan"),
            "mean_true_mass": float("nan"),
            "median_N_eff_fraction": float("nan"),
        }
    return {
        "n": int(block.shape[0]),
        "image_accuracy": float(block["image_correct"].mean()),
        "mean_feature_cosine": float(block["feature_cosine"].mean()),
        "mean_true_mass": float(block["candidate_posterior_true_mass"].mean()),
        "median_N_eff_fraction": float(block["candidate_posterior_N_eff_fraction"].median()),
    }


def _best_temperature(rows: pd.DataFrame) -> float:
    grouped = (
        rows.groupby("posterior_temperature", as_index=False)
        .agg(mean_feature_cosine=("feature_cosine", "mean"))
        .sort_values(["mean_feature_cosine", "posterior_temperature"], ascending=[False, True])
    )
    if grouped.empty:
        return float("nan")
    return float(grouped.iloc[0]["posterior_temperature"])


def _source_row_twofold(values: pd.Series) -> pd.Series:
    sources = pd.to_numeric(values, errors="raise").astype(int)
    unique = sorted({int(value) for value in sources.tolist()})
    fold_by_source = {source: int(index % 2) for index, source in enumerate(unique)}
    return sources.map(fold_by_source).astype(int)


def _compute_temperature_cv(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = sweep[
        sweep["latent"].eq(PRIMARY_LATENT)
        & sweep["observer_mode"].eq("continuous_joint")
        & sweep["table_index"].notna()
    ].copy()
    out: list[dict[str, object]] = []
    split_specs = {"table_index": rows["table_index"].astype(int) % 2}
    if "trial_id" in rows.columns:
        split_specs["trial_id"] = rows["trial_id"].astype(int) % 2
    if "true_source_row" in rows.columns:
        split_specs["source_row"] = _source_row_twofold(rows["true_source_row"])
    for split_key, split_values in split_specs.items():
        split_rows = rows.copy()
        split_rows["split"] = split_values
        for (run_slug, run_label), run_rows in split_rows.groupby(["run_slug", "run_label"], sort=False):
            for eval_split in [0, 1]:
                train = run_rows[run_rows["split"].ne(eval_split)]
                eval_rows = run_rows[run_rows["split"].eq(eval_split)]
                default_metrics = _mean_at_temperature(eval_rows, 1.0)

                global_temp = _best_temperature(train)
                global_metrics = _mean_at_temperature(eval_rows, global_temp)
                out.append(
                        {
                            "run_slug": str(run_slug),
                            "run_label": str(run_label),
                            "split_key": str(split_key),
                            "calibration_mode": "global",
                            "eval_split": int(eval_split),
                            "prior_scale": "all",
                            "prior_family": "all",
                            "selected_temperature": f"{global_temp:g}",
                            "default_temperature": 1.0,
                        "default_mean_feature_cosine": default_metrics["mean_feature_cosine"],
                        "eval_mean_feature_cosine": global_metrics["mean_feature_cosine"],
                        "delta_vs_default": global_metrics["mean_feature_cosine"] - default_metrics["mean_feature_cosine"],
                        "eval_image_accuracy": global_metrics["image_accuracy"],
                        "eval_mean_true_mass": global_metrics["mean_true_mass"],
                        "eval_median_N_eff_fraction": global_metrics["median_N_eff_fraction"],
                        "n_eval": int(global_metrics["n"]),
                    }
                )

                scale_rows = []
                for scale, scale_eval in eval_rows.groupby("prior_scale", sort=True):
                    scale_train = train[train["prior_scale"].eq(scale)]
                    scale_temp = _best_temperature(scale_train)
                    scale_default = _mean_at_temperature(scale_eval, 1.0)
                    scale_metrics = _mean_at_temperature(scale_eval, scale_temp)
                    scale_row = {
                        "run_slug": str(run_slug),
                        "run_label": str(run_label),
                        "split_key": str(split_key),
                        "calibration_mode": "scale_specific",
                        "eval_split": int(eval_split),
                        "prior_scale": float(scale),
                        "prior_family": "all",
                        "selected_temperature": f"{scale_temp:g}",
                        "default_temperature": 1.0,
                        "default_mean_feature_cosine": scale_default["mean_feature_cosine"],
                        "eval_mean_feature_cosine": scale_metrics["mean_feature_cosine"],
                        "delta_vs_default": scale_metrics["mean_feature_cosine"] - scale_default["mean_feature_cosine"],
                        "eval_image_accuracy": scale_metrics["image_accuracy"],
                        "eval_mean_true_mass": scale_metrics["mean_true_mass"],
                        "eval_median_N_eff_fraction": scale_metrics["median_N_eff_fraction"],
                        "n_eval": int(scale_metrics["n"]),
                    }
                    scale_rows.append(scale_row)
                    out.append(scale_row)
                if scale_rows:
                    weights = np.asarray([row["n_eval"] for row in scale_rows], dtype=np.float64)
                    selected = ",".join(f"{row['prior_scale']}:{row['selected_temperature']}" for row in scale_rows)
                    out.append(
                        {
                            "run_slug": str(run_slug),
                            "run_label": str(run_label),
                            "split_key": str(split_key),
                            "calibration_mode": "scale_specific",
                            "eval_split": int(eval_split),
                            "prior_scale": "all",
                            "prior_family": "all",
                            "selected_temperature": selected,
                            "default_temperature": 1.0,
                            "default_mean_feature_cosine": float(
                                np.average([row["default_mean_feature_cosine"] for row in scale_rows], weights=weights)
                            ),
                            "eval_mean_feature_cosine": float(
                                np.average([row["eval_mean_feature_cosine"] for row in scale_rows], weights=weights)
                            ),
                            "delta_vs_default": float(
                                np.average([row["delta_vs_default"] for row in scale_rows], weights=weights)
                            ),
                            "eval_image_accuracy": float(
                                np.average([row["eval_image_accuracy"] for row in scale_rows], weights=weights)
                            ),
                            "eval_mean_true_mass": float(
                                np.average([row["eval_mean_true_mass"] for row in scale_rows], weights=weights)
                            ),
                            "eval_median_N_eff_fraction": float(
                                np.median([row["eval_median_N_eff_fraction"] for row in scale_rows])
                            ),
                            "n_eval": int(np.sum(weights)),
                        }
                    )
                if split_key in {"trial_id", "source_row"} and "prior_family" in eval_rows.columns:
                    slice_rows = []
                    for (scale, prior_family), slice_eval in eval_rows.groupby(
                        ["prior_scale", "prior_family"],
                        sort=True,
                    ):
                        slice_train = train[
                            train["prior_scale"].eq(scale)
                            & train["prior_family"].eq(prior_family)
                        ]
                        slice_temp = _best_temperature(slice_train)
                        slice_default = _mean_at_temperature(slice_eval, 1.0)
                        slice_metrics = _mean_at_temperature(slice_eval, slice_temp)
                        if slice_metrics["n"] == 0:
                            continue
                        slice_row = {
                            "run_slug": str(run_slug),
                            "run_label": str(run_label),
                            "split_key": str(split_key),
                            "calibration_mode": "scale_family_specific",
                            "eval_split": int(eval_split),
                            "prior_scale": float(scale),
                            "prior_family": str(prior_family),
                            "selected_temperature": f"{slice_temp:g}",
                            "default_temperature": 1.0,
                            "default_mean_feature_cosine": slice_default["mean_feature_cosine"],
                            "eval_mean_feature_cosine": slice_metrics["mean_feature_cosine"],
                            "delta_vs_default": slice_metrics["mean_feature_cosine"] - slice_default["mean_feature_cosine"],
                            "eval_image_accuracy": slice_metrics["image_accuracy"],
                            "eval_mean_true_mass": slice_metrics["mean_true_mass"],
                            "eval_median_N_eff_fraction": slice_metrics["median_N_eff_fraction"],
                            "n_eval": int(slice_metrics["n"]),
                        }
                        slice_rows.append(slice_row)
                        out.append(slice_row)
                    if slice_rows:
                        weights = np.asarray([row["n_eval"] for row in slice_rows], dtype=np.float64)
                        selected = ",".join(
                            f"{row['prior_scale']}:{row['prior_family']}:{row['selected_temperature']}"
                            for row in slice_rows
                        )
                        out.append(
                            {
                                "run_slug": str(run_slug),
                                "run_label": str(run_label),
                                "split_key": str(split_key),
                                "calibration_mode": "scale_family_specific",
                                "eval_split": int(eval_split),
                                "prior_scale": "all",
                                "prior_family": "all",
                                "selected_temperature": selected,
                                "default_temperature": 1.0,
                                "default_mean_feature_cosine": float(
                                    np.average(
                                        [row["default_mean_feature_cosine"] for row in slice_rows],
                                        weights=weights,
                                    )
                                ),
                                "eval_mean_feature_cosine": float(
                                    np.average(
                                        [row["eval_mean_feature_cosine"] for row in slice_rows],
                                        weights=weights,
                                    )
                                ),
                                "delta_vs_default": float(
                                    np.average([row["delta_vs_default"] for row in slice_rows], weights=weights)
                                ),
                                "eval_image_accuracy": float(
                                    np.average([row["eval_image_accuracy"] for row in slice_rows], weights=weights)
                                ),
                                "eval_mean_true_mass": float(
                                    np.average([row["eval_mean_true_mass"] for row in slice_rows], weights=weights)
                                ),
                                "eval_median_N_eff_fraction": float(
                                    np.median([row["eval_median_N_eff_fraction"] for row in slice_rows])
                                ),
                                "n_eval": int(np.sum(weights)),
                            }
                        )
    return pd.DataFrame(out)


def _summarize_temperature_cv(cv_rows: pd.DataFrame) -> pd.DataFrame:
    if cv_rows.empty:
        return cv_rows
    overall = cv_rows[cv_rows["prior_scale"].astype(str).eq("all")].copy()
    grouped_rows = []
    for (run_slug, run_label, split_key, mode), group in overall.groupby(
        ["run_slug", "run_label", "split_key", "calibration_mode"],
        sort=False,
    ):
        weights = group["n_eval"].astype(float).to_numpy()
        grouped_rows.append(
            {
                "run_slug": str(run_slug),
                "run_label": str(run_label),
                "split_key": str(split_key),
                "calibration_mode": str(mode),
                "n_eval": int(group["n_eval"].sum()),
                "selected_temperature_by_split": ";".join(group["selected_temperature"].astype(str).tolist()),
                "default_mean_feature_cosine": float(np.average(group["default_mean_feature_cosine"], weights=weights)),
                "eval_mean_feature_cosine": float(np.average(group["eval_mean_feature_cosine"], weights=weights)),
                "delta_vs_default": float(np.average(group["delta_vs_default"], weights=weights)),
                "eval_image_accuracy": float(np.average(group["eval_image_accuracy"], weights=weights)),
                "eval_mean_true_mass": float(np.average(group["eval_mean_true_mass"], weights=weights)),
                "eval_median_N_eff_fraction": float(group["eval_median_N_eff_fraction"].median()),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(["eval_mean_feature_cosine", "run_slug"], ascending=[False, True])


def _plot_temperature_cv(cv_summary: pd.DataFrame) -> None:
    if cv_summary.empty:
        return
    chosen = [
        "noanchor_ar1",
        "noanchor_residual_ctf",
        "noanchor_quadratic_poisson",
        "noanchor_quadratic_scale_conditioned",
        "noanchor_quadratic_scale_conditioned_iter160",
        "catalog_residual_top2_shrink",
    ]
    labels = {
        "noanchor_ar1": "AR(1)",
        "noanchor_residual_ctf": "residual CTF",
        "noanchor_quadratic_poisson": "quadratic",
        "noanchor_quadratic_scale_conditioned": "quadratic scale",
        "noanchor_quadratic_scale_conditioned_iter160": "quadratic scale 160",
        "catalog_residual_top2_shrink": "catalog residual",
    }
    preferred_split = "source_row" if cv_summary["split_key"].eq("source_row").any() else "trial_id"
    rows = cv_summary[
        cv_summary["run_slug"].isin(chosen)
        & cv_summary["split_key"].eq(preferred_split)
        & cv_summary["calibration_mode"].isin(["global", "scale_specific"])
    ].copy()
    if rows.empty:
        rows = cv_summary[
            cv_summary["run_slug"].isin(chosen)
            & cv_summary["calibration_mode"].isin(["global", "scale_specific"])
        ].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.6), constrained_layout=True)
    x = np.arange(len(chosen), dtype=float)
    width = 0.24
    default = rows.drop_duplicates("run_slug").set_index("run_slug").reindex(chosen)
    global_rows = rows[rows["calibration_mode"].eq("global")].set_index("run_slug").reindex(chosen)
    scale_rows = rows[rows["calibration_mode"].eq("scale_specific")].set_index("run_slug").reindex(chosen)
    ax.bar(x - width, default["default_mean_feature_cosine"], width=width, color="#6b7280", label="default")
    ax.bar(x, global_rows["eval_mean_feature_cosine"], width=width, color="#235789", label="global temp")
    ax.bar(x + width, scale_rows["eval_mean_feature_cosine"], width=width, color="#8a5ca8", label="scale temp")
    ax.set_xticks(x, [labels.get(slug, slug) for slug in chosen], rotation=20, ha="right")
    ax.set_ylabel("heldout mean feature cosine")
    split_label = "source-row-disjoint" if preferred_split == "source_row" else "trial-disjoint"
    ax.set_title(f"{split_label.capitalize()} posterior-temperature calibration")
    ax.set_ylim(0.82, 0.99)
    ax.legend(frameon=False)
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_feature_temperature_cv.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_feature_temperature_cv.pdf")
    plt.close(fig)


def _summarize_quadratic_slice_cv(cv_rows: pd.DataFrame) -> pd.DataFrame:
    if cv_rows.empty or "prior_family" not in cv_rows.columns:
        return pd.DataFrame()
    preferred_split = "source_row" if cv_rows["split_key"].eq("source_row").any() else "trial_id"
    rows = cv_rows[
        cv_rows["run_slug"].eq("noanchor_quadratic_poisson")
        & cv_rows["split_key"].eq(preferred_split)
        & cv_rows["calibration_mode"].eq("scale_family_specific")
        & ~cv_rows["prior_scale"].astype(str).eq("all")
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    out = []
    for (scale, prior_family), group in rows.groupby(["prior_scale", "prior_family"], sort=True):
        weights = group["n_eval"].astype(float).to_numpy()
        out.append(
            {
                "prior_scale": float(scale),
                "prior_family": str(prior_family),
                "n_eval": int(group["n_eval"].sum()),
                "selected_temperature_by_split": ";".join(group["selected_temperature"].astype(str).tolist()),
                "default_mean_feature_cosine": float(np.average(group["default_mean_feature_cosine"], weights=weights)),
                "heldout_mean_feature_cosine": float(np.average(group["eval_mean_feature_cosine"], weights=weights)),
                "delta_vs_default": float(np.average(group["delta_vs_default"], weights=weights)),
                "heldout_mean_true_mass": float(np.average(group["eval_mean_true_mass"], weights=weights)),
                "heldout_median_N_eff_fraction": float(group["eval_median_N_eff_fraction"].median()),
            }
        )
    return pd.DataFrame(out).sort_values(["prior_scale", "prior_family"])


def _plot_quadratic_slice_cv(slice_summary: pd.DataFrame) -> None:
    if slice_summary.empty:
        return
    rows = slice_summary.sort_values(["prior_scale", "prior_family"]).copy()
    family_labels = {
        "axis_edge_parallel": "parallel",
        "axis_edge_orthogonal": "orthogonal",
    }
    labels = [
        f"{scale:g}x\n{family_labels.get(str(family), str(family))}"
        for scale, family in zip(rows["prior_scale"], rows["prior_family"])
    ]
    x = np.arange(rows.shape[0], dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.7), constrained_layout=True)
    ax.bar(
        x - width / 2,
        rows["default_mean_feature_cosine"],
        width=width,
        color="#6b7280",
        label="default temp",
    )
    ax.bar(
        x + width / 2,
        rows["heldout_mean_feature_cosine"],
        width=width,
        color="#8a5ca8",
        label="slice-heldout temp",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean feature cosine")
    ax.set_title("Quadratic no-anchor calibration by scale and axis prior")
    ax.set_ylim(0.88, 0.97)
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_temperature_slice_cv.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_quadratic_temperature_slice_cv.pdf")
    plt.close(fig)


def _compute_endpoint_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[(summary["latent"].eq(PRIMARY_LATENT)) & (summary["prior_scale"].astype(str).eq("all"))].copy()
    if "posterior_temperature" in rows.columns:
        rows = rows[rows["posterior_temperature"].astype(float).eq(1.0)].copy()
    continuous = rows[rows["observer_mode"].eq("continuous_joint")].copy()
    if continuous.empty:
        return pd.DataFrame()

    baseline_cols = ["run_slug", "observer_mode", "image_accuracy", "mean_feature_cosine"]
    baselines = rows[baseline_cols].copy()
    wide = baselines.pivot_table(
        index="run_slug",
        columns="observer_mode",
        values=["image_accuracy", "mean_feature_cosine"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{observer}" for metric, observer in wide.columns]
    out = continuous.merge(wide.reset_index(), on="run_slug", how="left")
    out["feature_rank"] = out["mean_feature_cosine"].rank(method="min", ascending=False).astype(int)
    out["image_rank"] = out["image_accuracy"].rank(method="min", ascending=False).astype(int)
    out["feature_minus_image_rank"] = out["feature_rank"] - out["image_rank"]
    out["feature_gain_vs_zero"] = out["mean_feature_cosine"] - out.get("mean_feature_cosine__zero", np.nan)
    out["image_gain_vs_zero"] = out["image_accuracy"] - out.get("image_accuracy__zero", np.nan)
    out["feature_gap_to_finite_joint"] = out.get("mean_feature_cosine__joint", np.nan) - out["mean_feature_cosine"]
    out["feature_gap_to_known_eye"] = out.get("mean_feature_cosine__known", np.nan) - out["mean_feature_cosine"]
    out["image_gap_to_finite_joint"] = out.get("image_accuracy__joint", np.nan) - out["image_accuracy"]
    keep_cols = [
        "run_family",
        "run_slug",
        "run_label",
        "n",
        "feature_rank",
        "image_rank",
        "feature_minus_image_rank",
        "image_accuracy",
        "mean_feature_cosine",
        "mean_map_feature_cosine",
        "feature_gain_vs_zero",
        "image_gain_vs_zero",
        "feature_gap_to_finite_joint",
        "feature_gap_to_known_eye",
        "image_gap_to_finite_joint",
        "mean_true_mass",
        "median_N_eff_fraction",
    ]
    return out[keep_cols].sort_values(["feature_rank", "image_rank", "run_slug"]).reset_index(drop=True)


def _plot_endpoint_comparison(endpoint: pd.DataFrame) -> None:
    if endpoint.empty:
        return
    family_colors = {
        "pure_continuous": "#235789",
        "no_anchor": "#8a5ca8",
        "catalog_residual": "#2f8f6a",
    }
    family_labels = {
        "pure_continuous": "pure continuous",
        "no_anchor": "no anchor",
        "catalog_residual": "catalog residual",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    for family, group in endpoint.groupby("run_family", sort=False):
        axes[0].scatter(
            group["image_accuracy"],
            group["mean_feature_cosine"],
            s=40,
            color=family_colors.get(str(family), "#6b7280"),
            label=family_labels.get(str(family), str(family)),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )
    label_slugs = {
        "noanchor_ar1",
        "noanchor_quadratic_scale_conditioned",
        "noanchor_quadratic_scale_conditioned_calibrated",
        "catalog_residual_top2_shrink",
    }
    label_offsets = {
        "noanchor_ar1": (5, 5),
        "noanchor_quadratic_scale_conditioned": (5, 3),
        "noanchor_quadratic_scale_conditioned_calibrated": (5, 5),
        "catalog_residual_top2_shrink": (-70, 6),
    }
    for _, row in endpoint[endpoint["run_slug"].isin(label_slugs)].iterrows():
        axes[0].annotate(
            str(row["run_label"]).replace("No-anchor ", "").replace("Catalog residual ", "cat. "),
            (row["image_accuracy"], row["mean_feature_cosine"]),
            textcoords="offset points",
            xytext=label_offsets.get(str(row["run_slug"]), (4, 3)),
            fontsize=7.0,
        )
    axes[0].set_xlabel("hard-negative image accuracy")
    axes[0].set_ylabel("posterior-weighted feature cosine")
    axes[0].set_title("Endpoint comparison")
    axes[0].legend(frameon=False, loc="lower right")
    _clean_axis(axes[0])

    chosen = endpoint.sort_values("feature_rank").head(10).copy()
    y = np.arange(chosen.shape[0], dtype=float)
    axes[1].barh(y - 0.18, chosen["mean_feature_cosine"], height=0.32, color="#8a5ca8", label="feature cosine")
    axes[1].barh(y + 0.18, chosen["image_accuracy"], height=0.32, color="#6b7280", label="image accuracy")
    axes[1].set_yticks(y, chosen["run_label"].str.replace("No-anchor ", "", regex=False))
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.50, 1.0)
    axes[1].set_xlabel("metric value")
    axes[1].set_title("Rank by feature recovery")
    axes[1].legend(frameon=False, loc="lower right")
    _clean_axis(axes[1])

    fig.suptitle("Use feature recovery as the development endpoint; image ID remains the hard endpoint")
    fig.savefig(OUT_DIR / "continuous_joint_endpoint_metric_comparison.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_endpoint_metric_comparison.pdf")
    plt.close(fig)


def _write_readme(summary: pd.DataFrame) -> None:
    primary = summary[(summary["latent"].eq(PRIMARY_LATENT)) & (summary["prior_scale"].astype(str).eq("all"))]
    if "posterior_temperature" in primary.columns:
        primary = primary[primary["posterior_temperature"].astype(float).eq(1.0)]
    wide = primary.pivot_table(index=["run_slug", "run_label"], columns="observer_mode", values="mean_feature_cosine")
    best_row = primary[primary["observer_mode"].eq("continuous_joint")].sort_values(
        "mean_feature_cosine", ascending=False
    ).iloc[0]
    table = wide.reset_index()
    table_lines = ["| " + " | ".join(str(col) for col in table.columns) + " |"]
    table_lines.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
    for _, row in table.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")

    lines = [
        "# Continuous-Joint Feature-Recovery Diagnostics",
        "",
        "This diagnostic asks whether each observer recovers the true candidate's local image feature vector, even when the discrete image-identification decision is wrong.",
        "",
        f"Feature source: `{FEATURE_NPZ}`.",
        f"Primary latent for the plot: `{PRIMARY_LATENT}`.",
        "",
        f"Best continuous-joint mean feature cosine: {best_row['mean_feature_cosine']:.4f} ({best_row['run_label']}).",
        "",
        "Posterior feature cosine is the cosine between the true feature vector and the posterior-weighted candidate feature vector. MAP feature cosine is also written to the trial CSV so near-miss wrong-image choices can be inspected.",
        "",
        "Model-development ranking should use posterior feature cosine first. Hard-negative image accuracy is retained as the stricter top-1 identity endpoint. See `continuous_joint_endpoint_metric_comparison.csv` and `continuous_joint_endpoint_metric_comparison.png` for the explicit rank comparison.",
        "",
        "Production analyzer runs can emit calibrated continuous-joint posterior scores with `--continuous-posterior-temperature-by-scale 0.5:0.125,1.0:0.125,2.0:0.5`; raw scores remain available as `candidate_score_raw`.",
        "",
        "In these diagnostics, `posterior_temperature` is the additional posthoc scorer temperature. `analyzer_posterior_temperature` preserves the temperature already emitted by analyzer rows.",
        "",
        "Full calibrated analyzer artifact: `continuous_joint_quadratic_poisson_scale_conditioned_calibrated_full`; summary CSV: `continuous_joint_quadratic_scale_conditioned_calibrated_full_summary.csv`; all-scale feature cosine 0.93584 at unchanged image accuracy 0.7083.",
        "",
        "Primary all-scale mean feature cosine:",
        "",
        "\n".join(table_lines),
        "",
    ]
    (OUT_DIR / "continuous_joint_feature_recovery_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    feature_tables = _load_feature_tables()
    trials = _compute_rows(feature_tables)
    summary = _summarize(trials)
    trials.to_csv(OUT_DIR / "continuous_joint_feature_recovery_trials.csv", index=False)
    summary.to_csv(OUT_DIR / "continuous_joint_feature_recovery_summary.csv", index=False)
    _plot_summary(summary)
    endpoint = _compute_endpoint_comparison(summary)
    endpoint.to_csv(OUT_DIR / "continuous_joint_endpoint_metric_comparison.csv", index=False)
    _plot_endpoint_comparison(endpoint)
    _write_readme(summary)
    temperature_sweep = _compute_temperature_sweep(feature_tables)
    temperature_summary = _summarize_temperature_sweep(temperature_sweep)
    temperature_sweep.to_csv(OUT_DIR / "continuous_joint_feature_temperature_trials.csv", index=False)
    temperature_summary.to_csv(OUT_DIR / "continuous_joint_feature_temperature_summary.csv", index=False)
    _plot_temperature_sweep(temperature_summary)
    temperature_cv = _compute_temperature_cv(temperature_sweep)
    temperature_cv_summary = _summarize_temperature_cv(temperature_cv)
    temperature_cv.to_csv(OUT_DIR / "continuous_joint_feature_temperature_cv.csv", index=False)
    temperature_cv_summary.to_csv(OUT_DIR / "continuous_joint_feature_temperature_cv_summary.csv", index=False)
    _plot_temperature_cv(temperature_cv_summary)
    quadratic_slice_cv = _summarize_quadratic_slice_cv(temperature_cv)
    quadratic_slice_cv.to_csv(OUT_DIR / "continuous_joint_quadratic_temperature_slice_cv_summary.csv", index=False)
    _plot_quadratic_slice_cv(quadratic_slice_cv)


if __name__ == "__main__":
    main()

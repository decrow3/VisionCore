"""Build a continuous feature-embedding reconstruction diagnostic for 4C.

This is the candidate-free feature endpoint proposed after the current
candidate-posterior observer. It still uses the existing response-table caches
to fit the readout, but at test time it infers a continuous feature embedding
``z`` rather than selecting or averaging over candidate images. The response
movie can be represented in full unit coordinates or projected through a
compact basis; compact coordinates should be treated as an intervention/control
unless explicitly promoted by the analysis design.

The default model is deliberately small and linear-Gaussian:

    response_features = A z + noise
    z ~ N(0, I)

The posterior mean is used as ``z_hat`` and scored against the true compact
feature embedding. For information-upper-bound probes, the script can also fit
a Tejas-style nonlinear MLP decoder from response features to ``z``. Cross-
fitting is by source image, so response samples for a held-out source row are
not used to fit the response model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.feature_recovery_scores import per_sample_sse_sst
from declan.redundancy_resolved_v1_population import PopulationView, full_population_view, load_population_view


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
COMPACT_BASIS = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_disjoint_compact_basis_delta025_v1"
    / "image_disjoint_compact_basis_delta0p25_fold0of2.npz"
)
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_feature_embedding"
)

RESPONSE_BASIS_MODES = ("compact", "full_units")
RESPONSE_POPULATION_MODES = ("full756", "rr100")
RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
STATIC_MATCHED_MEAN_MODE = "static_matched_mean"
STATIC_MATCHED_MEAN_ALIASES = {
    "zero_static",
    "static_zero",
    "static_crop_center",
    "static_trace_mean_centered",
    "stabilized_at_mean_position",
}
PRIMARY_LATENT = "pyramid_local_field"
DEFAULT_DECODER_MODES = ("linear_gaussian",)
DEFAULT_FEATURE_SPACE_MODES = (
    "fold_zscore_whitened_pca",
    "fold_centered_whitened_pca",
    "fold_zscore_pca",
)
SOURCE_WEIGHTING_MODES = ("source_balanced", "row_unweighted")
OBSERVER_ORDER = [
    "known_eye",
    "hidden_eye_tau_marginal",
    "hidden_eye_prior_mean",
    "zero_eye_on_motion",
    STATIC_MATCHED_MEAN_MODE,
]
OBSERVER_LABELS = {
    "known_eye": "known eye",
    "hidden_eye_tau_marginal": "hidden eye",
    "hidden_eye_prior_mean": "prior mean",
    "zero_eye_on_motion": "zero-eye model",
    STATIC_MATCHED_MEAN_MODE: "static matched mean",
}
OBSERVER_COLORS = {
    "known_eye": "#111827",
    "hidden_eye_tau_marginal": "#235789",
    "hidden_eye_prior_mean": "#4c78a8",
    "zero_eye_on_motion": "#8a5ca8",
    STATIC_MATCHED_MEAN_MODE: "#66717d",
}


@dataclass(frozen=True)
class ReconstructionSpec:
    slug: str
    label: str
    train_bank: str
    test_input: str
    interpretation: str


@dataclass
class SampleBank:
    x: np.ndarray
    source_rows: np.ndarray
    table_indices: np.ndarray


@dataclass
class TestSet:
    rows: pd.DataFrame
    observed_x: np.ndarray
    static_matched_mean_x: np.ndarray


@dataclass(frozen=True)
class FeatureTable:
    feature_npz: Path
    latent: str
    source_rows: np.ndarray
    features: np.ndarray
    source_to_index: dict[int, int]


@dataclass(frozen=True)
class FeatureTransform:
    latent: str
    feature_space_mode: str
    feature_dim: int
    mean: np.ndarray
    sd: np.ndarray
    components: np.ndarray
    denom: np.ndarray
    weights: np.ndarray | None
    fit_scope: str
    preprocessing: str
    whitened: bool
    weighted: bool
    n_fit_sources: int
    raw_feature_dim: int
    explained_variance_sum: float
    explained_variance_first5: list[float]


@dataclass
class ForwardPosteriorModel:
    response_mean: np.ndarray
    response_map: np.ndarray
    posterior_gain: np.ndarray
    noise_variance: float
    ridge: float
    n_train: int


@dataclass(frozen=True)
class MLPConfig:
    hidden_dim: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    validation_fraction: float
    max_train_samples: int
    device: str
    seed: int


SPECS = [
    ReconstructionSpec(
        slug="known_eye",
        label=OBSERVER_LABELS["known_eye"],
        train_bank="known",
        test_input="observed",
        interpretation="Feature posterior from measured-motion response family.",
    ),
    ReconstructionSpec(
        slug="hidden_eye_tau_marginal",
        label=OBSERVER_LABELS["hidden_eye_tau_marginal"],
        train_bank="prior_all",
        test_input="observed",
        interpretation="Feature posterior after treating trajectory as nuisance variation.",
    ),
    ReconstructionSpec(
        slug="hidden_eye_prior_mean",
        label=OBSERVER_LABELS["hidden_eye_prior_mean"],
        train_bank="prior_mean",
        test_input="observed",
        interpretation="A stricter hidden-eye baseline trained on trajectory-averaged responses.",
    ),
    ReconstructionSpec(
        slug="zero_eye_on_motion",
        label=OBSERVER_LABELS["zero_eye_on_motion"],
        train_bank="zero",
        test_input="observed",
        interpretation="Zero-eye response model applied to moving-response observations.",
    ),
    ReconstructionSpec(
        slug=STATIC_MATCHED_MEAN_MODE,
        label=OBSERVER_LABELS[STATIC_MATCHED_MEAN_MODE],
        train_bank="zero",
        test_input=STATIC_MATCHED_MEAN_MODE,
        interpretation=(
            "Static crop-center counterfactual: no within-patch displacement, "
            "with each BackImage crop centered at its own mean fixation position."
        ),
    ),
]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.3,
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_row_from_candidate_id(candidate_id: object) -> int:
    text = str(candidate_id)
    prefix = "source_row:"
    if text.startswith(prefix):
        return int(text[len(prefix) :])
    return int(text)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_basis(path: Path, *, n_units: int, basis_key: str, max_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        if basis_key == "auto":
            preferred = ["basis", "U", "basis_delta_0p25", "basis_uncentered"]
            chosen = next((key for key in preferred if key in data.files), None)
            if chosen is None:
                chosen = next((key for key in data.files if key.startswith("basis")), None)
            if chosen is None:
                raise ValueError(f"No basis-like key found in {path}; available keys={list(data.files)}")
        else:
            chosen = str(basis_key)
            if chosen not in data.files:
                raise ValueError(f"basis key {chosen!r} not found in {path}; available keys={list(data.files)}")
        basis = np.asarray(data[chosen], dtype=np.float64)
        meta = {
            "basis_source": str(path),
            "basis_key": chosen,
            "basis_file_keys": list(data.files),
        }
        for key in ["image_disjoint", "basis_mode", "basis_provenance", "delta_arcmin"]:
            if key in data.files:
                meta[key] = np.asarray(data[key]).reshape(-1).tolist()
    if basis.ndim != 2 or basis.shape[0] != int(n_units):
        raise ValueError(f"basis must be ({n_units}, k), got {basis.shape}")
    if not np.isfinite(basis).all():
        raise ValueError("basis contains non-finite values")
    gram = basis.T @ basis
    if float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro")) > 1e-5:
        basis, _r = np.linalg.qr(basis)
    keep = basis.shape[1] if int(max_dim) <= 0 else min(int(max_dim), basis.shape[1])
    basis = basis[:, :keep]
    meta["basis_dim"] = int(basis.shape[1])
    return basis, meta


def _response_population(
    *,
    mode: str,
    n_units: int,
    rr100_version: str,
) -> tuple[PopulationView, dict[str, Any]]:
    population_mode = str(mode)
    if population_mode == "full756":
        view = full_population_view(int(n_units), name=f"full_{int(n_units)}")
    elif population_mode == "rr100":
        view = load_population_view(version_name=str(rr100_version))
        if int(view.input_channels) != int(n_units):
            raise ValueError(
                "RR population view input channels do not match response cache: "
                f"{view.input_channels} vs {n_units}"
            )
    else:
        valid = ", ".join(RESPONSE_POPULATION_MODES)
        raise ValueError(f"Unknown response population mode {mode!r}; valid modes: {valid}")

    membership = view.membership
    meta = dict(view.meta or {})
    meta.update(
        {
            "response_population_mode": population_mode,
            "population_name": str(view.name),
            "population_input_channels": int(view.input_channels),
            "population_n_units": int(view.n_units),
            "population_pooling_mode": str(meta.get("pooling_mode", "identity" if membership is None else "unknown")),
            "population_membership_shape": None if membership is None else [int(v) for v in membership.shape],
        }
    )
    if membership is not None:
        meta["population_membership_max_nnz"] = int((np.asarray(membership) != 0).sum(axis=1).max())
    return view, meta


def _response_basis(
    *,
    mode: str,
    path: Path,
    n_units: int,
    basis_key: str,
    max_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    basis_mode = str(mode)
    if basis_mode == "compact":
        basis, meta = _load_basis(path, n_units=int(n_units), basis_key=str(basis_key), max_dim=int(max_dim))
        meta["response_basis_mode"] = "compact"
        meta["response_coordinate_contract"] = "response_counts_projected_onto_declared_compact_basis"
        return basis, meta
    if basis_mode == "full_units":
        basis = np.eye(int(n_units), dtype=np.float64)
        return (
            basis,
            {
                "response_basis_mode": "full_units",
                "response_coordinate_contract": "full_response_movie_flattened_over_time_and_units",
                "basis_source": None,
                "basis_key": "identity",
                "basis_dim": int(n_units),
                "basis_max_dim": "ignored_for_full_units",
            },
        )
    valid = ", ".join(RESPONSE_BASIS_MODES)
    raise ValueError(f"Unknown response basis mode {mode!r}; valid modes: {valid}")


def _apply_response_population(response_counts: np.ndarray, population: PopulationView) -> np.ndarray:
    response = np.asarray(response_counts, dtype=np.float64)
    if response.ndim < 2:
        raise ValueError(f"response must have at least time and unit axes, got {response.shape}")
    if response.shape[-1] != int(population.input_channels):
        raise ValueError(
            f"response unit count {response.shape[-1]} does not match population input "
            f"{population.input_channels}"
        )
    if population.membership is None:
        return response
    membership = np.asarray(population.membership, dtype=np.float64)
    if membership.ndim != 2 or membership.shape[1] != response.shape[-1]:
        raise ValueError(f"population membership shape {membership.shape} is incompatible with {response.shape}")
    return np.einsum("...tc,rc->...tr", response, membership, optimize=True)


def _project_response(
    response_counts: np.ndarray,
    basis: np.ndarray,
    *,
    population: PopulationView | None = None,
) -> np.ndarray:
    response = np.asarray(response_counts, dtype=np.float64)
    if population is not None:
        response = _apply_response_population(response, population)
    if response.ndim != 2:
        raise ValueError(f"response must be (time, unit), got {response.shape}")
    if response.shape[1] != basis.shape[0]:
        raise ValueError(f"response unit count {response.shape[1]} does not match basis {basis.shape}")
    if not np.isfinite(response).all():
        raise ValueError("response contains non-finite values")
    return (response @ basis).reshape(-1).astype(np.float32)


def _parse_str_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _load_feature_table(feature_npz: Path, *, latent: str) -> tuple[FeatureTable, dict[str, Any]]:
    with np.load(feature_npz, allow_pickle=False) as data:
        if latent not in data.files:
            raise ValueError(f"{feature_npz} does not contain latent {latent!r}; keys={list(data.files)}")
        if "source_row" not in data.files:
            raise ValueError(f"{feature_npz} must contain source_row identities")
        source_rows = np.asarray(data["source_row"], dtype=int)
        features = np.asarray(data[latent], dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"feature array must be 2D, got {features.shape}")
    if source_rows.shape[0] != features.shape[0]:
        raise ValueError("source_row and feature array length mismatch")
    if not np.isfinite(features).all():
        raise ValueError("feature array contains non-finite values")
    source_to_index: dict[int, int] = {}
    duplicates: list[int] = []
    for index, source in enumerate(source_rows.tolist()):
        source_int = int(source)
        if source_int in source_to_index:
            duplicates.append(source_int)
        source_to_index[source_int] = int(index)
    if duplicates:
        preview = ", ".join(str(value) for value in sorted(set(duplicates))[:8])
        raise ValueError(f"feature table has duplicate source_row identities: {preview}")
    table = FeatureTable(
        feature_npz=Path(feature_npz),
        latent=str(latent),
        source_rows=source_rows.astype(int, copy=False),
        features=features,
        source_to_index=source_to_index,
    )
    meta = {
        "feature_npz": str(feature_npz),
        "latent": str(latent),
        "source_rows": source_rows.tolist(),
        "raw_feature_dim": int(features.shape[1]),
        "n_feature_rows": int(features.shape[0]),
    }
    return table, meta


def _load_feature_weights(
    path: Path | None,
    *,
    latent: str,
    raw_feature_dim: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if path is None:
        return None, {"feature_weights_npz": None, "feature_weights_status": "not_requested"}
    with np.load(path, allow_pickle=False) as data:
        if latent not in data.files:
            return (
                None,
                {
                    "feature_weights_npz": str(path),
                    "feature_weights_status": "latent_missing",
                    "available_weight_keys": list(data.files),
                },
            )
        weights = np.asarray(data[latent], dtype=np.float64).reshape(-1)
    if weights.shape[0] != int(raw_feature_dim):
        return (
            None,
            {
                "feature_weights_npz": str(path),
                "feature_weights_status": "dimension_mismatch",
                "weight_dim": int(weights.shape[0]),
                "raw_feature_dim": int(raw_feature_dim),
            },
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError(f"feature weights for {latent!r} must be positive and finite")
    return (
        weights,
        {
            "feature_weights_npz": str(path),
            "feature_weights_status": "loaded",
            "weight_dim": int(weights.shape[0]),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
        },
    )


def _feature_matrix_for_sources(table: FeatureTable, source_rows: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    missing: list[int] = []
    for source in np.asarray(source_rows, dtype=int).reshape(-1).tolist():
        source_int = int(source)
        if source_int not in table.source_to_index:
            missing.append(source_int)
        else:
            indices.append(table.source_to_index[source_int])
    if missing:
        preview = ", ".join(str(value) for value in sorted(set(missing))[:8])
        raise KeyError(f"Missing feature embeddings for source rows {preview}")
    return table.features[np.asarray(indices, dtype=int)]


def _feature_space_config(mode: str) -> dict[str, Any]:
    configs = {
        "global_centered_whitened_pca": {
            "fit_scope": "global",
            "preprocessing": "centered",
            "whitened": True,
            "weighted": False,
        },
        "fold_centered_whitened_pca": {
            "fit_scope": "fold",
            "preprocessing": "centered",
            "whitened": True,
            "weighted": False,
        },
        "fold_zscore_whitened_pca": {
            "fit_scope": "fold",
            "preprocessing": "zscore",
            "whitened": True,
            "weighted": False,
        },
        "fold_zscore_pca": {
            "fit_scope": "fold",
            "preprocessing": "zscore",
            "whitened": False,
            "weighted": False,
        },
        "fold_weighted_zscore_whitened_pca": {
            "fit_scope": "fold",
            "preprocessing": "zscore",
            "whitened": True,
            "weighted": True,
        },
        "fold_weighted_zscore_pca": {
            "fit_scope": "fold",
            "preprocessing": "zscore",
            "whitened": False,
            "weighted": True,
        },
    }
    aliases = {
        "global_whitened_pca": "global_centered_whitened_pca",
        "fold_whitened_pca": "fold_centered_whitened_pca",
        "fold_4b_whitened_pca": "fold_zscore_whitened_pca",
        "fold_4b_pca": "fold_zscore_pca",
    }
    canonical = aliases.get(str(mode), str(mode))
    if canonical not in configs:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown feature-space mode {mode!r}; valid modes: {valid}")
    out = dict(configs[canonical])
    out["canonical_mode"] = canonical
    return out


def _fit_feature_transform(
    table: FeatureTable,
    *,
    fit_sources: np.ndarray,
    feature_dim: int,
    feature_space_mode: str,
    feature_weights: np.ndarray | None,
) -> FeatureTransform:
    config = _feature_space_config(feature_space_mode)
    fit_sources_unique = np.asarray(sorted({int(value) for value in np.asarray(fit_sources, dtype=int)}), dtype=int)
    fit_features = _feature_matrix_for_sources(table, fit_sources_unique)
    if fit_features.shape[0] < 2:
        raise ValueError(f"Need at least two feature rows to fit PCA for {feature_space_mode}")
    if bool(config["weighted"]) and feature_weights is None:
        raise ValueError(
            f"feature-space mode {feature_space_mode!r} requires --feature-weights-npz with key {table.latent!r}"
        )
    mean = np.mean(fit_features, axis=0)
    if config["preprocessing"] == "zscore":
        sd = np.std(fit_features, axis=0)
        sd[~np.isfinite(sd) | (sd <= 1e-8)] = 1.0
    else:
        sd = np.ones(fit_features.shape[1], dtype=np.float64)
    transformed = (fit_features - mean[None, :]) / sd[None, :]
    weights = None
    if bool(config["weighted"]):
        weights = np.asarray(feature_weights, dtype=np.float64)
        transformed = transformed * weights[None, :]
    _u, singular_values, vt = np.linalg.svd(transformed, full_matrices=False)
    dim = min(max(1, int(feature_dim)), transformed.shape[0] - 1, transformed.shape[1], vt.shape[0])
    components = vt[:dim]
    if bool(config["whitened"]):
        denom = np.maximum(singular_values[:dim] / np.sqrt(max(transformed.shape[0] - 1, 1)), 1e-8)
    else:
        denom = np.ones(dim, dtype=np.float64)
    explained = (singular_values[:dim] ** 2) / np.maximum(np.sum(singular_values**2), 1e-12)
    return FeatureTransform(
        latent=table.latent,
        feature_space_mode=str(config["canonical_mode"]),
        feature_dim=int(dim),
        mean=mean,
        sd=sd,
        components=components,
        denom=denom,
        weights=weights,
        fit_scope=str(config["fit_scope"]),
        preprocessing=str(config["preprocessing"]),
        whitened=bool(config["whitened"]),
        weighted=bool(config["weighted"]),
        n_fit_sources=int(fit_features.shape[0]),
        raw_feature_dim=int(fit_features.shape[1]),
        explained_variance_sum=float(np.sum(explained)),
        explained_variance_first5=explained[:5].tolist(),
    )


def _transform_feature_sources(
    transform: FeatureTransform,
    table: FeatureTable,
    source_rows: np.ndarray,
) -> np.ndarray:
    features = _feature_matrix_for_sources(table, np.asarray(source_rows, dtype=int))
    transformed = (features - transform.mean[None, :]) / transform.sd[None, :]
    if transform.weights is not None:
        transformed = transformed * transform.weights[None, :]
    scores = transformed @ transform.components.T
    scores = scores / transform.denom[None, :]
    return scores.astype(np.float64, copy=False)


def _read_manifest(path: Path, *, scales: set[float], max_tables: int) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[rows["response_cache_path"].astype(str).str.len() > 0].copy()
    if scales:
        rows = rows[rows["scale"].astype(float).isin(scales)].copy()
    rows = rows.reset_index(drop=True)
    if int(max_tables) > 0:
        rows = rows.iloc[: int(max_tables)].copy()
    if rows.empty:
        raise ValueError(f"No response tables selected from {path}")
    rows["table_index"] = np.arange(rows.shape[0], dtype=int)
    return rows


def _new_sample_parts() -> dict[str, dict[str, list[Any]]]:
    return {
        name: {"x": [], "source_rows": [], "table_indices": []}
        for name in ["known", "zero", "prior_mean", "prior_all"]
    }


def _append_sample(
    parts: dict[str, list[Any]],
    *,
    x: np.ndarray,
    source_row: int,
    table_index: int,
) -> None:
    parts["x"].append(x)
    parts["source_rows"].append(int(source_row))
    parts["table_indices"].append(int(table_index))


def _bank_from_parts(parts: dict[str, list[Any]]) -> SampleBank:
    if not parts["x"]:
        raise ValueError("empty sample bank")
    return SampleBank(
        x=np.stack(parts["x"], axis=0).astype(np.float32),
        source_rows=np.asarray(parts["source_rows"], dtype=int),
        table_indices=np.asarray(parts["table_indices"], dtype=int),
    )


def _build_sample_banks(
    *,
    run_dir: Path,
    manifest: pd.DataFrame,
    population: PopulationView,
    population_meta: dict[str, Any],
    basis: np.ndarray,
    feature_sources: set[int],
    progress_every: int,
) -> tuple[dict[str, SampleBank], TestSet]:
    parts = _new_sample_parts()
    test_rows: list[dict[str, Any]] = []
    observed_x: list[np.ndarray] = []
    static_matched_mean_x: list[np.ndarray] = []

    for local_index, man_row in manifest.iterrows():
        table_index = int(man_row["table_index"])
        if progress_every > 0 and (local_index + 1) % int(progress_every) == 0:
            print(f"loaded {local_index + 1} / {manifest.shape[0]} response tables")
        table_path = run_dir / str(man_row["response_cache_path"])
        table = _load_npz(table_path)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        static_reference_mode = (
            str(np.asarray(table["zero_reference_mode"]).reshape(-1)[0])
            if "zero_reference_mode" in table
            else "patch_center_static_tau_zero"
        )
        candidate_ids = [str(value) for value in np.asarray(table["candidate_ids"]).tolist()]
        source_rows = [_source_row_from_candidate_id(candidate_id) for candidate_id in candidate_ids]
        missing = [source for source in source_rows if source not in feature_sources]
        if missing:
            raise KeyError(f"Missing feature embeddings for source rows {missing} in {table_path}")
        true_idx = int(np.asarray(table["true_candidate_index"]).reshape(-1)[0])
        if true_idx < 0 or true_idx >= len(source_rows):
            raise ValueError(f"true_candidate_index {true_idx} outside candidate list for {table_path}")
        true_source = int(source_rows[true_idx])

        for candidate_index, source_row in enumerate(source_rows):
            _append_sample(
                parts["known"],
                x=_project_response(known[candidate_index], basis, population=population),
                source_row=source_row,
                table_index=table_index,
            )
            _append_sample(
                parts["zero"],
                x=_project_response(zero[candidate_index], basis, population=population),
                source_row=source_row,
                table_index=table_index,
            )
            _append_sample(
                parts["prior_mean"],
                x=_project_response(np.mean(prior[candidate_index], axis=0), basis, population=population),
                source_row=source_row,
                table_index=table_index,
            )
            for trajectory_index in range(prior.shape[1]):
                _append_sample(
                    parts["prior_all"],
                    x=_project_response(prior[candidate_index, trajectory_index], basis, population=population),
                    source_row=source_row,
                    table_index=table_index,
                )

        observed_x.append(_project_response(obs, basis, population=population))
        static_matched_mean_x.append(_project_response(zero[true_idx], basis, population=population))
        test_rows.append(
            {
                "table_index": table_index,
                "trial_id": int(man_row["trial_id"]),
                "response_cache_path": str(man_row["response_cache_path"]),
                "candidate_set_mode": str(man_row["candidate_set_mode"]),
                "observation_family": str(man_row["observation_family"]),
                "prior_family": str(man_row["prior_family"]),
                "observation_scale": float(man_row["scale"]),
                "axis_catalog_mode": str(man_row["axis_catalog_mode"]),
                "n_candidates": int(man_row["n_candidates"]),
                "n_prior_trajectories": int(man_row["n_prior_trajectories"]),
                "n_timebins": int(man_row["n_timebins"]),
                "n_units": int(man_row["n_units"]),
                "response_population_mode": str(population_meta.get("response_population_mode", "")),
                "response_population_name": str(population.name),
                "observer_n_units": int(population.n_units),
                "static_matched_mean_reference_mode": static_reference_mode,
                "static_matched_mean_source": "zero_lambda_counts[true_candidate_index]",
                "true_candidate_index": int(true_idx),
                "true_source_row": true_source,
                "true_candidate_id": candidate_ids[true_idx],
            }
        )

    banks = {name: _bank_from_parts(value) for name, value in parts.items()}
    tests = TestSet(
        rows=pd.DataFrame(test_rows),
        observed_x=np.stack(observed_x, axis=0).astype(np.float32),
        static_matched_mean_x=np.stack(static_matched_mean_x, axis=0).astype(np.float32),
    )
    return banks, tests


def _assign_source_folds(source_rows: np.ndarray, *, n_folds: int, seed: int) -> dict[int, int]:
    unique = np.asarray(sorted({int(value) for value in source_rows}), dtype=int)
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_count = max(2, min(int(n_folds), unique.size))
    return {int(source): int(index % fold_count) for index, source in enumerate(shuffled)}


def _fit_forward_posterior(
    *,
    z_train: np.ndarray,
    x_train: np.ndarray,
    ridge: float,
    noise_floor: float,
    sample_weight: np.ndarray | None = None,
) -> ForwardPosteriorModel:
    z = np.asarray(z_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2 or z.shape[0] != x.shape[0]:
        raise ValueError(f"Expected z/x train matrices with shared rows, got {z.shape} and {x.shape}")
    if z.shape[0] <= z.shape[1]:
        raise ValueError(f"Need more training samples than feature dimensions, got {z.shape}")
    if sample_weight is None:
        weights = np.ones(z.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (z.shape[0],):
            raise ValueError(f"sample_weight must have shape ({z.shape[0]},), got {weights.shape}")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample_weight must be finite and non-negative")
        if float(np.sum(weights)) <= 1e-12:
            raise ValueError("sample_weight sum must be positive")
    weights = weights / (float(np.mean(weights)) + 1e-12)
    weight_sum = float(np.sum(weights))
    response_mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    y = x - response_mean[None, :]
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    wz = z * weights[:, None]
    normal = z.T @ wz + ridge_value * np.eye(z.shape[1], dtype=np.float64)
    response_map = np.linalg.solve(normal, wz.T @ y)
    residual = y - z @ response_map
    noise_variance = max(float(np.sum((residual * residual) * weights[:, None]) / (weight_sum * residual.shape[1])), float(noise_floor))
    precision = np.eye(z.shape[1], dtype=np.float64) + (response_map @ response_map.T) / noise_variance
    posterior_gain = np.linalg.solve(precision, response_map) / noise_variance
    return ForwardPosteriorModel(
        response_mean=response_mean,
        response_map=response_map,
        posterior_gain=posterior_gain,
        noise_variance=float(noise_variance),
        ridge=ridge_value,
        n_train=int(z.shape[0]),
    )


def _source_balanced_weights(source_rows: np.ndarray) -> np.ndarray:
    sources = np.asarray(source_rows, dtype=int)
    if sources.ndim != 1:
        raise ValueError(f"source_rows must be 1D, got {sources.shape}")
    if sources.size == 0:
        return np.empty(0, dtype=np.float64)
    unique, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= float(sources.size) / float(max(1, unique.size))
    return weights.astype(np.float64, copy=False)


def _predict_z(model: ForwardPosteriorModel, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    centered = values - model.response_mean[None, :]
    return centered @ model.posterior_gain.T


def _stable_token_value(text: str) -> int:
    total = 0
    for index, char in enumerate(str(text)):
        total += (index + 1) * ord(char)
    return int(total)


def _resolve_torch_device(device: str) -> str:
    import torch

    text = str(device)
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if text.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return text


def _make_mlp(input_dim: int, output_dim: int, config: MLPConfig):
    import torch

    hidden = int(config.hidden_dim)
    layers: list[torch.nn.Module] = []
    prev = int(input_dim)
    for _ in range(max(1, int(config.layers))):
        layers.append(torch.nn.Linear(prev, hidden))
        layers.append(torch.nn.ReLU())
        if float(config.dropout) > 0.0:
            layers.append(torch.nn.Dropout(float(config.dropout)))
        prev = hidden
    layers.append(torch.nn.Linear(prev, int(output_dim)))
    return torch.nn.Sequential(*layers)


def _source_group_validation_mask(
    source_rows: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    sources = np.asarray(source_rows, dtype=int)
    unique = np.asarray(sorted(set(sources.tolist())), dtype=int)
    if unique.size < 3 or float(validation_fraction) <= 0.0:
        return np.zeros(sources.shape[0], dtype=bool)
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(float(validation_fraction) * shuffled.size)))
    n_val = min(n_val, shuffled.size - 1)
    val_sources = set(int(value) for value in shuffled[:n_val].tolist())
    return np.asarray([int(source) in val_sources for source in sources.tolist()], dtype=bool)


def _fit_predict_mlp(
    *,
    x_all: np.ndarray,
    z_all: np.ndarray,
    source_rows: np.ndarray,
    train_mask: np.ndarray,
    x_test: np.ndarray,
    config: MLPConfig,
    fold: int,
    spec_slug: str,
    feature_space_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    x = np.asarray(x_all, dtype=np.float32)
    z = np.asarray(z_all, dtype=np.float32)
    train_mask_arr = np.asarray(train_mask, dtype=bool)
    if x.ndim != 2 or z.ndim != 2 or x.shape[0] != z.shape[0]:
        raise ValueError(f"MLP expects aligned 2D x/z matrices, got {x.shape} and {z.shape}")
    if int(np.sum(train_mask_arr)) <= max(4, z.shape[1]):
        raise ValueError(f"Too few MLP training rows for {spec_slug} / {feature_space_mode}")

    train_indices_all = np.flatnonzero(train_mask_arr)
    val_rel_mask = _source_group_validation_mask(
        np.asarray(source_rows, dtype=int)[train_indices_all],
        validation_fraction=float(config.validation_fraction),
        seed=int(config.seed) + 31 * int(fold) + _stable_token_value(spec_slug) + _stable_token_value(feature_space_mode),
    )
    train_fit_indices = train_indices_all[~val_rel_mask]
    val_indices = train_indices_all[val_rel_mask]
    if train_fit_indices.size == 0:
        train_fit_indices = train_indices_all
        val_indices = train_indices_all
    if val_indices.size == 0:
        val_indices = train_fit_indices

    if int(config.max_train_samples) > 0 and train_fit_indices.size > int(config.max_train_samples):
        rng = np.random.default_rng(
            int(config.seed) + 101 * int(fold) + _stable_token_value(spec_slug) + _stable_token_value(feature_space_mode)
        )
        train_fit_indices = np.sort(rng.choice(train_fit_indices, size=int(config.max_train_samples), replace=False))

    x_fit = x[train_fit_indices]
    z_fit = z[train_fit_indices]
    x_val = x[val_indices]
    z_val = z[val_indices]

    x_mean = np.mean(x_fit, axis=0, keepdims=True)
    x_sd = np.std(x_fit, axis=0, keepdims=True)
    x_sd[~np.isfinite(x_sd) | (x_sd <= 1e-6)] = 1.0
    z_mean = np.mean(z_fit, axis=0, keepdims=True)
    z_sd = np.std(z_fit, axis=0, keepdims=True)
    z_sd[~np.isfinite(z_sd) | (z_sd <= 1e-6)] = 1.0

    x_fit_std = (x_fit - x_mean) / x_sd
    z_fit_std = (z_fit - z_mean) / z_sd
    x_val_std = (x_val - x_mean) / x_sd
    z_val_std = (z_val - z_mean) / z_sd
    x_test_std = (np.asarray(x_test, dtype=np.float32) - x_mean) / x_sd

    torch.manual_seed(int(config.seed) + int(fold) + _stable_token_value(spec_slug) + _stable_token_value(feature_space_mode))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed) + int(fold))
    device = torch.device(_resolve_torch_device(config.device))
    model = _make_mlp(x.shape[1], z.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()

    x_train_tensor = torch.as_tensor(x_fit_std, dtype=torch.float32, device=device)
    z_train_tensor = torch.as_tensor(z_fit_std, dtype=torch.float32, device=device)
    x_val_tensor = torch.as_tensor(x_val_std, dtype=torch.float32, device=device)
    z_val_tensor = torch.as_tensor(z_val_std, dtype=torch.float32, device=device)

    n_train = int(x_train_tensor.shape[0])
    batch_size = max(1, min(int(config.batch_size), n_train))
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    final_train = float("nan")
    final_val = float("nan")

    for epoch in range(max(1, int(config.epochs))):
        model.train()
        order = torch.randperm(n_train, device=device)
        batch_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train_tensor[batch_idx])
            loss = loss_fn(pred, z_train_tensor[batch_idx])
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        final_train = float(np.mean(batch_losses)) if batch_losses else float("nan")
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_tensor)
            final_val = float(loss_fn(val_pred, z_val_tensor).detach().cpu())
        if final_val < best_val - 1e-7:
            best_val = final_val
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if int(config.patience) > 0 and stale >= int(config.patience):
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    preds: list[np.ndarray] = []
    x_test_tensor = torch.as_tensor(x_test_std, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, int(x_test_tensor.shape[0]), max(1, int(config.batch_size))):
            pred = model(x_test_tensor[start : start + int(config.batch_size)])
            preds.append(pred.detach().cpu().numpy())
    z_hat_std = np.concatenate(preds, axis=0) if preds else np.empty((0, z.shape[1]), dtype=np.float32)
    z_hat = z_hat_std * z_sd + z_mean
    stats = {
        "mlp_hidden_dim": int(config.hidden_dim),
        "mlp_layers": int(config.layers),
        "mlp_dropout": float(config.dropout),
        "mlp_learning_rate": float(config.learning_rate),
        "mlp_weight_decay": float(config.weight_decay),
        "mlp_batch_size": int(config.batch_size),
        "mlp_epochs_requested": int(config.epochs),
        "mlp_epochs_trained": int(max(best_epoch + 1, epoch + 1 if "epoch" in locals() else 0)),
        "mlp_best_epoch": int(best_epoch),
        "mlp_best_val_loss": float(best_val),
        "mlp_final_train_loss": float(final_train),
        "mlp_final_val_loss": float(final_val),
        "mlp_n_fit_rows": int(train_fit_indices.size),
        "mlp_n_val_rows": int(val_indices.size),
        "mlp_device": str(device),
        "target_standardized_for_training": True,
        "response_standardized_for_training": True,
    }
    return z_hat.astype(np.float64, copy=False), stats


def _selected_specs(observer_modes: list[str]) -> list[ReconstructionSpec]:
    if not observer_modes:
        return list(SPECS)
    alias = {name: STATIC_MATCHED_MEAN_MODE for name in STATIC_MATCHED_MEAN_ALIASES}
    requested = {alias.get(str(value), str(value)) for value in observer_modes}
    specs = [spec for spec in SPECS if spec.slug in requested]
    missing = sorted(requested.difference({spec.slug for spec in specs}))
    if missing:
        valid = ", ".join([spec.slug for spec in SPECS] + sorted(STATIC_MATCHED_MEAN_ALIASES))
        raise ValueError(f"Unknown observer mode(s) {missing}; valid modes: {valid}")
    return specs


def _metrics(z_hat: np.ndarray, z_true: np.ndarray, *, train_mean: np.ndarray | None = None) -> dict[str, float]:
    pred = np.asarray(z_hat, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    diff = pred - true
    mse = float(np.mean(diff * diff))
    pred_norm = float(np.linalg.norm(pred))
    true_norm = float(np.linalg.norm(true))
    denom = pred_norm * true_norm
    cosine = float(np.dot(pred, true) / denom) if denom > 1e-12 else float("nan")
    out = {
        "feature_mse": mse,
        "feature_neg_mse": -mse,
        "feature_rmse": float(np.sqrt(mse)),
        "feature_l2_error": float(np.linalg.norm(diff)),
        "feature_cosine": cosine,
        "feature_true_norm": true_norm,
        "feature_pred_norm": pred_norm,
    }
    if train_mean is not None:
        row_sse, row_sst, row_valid = per_sample_sse_sst(true[None, :], pred[None, :], train_mean=train_mean)
        out["feature_sse"] = float(row_sse[0])
        out["feature_sst_train_baseline"] = float(row_sst[0])
        out["feature_r2_row_diagnostic"] = (
            float(1.0 - row_sse[0] / row_sst[0]) if bool(row_valid[0]) and row_sst[0] > 1e-12 else float("nan")
        )
        out["feature_r2_baseline"] = "train_fold_feature_mean"
    return out


def _run_crossfit(
    *,
    banks: dict[str, SampleBank],
    tests: TestSet,
    feature_table: FeatureTable,
    feature_weights: np.ndarray | None,
    feature_space_modes: list[str],
    decoder_modes: list[str],
    specs: list[ReconstructionSpec],
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    ridge: float,
    noise_floor: float,
    source_weighting: str,
    mlp_config: MLPConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_by_source = _assign_source_folds(tests.rows["true_source_row"].to_numpy(dtype=int), n_folds=n_folds, seed=fold_seed)
    test_folds = tests.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    trial_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    canonical_modes = list(dict.fromkeys(_feature_space_config(mode)["canonical_mode"] for mode in feature_space_modes))
    canonical_decoders = list(dict.fromkeys(str(mode) for mode in decoder_modes))
    valid_decoders = {"linear_gaussian", "mlp"}
    unknown_decoders = sorted(set(canonical_decoders).difference(valid_decoders))
    if unknown_decoders:
        raise ValueError(f"Unknown decoder mode(s) {unknown_decoders}; valid modes: {sorted(valid_decoders)}")
    global_transforms: dict[str, FeatureTransform] = {}

    for mode in canonical_modes:
        if _feature_space_config(mode)["fit_scope"] == "global":
            global_transforms[mode] = _fit_feature_transform(
                feature_table,
                fit_sources=feature_table.source_rows,
                feature_dim=int(feature_dim),
                feature_space_mode=mode,
                feature_weights=feature_weights,
            )

    for mode in canonical_modes:
        for fold in sorted(set(test_folds.tolist())):
            test_mask = test_folds == int(fold)
            if int(np.sum(test_mask)) == 0:
                continue
            if mode in global_transforms:
                transform = global_transforms[mode]
            else:
                heldout_sources = {
                    int(source)
                    for source, source_fold in fold_by_source.items()
                    if int(source_fold) == int(fold)
                }
                fold_train_sources = np.asarray(
                    [int(source) for source in feature_table.source_rows.tolist() if int(source) not in heldout_sources],
                    dtype=int,
                )
                transform = _fit_feature_transform(
                    feature_table,
                    fit_sources=fold_train_sources,
                    feature_dim=int(feature_dim),
                    feature_space_mode=mode,
                    feature_weights=feature_weights,
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
            test_sources = tests.rows.loc[test_mask, "true_source_row"].to_numpy(dtype=int)
            z_true = _transform_feature_sources(transform, feature_table, test_sources)

            for spec in specs:
                bank = banks[spec.train_bank]
                train_mask = np.asarray([fold_by_source.get(int(source), -1) != int(fold) for source in bank.source_rows])
                if int(np.sum(train_mask)) <= transform.feature_dim:
                    raise ValueError(f"Fold {fold} has too few training samples for {spec.slug} / {mode}")
                bank_z = _transform_feature_sources(transform, feature_table, bank.source_rows)
                train_source_rows = bank.source_rows[train_mask]
                if str(source_weighting) == "source_balanced":
                    train_weights = _source_balanced_weights(train_source_rows)
                elif str(source_weighting) == "row_unweighted":
                    train_weights = np.ones(int(np.sum(train_mask)), dtype=np.float64)
                else:
                    valid = ", ".join(SOURCE_WEIGHTING_MODES)
                    raise ValueError(f"Unknown source_weighting={source_weighting!r}; valid modes: {valid}")
                x_test = (
                    tests.static_matched_mean_x[test_mask]
                    if spec.test_input == STATIC_MATCHED_MEAN_MODE
                    else tests.observed_x[test_mask]
                )
                for decoder_mode in canonical_decoders:
                    decoder_stats: dict[str, Any]
                    if decoder_mode == "linear_gaussian":
                        model = _fit_forward_posterior(
                            z_train=bank_z[train_mask],
                            x_train=bank.x[train_mask],
                            ridge=float(ridge),
                            noise_floor=float(noise_floor),
                            sample_weight=train_weights,
                        )
                        z_hat = _predict_z(model, x_test)
                        decoder_stats = {
                            "n_train_samples": int(model.n_train),
                            "n_train_sources": int(len(set(train_source_rows.tolist()))),
                            "source_weighting": str(source_weighting),
                            "train_weight_min": float(np.min(train_weights)) if train_weights.size else float("nan"),
                            "train_weight_max": float(np.max(train_weights)) if train_weights.size else float("nan"),
                            "ridge": float(model.ridge),
                            "noise_variance": float(model.noise_variance),
                            "response_map_fro_norm": float(np.linalg.norm(model.response_map)),
                            "posterior_gain_fro_norm": float(np.linalg.norm(model.posterior_gain)),
                        }
                    elif decoder_mode == "mlp":
                        z_hat, mlp_stats = _fit_predict_mlp(
                            x_all=bank.x,
                            z_all=bank_z,
                            source_rows=bank.source_rows,
                            train_mask=train_mask,
                            x_test=x_test,
                            config=mlp_config,
                            fold=int(fold),
                            spec_slug=spec.slug,
                            feature_space_mode=transform.feature_space_mode,
                        )
                        decoder_stats = {
                            "n_train_samples": int(np.sum(train_mask)),
                            "n_train_sources": int(len(set(train_source_rows.tolist()))),
                            "source_weighting": "row_unweighted_mlp",
                            "train_weight_min": float("nan"),
                            "train_weight_max": float("nan"),
                            "ridge": float("nan"),
                            "noise_variance": float("nan"),
                            "response_map_fro_norm": float("nan"),
                            "posterior_gain_fro_norm": float("nan"),
                            **mlp_stats,
                        }
                    else:  # pragma: no cover - guarded above
                        raise AssertionError(decoder_mode)
                    model_row = {
                        "decoder_mode": decoder_mode,
                        "observer_mode": spec.slug,
                        "observer_label": spec.label,
                        "train_bank": spec.train_bank,
                        "test_input": spec.test_input,
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "fold": int(fold),
                        "n_fit_sources": int(transform.n_fit_sources),
                        "n_test_rows": int(np.sum(test_mask)),
                        "feature_dim": int(transform.feature_dim),
                        "raw_feature_dim": int(transform.raw_feature_dim),
                        "response_dim": int(bank.x.shape[1]),
                        "feature_fit_scope": transform.fit_scope,
                        "feature_preprocessing": transform.preprocessing,
                        "feature_whitened": bool(transform.whitened),
                        "feature_weighted": bool(transform.weighted),
                        "feature_variance_fraction": float(transform.explained_variance_sum),
                        "r2_cv_train_baseline": "source_fold_train_feature_mean",
                        "interpretation": spec.interpretation,
                    }
                    model_row.update(decoder_stats)
                    model_rows.append(model_row)
                    test_meta = tests.rows.loc[test_mask].reset_index(drop=True)
                    for row_index, meta in enumerate(test_meta.to_dict(orient="records")):
                        row = dict(meta)
                        row.update(
                            {
                                "decoder_mode": decoder_mode,
                                "observer_mode": spec.slug,
                                "observer_label": spec.label,
                                "train_bank": spec.train_bank,
                                "test_input": spec.test_input,
                                "latent": transform.latent,
                                "feature_space_mode": transform.feature_space_mode,
                                "feature_fit_scope": transform.fit_scope,
                                "feature_preprocessing": transform.preprocessing,
                                "feature_whitened": bool(transform.whitened),
                                "feature_weighted": bool(transform.weighted),
                                "feature_variance_fraction": float(transform.explained_variance_sum),
                                "r2_cv_train_baseline": "source_fold_train_feature_mean",
                                "fold": int(fold),
                                "n_train_samples": int(decoder_stats["n_train_samples"]),
                                "n_train_sources": int(decoder_stats["n_train_sources"]),
                                "source_weighting": str(decoder_stats["source_weighting"]),
                                "n_fit_sources": int(transform.n_fit_sources),
                            }
                        )
                        row.update(_metrics(z_hat[row_index], z_true[row_index], train_mean=z_train_mean))
                        trial_rows.append(row)
    return pd.DataFrame(trial_rows), pd.DataFrame(model_rows)


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "observer_mode",
        "observer_label",
        "observation_scale",
        "prior_family",
    ]
    summary = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_mse=("feature_mse", "median"),
            feature_sse=("feature_sse", "sum"),
            feature_sst_train_baseline=("feature_sst_train_baseline", "sum"),
            mean_feature_rmse=("feature_rmse", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            median_feature_true_norm=("feature_true_norm", "median"),
        )
        .sort_values(["observation_scale", "prior_family", "observer_mode"])
    )
    overall = (
        trials.groupby(["decoder_mode", "latent", "feature_space_mode", "observer_mode", "observer_label"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_mse=("feature_mse", "median"),
            feature_sse=("feature_sse", "sum"),
            feature_sst_train_baseline=("feature_sst_train_baseline", "sum"),
            mean_feature_rmse=("feature_rmse", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            median_feature_true_norm=("feature_true_norm", "median"),
        )
        .sort_values("observer_mode")
    )
    summary["R2_cv"] = 1.0 - summary["feature_sse"] / summary["feature_sst_train_baseline"]
    summary.loc[summary["feature_sst_train_baseline"] <= 1e-12, "R2_cv"] = np.nan
    overall["R2_cv"] = 1.0 - overall["feature_sse"] / overall["feature_sst_train_baseline"]
    overall.loc[overall["feature_sst_train_baseline"] <= 1e-12, "R2_cv"] = np.nan
    overall["observation_scale"] = "all"
    overall["prior_family"] = "all"
    return pd.concat([summary, overall[summary.columns]], ignore_index=True)


def _bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    clusters: np.ndarray | None = None,
) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    vals = vals[finite]
    if clusters is not None:
        cluster_values = np.asarray(clusters, dtype=object)[finite]
    else:
        cluster_values = None
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if cluster_values is not None:
        grouped = (
            pd.DataFrame({"cluster": cluster_values, "value": vals})
            .groupby("cluster", sort=False)["value"]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        vals_for_bootstrap = grouped[np.isfinite(grouped)]
    else:
        vals_for_bootstrap = vals
    if vals_for_bootstrap.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals_for_bootstrap.size == 1 or int(n_boot) <= 0:
        value = float(np.mean(vals_for_bootstrap))
        return value, value, value
    draws = rng.choice(vals_for_bootstrap, size=(int(n_boot), vals_for_bootstrap.size), replace=True).mean(axis=1)
    return float(np.mean(vals_for_bootstrap)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _contrasts(trials: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    key_cols = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "table_index",
        "trial_id",
        "observation_scale",
        "prior_family",
        "true_source_row",
    ]
    pivot = trials.pivot_table(index=key_cols, columns="observer_mode", values="feature_cosine", aggfunc="first")
    pairs = [
        ("known_eye", "hidden_eye_tau_marginal", "known_minus_hidden"),
        ("hidden_eye_tau_marginal", "zero_eye_on_motion", "hidden_minus_zero_eye_model"),
        ("known_eye", "zero_eye_on_motion", "known_minus_zero_eye_model"),
        ("known_eye", STATIC_MATCHED_MEAN_MODE, "known_motion_minus_static_matched_mean"),
        ("hidden_eye_tau_marginal", STATIC_MATCHED_MEAN_MODE, "hidden_motion_minus_static_matched_mean"),
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for lhs, rhs, contrast in pairs:
        if lhs not in pivot.columns or rhs not in pivot.columns:
            continue
        vals = (pivot[lhs] - pivot[rhs]).rename("delta").reset_index()
        vals = vals[np.isfinite(vals["delta"].to_numpy(dtype=float))]
        for scale_value, scale_rows in vals.groupby("observation_scale", sort=True):
            for (decoder_mode, latent, feature_space_mode), mode_rows in scale_rows.groupby(
                ["decoder_mode", "latent", "feature_space_mode"],
                sort=True,
            ):
                mode_values = mode_rows["delta"].to_numpy(dtype=float)
                mode_mean, mode_lo, mode_hi = _bootstrap_mean(
                    mode_values,
                    rng,
                    n_boot,
                    clusters=mode_rows["true_source_row"].to_numpy(dtype=int),
                )
                rows.append(
                    {
                        "decoder_mode": str(decoder_mode),
                        "latent": str(latent),
                        "feature_space_mode": str(feature_space_mode),
                        "contrast": contrast,
                        "lhs": lhs,
                        "rhs": rhs,
                        "observation_scale": float(scale_value),
                        "prior_family": "all",
                        "mean_feature_cosine_delta": mode_mean,
                        "ci_low": mode_lo,
                        "ci_high": mode_hi,
                        "bootstrap_unit": "true_source_row",
                        "fraction_positive": float(np.mean(mode_values > 0.0)) if mode_values.size else float("nan"),
                        "n": int(mode_values.size),
                        "n_bootstrap_clusters": int(mode_rows["true_source_row"].nunique()),
                    }
                )
        for (decoder_mode, latent, feature_space_mode), mode_rows in vals.groupby(
            ["decoder_mode", "latent", "feature_space_mode"],
            sort=True,
        ):
            values = mode_rows["delta"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(
                values,
                rng,
                n_boot,
                clusters=mode_rows["true_source_row"].to_numpy(dtype=int),
            )
            rows.append(
                {
                    "decoder_mode": str(decoder_mode),
                    "latent": str(latent),
                    "feature_space_mode": str(feature_space_mode),
                    "contrast": contrast,
                    "lhs": lhs,
                    "rhs": rhs,
                    "observation_scale": "all",
                    "prior_family": "all",
                    "mean_feature_cosine_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_unit": "true_source_row",
                    "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                    "n": int(values.size),
                    "n_bootstrap_clusters": int(mode_rows["true_source_row"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _scale_x(values: pd.Series) -> np.ndarray:
    return values.astype(float).map({0.5: 0.0, 1.0: 1.0, 2.0: 2.0}).to_numpy()


def _filter_plot_tables(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    decoder_mode: str,
    latent: str,
    feature_space_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    available = summary[["decoder_mode", "latent", "feature_space_mode"]].drop_duplicates()
    if available.empty:
        raise ValueError("No summary rows are available for plotting")
    selected_decoder = str(decoder_mode)
    if selected_decoder == "auto":
        selected_decoder = str(available.iloc[0]["decoder_mode"]) if not available.empty else selected_decoder
    elif selected_decoder not in set(available["decoder_mode"].astype(str)):
        choices = available[["decoder_mode", "latent", "feature_space_mode"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"Requested plot decoder {selected_decoder!r} is absent; available={choices}")
    selected_mode = str(feature_space_mode)
    if selected_mode == "auto":
        match = available[
            available["decoder_mode"].astype(str).eq(selected_decoder)
            & available["latent"].astype(str).eq(str(latent))
        ]
        if match.empty:
            match = available[available["decoder_mode"].astype(str).eq(selected_decoder)]
        if match.empty:
            match = available
        if match.empty:
            raise ValueError("No available decoder/feature-space mode for plotting")
        selected_mode = str(match.iloc[0]["feature_space_mode"])
    elif selected_mode not in set(available["feature_space_mode"].astype(str)):
        choices = available[["decoder_mode", "latent", "feature_space_mode"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"Requested plot feature-space mode {selected_mode!r} is absent; available={choices}")
    plot_summary = summary[
        summary["decoder_mode"].astype(str).eq(selected_decoder)
        & summary["latent"].astype(str).eq(str(latent))
        & summary["feature_space_mode"].astype(str).eq(selected_mode)
    ].copy()
    plot_contrasts = contrasts[
        contrasts["decoder_mode"].astype(str).eq(selected_decoder)
        & contrasts["latent"].astype(str).eq(str(latent))
        & contrasts["feature_space_mode"].astype(str).eq(selected_mode)
    ].copy()
    if plot_summary.empty:
        choices = available[["decoder_mode", "latent", "feature_space_mode"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(
            f"Requested plot combination decoder={selected_decoder!r}, latent={latent!r}, "
            f"feature_space_mode={selected_mode!r} is absent; available={choices}"
        )
    return plot_summary, plot_contrasts, selected_decoder, selected_mode


def _plot(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    *,
    decoder_mode: str,
    latent: str,
    feature_space_mode: str,
) -> tuple[Path, Path, str, str]:
    _configure_matplotlib()
    summary, contrasts, selected_decoder, selected_mode = _filter_plot_tables(
        summary,
        contrasts,
        decoder_mode=str(decoder_mode),
        latent=str(latent),
        feature_space_mode=str(feature_space_mode),
    )
    scale_summary = summary[(summary["prior_family"] == "all") & (summary["observation_scale"] != "all")].copy()
    if scale_summary.empty:
        scale_summary = (
            summary[summary["observation_scale"] != "all"]
            .groupby(["observer_mode", "observer_label", "observation_scale"], as_index=False)
            .agg(mean_feature_cosine=("mean_feature_cosine", "mean"), mean_feature_mse=("mean_feature_mse", "mean"), n=("n", "sum"))
        )
        scale_summary["prior_family"] = "all"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    ax = axes[0]
    for observer in OBSERVER_ORDER:
        block = scale_summary[scale_summary["observer_mode"] == observer].sort_values("observation_scale")
        if block.empty:
            continue
        ax.plot(
            _scale_x(block["observation_scale"]),
            block["mean_feature_cosine"],
            marker="o",
            lw=2.0 if observer in {"known_eye", "hidden_eye_tau_marginal", "zero_eye_on_motion"} else 1.5,
            linestyle=":" if observer in {STATIC_MATCHED_MEAN_MODE, "hidden_eye_prior_mean"} else "-",
            color=OBSERVER_COLORS[observer],
            label=OBSERVER_LABELS[observer],
        )
    ax.set_title("A. continuous z recovery")
    ax.set_ylabel("feature cosine")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(-0.05, 1.0)
    ax.legend(frameon=False, loc="upper left")
    _clean_axis(ax)

    ax = axes[1]
    key_contrasts = [
        ("known_minus_hidden", "known - hidden", "#111827"),
        ("hidden_minus_zero_eye_model", "hidden - zero model", "#235789"),
        ("known_motion_minus_static_matched_mean", "known - static mean", "#2f8f6a"),
        ("hidden_motion_minus_static_matched_mean", "hidden - static mean", "#4c78a8"),
    ]
    offsets = np.linspace(-0.21, 0.21, len(key_contrasts))
    for offset, (contrast, label, color) in zip(offsets, key_contrasts, strict=True):
        block = contrasts[
            (contrasts["contrast"] == contrast)
            & (contrasts["prior_family"] == "all")
            & (contrasts["observation_scale"].astype(str) != "all")
        ].copy()
        if block.empty:
            continue
        block = block.sort_values("observation_scale")
        x = _scale_x(block["observation_scale"]) + offset
        y = block["mean_feature_cosine_delta"].to_numpy(dtype=float)
        yerr = np.vstack([y - block["ci_low"].to_numpy(dtype=float), block["ci_high"].to_numpy(dtype=float) - y])
        ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.5, capsize=2.5, color=color, label=label)
    ax.axhline(0.0, color="#6b7280", lw=0.9)
    ax.set_title("B. paired contrasts")
    ax.set_ylabel("cosine difference")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.legend(frameon=False, loc="best")
    _clean_axis(ax)

    png = out_dir / "continuous_feature_embedding_reconstruction.png"
    pdf = out_dir / "continuous_feature_embedding_reconstruction.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, selected_decoder, selected_mode


def _write_readme(
    *,
    out_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    all_summary = summary.copy()
    summary, contrasts, selected_decoder, selected_mode = _filter_plot_tables(
        summary,
        contrasts,
        decoder_mode=str(manifest["plot_decoder_mode"]),
        latent=str(manifest["feature"]["latent"]),
        feature_space_mode=str(manifest["plot_feature_space_mode"]),
    )
    overall = summary[summary["observation_scale"].astype(str) == "all"].set_index("observer_mode")
    scale1 = summary[
        (summary["observation_scale"].astype(str) == "1.0") & (summary["prior_family"].astype(str) == "all")
    ].set_index("observer_mode")
    if scale1.empty:
        scale1 = (
            summary[summary["observation_scale"].astype(str) == "1.0"]
            .groupby("observer_mode")
            .agg(mean_feature_cosine=("mean_feature_cosine", "mean"))
        )
    def _value(frame: pd.DataFrame, observer: str) -> float:
        if observer not in frame.index:
            return float("nan")
        value = frame.loc[observer, "mean_feature_cosine"]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value)

    contrast_overall = contrasts[
        (contrasts["observation_scale"].astype(str) == "all") & (contrasts["prior_family"].astype(str) == "all")
    ].set_index("contrast")

    def _contrast(name: str) -> float:
        if name not in contrast_overall.index:
            return float("nan")
        return float(contrast_overall.loc[name, "mean_feature_cosine_delta"])

    option_rows = all_summary[
        (all_summary["observation_scale"].astype(str) == "all")
        & (all_summary["prior_family"].astype(str) == "all")
        & (
            all_summary["observer_mode"]
            .astype(str)
            .isin(["known_eye", "hidden_eye_tau_marginal", "zero_eye_on_motion", STATIC_MATCHED_MEAN_MODE])
        )
    ].copy()
    option_lines = ["decoder_mode,feature_space_mode,known,hidden,zero_eye_model,static_matched_mean"]
    if not option_rows.empty:
        wide = option_rows.pivot_table(
            index=["decoder_mode", "feature_space_mode"],
            columns="observer_mode",
            values="mean_feature_cosine",
            aggfunc="first",
        )
        for (decoder, mode), row in wide.sort_index().iterrows():
            option_lines.append(
                ",".join(
                    [
                        str(decoder),
                        str(mode),
                        f"{float(row.get('known_eye', np.nan)):.4f}",
                        f"{float(row.get('hidden_eye_tau_marginal', np.nan)):.4f}",
                        f"{float(row.get('zero_eye_on_motion', np.nan)):.4f}",
                        f"{float(row.get(STATIC_MATCHED_MEAN_MODE, np.nan)):.4f}",
                    ]
                )
            )

    population_meta = manifest.get("population", {})
    population_name = str(population_meta.get("population_name", "full response population"))
    population_n_units = population_meta.get("population_n_units", "unknown")
    population_mode = str(population_meta.get("response_population_mode", "full756"))
    population_line = f"response population = `{population_name}` ({population_n_units} units; mode `{population_mode}`)"
    basis_meta = manifest.get("basis", {})
    response_basis_mode = str(basis_meta.get("response_basis_mode", "compact"))
    if response_basis_mode == "full_units":
        response_feature_line = "response_features = flatten(selected response population movie over time and units)"
        response_target_lines = [
            "the selected response population movie, flattened over time and unit identity.",
            "This is the geometry-uncommitted primary 4C response representation.",
        ]
    else:
        response_feature_line = "response_features = compact_basis(response movie)"
        response_target_lines = [
            "the declared compact response basis.",
            "This is a compact intervention/control representation, not a prerequisite for the primary 4C claim.",
        ]

    if selected_decoder == "mlp":
        model_lines = [
            response_feature_line,
            "z_hat = MLP(response_features)",
        ]
        decoder_note = "Tejas-style ReLU MLP decoder trained as a nonlinear information upper-bound readout."
    else:
        model_lines = [
            f"{response_feature_line}",
            "response_features = A z + noise",
            "z ~ N(0, I)",
            "z_hat = E[z | response]",
        ]
        decoder_note = "Linear-Gaussian decoder with a feature-space prior."

    lines = [
        "# Continuous Feature-Embedding Reconstruction",
        "",
        "This diagnostic infers a continuous feature embedding instead of",
        "selecting or posterior-averaging over the candidate image list.",
        "",
        "Model:",
        "",
        "```text",
        population_line,
        *model_lines,
        "```",
        "",
        f"Decoder mode: `{selected_decoder}`",
        decoder_note,
        "",
        "The feature target is a compact PCA-space embedding of the existing local",
        f"`{manifest['feature']['latent']}` feature array. The plotted feature-space",
        f"option is `{selected_mode}`. The response representation is",
        *response_target_lines,
        "Cross-fitting is by source image: no response sample",
        "whose target source row is in the held-out fold is used to fit that fold.",
        "Linear-Gaussian fits use inverse-source-frequency weighting by default,",
        f"with source weighting mode `{manifest.get('source_weighting', 'unknown')}`.",
        "Contrast intervals are bootstrapped over `true_source_row` clusters.",
        "",
        "Coordinate contract: BackImage crops are extracted at each window's mean",
        "fixation position. The `static_matched_mean` observer uses the cached",
        "`zero_lambda_counts[true_candidate_index]`, i.e. zero residual displacement",
        "at that crop center. This is static at the movie/window mean position, not",
        "a global absolute-eye-position zero oracle. Legacy names such as",
        "`zero_static` are accepted as aliases but should not be used in manuscript",
        "wording.",
        "",
        "Axis note: this runner does not report edge-parallel minus",
        "edge-orthogonal contrasts. The current response-feature model is fit",
        "pooled across prior-family labels and evaluates the same observed",
        "response rows for both labels, so axis-prior contrasts would be a",
        "structural bookkeeping artifact. Use a dedicated per-axis observer for",
        "along/across claims.",
        "",
        "At the 1x scale:",
        "",
        "```text",
        f"known eye feature cosine:          {_value(scale1, 'known_eye'):.4f}",
        f"hidden eye feature cosine:         {_value(scale1, 'hidden_eye_tau_marginal'):.4f}",
        f"zero-eye model on motion:          {_value(scale1, 'zero_eye_on_motion'):.4f}",
        f"static matched mean feature cosine:{_value(scale1, STATIC_MATCHED_MEAN_MODE):.4f}",
        "```",
        "",
        "All-scale paired contrasts:",
        "",
        "```text",
        f"known - hidden:                    {_contrast('known_minus_hidden'):.4f}",
        f"hidden - zero-eye model:           {_contrast('hidden_minus_zero_eye_model'):.4f}",
        f"known motion - static mean:        {_contrast('known_motion_minus_static_matched_mean'):.4f}",
        f"hidden motion - static mean:       {_contrast('hidden_motion_minus_static_matched_mean'):.4f}",
        "```",
        "",
        "All-scale option means:",
        "",
        "```csv",
        *option_lines,
        "```",
        "",
        "Interpretation boundary: this is a continuous feature posterior, not a",
        "pixel MAP reconstruction and not a candidate posterior. The finite image",
        "set is still used to fit the empirical feature prior/encoder and to score",
        "held-out source rows.",
        "",
        "Outputs:",
        "",
        "- `continuous_feature_embedding_reconstruction_trials.csv`",
        "- `continuous_feature_embedding_reconstruction_summary.csv`",
        "- `continuous_feature_embedding_reconstruction_contrasts.csv`",
        "- `continuous_feature_embedding_reconstruction_models.csv`",
        "- `continuous_feature_embedding_reconstruction_manifest.json`",
        "- `continuous_feature_embedding_reconstruction.png`",
    ]
    (out_dir / "continuous_feature_embedding_reconstruction_README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _parse_scales(text: str) -> set[float]:
    return {float(part.strip()) for part in str(text).split(",") if part.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--response-manifest", type=Path, default=None)
    parser.add_argument("--feature-npz", type=Path, default=FEATURE_NPZ)
    parser.add_argument("--feature-weights-npz", type=Path, default=None)
    parser.add_argument("--latent", default=PRIMARY_LATENT)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--feature-space-modes", default=",".join(DEFAULT_FEATURE_SPACE_MODES))
    parser.add_argument(
        "--allow-global-feature-transforms",
        action="store_true",
        help=(
            "Allow global feature-space transforms fit on all feature rows. This is a diagnostic "
            "coordinate system, not a fully source-disjoint promoted score."
        ),
    )
    parser.add_argument("--decoder-modes", default=",".join(DEFAULT_DECODER_MODES))
    parser.add_argument("--observer-modes", default="")
    parser.add_argument("--plot-decoder-mode", default="linear_gaussian")
    parser.add_argument("--plot-feature-space-mode", default="fold_zscore_whitened_pca")
    parser.add_argument(
        "--response-population-mode",
        choices=RESPONSE_POPULATION_MODES,
        default="full756",
        help=(
            "Twin population before response-basis projection. 'full756' keeps the canonical "
            "population; 'rr100' applies the saved RR100 movie-medoid population view."
        ),
    )
    parser.add_argument(
        "--rr100-version",
        default=RR100_MOVIE_MEDOID_VERSION,
        help="Saved redundancy-resolved population version used when --response-population-mode rr100.",
    )
    parser.add_argument(
        "--response-basis-mode",
        choices=RESPONSE_BASIS_MODES,
        default="compact",
        help=(
            "Response representation for the observer. 'full_units' is the geometry-uncommitted "
            "primary path; 'compact' is a compact-basis intervention/control path."
        ),
    )
    parser.add_argument("--compact-basis-path", type=Path, default=COMPACT_BASIS)
    parser.add_argument("--basis-key", default="basis")
    parser.add_argument("--basis-max-dim", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--noise-floor", type=float, default=1e-8)
    parser.add_argument(
        "--source-weighting",
        choices=SOURCE_WEIGHTING_MODES,
        default="source_balanced",
        help=(
            "Training-row weighting for linear-Gaussian fits. source_balanced gives each "
            "source equal total weight inside each fold/spec; row_unweighted reproduces "
            "the older candidate/trajectory-frequency-weighted fit."
        ),
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=4)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-batch-size", type=int, default=512)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-patience", type=int, default=40)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.2)
    parser.add_argument("--mlp-max-train-samples", type=int, default=0)
    parser.add_argument("--mlp-device", default="auto")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260624)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--scales", default="0.5,1.0,2.0")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.response_manifest) if args.response_manifest else run_dir / "response_cache_manifest.csv"
    manifest = _read_manifest(manifest_path, scales=_parse_scales(args.scales), max_tables=int(args.max_tables))

    first_table = _load_npz(run_dir / str(manifest.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first_table["y_obs_counts"]).shape[1])
    population, population_meta = _response_population(
        mode=str(args.response_population_mode),
        n_units=n_units,
        rr100_version=str(args.rr100_version),
    )
    if str(args.response_basis_mode) == "compact" and str(args.response_population_mode) != "full756":
        raise ValueError(
            "The declared compact basis is defined in the canonical full-756 unit space. "
            "Use --response-population-mode full756 with --response-basis-mode compact, "
            "or use --response-basis-mode full_units for RR100."
        )
    basis, basis_meta = _response_basis(
        mode=str(args.response_basis_mode),
        path=Path(args.compact_basis_path),
        n_units=int(population.n_units),
        basis_key=str(args.basis_key),
        max_dim=int(args.basis_max_dim),
    )
    feature_table, feature_meta = _load_feature_table(
        Path(args.feature_npz),
        latent=str(args.latent),
    )
    feature_weights, feature_weight_meta = _load_feature_weights(
        Path(args.feature_weights_npz) if args.feature_weights_npz is not None else None,
        latent=str(args.latent),
        raw_feature_dim=int(feature_table.features.shape[1]),
    )
    feature_space_modes = _parse_str_list(args.feature_space_modes)
    if not feature_space_modes:
        raise ValueError("--feature-space-modes must list at least one mode")
    global_modes = [
        mode
        for mode in feature_space_modes
        if _feature_space_config(mode)["fit_scope"] == "global"
    ]
    if global_modes and not bool(args.allow_global_feature_transforms):
        raise ValueError(
            "Global feature transforms include held-out feature rows and are diagnostic only. "
            f"Requested global mode(s): {global_modes}. Pass --allow-global-feature-transforms "
            "to run them explicitly."
        )
    decoder_modes = _parse_str_list(args.decoder_modes)
    if not decoder_modes:
        raise ValueError("--decoder-modes must list at least one mode")
    specs = _selected_specs(_parse_str_list(args.observer_modes))
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
    banks, tests = _build_sample_banks(
        run_dir=run_dir,
        manifest=manifest,
        population=population,
        population_meta=population_meta,
        basis=basis,
        feature_sources=set(int(value) for value in feature_table.source_rows.tolist()),
        progress_every=int(args.progress_every),
    )
    trials, models = _run_crossfit(
        banks=banks,
        tests=tests,
        feature_table=feature_table,
        feature_weights=feature_weights,
        feature_space_modes=feature_space_modes,
        decoder_modes=decoder_modes,
        specs=specs,
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        ridge=float(args.ridge),
        noise_floor=float(args.noise_floor),
        source_weighting=str(args.source_weighting),
        mlp_config=mlp_config,
    )
    summary = _summarize(trials)
    contrasts = _contrasts(trials, n_boot=int(args.n_bootstrap), seed=int(args.fold_seed) + 17)

    trials_path = out_dir / "continuous_feature_embedding_reconstruction_trials.csv"
    summary_path = out_dir / "continuous_feature_embedding_reconstruction_summary.csv"
    contrasts_path = out_dir / "continuous_feature_embedding_reconstruction_contrasts.csv"
    models_path = out_dir / "continuous_feature_embedding_reconstruction_models.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    models.to_csv(models_path, index=False)
    png, pdf, plotted_decoder, plotted_mode = _plot(
        summary,
        contrasts,
        out_dir,
        decoder_mode=str(args.plot_decoder_mode),
        latent=str(args.latent),
        feature_space_mode=str(args.plot_feature_space_mode),
    )

    manifest_payload = {
        "analysis": "continuous_feature_embedding_reconstruction",
        "run_dir": str(run_dir),
        "response_manifest": str(manifest_path),
        "n_response_tables": int(manifest.shape[0]),
        "feature": {
            **feature_meta,
            "feature_dim_requested": int(args.feature_dim),
            "feature_space_modes_requested": feature_space_modes,
            "feature_space_modes_canonical": sorted(set(models["feature_space_mode"].astype(str).tolist())),
            "feature_weights": feature_weight_meta,
        },
        "decoder_modes_requested": decoder_modes,
        "decoder_modes_canonical": sorted(set(models["decoder_mode"].astype(str).tolist())),
        "observer_modes": [spec.slug for spec in specs],
        "source_weighting": str(args.source_weighting),
        "allow_global_feature_transforms": bool(args.allow_global_feature_transforms),
        "axis_prior_family_contrasts": {
            "reported": False,
            "reason": (
                "This runner fits response-feature models pooled across prior-family labels "
                "and evaluates the same observed response rows for axis_edge_parallel and "
                "axis_edge_orthogonal labels. Axis contrasts require a dedicated per-axis "
                "fit/evaluation contract."
            ),
        },
        "mlp": _json_ready(mlp_config.__dict__),
        "plot_decoder_mode": plotted_decoder,
        "plot_feature_space_mode": plotted_mode,
        "population": population_meta,
        "basis": basis_meta,
        "trajectory_coordinate_contract": {
            "rendering_coordinate": "crop_centered_displacement",
            "motion_condition": "motion_mean_centered_within_each_backimage_crop",
            "static_matched_mean_mode": STATIC_MATCHED_MEAN_MODE,
            "static_matched_mean_source": "zero_lambda_counts[true_candidate_index]",
            "static_matched_mean_reference_mode": "patch_center_static_tau_zero",
            "legacy_aliases": sorted(STATIC_MATCHED_MEAN_ALIASES),
            "interpretation": (
                "BackImage crops are extracted at each window's mean fixation position; "
                "tau=0 is therefore static at that crop/mean position, not a global-zero "
                "absolute eye-position oracle."
            ),
        },
        "ridge": float(args.ridge),
        "noise_floor": float(args.noise_floor),
        "n_folds": int(args.n_folds),
        "fold_seed": int(args.fold_seed),
        "n_bootstrap": int(args.n_bootstrap),
        "sample_banks": {
            name: {
                "n_samples": int(bank.x.shape[0]),
                "response_dim": int(bank.x.shape[1]),
                "n_source_rows": int(len(set(bank.source_rows.tolist()))),
            }
            for name, bank in banks.items()
        },
        "outputs": {
            "trials": trials_path,
            "summary": summary_path,
            "contrasts": contrasts_path,
            "models": models_path,
            "figure_png": png,
            "figure_pdf": pdf,
        },
    }
    manifest_json = out_dir / "continuous_feature_embedding_reconstruction_manifest.json"
    _write_json(manifest_json, manifest_payload)
    _write_readme(out_dir=out_dir, summary=summary, contrasts=contrasts, manifest=manifest_payload)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()

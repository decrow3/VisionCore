"""Run the covariance-aware FEM optimality extension for a twininfo run.

The runner reuses the selected traces, image crops, and sampled population from
an existing production ``jake.twininfo`` run, then writes all new artifacts
under ``<run-dir>/covariance_optimality/<run-name>/``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import DT, OUTPUT_DIR, extract_fixrsvp_eye_traces, load_digital_twin, write_json
from .covariance_optimality import (
    DEFAULT_K_LIST,
    DEFAULT_NOISE_FLOOR_MULTIPLIERS,
    DEFAULT_RATE_GAINS,
    DEFAULT_SCALES,
    SCALED_FAMILIES,
    CovarianceEstimate,
    alignment_rows,
    coding_covariance_from_j,
    covariance_residual_after_subspace,
    counts_and_derivatives_from_shifted_rates,
    covariance_fisher_by_time,
    covariance_spectrum_row,
    fisher_metric_row,
    geometry_covariance_rows,
    independent_fisher_by_time,
    movement_covariance_pooled_residual,
    movement_covariance_within_pair,
    parse_csv_list,
    parse_float_list,
    read_csv_rows,
    scale_label,
    scaled_condition_name,
    sensitivity_metric_rows,
    signal_covariance_from_pair_means,
    top_eigenvectors,
    trajectories_for_scaled_family,
    write_csv_rows,
)
from .image_selection import load_natural_images
from .lagcube_information import finite_difference_shift_set, run_shifted_lag_cube_rates
from .pipeline import _example_seed
from .population import build_analysis_population
from .retinal_examples import model_lag_cubes_from_image_trace


@dataclass(frozen=True)
class CovOptConfig:
    from_run_dir: Path
    run_name: str = "covopt"
    device: str | None = None
    scales: tuple[float, ...] = DEFAULT_SCALES
    condition_families: tuple[str, ...] = ("scaled_real",)
    max_pairs: int = 0
    population_source: str = "metadata"
    population_mode: str = "sampled_units"
    analysis_population_size: int = 0
    analysis_population_selection: str = "top_performance"
    analysis_grid_position_mode: str = "center"
    analysis_grid_stride: int = 1
    analysis_performance_metric: str = "ccnorm"
    analysis_deduplicate_units: bool = True
    center_rate_cache_dir: Path | None = Path("outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9")
    use_center_rate_cache: bool = True
    batch_size: int | None = None
    fisher_step_arcmin: float | None = None
    ridge_frac: float = 1e-4
    rate_gains: tuple[float, ...] = DEFAULT_RATE_GAINS
    noise_floor_multipliers: tuple[float, ...] = DEFAULT_NOISE_FLOOR_MULTIPLIERS
    k_list: tuple[int, ...] = DEFAULT_K_LIST
    geometry_k_list: tuple[int, ...] = DEFAULT_K_LIST
    recompute: bool = False
    refresh_results: bool = False
    skip_sensitivity: bool = False


def _as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return int(default)
    return int(float(value))


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_trace_examples_from_metadata(run_dir: Path, model: Any, *, t_max: int) -> dict[str, dict[str, Any]]:
    trace_rows = read_csv_rows(run_dir / "metadata" / "01_trace_examples.csv")
    used_rows = read_csv_rows(run_dir / "metadata" / "01_trace_examples_used.csv")
    if not trace_rows and not used_rows:
        raise FileNotFoundError(f"No trace example metadata found under {run_dir / 'metadata'}")
    by_id = {str(row["example_id"]): row for row in trace_rows}
    rows = used_rows or trace_rows
    eye_traces, _durations = extract_fixrsvp_eye_traces(model, min_fix_dur=int(t_max))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        full = dict(by_id.get(example_id, {}))
        full.update(row)
        source_idx = _as_int(full, "source_trace_index")
        start = _as_int(full, "window_start")
        stop = _as_int(full, "window_stop", start + int(t_max))
        trace = np.asarray(eye_traces[source_idx, start:stop], dtype=np.float32)
        if trace.shape[0] < int(t_max):
            raise ValueError(f"Trace {example_id} has only {trace.shape[0]} samples, expected {t_max}")
        full["trace"] = trace[: int(t_max)].astype(np.float32)
        out[example_id] = full
    return out


def _load_images_for_crops(crop_rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    image_indices = sorted({_as_int(row, "image_index") for row in crop_rows})
    loaded = load_natural_images(len(image_indices), indices=tuple(image_indices))
    return {int(spec.image_index): image for spec, image in loaded if spec.image_index is not None}


def _population_metadata_subset(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    """Return metadata rows used as covariance-analysis channels."""
    if str(mode) == "metadata_all":
        return sorted(rows, key=lambda row: _as_int(row, "simulated_unit_idx"))
    if str(mode) != "sampled_units":
        raise ValueError("population_mode must be 'sampled_units' or 'metadata_all'.")
    by_global: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_global.setdefault(_as_int(row, "global_unit_idx"), []).append(row)
    out: list[dict[str, str]] = []
    for global_idx in sorted(by_global):
        candidates = by_global[global_idx]
        h = _as_int(candidates[0], "grid_shape_h")
        w = _as_int(candidates[0], "grid_shape_w")
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        best = min(
            candidates,
            key=lambda row: (
                (_as_int(row, "grid_row") - cy) ** 2 + (_as_int(row, "grid_col") - cx) ** 2,
                _as_int(row, "simulated_unit_idx"),
            ),
        )
        out.append(best)
    return out


def _population_from_metadata_rows(model: Any, rows: list[dict[str, str]], *, mode: str = "sampled_units"):
    """Reconstruct the exact sampled population represented by metadata rows."""
    if not rows:
        raise ValueError("Cannot reconstruct population from empty 00_population_units.csv")

    from _common import SimulatedPopulation
    from .population import _readout_from_ranked_rows

    sorted_rows = _population_metadata_subset(rows, mode)
    unique_by_global: dict[int, dict[str, Any]] = {}
    for row in sorted_rows:
        global_idx = _as_int(row, "global_unit_idx")
        unique_by_global.setdefault(global_idx, dict(row))
    biological_rows = [unique_by_global[idx] for idx in sorted(unique_by_global)]
    readout, session_names, _metadata = _readout_from_ranked_rows(
        model,
        biological_rows,
        feature_grid=(14, 14),
    )
    unit_ids = np.asarray(
        [
            [_as_int(row, "global_unit_idx"), _as_int(row, "grid_row"), _as_int(row, "grid_col")]
            for row in sorted_rows
        ],
        dtype=np.int64,
    )
    grid_shape = (
        _as_int(sorted_rows[0], "grid_shape_h"),
        _as_int(sorted_rows[0], "grid_shape_w"),
    )
    return SimulatedPopulation(
        readout=readout,
        unit_ids=unit_ids,
        session_names=session_names,
        grid_shape=grid_shape,
        N=int(unit_ids.shape[0]),
    )


def _build_covopt_population(
    *,
    config: CovOptConfig,
    model: Any,
    population_rows: list[dict[str, str]],
    seed: int,
    out_dir: Path,
):
    """Build the response population for covariance-optimality rates."""
    source = str(config.population_source)
    if source == "metadata":
        population = _population_from_metadata_rows(model, population_rows, mode=config.population_mode)
        write_csv_rows(out_dir / "metadata" / "covopt_population_units.csv", _population_metadata_subset(population_rows, config.population_mode))
        return population
    if source != "analysis":
        raise ValueError("population_source must be 'metadata' or 'analysis'.")
    n = int(config.analysis_population_size)
    if n <= 0:
        raise ValueError("--analysis-population-size must be positive when --population-source=analysis.")
    population, rows = build_analysis_population(
        model,
        N=n,
        rng=np.random.default_rng(int(seed)),
        selection=str(config.analysis_population_selection),
        performance_metric=str(config.analysis_performance_metric),
        grid_position_mode=str(config.analysis_grid_position_mode),
        grid_stride=int(config.analysis_grid_stride),
        deduplicate_units=bool(config.analysis_deduplicate_units),
    )
    write_csv_rows(out_dir / "metadata" / "covopt_population_units.csv", rows)
    return population


def _pair_records(run_dir: Path, *, max_pairs: int = 0) -> list[dict[str, Any]]:
    """Return unique paired image/crop/trace records covered by Figure 5 outputs."""
    rows = read_csv_rows(run_dir / "metadata" / "05_information_series_records.csv")
    if not rows:
        rows = read_csv_rows(run_dir / "metadata" / "05_lagcube_information_summary.csv")
    if not rows:
        raise FileNotFoundError(f"No Figure 5 information records found under {run_dir / 'metadata'}")
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["example_id"]), _as_int(row, "image_index"), _as_int(row, "crop_rank"))
        if key not in by_key:
            by_key[key] = {
                "example_id": str(row["example_id"]),
                "kind": str(row.get("kind", "")),
                "image_index": key[1],
                "crop_rank": key[2],
            }
    out = list(by_key.values())
    if int(max_pairs) > 0:
        out = out[: int(max_pairs)]
    return out


def _output_dir(config: CovOptConfig) -> Path:
    return Path(config.from_run_dir) / "covariance_optimality" / str(config.run_name)


def _load_existing_arrays(out_dir: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]] | None:
    records_path = out_dir / "metadata" / "covopt_rate_records.csv"
    arrays_path = out_dir / "cache" / "covopt_mu_j.npz"
    if not records_path.exists() or not arrays_path.exists():
        return None
    with np.load(arrays_path) as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    return read_csv_rows(records_path), arrays


def _row_cache_id(record: dict[str, Any]) -> str:
    """Stable row-cache identifier for one pair/family/scale."""
    return (
        f"{record['example_id']}__{record.get('kind', '')}__image{int(record['image_index']):03d}"
        f"__crop{int(record['crop_rank']):02d}__{record['family']}__D{scale_label(float(record['scale_D']))}"
        f"__step{scale_label(float(record.get('fisher_step_arcmin', 0.0)))}"
        f"__pop{record.get('population_source', 'metadata')}_{record.get('population_mode', 'unknown')}"
        f"_{record.get('population_selection', '')}_{record.get('grid_position_mode', '')}"
        f"_N{int(record.get('population_n', 0))}"
    )


def _save_row_cache(path: Path, record: dict[str, Any], mu: np.ndarray, jac: np.ndarray, expected: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mu=np.asarray(mu, dtype=np.float32),
        J=np.asarray(jac, dtype=np.float32),
        expected_spikes_t=np.asarray(expected, dtype=np.float32),
        cache_version=np.asarray("covopt_mu_j_row_v1"),
        record_json=np.asarray(json.dumps(record, sort_keys=True)),
    )


def _load_row_cache(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        record = json.loads(str(npz["record_json"]))
        mu = np.asarray(npz["mu"], dtype=np.float32)
        jac = np.asarray(npz["J"], dtype=np.float32)
        expected = np.asarray(npz["expected_spikes_t"], dtype=np.float32)
    return record, mu, jac, expected


def _center_cache_condition(family: str, scale: float) -> str | None:
    """Return the standard cached condition matching a scaled center response."""
    if np.isclose(float(scale), 0.0):
        return "stabilized"
    if family == "scaled_real" and np.isclose(float(scale), 1.0):
        return "real"
    if family == "random_amp_scaled" and np.isclose(float(scale), 1.0):
        return "random_amp"
    if family == "random_amp_cloud_matched_scaled" and np.isclose(float(scale), 1.0):
        return "random_amp_cloud_matched"
    if family == "trajectory_order_shuffle_scaled" and np.isclose(float(scale), 1.0):
        return "trajectory_order_shuffle"
    return None


def _load_center_rate_cache(
    cache_dir: Path | None,
) -> tuple[dict[tuple[str, str, int, int, str], int], np.ndarray, dict[str, Any]] | None:
    if cache_dir is None:
        return None
    root = Path(cache_dir)
    records_path = root / "natural_image_center_rate_records.csv"
    rates_path = root / "natural_image_center_rates.npz"
    manifest_path = root / "manifest.json"
    if not records_path.exists() or not rates_path.exists():
        return None
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    records = read_csv_rows(records_path)
    with np.load(rates_path) as npz:
        rates = np.asarray(npz["rates"], dtype=np.float32)
    if len(records) != int(rates.shape[0]):
        raise ValueError(f"Center-rate cache mismatch: {len(records)} records for {rates.shape[0]} rate rows.")
    lookup = {
        (
            str(row["example_id"]),
            str(row.get("kind", "")),
            _as_int(row, "image_index"),
            _as_int(row, "crop_rank"),
            str(row["condition"]),
        ): i
        for i, row in enumerate(records)
    }
    return lookup, rates, manifest


def _path_matches(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.resolve() == path_b.resolve()
    except FileNotFoundError:
        return Path(path_a).absolute() == Path(path_b).absolute()


def _compute_mu_j_cache(
    *,
    config: CovOptConfig,
    out_dir: Path,
    model: Any,
    population: Any,
    device: Any,
    trace_by_id: dict[str, dict[str, Any]],
    image_by_index: dict[int, np.ndarray],
    crop_rows: list[dict[str, str]],
    pair_records: list[dict[str, Any]],
    seed: int,
    t_max: int,
    batch_size: int,
    fisher_step_arcmin: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    crop_by_key = {
        (_as_int(row, "image_index"), _as_int(row, "crop_rank")): row
        for row in crop_rows
    }
    records: list[dict[str, Any]] = []
    mu_rows: list[np.ndarray] = []
    j_rows: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    shifts = finite_difference_shift_set(float(fisher_step_arcmin))
    noncenter_shifts = np.asarray(
        [row for row in shifts if not (np.isclose(row[0], 0.0) and np.isclose(row[1], 0.0))],
        dtype=np.float64,
    )
    row_cache_dir = out_dir / "cache" / "covopt_mu_j_rows"
    center_cache = (
        _load_center_rate_cache(config.center_rate_cache_dir)
        if bool(config.use_center_rate_cache)
        else None
    )
    center_lookup: dict[tuple[str, str, int, int, str], int] = {}
    center_rates: np.ndarray | None = None
    center_manifest: dict[str, Any] = {}
    if center_cache is not None:
        center_lookup, center_rates, center_manifest = center_cache
        manifest_run_dir = center_manifest.get("run_dir")
        if manifest_run_dir:
            manifest_run_path = Path(manifest_run_dir)
            if not manifest_run_path.is_absolute():
                manifest_run_path = Path.cwd() / manifest_run_path
            source_run = Path(config.from_run_dir)
            if not _path_matches(manifest_run_path, source_run):
                print(
                    f"Center-rate cache ignored: manifest run_dir={manifest_run_dir} "
                    f"does not match source run {source_run}",
                    flush=True,
                )
                center_lookup = {}
                center_rates = None
        if center_rates is not None:
            print(
                f"Center-rate cache enabled: {config.center_rate_cache_dir} "
                f"({len(center_lookup)} indexed rows)",
                flush=True,
            )

    total = len(pair_records) * len(config.condition_families) * len(config.scales)
    done = 0
    for pair in pair_records:
        example_id = str(pair["example_id"])
        image_index = int(pair["image_index"])
        crop_rank = int(pair["crop_rank"])
        example = trace_by_id[example_id]
        crop = crop_by_key[(image_index, crop_rank)]
        image = image_by_index[image_index]
        crop_offset = (_as_float(crop, "offset_x_px"), _as_float(crop, "offset_y_px"))
        pair_seed = _example_seed(int(seed), example_id, image_index, crop_rank)
        for family in config.condition_families:
            trajectories = trajectories_for_scaled_family(
                example["trace"],
                family,
                config.scales,
                t_max=int(t_max),
                seed=pair_seed,
            )
            for scale in config.scales:
                done += 1
                condition = scaled_condition_name(family, scale)
                record = {
                    "row_id": len(records),
                    "example_id": example_id,
                    "kind": str(example.get("kind", pair.get("kind", ""))),
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "family": family,
                    "condition": condition,
                    "scale_D": float(scale),
                    "trajectory_description": trajectories[float(scale)][1],
                    "crop_center_offset_x_px": crop_offset[0],
                    "crop_center_offset_y_px": crop_offset[1],
                    "fisher_step_arcmin": float(fisher_step_arcmin),
                    "population_source": str(config.population_source),
                    "population_mode": str(config.population_mode),
                    "population_selection": str(config.analysis_population_selection),
                    "grid_position_mode": str(config.analysis_grid_position_mode),
                    "population_n": int(population.N),
                    "t_max": int(t_max),
                }
                row_cache_path = row_cache_dir / f"{_row_cache_id(record)}.npz"
                if row_cache_path.exists():
                    cached_record, mu, jac, expected = _load_row_cache(row_cache_path)
                    cached_record["row_id"] = len(records)
                    records.append(cached_record)
                    mu_rows.append(mu.astype(np.float32))
                    j_rows.append(jac.astype(np.float32))
                    expected_rows.append(expected.astype(np.float32))
                    print(
                        f"[{done}/{total}] {condition} example={example_id} image={image_index} crop={crop_rank} cached-row",
                        flush=True,
                    )
                    continue
                print(
                    f"[{done}/{total}] {condition} example={example_id} image={image_index} crop={crop_rank}",
                    flush=True,
                )
                trace, desc = trajectories[float(scale)]
                cubes = model_lag_cubes_from_image_trace(
                    image,
                    trace,
                    t_max=int(t_max),
                    crop_center_offset_px=crop_offset,
                )
                center_condition = _center_cache_condition(family, float(scale))
                center_rates_tn = None
                if center_condition is not None and center_rates is not None:
                    center_idx = center_lookup.get(
                        (
                            example_id,
                            str(example.get("kind", pair.get("kind", ""))),
                            image_index,
                            crop_rank,
                            center_condition,
                        )
                    )
                    if center_idx is not None:
                        center_rates_tn = np.asarray(center_rates[center_idx], dtype=np.float32)
                        if center_rates_tn.shape != (int(t_max), int(population.N)):
                            center_rates_tn = None
                shift_arg = noncenter_shifts if center_rates_tn is not None else shifts
                shifted_rates = run_shifted_lag_cube_rates(
                    model,
                    population,
                    device,
                    cubes,
                    shift_arg,
                    batch_size=int(batch_size),
                )
                rates_by_shift = dict(shifted_rates)
                if center_rates_tn is not None:
                    rates_by_shift[(0.0, 0.0)] = center_rates_tn
                mu, jac = counts_and_derivatives_from_shifted_rates(
                    rates_by_shift,
                    fisher_step_arcmin=float(fisher_step_arcmin),
                    dt=DT,
                )
                record["trajectory_description"] = desc
                if center_rates_tn is not None:
                    record["center_rate_cache_condition"] = center_condition
                    print(
                        f"  reused center-rate cache condition={center_condition}; "
                        f"ran {len(shift_arg)}/{len(shifts)} finite-difference shifts",
                        flush=True,
                    )
                expected = np.sum(mu, axis=1).astype(np.float32)
                _save_row_cache(row_cache_path, record, mu, jac, expected)
                records.append(record)
                mu_rows.append(mu.astype(np.float32))
                j_rows.append(jac.astype(np.float32))
                expected_rows.append(expected)

    arrays = {
        "mu": np.stack(mu_rows, axis=0).astype(np.float32),
        "J": np.stack(j_rows, axis=0).astype(np.float32),
        "expected_spikes_t": np.stack(expected_rows, axis=0).astype(np.float32),
        "scale_D": np.asarray([float(row["scale_D"]) for row in records], dtype=np.float32),
        "row_id": np.asarray([int(row["row_id"]) for row in records], dtype=np.int32),
    }
    out_dir.joinpath("cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "cache" / "covopt_mu_j.npz", **arrays)
    write_csv_rows(out_dir / "metadata" / "covopt_rate_records.csv", records)
    return records, arrays


def _compute_results(
    *,
    config: CovOptConfig,
    out_dir: Path,
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> dict[str, str]:
    mu_all = np.asarray(arrays["mu"], dtype=np.float32)
    j_all = np.asarray(arrays["J"], dtype=np.float32)
    expected_all = np.asarray(arrays["expected_spikes_t"], dtype=np.float32)
    by_group: dict[tuple[str, float, str], list[int]] = {}
    for i, row in enumerate(records):
        key = (str(row["family"]), float(row["scale_D"]), str(row.get("kind", "")))
        by_group.setdefault(key, []).append(i)
    print(
        f"Computing covariance/Fisher results for {len(records)} rows across {len(by_group)} groups",
        flush=True,
    )

    signal_reference: dict[tuple[str, str], tuple[float, np.ndarray]] = {}
    for family in sorted({key[0] for key in by_group}):
        for kind in sorted({key[2] for key in by_group if key[0] == family}):
            candidate_scales = sorted(scale for fam, scale, kd in by_group if fam == family and kd == kind)
            if not candidate_scales:
                continue
            reference_scale = min(candidate_scales, key=lambda scale: (abs(float(scale) - 1.0), float(scale)))
            signal_reference[(family, kind)] = (
                reference_scale,
                signal_covariance_from_pair_means(mu_all[by_group[(family, reference_scale, kind)]]),
            )

    covariance_estimates: dict[tuple[str, float, str, str], CovarianceEstimate] = {}
    cov_rows: list[dict[str, Any]] = []
    for (family, scale, kind), ix in by_group.items():
        pooled, n_samples, n_pairs = movement_covariance_pooled_residual(mu_all[ix])
        within, n_samples_w, n_pairs_w = movement_covariance_within_pair(mu_all[ix])
        covariance_estimates[(family, scale, kind, "pooled_residual")] = CovarianceEstimate(
            family=family,
            scale=scale,
            kind=kind,
            estimator="pooled_residual",
            covariance=pooled,
            n_samples=n_samples,
            n_pairs=n_pairs,
        )
        covariance_estimates[(family, scale, kind, "within_pair")] = CovarianceEstimate(
            family=family,
            scale=scale,
            kind=kind,
            estimator="within_pair",
            covariance=within,
            n_samples=n_samples_w,
            n_pairs=n_pairs_w,
        )

    d1_trace = {
        (family, kind, estimator): float(np.trace(est.covariance))
        for (family, scale, kind, estimator), est in covariance_estimates.items()
        if np.isclose(scale, 1.0)
    }
    cov_arrays: dict[str, np.ndarray] = {}
    for key, est in covariance_estimates.items():
        family, scale, kind, estimator = key
        cov_rows.append(
            covariance_spectrum_row(
                est,
                reference_trace=d1_trace.get((family, kind, estimator)),
            )
        )
        cov_arrays[f"{family}__{kind or 'all'}__D{str(scale).replace('.', 'p')}__{estimator}"] = est.covariance.astype(np.float32)

    row_metrics: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    group_items = list(by_group.items())
    for group_idx, ((family, scale, kind), ix) in enumerate(group_items, start=1):
        print(
            f"[results {group_idx}/{len(group_items)}] family={family} kind={kind} "
            f"D={scale:g} rows={len(ix)}",
            flush=True,
        )
        primary_cov = covariance_estimates[(family, scale, kind, "pooled_residual")].covariance
        coding_cov = coding_covariance_from_j(j_all[ix])
        if (family, kind) in signal_reference:
            reference_scale, signal_cov = signal_reference[(family, kind)]
        else:
            reference_scale = scale
            signal_cov = signal_covariance_from_pair_means(mu_all[ix])
        group_alignment = alignment_rows(
            family=family,
            scale=scale,
            kind=kind,
            sigma_fem=primary_cov,
            coding_cov=coding_cov,
            signal_cov=signal_cov,
            k_list=config.k_list,
        )
        for row in group_alignment:
            row["signal_reference_scale_D"] = float(reference_scale)
        alignment.extend(group_alignment)
        group_geometry_rows = geometry_covariance_rows(
            family=family,
            scale=scale,
            kind=kind,
            sigma_fem=primary_cov,
            coding_cov=coding_cov,
            signal_cov=signal_cov,
            k_list=config.geometry_k_list,
        )
        for row in group_geometry_rows:
            row["signal_reference_scale_D"] = float(reference_scale)
        geometry_rows.extend(group_geometry_rows)
        geometry_residuals: dict[int, np.ndarray] = {}
        for k in config.geometry_k_list:
            basis = top_eigenvectors(primary_cov, int(k))
            _compact_cov, residual_cov = covariance_residual_after_subspace(primary_cov, basis)
            geometry_residuals[int(k)] = residual_cov
        for i in ix:
            row = records[i]
            f_ind = independent_fisher_by_time(mu_all[i], j_all[i])
            f_cov_pose = covariance_fisher_by_time(
                mu_all[i],
                j_all[i],
                None,
                ridge_frac=float(config.ridge_frac),
            )
            f_cov_blind = covariance_fisher_by_time(
                mu_all[i],
                j_all[i],
                primary_cov,
                ridge_frac=float(config.ridge_frac),
            )
            f_geometry_by_k = {
                int(k): covariance_fisher_by_time(
                    mu_all[i],
                    j_all[i],
                    residual_cov,
                    ridge_frac=float(config.ridge_frac),
                )
                for k, residual_cov in geometry_residuals.items()
            }
            row_metrics.append(
                fisher_metric_row(
                    row_id=i,
                    record=row,
                    family=family,
                    scale=scale,
                    regime="independent_pose_aware",
                    f_by_time=f_ind,
                    expected_spikes_t=expected_all[i],
                )
            )
            for k, f_geom in sorted(f_geometry_by_k.items()):
                geom_row = fisher_metric_row(
                    row_id=i,
                    record=row,
                    family=family,
                    scale=scale,
                    regime=f"cov_geometry_aware_k{k}",
                    f_by_time=f_geom,
                    expected_spikes_t=expected_all[i],
                )
                geom_row["geometry_k"] = int(k)
                geom_row["geometry_mode"] = "movement_covariance_top_eigen_residual"
                row_metrics.append(geom_row)
            row_metrics.append(
                fisher_metric_row(
                    row_id=i,
                    record=row,
                    family=family,
                    scale=scale,
                    regime="cov_pose_aware",
                    f_by_time=f_cov_pose,
                    expected_spikes_t=expected_all[i],
                )
            )
            row_metrics.append(
                fisher_metric_row(
                    row_id=i,
                    record=row,
                    family=family,
                    scale=scale,
                    regime="cov_pose_blind",
                    f_by_time=f_cov_blind,
                    expected_spikes_t=expected_all[i],
                )
            )
            if not config.skip_sensitivity:
                sensitivity_rows.extend(
                    sensitivity_metric_rows(
                        row_id=i,
                        record=row,
                        family=family,
                        scale=scale,
                        mu_tn=mu_all[i],
                        j_tnd=j_all[i],
                        sigma_extra=primary_cov,
                        rate_gains=config.rate_gains,
                        noise_floor_multipliers=config.noise_floor_multipliers,
                        ridge_frac=float(config.ridge_frac),
                    )
                )
        print(
            f"[results {group_idx}/{len(group_items)}] done family={family} kind={kind} D={scale:g}",
            flush=True,
        )

    out_dir.joinpath("cache").mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("results").mkdir(parents=True, exist_ok=True)
    print("Writing covariance/Fisher result tables", flush=True)
    sensitivity_path = out_dir / "results" / "covopt_sensitivity_row_metrics.csv"
    np.savez_compressed(out_dir / "cache" / "covopt_covariances.npz", **cov_arrays)
    write_csv_rows(out_dir / "results" / "covopt_covariance_spectra.csv", cov_rows)
    write_csv_rows(out_dir / "results" / "covopt_alignment_diagnostics.csv", alignment)
    write_csv_rows(out_dir / "results" / "covopt_geometry_diagnostics.csv", geometry_rows)
    write_csv_rows(out_dir / "results" / "covopt_row_metrics.csv", row_metrics)
    if not config.skip_sensitivity or not sensitivity_path.exists():
        write_csv_rows(sensitivity_path, sensitivity_rows)
    return {
        "row_metrics": str(out_dir / "results" / "covopt_row_metrics.csv"),
        "sensitivity_row_metrics": str(out_dir / "results" / "covopt_sensitivity_row_metrics.csv"),
        "covariance_spectra": str(out_dir / "results" / "covopt_covariance_spectra.csv"),
        "alignment_diagnostics": str(out_dir / "results" / "covopt_alignment_diagnostics.csv"),
        "geometry_diagnostics": str(out_dir / "results" / "covopt_geometry_diagnostics.csv"),
        "mu_j_cache": str(out_dir / "cache" / "covopt_mu_j.npz"),
        "covariance_cache": str(out_dir / "cache" / "covopt_covariances.npz"),
    }


def run_covariance_optimality(config: CovOptConfig) -> dict[str, Any]:
    from_run_dir = Path(config.from_run_dir)
    if not from_run_dir.exists():
        candidate = OUTPUT_DIR / str(config.from_run_dir)
        if candidate.exists():
            from_run_dir = candidate
        else:
            raise FileNotFoundError(f"Could not find source run directory: {config.from_run_dir}")
    config = CovOptConfig(**{**asdict(config), "from_run_dir": from_run_dir})
    out_dir = _output_dir(config)
    summary_path = out_dir / "metadata" / "covopt_run_summary.json"
    if summary_path.exists() and not config.recompute and not config.refresh_results:
        return _load_json(summary_path)

    out_dir.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("cache").mkdir(parents=True, exist_ok=True)
    run_config = _load_json(from_run_dir / "metadata" / "run_config.json")
    t_max = int(run_config.get("t_max", 128))
    seed = int(run_config.get("seed", 0))
    batch_size = int(config.batch_size or run_config.get("batch_size", 64))
    fisher_step_arcmin = float(config.fisher_step_arcmin or run_config.get("fisher_step_arcmin", 0.5))
    crop_rows = read_csv_rows(from_run_dir / "metadata" / "02_image_crop_hotspots.csv")
    population_rows = read_csv_rows(from_run_dir / "metadata" / "00_population_units.csv")
    pair_records = _pair_records(from_run_dir, max_pairs=config.max_pairs)

    run_config_payload = {
        **asdict(config),
        "from_run_dir": str(from_run_dir),
        "device": config.device,
        "effective_model_device": None,
        "t_max": t_max,
        "seed": seed,
        "batch_size": batch_size,
        "fisher_step_arcmin": fisher_step_arcmin,
        "n_pairs": len(pair_records),
        "population_source": config.population_source,
        "population_mode": config.population_mode,
        "analysis_population_size": int(config.analysis_population_size),
        "analysis_population_selection": str(config.analysis_population_selection),
        "analysis_grid_position_mode": str(config.analysis_grid_position_mode),
        "analysis_grid_stride": int(config.analysis_grid_stride),
        "analysis_performance_metric": str(config.analysis_performance_metric),
        "analysis_deduplicate_units": bool(config.analysis_deduplicate_units),
        "geometry_k_list": list(config.geometry_k_list),
        "center_rate_cache_dir": None if config.center_rate_cache_dir is None else str(config.center_rate_cache_dir),
        "use_center_rate_cache": bool(config.use_center_rate_cache),
    }
    write_json(out_dir / "metadata" / "covopt_run_config.json", run_config_payload)

    existing = None if config.recompute else _load_existing_arrays(out_dir)
    if existing is None:
        model, _model_info, device = load_digital_twin(device=config.device)
        effective_model_device = str(next(model.model.parameters()).device)
        print(f"CovOpt requested device={config.device!r}; effective model device={effective_model_device}")
        run_config_payload["effective_model_device"] = effective_model_device
        write_json(out_dir / "metadata" / "covopt_run_config.json", run_config_payload)
        population = _build_covopt_population(
            config=config,
            model=model,
            population_rows=population_rows,
            seed=seed,
            out_dir=out_dir,
        )
        trace_by_id = _load_trace_examples_from_metadata(from_run_dir, model, t_max=t_max)
        image_by_index = _load_images_for_crops(crop_rows)
        records, arrays = _compute_mu_j_cache(
            config=config,
            out_dir=out_dir,
            model=model,
            population=population,
            device=device,
            trace_by_id=trace_by_id,
            image_by_index=image_by_index,
            crop_rows=crop_rows,
            pair_records=pair_records,
            seed=seed,
            t_max=t_max,
            batch_size=batch_size,
            fisher_step_arcmin=fisher_step_arcmin,
        )
    else:
        records, arrays = existing

    outputs = _compute_results(config=config, out_dir=out_dir, records=records, arrays=arrays)
    summary = {
        "covopt_dir": str(out_dir),
        "from_run_dir": str(from_run_dir),
        "n_rate_rows": len(records),
        "n_pairs": len(pair_records),
        "scales": list(config.scales),
        "condition_families": list(config.condition_families),
        "population_source": str(config.population_source),
        "population_mode": str(config.population_mode),
        "geometry_k_list": list(config.geometry_k_list),
        "outputs": outputs,
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="covopt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scales", default=",".join(str(v) for v in DEFAULT_SCALES))
    parser.add_argument("--condition-families", default="scaled_real")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--population-source", choices=("metadata", "analysis"), default="metadata")
    parser.add_argument("--population-mode", choices=("sampled_units", "metadata_all"), default="sampled_units")
    parser.add_argument("--analysis-population-size", type=int, default=0)
    parser.add_argument("--analysis-population-selection", choices=("top_performance", "random_reliable"), default="top_performance")
    parser.add_argument("--analysis-grid-position-mode", choices=("random", "center", "full_grid"), default="center")
    parser.add_argument("--analysis-grid-stride", type=int, default=1)
    parser.add_argument("--analysis-performance-metric", default="ccnorm")
    parser.add_argument("--analysis-deduplicate-units", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--center-rate-cache-dir", type=Path, default=CovOptConfig.center_rate_cache_dir)
    parser.add_argument("--use-center-rate-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fisher-step-arcmin", type=float, default=None)
    parser.add_argument("--ridge-frac", type=float, default=1e-4)
    parser.add_argument("--rate-gains", default=",".join(str(v) for v in DEFAULT_RATE_GAINS))
    parser.add_argument("--noise-floor-multipliers", default=",".join(str(v) for v in DEFAULT_NOISE_FLOOR_MULTIPLIERS))
    parser.add_argument("--k-list", default=",".join(str(v) for v in DEFAULT_K_LIST))
    parser.add_argument("--geometry-k-list", default=",".join(str(v) for v in DEFAULT_K_LIST))
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--refresh-results",
        action="store_true",
        help="Recompute result tables from the saved mu/J cache without rerendering rate rows.",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Preserve an existing sensitivity table instead of recomputing the gain/noise sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(parse_csv_list(args.condition_families))
    unknown = [family for family in families if family not in SCALED_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown condition families {unknown}. Use one of {SCALED_FAMILIES}.")
    summary = run_covariance_optimality(
        CovOptConfig(
            from_run_dir=args.from_run_dir,
            run_name=args.run_name,
            device=args.device,
            scales=tuple(parse_float_list(args.scales)),
            condition_families=families,
            max_pairs=int(args.max_pairs),
            population_source=str(args.population_source),
            population_mode=str(args.population_mode),
            analysis_population_size=int(args.analysis_population_size),
            analysis_population_selection=str(args.analysis_population_selection),
            analysis_grid_position_mode=str(args.analysis_grid_position_mode),
            analysis_grid_stride=int(args.analysis_grid_stride),
            analysis_performance_metric=str(args.analysis_performance_metric),
            analysis_deduplicate_units=bool(args.analysis_deduplicate_units),
            center_rate_cache_dir=args.center_rate_cache_dir,
            use_center_rate_cache=bool(args.use_center_rate_cache),
            batch_size=args.batch_size,
            fisher_step_arcmin=args.fisher_step_arcmin,
            ridge_frac=float(args.ridge_frac),
            rate_gains=tuple(parse_float_list(args.rate_gains)),
            noise_floor_multipliers=tuple(parse_float_list(args.noise_floor_multipliers)),
            k_list=tuple(int(v) for v in parse_float_list(args.k_list)),
            geometry_k_list=tuple(int(v) for v in parse_float_list(args.geometry_k_list)),
            recompute=bool(args.recompute),
            refresh_results=bool(args.refresh_results),
            skip_sensitivity=bool(args.skip_sensitivity),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

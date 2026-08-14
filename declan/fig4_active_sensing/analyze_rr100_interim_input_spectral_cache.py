#!/usr/bin/env python3
"""Summarize the corrected 49 x 973 input-only retinal spectral cache.

This analysis is deliberately restricted to retinal inputs. It does not load
the frozen RR100 model and makes no statements about neural sensitivity or
responses. The crossed design permits descriptive separation of image, trace,
and image-by-trace contributions to scalar spectral quantities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs/fig4_active_sensing/rr100_interim49x973_input_spectral_cache_checkpoint_34_v1"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_interim49x973_bridge_cohort_checkpoint_28_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_interim49x973_input_spectral_summary_checkpoint_35_v2"
SF_EDGES_CPD = np.asarray(
    [0.0, 0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0,
     5.65685425, 8.0, 11.3137085, 16.0, 22.627417, 32.0],
    dtype=float,
)
TF_HZ = np.arange(3.0, 60.0 + 0.1, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def load_and_validate(cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "input_only_spectral_cache_complete":
        raise RuntimeError(f"Production cache is not complete: {manifest.get('status')}")
    metrics = pd.read_csv(cache_dir / "movie_spectral_metrics.csv")
    if len(metrics) != 49 * 973:
        raise ValueError(f"Expected 47,677 rows, found {len(metrics)}")
    if metrics["image_index"].nunique() != 49 or metrics["trace_index"].nunique() != 973:
        raise ValueError("The cache is not the expected complete 49 x 973 crossing")
    if metrics.duplicated(["image_index", "trace_index"]).any():
        raise ValueError("Duplicate image-trace identities found")
    if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Non-finite numeric cache values found")

    radial_rows: list[np.ndarray] = []
    identity_rows: list[tuple[int, int]] = []
    for shard_path in sorted((cache_dir / "shards").glob("image_*.npz")):
        with np.load(shard_path) as data:
            image_index = int(np.asarray(data["image_index"]).ravel()[0])
            trace_index = np.asarray(data["trace_index"], dtype=int)
            radial = np.asarray(data["radial_power"], dtype=np.float32)
            oriented = np.asarray(data["orientation_power"], dtype=np.float32)
        if radial.shape != (973, 20, 13) or oriented.shape != (973, 20, 13, 12):
            raise ValueError(f"Unexpected shard shape in {shard_path}")
        relative_error = np.max(
            np.abs(radial.astype(float) - oriented.astype(float).sum(axis=3))
            / np.maximum(radial.astype(float), 1.0)
        )
        if float(relative_error) > 1e-5:
            raise ValueError(f"Radial/orientation conservation failed in {shard_path}: {relative_error}")
        radial_rows.append(radial)
        identity_rows.extend((image_index, int(value)) for value in trace_index)
    radial_all = np.concatenate(radial_rows, axis=0)
    identity = pd.DataFrame(identity_rows, columns=["image_index", "trace_index"])
    order = metrics[["image_index", "trace_index"]].merge(
        identity.reset_index(names="radial_row"),
        on=["image_index", "trace_index"],
        how="left",
        validate="one_to_one",
    )["radial_row"].to_numpy(int)
    radial_all = radial_all[order]
    radial_sum = radial_all.astype(np.float64).sum(axis=(1, 2))
    scalar = metrics["total_positive_tf_power"].to_numpy(float)
    maximum_sum_error = float(np.max(np.abs(radial_sum - scalar) / np.maximum(scalar, 1.0)))
    if maximum_sum_error > 1e-5:
        raise ValueError(f"Radial/scalar power conservation failed: {maximum_sum_error}")
    return metrics, radial_all


def add_derived_metrics(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    total = out["total_positive_tf_power"].clip(lower=np.finfo(float).tiny)
    fitted = out["total_positive_tf_power_fitted_sf"].clip(lower=np.finfo(float).tiny)
    out["log10_total_positive_tf_power"] = np.log10(total)
    out["fraction_fitted_sf"] = out["total_positive_tf_power_fitted_sf"] / total
    out["fraction_above_32_all_sf"] = (
        out["power_32_45p25_all_sf"] + out["power_45p25_60_all_sf"]
    ) / total
    out["fraction_above_45p25_all_sf"] = out["power_45p25_60_all_sf"] / total
    out["fraction_at_60_all_sf"] = out["power_at_60_all_sf"] / total
    out["fraction_above_32_fitted_sf"] = (
        out["power_32_45p25_fitted_sf"] + out["power_45p25_60_fitted_sf"]
    ) / fitted
    out["fraction_above_45p25_fitted_sf"] = out["power_45p25_60_fitted_sf"] / fitted
    out["fraction_at_60_fitted_sf"] = out["power_at_60_fitted_sf"] / fitted
    return out


def descriptive_summary(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    quantiles = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    for column in columns:
        values = table[column].to_numpy(float)
        record: dict[str, object] = {
            "metric": column,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
        }
        for quantile, value in zip(quantiles, np.quantile(values, quantiles), strict=True):
            record[f"q{int(round(100 * quantile)):02d}"] = float(value)
        rows.append(record)
    return pd.DataFrame(rows)


def variance_decomposition(table: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    n_images = int(table["image_index"].nunique())
    n_traces = int(table["trace_index"].nunique())
    for metric in metrics:
        pivot = table.pivot(index="image_index", columns="trace_index", values=metric).to_numpy(float)
        grand = float(pivot.mean())
        total = float(np.sum((pivot - grand) ** 2))
        image = float(n_traces * np.sum((pivot.mean(axis=1) - grand) ** 2))
        trace = float(n_images * np.sum((pivot.mean(axis=0) - grand) ** 2))
        interaction = max(total - image - trace, 0.0)
        for component, value in (("image", image), ("trace", trace), ("image_x_trace_remainder", interaction)):
            rows.append(
                {
                    "metric": metric,
                    "component": component,
                    "sum_squares": value,
                    "fraction_total_sum_squares": value / max(total, np.finfo(float).tiny),
                }
            )
    return pd.DataFrame(rows)


def driver_correlations(table: pd.DataFrame, cohort_dir: Path) -> pd.DataFrame:
    images = pd.read_csv(cohort_dir / "interim49_images.csv")
    traces = pd.read_csv(cohort_dir / "interim973_traces.csv")
    image_means = table.groupby("image_index", as_index=False).mean(numeric_only=True).merge(
        images, on="image_index", validate="one_to_one", suffixes=("", "_descriptor")
    )
    trace_means = table.groupby("trace_index", as_index=False).mean(numeric_only=True).merge(
        traces, on="trace_index", validate="one_to_one", suffixes=("", "_descriptor")
    )
    outcomes = [
        "log10_total_positive_tf_power",
        "fraction_above_32_all_sf",
        "fraction_above_45p25_all_sf",
        "fraction_fitted_sf",
    ]
    families = [
        (
            "image",
            image_means,
            [
                "reconstruction_exact_pixel_r",
                "corrected_reconstruction_rms_contrast",
                "corrected_reconstruction_orientation_coherence",
                "corrected_reconstruction_sf_centroid_cpd",
                "corrected_reconstruction_high_sf_fraction",
                "exact_saved_rms_contrast",
                "exact_saved_orientation_coherence",
                "exact_saved_sf_centroid_cpd",
                "exact_saved_high_sf_fraction",
            ],
        ),
        (
            "trace",
            trace_means,
            [
                "corrected_dpi_crop120_path_length_arcmin",
                "corrected_dpi_crop120_rms_radius_arcmin",
                "corrected_dpi_crop120_position_power_fraction_32plus_hz",
                "corrected_dpi_crop120_position_power_centroid_hz",
                "corrected_dpi_crop120_cov_anisotropy",
                "corrected_minus_legacy_path_rank",
            ],
        ),
    ]
    rows = []
    for kind, data, predictors in families:
        for predictor in predictors:
            for outcome in outcomes:
                rows.append(
                    {
                        "kind": kind,
                        "predictor": predictor,
                        "outcome": outcome,
                        "n": int(len(data)),
                        "spearman_rho": float(data[predictor].corr(data[outcome], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def spectral_shape_summary(radial: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    matrix = radial.reshape(len(radial), -1).astype(np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(totals, np.finfo(float).tiny)
    _, raw_singular, _ = randomized_svd(matrix, n_components=10, random_state=20260813)
    _, norm_singular, _ = randomized_svd(normalized, n_components=10, random_state=20260813)
    raw_rank1_energy = float(raw_singular[0] ** 2 / np.sum(matrix ** 2))
    normalized_rank1_energy = float(norm_singular[0] ** 2 / np.sum(normalized ** 2))
    pca = PCA(n_components=10, svd_solver="randomized", random_state=20260813).fit(normalized)
    mean_template = normalized.mean(axis=0)
    cosine = (normalized @ mean_template) / (
        np.linalg.norm(normalized, axis=1) * max(np.linalg.norm(mean_template), np.finfo(float).tiny)
    )
    summary = {
        "raw_uncentered_rank1_energy_fraction": raw_rank1_energy,
        "l1_normalized_uncentered_rank1_energy_fraction": normalized_rank1_energy,
        "l1_normalized_centered_pc1_variance_fraction": float(pca.explained_variance_ratio_[0]),
        "l1_normalized_centered_first5_variance_fraction": float(pca.explained_variance_ratio_[:5].sum()),
        "cosine_to_mean_template_median": float(np.median(cosine)),
        "cosine_to_mean_template_q05": float(np.quantile(cosine, 0.05)),
        "cosine_to_mean_template_min": float(np.min(cosine)),
    }
    return summary, normalized.mean(axis=0).reshape(20, 13), normalized.std(axis=0).reshape(20, 13)


def select_conditions(table: pd.DataFrame) -> pd.DataFrame:
    roles = [
        ("lowest_above32_fraction", "fraction_above_32_all_sf", "min"),
        ("median_above32_fraction", "fraction_above_32_all_sf", "median"),
        ("highest_above32_fraction", "fraction_above_32_all_sf", "max"),
        ("lowest_total_dynamic_power", "log10_total_positive_tf_power", "min"),
        ("highest_total_dynamic_power", "log10_total_positive_tf_power", "max"),
    ]
    selected = []
    used: set[tuple[int, int]] = set()
    for role, metric, rule in roles:
        values = table[metric].to_numpy(float)
        target = float(np.median(values)) if rule == "median" else float(np.min(values) if rule == "min" else np.max(values))
        order = np.argsort(np.abs(values - target), kind="stable")
        for index in order:
            row = table.iloc[int(index)]
            identity = (int(row.image_index), int(row.trace_index))
            if identity not in used:
                used.add(identity)
                record = row.to_dict()
                record.update(
                    {
                        "selection_role": role,
                        "selection_metric": metric,
                        "selection_rule": rule,
                        "selection_target": target,
                        "selection_is_algorithmic": True,
                    }
                )
                selected.append(record)
                break
    return pd.DataFrame(selected)


def plot_population(
    table: pd.DataFrame,
    variance: pd.DataFrame,
    mean_shape: np.ndarray,
    shape_sd: np.ndarray,
    out_dir: Path,
) -> None:
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    mean_db = 10 * np.log10(np.maximum(mean_shape / max(float(mean_shape.max()), 1e-30), 1e-5))
    mesh = axes[0, 0].pcolormesh(sf_centers, TF_HZ, mean_db, cmap="magma", shading="nearest", vmin=-40, vmax=0)
    axes[0, 0].set_xscale("log")
    axes[0, 0].axhline(32, color="cyan", ls="--", lw=1)
    axes[0, 0].set_title("Mean normalized SF x TF shape")
    axes[0, 0].set_xlabel("SF (cpd)")
    axes[0, 0].set_ylabel("TF (Hz)")
    fig.colorbar(mesh, ax=axes[0, 0], label="relative share (dB)")
    cv = shape_sd / np.maximum(mean_shape, 1e-12)
    mesh_cv = axes[0, 1].pcolormesh(sf_centers, TF_HZ, np.clip(cv, 0, 3), cmap="viridis", shading="nearest", vmin=0, vmax=3)
    axes[0, 1].set_xscale("log")
    axes[0, 1].axhline(32, color="white", ls="--", lw=1)
    axes[0, 1].set_title("Across-condition shape variability")
    axes[0, 1].set_xlabel("SF (cpd)")
    axes[0, 1].set_ylabel("TF (Hz)")
    fig.colorbar(mesh_cv, ax=axes[0, 1], label="coefficient of variation")

    distributions = [
        ("fraction_above_32_all_sf", "Power above 32 Hz", axes[0, 2]),
        ("fraction_above_45p25_all_sf", "Power above 45.25 Hz", axes[1, 0]),
        ("fraction_fitted_sf", "Power inside fitted SF support", axes[1, 1]),
    ]
    for metric, title, axis in distributions:
        values = table[metric].to_numpy(float)
        axis.hist(values, bins=45, color="#b2182b", alpha=0.78, density=True)
        axis.axvline(np.median(values), color="black", lw=1.5, label=f"median {np.median(values):.2f}")
        axis.set_xlabel("fraction")
        axis.set_ylabel("density")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=8)

    variance_plot = variance.pivot(index="metric", columns="component", values="fraction_total_sum_squares")
    variance_plot = variance_plot[["image", "trace", "image_x_trace_remainder"]]
    variance_plot.plot(kind="bar", stacked=True, ax=axes[1, 2], color=["#2166ac", "#b2182b", "#bdbdbd"])
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_ylabel("fraction of crossed-design sum of squares")
    axes[1, 2].set_xlabel("")
    axes[1, 2].set_xticklabels([value.replace("_", "\n") for value in variance_plot.index], rotation=0, fontsize=8)
    axes[1, 2].set_title("What varies: image, trace, or interaction")
    axes[1, 2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Corrected retinal-input spectra across 49 images x 973 traces\n"
        "Input power only; no RR100 neural responses",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out_dir / "input_spectral_population_summary.png", dpi=180)
    fig.savefig(out_dir / "input_spectral_population_summary.pdf")
    plt.close(fig)


def plot_selected_maps(selected: pd.DataFrame, radial: np.ndarray, table: pd.DataFrame, out_dir: Path) -> None:
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    lookup = {
        (int(row.image_index), int(row.trace_index)): index
        for index, row in enumerate(table.itertuples(index=False))
    }
    fig, axes = plt.subplots(1, len(selected), figsize=(3.6 * len(selected), 4.2), constrained_layout=True)
    for axis, row in zip(np.atleast_1d(axes), selected.itertuples(index=False), strict=True):
        spectrum = radial[lookup[(int(row.image_index), int(row.trace_index))]].astype(float)
        db = 10 * np.log10(np.maximum(spectrum / max(float(spectrum.max()), 1e-30), 1e-5))
        mesh = axis.pcolormesh(sf_centers, TF_HZ, db, cmap="magma", shading="nearest", vmin=-40, vmax=0)
        axis.set_xscale("log")
        axis.axhline(32, color="cyan", ls="--", lw=1)
        axis.set_xlabel("SF (cpd)")
        axis.set_ylabel("TF (Hz)")
        axis.set_title(
            f"{str(row.selection_role).replace('_', ' ')}\n"
            f"image {int(row.image_index)}, trace {int(row.trace_index)}\n"
            f">32 Hz={float(row.fraction_above_32_all_sf):.2f}, fitted SF={float(row.fraction_fitted_sf):.2f}",
            fontsize=9,
        )
    fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), label="within-condition relative power (dB)")
    fig.suptitle("Algorithmically selected spectral examples", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "selected_input_spectral_maps.png", dpi=180)
    fig.savefig(out_dir / "selected_input_spectral_maps.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite summary checkpoint: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    table, radial = load_and_validate(args.cache_dir)
    table = add_derived_metrics(table)
    table.to_csv(args.out_dir / "movie_spectral_metrics_derived.csv", index=False)
    summary_metrics = [
        "fraction_fitted_sf",
        "fraction_above_32_all_sf",
        "fraction_above_45p25_all_sf",
        "fraction_at_60_all_sf",
        "fraction_above_32_fitted_sf",
        "fraction_above_45p25_fitted_sf",
        "fraction_at_60_fitted_sf",
        "log10_total_positive_tf_power",
    ]
    descriptive = descriptive_summary(table, summary_metrics)
    descriptive.to_csv(args.out_dir / "spectral_fraction_summary.csv", index=False)
    variance_metrics = [
        "log10_total_positive_tf_power",
        "fraction_above_32_all_sf",
        "fraction_above_45p25_all_sf",
        "fraction_fitted_sf",
    ]
    variance = variance_decomposition(table, variance_metrics)
    variance.to_csv(args.out_dir / "image_trace_variance_decomposition.csv", index=False)
    correlations = driver_correlations(table, args.cohort_dir)
    correlations.to_csv(args.out_dir / "descriptor_spectral_correlations.csv", index=False)
    shape, mean_shape, shape_sd = spectral_shape_summary(radial)
    (args.out_dir / "spectral_shape_summary.json").write_text(
        json.dumps(shape, indent=2) + "\n", encoding="utf-8"
    )
    selected = select_conditions(table)
    selected.to_csv(args.out_dir / "selected_spectral_conditions.csv", index=False)
    plot_population(table, variance, mean_shape, shape_sd, args.out_dir)
    plot_selected_maps(selected, radial, table, args.out_dir)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "corrected_input_only_population_summary_complete_stop_before_neural_inference",
        "scope": {"images": 49, "traces": 973, "movies": 47677},
        "qa": {
            "complete_crossing": True,
            "finite_numeric_values": True,
            "radial_orientation_power_conserved": True,
            "radial_scalar_power_conserved": True,
        },
        "sources": {
            "cache_manifest": file_identity(args.cache_dir / "manifest.json"),
            "movie_metrics": file_identity(args.cache_dir / "movie_spectral_metrics.csv"),
            "image_cohort": file_identity(args.cohort_dir / "interim49_images.csv"),
            "trace_cohort": file_identity(args.cohort_dir / "interim973_traces.csv"),
            "analysis_script": file_identity(Path(__file__)),
        },
        "spectral_shape": shape,
        "guardrail": (
            "Retinal input power only. No frozen-model response, neural gain, SSI, "
            "or tuning-weighted prediction is tested here."
        ),
        "next_gate": "review population and selected spectral maps before neural-model scoring",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

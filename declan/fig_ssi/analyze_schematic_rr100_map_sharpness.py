#!/usr/bin/env python3
"""Audit real-vs-stabilized RR100 final-map sharpness for the SSI schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "outputs" / "fig_ssi" / "rr100_schematic_endpoint_final_maps"
MAP_NPZ = MAP_DIR / "cache" / "schematic_rr100_final_maps.npz"
METRICS_CSV = MAP_DIR / "schematic_rr100_final_map_unit_metrics.csv"
OUT_CSV = MAP_DIR / "schematic_rr100_final_map_sharpness_metric_audit.csv"
OUT_SUMMARY_CSV = MAP_DIR / "schematic_rr100_final_map_sharpness_metric_summary.csv"
OUT_PNG = MAP_DIR / "schematic_rr100_final_map_sharpness_metric_audit.png"
OUT_PDF = MAP_DIR / "schematic_rr100_final_map_sharpness_metric_audit.pdf"

EPS = 1e-12
CURRENT_FIGURE_UNIT_INDEX = 56
TOP_MASS_FRACTION = 0.05


def probability_map(image: np.ndarray) -> np.ndarray | None:
    values = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    values = np.where(np.isfinite(values), values, 0.0)
    total = float(np.sum(values))
    if total <= EPS:
        return None
    return values / total


def spatial_ssi_bits_per_spike(image: np.ndarray) -> float:
    values = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    mean_rate = float(np.nanmean(values))
    if not np.isfinite(mean_rate) or mean_rate <= EPS:
        return 0.0
    gain = values / mean_rate
    return float(np.nanmean(gain * np.log2(gain + EPS)))


def high_frequency_power_fraction(image: np.ndarray, radius_quantile: float = 0.65) -> float:
    values = np.asarray(image, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    centered = values - float(np.mean(values))
    denom = float(np.sqrt(np.mean(centered * centered)))
    if denom <= EPS:
        return 0.0
    z = centered / denom
    power = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((yy - 0.5 * (h - 1)) ** 2 + (xx - 0.5 * (w - 1)) ** 2)
    cutoff = float(np.quantile(rr.ravel(), radius_quantile))
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0
    return float(np.sum(power[rr >= cutoff]) / total)


def laplacian_rms(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    centered = values - float(np.mean(values))
    denom = float(np.sqrt(np.mean(centered * centered)))
    if denom <= EPS:
        return 0.0
    z = centered / denom
    lap = (
        -4.0 * z
        + np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
    )
    return float(np.sqrt(np.mean(lap * lap)))


def map_sharpness_metrics(image: np.ndarray) -> dict[str, float]:
    values = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    p = probability_map(values)
    n_pix = int(values.size)
    mean_rate = float(np.nanmean(values))
    vmax = float(np.nanmax(values))
    if p is None:
        return {
            "mean_rate": mean_rate,
            "ssi_bits_per_spike": 0.0,
            "entropy_sharpness": 0.0,
            "inverse_participation": 0.0,
            "effective_area_px": float("nan"),
            "effective_area_fraction": float("nan"),
            "top5_mass_fraction": 0.0,
            "peak_to_mean": 0.0,
            "high_frequency_power_fraction": 0.0,
            "laplacian_rms": 0.0,
        }

    entropy = -float(np.sum(p * np.log(p + EPS)))
    entropy_sharpness = 1.0 - entropy / float(np.log(n_pix))
    inverse_pr = float(np.sum(p * p))
    effective_area = 1.0 / max(inverse_pr, EPS)
    flat = np.sort(p.ravel())[::-1]
    n_top = max(1, int(round(TOP_MASS_FRACTION * flat.size)))
    top_mass = float(np.sum(flat[:n_top]))
    return {
        "mean_rate": mean_rate,
        "ssi_bits_per_spike": spatial_ssi_bits_per_spike(values),
        "entropy_sharpness": float(entropy_sharpness),
        "inverse_participation": inverse_pr,
        "effective_area_px": float(effective_area),
        "effective_area_fraction": float(effective_area / n_pix),
        "top5_mass_fraction": top_mass,
        "peak_to_mean": float(vmax / max(mean_rate, EPS)),
        "high_frequency_power_fraction": high_frequency_power_fraction(values),
        "laplacian_rms": laplacian_rms(values),
    }


def load_inputs() -> tuple[np.ndarray, pd.DataFrame]:
    if not MAP_NPZ.exists():
        raise FileNotFoundError(MAP_NPZ)
    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)
    with np.load(MAP_NPZ, allow_pickle=False) as data:
        final_maps = np.maximum(np.asarray(data["final_maps"], dtype=np.float64), 0.0)
    return final_maps, pd.read_csv(METRICS_CSV)


def build_metric_table(final_maps: np.ndarray, base_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for unit in range(final_maps.shape[1]):
        row: dict[str, float | int | str] = {"unit_index": int(unit), "unit_label": f"u{unit:03d}"}
        for condition_idx, condition in enumerate(["real", "stable"]):
            metrics = map_sharpness_metrics(final_maps[condition_idx, unit])
            for key, value in metrics.items():
                row[f"{condition}_{key}"] = value
        rows.append(row)

    audit = pd.DataFrame(rows)
    for metric in [
        "mean_rate",
        "ssi_bits_per_spike",
        "entropy_sharpness",
        "inverse_participation",
        "top5_mass_fraction",
        "peak_to_mean",
        "high_frequency_power_fraction",
        "laplacian_rms",
    ]:
        audit[f"delta_{metric}"] = audit[f"real_{metric}"] - audit[f"stable_{metric}"]
    audit["delta_effective_area_fraction"] = (
        audit["stable_effective_area_fraction"] - audit["real_effective_area_fraction"]
    )
    audit["delta_effective_area_px"] = audit["stable_effective_area_px"] - audit["real_effective_area_px"]

    keep = [
        "unit_index",
        "sf_group",
        "orientation_group",
        "figure_candidate_score",
        "real_minus_stable_map_ssi",
    ]
    annotations = base_metrics[[col for col in keep if col in base_metrics.columns]].copy()
    return audit.merge(annotations, on="unit_index", how="left")


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("SSI bits/spike", "delta_ssi_bits_per_spike", "higher real"),
        ("entropy sharpness", "delta_entropy_sharpness", "higher real"),
        ("inverse participation", "delta_inverse_participation", "higher real"),
        ("effective area", "delta_effective_area_fraction", "lower real area"),
        ("top 5% mass", "delta_top5_mass_fraction", "higher real"),
        ("peak/mean", "delta_peak_to_mean", "higher real"),
        ("high-frequency power", "delta_high_frequency_power_fraction", "higher real"),
        ("Laplacian RMS", "delta_laplacian_rms", "higher real"),
    ]
    rows = []
    for label, col, direction in metric_specs:
        values = audit[col].astype(float)
        rows.append(
            {
                "metric": label,
                "delta_column": col,
                "positive_direction": direction,
                "n_real_sharper": int((values > 0).sum()),
                "n_stable_sharper": int((values < 0).sum()),
                "n_tied": int((values == 0).sum()),
                "median_delta": float(values.median()),
                "mean_delta": float(values.mean()),
                "q25_delta": float(values.quantile(0.25)),
                "q75_delta": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    labels = summary["metric"].tolist()
    y = np.arange(len(labels))
    real_counts = summary["n_real_sharper"].to_numpy(dtype=float)
    stable_counts = summary["n_stable_sharper"].to_numpy(dtype=float)
    ax.barh(y - 0.18, real_counts, height=0.34, color="#1e4ed8", label="real FEM sharper")
    ax.barh(y + 0.18, stable_counts, height=0.34, color="#777777", label="stabilized sharper")
    ax.axvline(50, color="#cdd1d6", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("units")
    ax.set_title("Direction by metric")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.scatter(
        audit["delta_ssi_bits_per_spike"],
        audit["delta_entropy_sharpness"],
        s=22,
        c=np.where(audit["unit_index"].eq(CURRENT_FIGURE_UNIT_INDEX), "#c51f27", "#333333"),
        alpha=0.78,
        linewidths=0,
    )
    ax.axhline(0, color="#bfc3c9", lw=0.9)
    ax.axvline(0, color="#bfc3c9", lw=0.9)
    ax.set_xlabel("Δ SSI bits/spike")
    ax.set_ylabel("Δ entropy sharpness")
    ax.set_title("SSI vs entropy concentration")
    current = audit[audit["unit_index"].eq(CURRENT_FIGURE_UNIT_INDEX)]
    if not current.empty:
        row = current.iloc[0]
        ax.annotate(
            f"u{CURRENT_FIGURE_UNIT_INDEX}",
            (row["delta_ssi_bits_per_spike"], row["delta_entropy_sharpness"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#c51f27",
        )

    ax = axes[1, 0]
    ax.scatter(
        audit["delta_effective_area_fraction"],
        audit["delta_top5_mass_fraction"],
        s=22,
        c=np.where(audit["unit_index"].eq(CURRENT_FIGURE_UNIT_INDEX), "#c51f27", "#333333"),
        alpha=0.78,
        linewidths=0,
    )
    ax.axhline(0, color="#bfc3c9", lw=0.9)
    ax.axvline(0, color="#bfc3c9", lw=0.9)
    ax.set_xlabel("stable - real effective area fraction")
    ax.set_ylabel("Δ top-5% mass")
    ax.set_title("Spatial concentration checks")

    ax = axes[1, 1]
    ax.scatter(
        audit["delta_entropy_sharpness"],
        audit["delta_high_frequency_power_fraction"],
        s=22,
        c=np.where(audit["unit_index"].eq(CURRENT_FIGURE_UNIT_INDEX), "#c51f27", "#333333"),
        alpha=0.78,
        linewidths=0,
    )
    ax.axhline(0, color="#bfc3c9", lw=0.9)
    ax.axvline(0, color="#bfc3c9", lw=0.9)
    ax.set_xlabel("Δ entropy sharpness")
    ax.set_ylabel("Δ high-frequency power fraction")
    ax.set_title("Concentration vs texture")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#eceff2", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("RR100 schematic final maps: real FEM minus endpoint-stabilized sharpness", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    final_maps, base_metrics = load_inputs()
    audit = build_metric_table(final_maps, base_metrics)
    summary = summarize(audit)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    plot_summary(audit, summary)

    print(OUT_CSV)
    print(OUT_SUMMARY_CSV)
    print(OUT_PNG)
    print(summary.to_string(index=False))
    current = audit[audit["unit_index"].eq(CURRENT_FIGURE_UNIT_INDEX)]
    if not current.empty:
        cols = [
            "unit_index",
            "delta_ssi_bits_per_spike",
            "delta_entropy_sharpness",
            "delta_inverse_participation",
            "delta_effective_area_fraction",
            "delta_top5_mass_fraction",
            "delta_peak_to_mean",
            "delta_high_frequency_power_fraction",
            "delta_laplacian_rms",
        ]
        print("\nCurrent figure unit:")
        print(current[cols].to_string(index=False))


if __name__ == "__main__":
    main()

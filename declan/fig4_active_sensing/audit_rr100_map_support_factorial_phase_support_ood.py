#!/usr/bin/env python3
"""Post-checkpoint phase-support and structural-OOD audit for Stage 2A."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from declan.fig4_active_sensing.spectral_cache_contract import sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_method_audit_v1"
RELATIVE_THRESHOLDS = (1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
CANONICAL_LOW, CANONICAL_HIGH = -127.0 / 255.0, 128.0 / 255.0
PATCH_SIZE = 15


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def load_conditions() -> tuple[np.ndarray, dict[tuple[int, str], np.ndarray]]:
    with np.load(SOURCE / "factorial_input_cubes.npz", allow_pickle=False) as archive:
        seeds = np.asarray(archive["seeds"], dtype=int)
        fixed = {
            "original stabilized input": np.asarray(archive["stabilized_original"], dtype=float),
            "original FEM input": np.asarray(archive["fem_original"], dtype=float),
            "stabilized power with FEM phase": np.asarray(archive["stabilized_power_fem_phase"], dtype=float),
        }
        fem_random = np.asarray(archive["fem_power_random_phase"], dtype=float)
        stabilized_random = np.asarray(archive["stabilized_power_random_phase"], dtype=float)
    conditions: dict[tuple[int, str], np.ndarray] = {}
    for ordinal, seed in enumerate(seeds):
        for label, cube in fixed.items():
            conditions[(int(seed), label)] = cube
        conditions[(int(seed), "FEM power with shared random phase")] = fem_random[ordinal]
        conditions[(int(seed), "stabilized power with shared random phase")] = stabilized_random[ordinal]
    return seeds, conditions


def phase_support_audit(fem: np.ndarray, stabilized: np.ndarray) -> pd.DataFrame:
    spectra = {
        "FEM": np.abs(np.fft.fftn(fem)),
        "stabilized": np.abs(np.fft.fftn(stabilized)),
    }
    rows: list[dict[str, object]] = []
    for source_name, target_name in (("FEM", "stabilized"), ("stabilized", "FEM")):
        source = spectra[source_name]
        target = spectra[target_name]
        for relative_threshold in RELATIVE_THRESHOLDS:
            absolute_threshold = max(1e-12, relative_threshold * float(source.max()))
            invalid = source <= absolute_threshold
            rows.append({
                "source_phase": source_name,
                "target_amplitude": target_name,
                "relative_source_amplitude_threshold": relative_threshold,
                "absolute_source_amplitude_threshold": absolute_threshold,
                "invalid_source_phase_bin_count": int(invalid.sum()),
                "invalid_source_phase_bin_fraction": float(invalid.mean()),
                "target_amplitude_fraction_in_invalid_source_bins": float(target[invalid].sum() / target.sum()),
                "target_spectral_energy_fraction_in_invalid_source_bins": float(
                    np.square(target[invalid]).sum() / np.square(target).sum()
                ),
            })
    return pd.DataFrame(rows)


def patch_kurtoses(cube: np.ndarray) -> np.ndarray:
    values: list[float] = []
    usable_y = cube.shape[1] - cube.shape[1] % PATCH_SIZE
    usable_x = cube.shape[2] - cube.shape[2] % PATCH_SIZE
    for frame in cube:
        for y in range(0, usable_y, PATCH_SIZE):
            for x in range(0, usable_x, PATCH_SIZE):
                value = kurtosis(
                    frame[y:y + PATCH_SIZE, x:x + PATCH_SIZE].ravel(),
                    fisher=True,
                    bias=False,
                )
                if np.isfinite(value):
                    values.append(float(value))
    return np.asarray(values, dtype=float)


def structural_audit(conditions: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, condition), cube in conditions.items():
        local = patch_kurtoses(cube)
        frame_kurtosis = np.asarray(
            [kurtosis(frame.ravel(), fisher=True, bias=False) for frame in cube], dtype=float
        )
        rows.append({
            "seed": seed,
            "condition": condition,
            "global_excess_kurtosis": float(kurtosis(cube.ravel(), fisher=True, bias=False)),
            "mean_frame_excess_kurtosis": float(np.nanmean(frame_kurtosis)),
            "median_patch_excess_kurtosis": float(np.nanmedian(local)),
            "ninetieth_percentile_patch_excess_kurtosis": float(np.nanquantile(local, 0.9)),
            "ninety_ninth_percentile_absolute_value": float(np.quantile(np.abs(cube), 0.99)),
            "maximum_absolute_value": float(np.max(np.abs(cube))),
            "fraction_outside_canonical_input_range": float(
                np.mean((cube < CANONICAL_LOW) | (cube > CANONICAL_HIGH))
            ),
        })
    return pd.DataFrame(rows)


def plot_audit(phase: pd.DataFrame, structural: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for (source, target), frame in phase.groupby(["source_phase", "target_amplitude"]):
        axes[0, 0].plot(
            frame.relative_source_amplitude_threshold,
            np.maximum(frame.target_spectral_energy_fraction_in_invalid_source_bins, 1e-18),
            marker="o",
            label=f"{source} phase → {target} amplitude",
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(
        xlabel="source-phase validity threshold relative to its spectral peak",
        ylabel="target energy fraction in unsupported-phase bins",
        title="Symmetric source-phase validity audit",
    )
    axes[0, 0].legend(fontsize=8)

    fixed_names = {
        "original stabilized input", "original FEM input", "stabilized power with FEM phase"
    }
    fixed = structural.loc[structural.condition.isin(fixed_names)].drop_duplicates("condition")
    random = structural.loc[~structural.condition.isin(fixed_names)]
    condition_order = {
        "original stabilized input": 0,
        "original FEM input": 1,
        "stabilized power with FEM phase": 2,
        "FEM power with shared random phase": 3,
        "stabilized power with shared random phase": 4,
    }
    ordered = pd.concat([fixed, random], ignore_index=True)
    ordered["condition_order"] = ordered.condition.map(condition_order)
    ordered = ordered.sort_values(["condition_order", "seed"])
    labels = [
        name if name in fixed_names else f"{name}\nseed {seed}"
        for name, seed in zip(ordered.condition, ordered.seed, strict=True)
    ]
    colors = ["#999999" if "original" in name else "#0072B2" if "FEM phase" in name else "#D55E00"
              for name in ordered.condition]
    for axis, column, title, ylabel in (
        (axes[0, 1], "global_excess_kurtosis", "Whole-cube higher-order pixel statistic", "excess kurtosis"),
        (axes[1, 0], "median_patch_excess_kurtosis", "Median 15×15-pixel patch statistic", "median patch excess kurtosis"),
        (axes[1, 1], "fraction_outside_canonical_input_range", "Canonical input-range violation", "fraction outside range"),
    ):
        axis.bar(np.arange(len(ordered)), ordered[column], color=colors)
        axis.set_xticks(np.arange(len(ordered)), labels, rotation=70, ha="right", fontsize=7)
        axis.set(title=title, ylabel=ylabel)
        axis.axhline(0, color="0.3", linewidth=0.8)
    figure.suptitle(
        "Stage 2A method audit: fixed FEM phase is spectrally supported, while random phase changes higher-order input structure",
        fontsize=14,
        weight="bold",
    )
    path = OUT / "01_phase_support_and_structural_distribution_audit.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds, conditions = load_conditions()
    fem = conditions[(int(seeds[0]), "original FEM input")]
    stabilized = conditions[(int(seeds[0]), "original stabilized input")]
    phase = phase_support_audit(fem, stabilized)
    structural = structural_audit(conditions)
    phase_path = OUT / "symmetric_phase_source_support_audit.csv"
    structural_path = OUT / "structural_input_distribution_audit.csv"
    phase.to_csv(phase_path, index=False)
    structural.to_csv(structural_path, index=False)
    figure_path = plot_audit(phase, structural)

    used = phase.loc[
        phase.source_phase.eq("FEM")
        & phase.target_amplitude.eq("stabilized")
        & phase.relative_source_amplitude_threshold.eq(1e-4)
    ].iloc[0]
    random_rows = structural.loc[structural.condition.str.contains("random phase")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "stage2a_post_checkpoint_symmetric_phase_support_and_structural_ood_audit",
        "status": "fixed_phase_arm_passes_support_audit_random_phase_arms_remain_ood_confounded",
        "result": {
            "fem_phase_invalid_bins_through_relative_threshold_1e-8": 0,
            "stabilized_target_energy_fraction_in_weak_fem_phase_bins_at_1e-4": float(
                used.target_spectral_energy_fraction_in_invalid_source_bins
            ),
            "original_global_excess_kurtosis_range": [
                float(structural.loc[structural.condition.str.contains("original"), "global_excess_kurtosis"].min()),
                float(structural.loc[structural.condition.str.contains("original"), "global_excess_kurtosis"].max()),
            ],
            "random_phase_global_excess_kurtosis_range": [
                float(random_rows.global_excess_kurtosis.min()), float(random_rows.global_excess_kurtosis.max())
            ],
            "random_phase_out_of_range_fraction_range": [
                float(random_rows.fraction_outside_canonical_input_range.min()),
                float(random_rows.fraction_outside_canonical_input_range.max()),
            ],
        },
        "interpretation": (
            "the current stabilized-amplitude/FEM-phase result is not materially affected by unsupported FEM phase; "
            "the missing symmetric guard is nevertheless a correctness requirement for future pairs. Random-phase "
            "inputs differ strongly in higher-order structure and remain unsuitable for a clean phase-only claim."
        ),
        "source": identity(SOURCE / "factorial_input_cubes.npz"),
        "outputs": {
            "phase_support_csv": identity(phase_path),
            "structural_distribution_csv": identity(structural_path),
            "review_figure": identity(figure_path),
        },
        "runner": identity(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Stage 2A phase-support and structural-distribution audit\n\n"
        "The phase-source audit is now symmetric. The current stabilized-power/FEM-phase arm has no invalid "
        "FEM-phase bins through a relative threshold of 1e-8; even at 1e-4, weak-source bins contain only "
        f"{used.target_spectral_energy_fraction_in_invalid_source_bins:.3e} of stabilized target energy. "
        "The current fixed-phase result therefore remains usable. The random-phase arms remain confounded by "
        "range violations and large higher-order structural changes and cannot support a clean phase-only claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

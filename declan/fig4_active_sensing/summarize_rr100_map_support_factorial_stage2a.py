#!/usr/bin/env python3
"""Human-review summary for the Stage 2A map-support factorial checkpoint."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.spectral_cache_contract import sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_review_v1"
LOW, HIGH = -127.0 / 255.0, 128.0 / 255.0
CONDITIONS = [
    ("stabilized_original", "original stabilized input"),
    ("fem_original", "original FEM input"),
    ("fem_power_random_phase", "FEM power with shared random phase"),
    ("stabilized_power_fem_phase", "stabilized power with FEM phase"),
    ("stabilized_power_random_phase", "stabilized power with shared random phase"),
]


def identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def rgb_with_out_of_range(values: np.ndarray) -> np.ndarray:
    normalized = np.clip((np.asarray(values, dtype=float) - LOW) / (HIGH - LOW), 0, 1)
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    rgb[values < LOW] = np.asarray([0.0, 0.92, 1.0])
    rgb[values > HIGH] = np.asarray([1.0, 0.0, 0.82])
    return rgb


def load_cubes() -> tuple[np.ndarray, dict[tuple[int, str], np.ndarray]]:
    with np.load(SOURCE / "factorial_input_cubes.npz", allow_pickle=False) as archive:
        seeds = np.asarray(archive["seeds"], dtype=int)
        fixed = {
            "stabilized_original": np.asarray(archive["stabilized_original"]),
            "fem_original": np.asarray(archive["fem_original"]),
            "stabilized_power_fem_phase": np.asarray(archive["stabilized_power_fem_phase"]),
        }
        fem_random = np.asarray(archive["fem_power_random_phase"])
        stabilized_random = np.asarray(archive["stabilized_power_random_phase"])
    cubes = {}
    for ordinal, seed in enumerate(seeds):
        for name, cube in fixed.items():
            cubes[(int(seed), name)] = cube
        cubes[(int(seed), "fem_power_random_phase")] = fem_random[ordinal]
        cubes[(int(seed), "stabilized_power_random_phase")] = stabilized_random[ordinal]
    return seeds, cubes


def plot_out_of_range(seeds: np.ndarray, cubes: dict[tuple[int, str], np.ndarray], audit: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(len(seeds), len(CONDITIONS), figsize=(18, 10), constrained_layout=True)
    for row, seed in enumerate(seeds):
        for column, (condition, label) in enumerate(CONDITIONS):
            cube = cubes[(int(seed), condition)]
            record = audit.loc[audit.seed.eq(seed) & audit.condition.eq(condition)].iloc[0]
            axes[row, column].imshow(rgb_with_out_of_range(cube[0]), origin="lower")
            axes[row, column].set_title(
                f"{label}\nfull-cube outside range={100*record.fraction_outside_canonical_input_range:.1f}%",
                fontsize=8.5,
            )
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"phase seed {seed}")
    figure.suptitle(
        "Stage 2A input-range audit: random phase creates substantial out-of-range structure\n"
        "Current-lag frame shown; percentages use each complete 32-lag cube; cyan is below range and magenta is above range",
        fontsize=15, weight="bold",
    )
    path = OUT / "01_input_range_overlay.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def summarize_effects(effects: pd.DataFrame) -> pd.DataFrame:
    return effects.groupby("effect", as_index=False).agg(
        n_unit_seed_pairs=("rr100_index", "size"),
        mean_rate_difference_hz=("difference__instantaneous_mean_rate_hz", "mean"),
        median_rate_difference_hz=("difference__instantaneous_mean_rate_hz", "median"),
        minimum_rate_difference_hz=("difference__instantaneous_mean_rate_hz", "min"),
        maximum_rate_difference_hz=("difference__instantaneous_mean_rate_hz", "max"),
        mean_ssi_difference_bits_per_spike=("difference__instantaneous_ssi_bits_per_spike", "mean"),
        median_ssi_difference_bits_per_spike=("difference__instantaneous_ssi_bits_per_spike", "median"),
        minimum_ssi_difference_bits_per_spike=("difference__instantaneous_ssi_bits_per_spike", "min"),
        maximum_ssi_difference_bits_per_spike=("difference__instantaneous_ssi_bits_per_spike", "max"),
    )


def plot_effect_review(effects: pd.DataFrame, audit: pd.DataFrame) -> Path:
    frame = effects.pivot_table(
        index=["seed", "rr100_index", "selection_role"], columns="effect",
        values=["difference__instantaneous_mean_rate_hz", "difference__instantaneous_ssi_bits_per_spike"],
    ).reset_index()
    original_rate = frame[("difference__instantaneous_mean_rate_hz", "original FEM minus original stabilized")]
    power_rate = frame[("difference__instantaneous_mean_rate_hz", "power effect under FEM phase")]
    original_ssi = frame[("difference__instantaneous_ssi_bits_per_spike", "original FEM minus original stabilized")]
    power_ssi = frame[("difference__instantaneous_ssi_bits_per_spike", "power effect under FEM phase")]
    random_phase_rate = frame[("difference__instantaneous_mean_rate_hz", "phase effect at FEM power")]
    random_phase_ssi = frame[("difference__instantaneous_ssi_bits_per_spike", "phase effect at FEM power")]

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    colors = {seed: color for seed, color in zip(sorted(effects.seed.unique()), ["#0072B2", "#D55E00", "#009E73"], strict=True)}
    point_colors = [colors[int(seed)] for seed in frame[("seed", "")]]
    for axis, x, y, title, label in (
        (axes[0, 0], original_rate, power_rate, "Rate: power contrast under FEM phase", "rate difference (Hz)"),
        (axes[0, 1], original_ssi, power_ssi, "SSI: power contrast under FEM phase", "SSI difference (bits/spike)"),
    ):
        axis.scatter(x, y, c=point_colors, alpha=0.8)
        low = float(min(x.min(), y.min())); high = float(max(x.max(), y.max()))
        axis.plot([low, high], [low, high], color="0.4", ls="--")
        axis.set(xlabel=f"original FEM − stabilized {label}", ylabel=f"FEM power − stabilized power {label}", title=title)

    seed_values = sorted(effects.seed.unique())
    seed_positions = {int(seed): index for index, seed in enumerate(seed_values)}
    for axis, values, title, xlabel in (
        (axes[0, 2], random_phase_rate, "Random-phase effect at FEM power", "FEM phase − random phase rate (Hz)"),
        (axes[1, 0], random_phase_ssi, "Random-phase effect on SSI at FEM power", "FEM phase − random phase SSI (bits/spike)"),
    ):
        for seed in seed_values:
            mask = frame[("seed", "")].to_numpy(int) == int(seed)
            axis.scatter(
                np.full(mask.sum(), seed_positions[int(seed)]), values[mask],
                color=colors[int(seed)], alpha=0.8,
            )
        axis.axhline(0, color="0.4", ls="--")
        axis.set_xticks(np.arange(len(seed_values)), [str(int(seed)) for seed in seed_values])
        axis.set(xlabel="predeclared phase seed", ylabel=xlabel, title=title)

    random_audit = audit.loc[audit.condition.str.contains("random_phase")]
    axes[1, 1].scatter(
        100 * random_audit.fraction_outside_canonical_input_range,
        random_audit.histogram_wasserstein_distance_from_same_power_original,
        c=[colors[int(seed)] for seed in random_audit.seed],
    )
    axes[1, 1].set(
        xlabel="complete-cube values outside canonical range (%)",
        ylabel="histogram Wasserstein distance",
        title="Random-phase input-distribution shift",
    )

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.98,
        "Checkpoint interpretation\n\n"
        "• The stabilized-power/FEM-phase arm stays close to the original stabilized input and map.\n\n"
        "• Therefore the clean power contrast under fixed FEM phase closely tracks the original FEM effect in this example.\n\n"
        "• Random phase changes local energy allocation, the pixel histogram, and the valid input range. Its seed-variable response is not a clean phase-only effect.\n\n"
        "• This is one frame, one development image–trace pair, and six pre-response-selected units.",
        va="top", fontsize=11, wrap=True,
    )
    figure.suptitle(
        "Stage 2A factorial review: a clean power contrast but a confounded random-phase contrast",
        fontsize=15, weight="bold",
    )
    path = OUT / "02_factorial_effect_review.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(SOURCE / "factorial_input_audit.csv")
    effects = pd.read_csv(SOURCE / "selected_unit_factorial_effects.csv")
    seeds, cubes = load_cubes()
    overlay = plot_out_of_range(seeds, cubes, audit)
    summary = summarize_effects(effects)
    summary_path = OUT / "factorial_effect_descriptive_summary.csv"
    summary.to_csv(summary_path, index=False)
    review = plot_effect_review(effects, audit)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "stage2a_human_review_checkpoint_complete_gate_not_passed",
        "conclusion": (
            "the fixed-FEM-phase amplitude contrast is interpretable and tracks the original FEM effect, but "
            "the shared-random-phase arms are confounded by 7-10% out-of-range values, histogram change, and "
            "near-complete redistribution of tiled local energy; do not interpret them as clean phase-only controls"
        ),
        "decision_gate": "do not expand Stage 2A or claim phase sufficiency; choose a separately labelled distribution-constrained phase control or proceed only with claims supported by the fixed-phase power contrast",
        "source": identity(SOURCE / "manifest.json"),
        "outputs": {
            "input_range_overlay": identity(overlay),
            "factorial_review": identity(review),
            "effect_summary": identity(summary_path),
        },
        "runner": identity(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Stage 2A human-review checkpoint\n\n"
        "The exact-power construction passed numerically. The fixed-FEM-phase amplitude contrast is "
        "interpretable and tracks the original FEM effect. The random-phase arms are not clean phase-only "
        "controls because they substantially alter the input range, histogram, and local energy allocation. "
        "The Stage 2A decision gate is therefore not passed.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Place each validated unit's Figure-4B path effect in preferred SF x TF space."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_031_n032_clean_history_snapshot_v1"
)
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_clean_history_rounds000_022_v4/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_clean_history_no_bottom_row_rounds000_022_v4_unit_assignments.csv"
)
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
MICROSACCADES = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected_microsaccade_audit_v1/"
    "corrected_scored_microsaccade_labels.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_unit_path_effect_sf_tf_plane_v3"
GROUPS = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
QCOLORS = {"sf_q1": "#0072B2", "sf_q2": "#009E73", "sf_q3": "#E69F00", "sf_q4": "#CC79A7"}
CONTEXTS = ("drift_only", "microsaccade")
CONTEXT_LABELS = {"drift_only": "Drift-only", "microsaccade": "Microsaccade-containing"}
N_BOOTSTRAP = 4000
SEED = 20260814


def image_fixed_effect_components(
    x: np.ndarray, y: np.ndarray, image_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-image slope numerator, denominator, and condition count."""
    unique = np.unique(image_id)
    numerator = np.zeros((len(unique), y.shape[1]), float)
    denominator = np.zeros(len(unique), float)
    count = np.zeros(len(unique), int)
    for row, image in enumerate(unique):
        take = image_id == image
        xx = x[take] - np.mean(x[take])
        yy = y[take] - np.mean(y[take], axis=0, keepdims=True)
        numerator[row] = xx @ yy
        denominator[row] = xx @ xx
        count[row] = int(take.sum())
    return numerator, denominator, count


def bootstrap_slopes(
    numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = numerator.sum(0) / denominator.sum()
    draws = np.empty((N_BOOTSTRAP, numerator.shape[1]), float)
    for start in range(0, N_BOOTSTRAP, 250):
        stop = min(start + 250, N_BOOTSTRAP)
        weights = rng.multinomial(
            numerator.shape[0], np.full(numerator.shape[0], 1 / numerator.shape[0]), size=stop - start
        )
        draws[start:stop] = (weights @ numerator) / (weights @ denominator)[:, None]
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return slope, low, high


def standardized_multiple_regression(frame: pd.DataFrame, response: str) -> dict[str, float]:
    clean = frame[[response, "preferred_sf_cpd", "preferred_tf_hz"]].dropna()
    y = clean[response].to_numpy(float)
    y = (y - y.mean()) / y.std(ddof=0)
    x = np.column_stack([
        np.log2(clean.preferred_sf_cpd.to_numpy(float)),
        np.log2(clean.preferred_tf_hz.to_numpy(float)),
    ])
    x = (x - x.mean(0)) / x.std(0, ddof=0)
    design = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    resid = y - fitted
    sigma2 = float(resid @ resid / (len(y) - design.shape[1]))
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    p = 2 * stats.t.sf(np.abs(t), len(y) - design.shape[1])
    return {
        "standardized_beta_log2_sf": float(coef[1]),
        "standardized_beta_log2_sf_p": float(p[1]),
        "standardized_beta_log2_tf": float(coef[2]),
        "standardized_beta_log2_tf_p": float(p[2]),
        "multiple_r2": float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)),
    }


def fixed_effect_slope(x: np.ndarray, y: np.ndarray, image_id: np.ndarray) -> float:
    xx = x - pd.Series(x).groupby(image_id).transform("mean").to_numpy()
    yy = y - pd.Series(y).groupby(image_id).transform("mean").to_numpy()
    return float(xx @ yy / (xx @ xx))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)

    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv")
    condition = condition.merge(
        traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin"]],
        on="trace_index", validate="many_to_one",
    )
    microsaccades = pd.read_csv(MICROSACCADES)
    condition = condition.merge(
        microsaccades[["trace_index", "scored_n_microsaccade_events"]],
        on="trace_index", validate="many_to_one",
    )
    condition["context"] = np.where(
        condition.scored_n_microsaccade_events.gt(0), "microsaccade", "drift_only"
    )
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index")
    strong_image = images.corrected_reconstruction_orientation_coherence.to_numpy(float) >= 0.20

    assignments = pd.read_csv(ASSIGNMENTS)
    assignments = assignments[
        assignments.recorded_validation_pass.astype(bool) & assignments.sf_quartile.isin(GROUPS)
    ].copy()
    unit_ids = assignments.rr100_index.to_numpy(int)
    models = pd.read_csv(MODELS)[[
        "rr100_index", "ccnorm", "preferred_sf_cpd", "preferred_tf_hz", "sf_fit_r2", "tf_fit_r2",
        "joint_parametric_surface_r2", "recorded_sf_curve_r_full_support",
    ]]
    units = assignments[["rr100_index", "sf_quartile"]].merge(models, on="rr100_index", validate="one_to_one")

    moving_ssi = np.load(ASSEMBLED / "moving_movie_ssi_bits_per_spike.npy", mmap_mode="r")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as archive:
        baseline_ssi = np.asarray(archive["movie_ssi_bits_per_spike"], float)
        baseline_spikes = np.asarray(archive["expected_spikes"], float)
    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")

    image_id = condition.image_index.to_numpy(int)
    path = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    delta = np.asarray(moving_ssi[:, unit_ids], float) - baseline_ssi[image_id][:, unit_ids]
    mean_moving_spikes = np.asarray(moving_spikes[:, unit_ids], float).mean(0)
    mean_baseline_spikes = baseline_spikes[:, unit_ids].mean(0)
    units["mean_expected_spikes_moving"] = mean_moving_spikes
    units["mean_expected_spikes_stabilized"] = mean_baseline_spikes

    context_meta: dict[str, dict[str, int]] = {}
    for context in CONTEXTS:
        use = strong_image[image_id] & condition.context.eq(context).to_numpy() & np.isfinite(path)
        numerator, denominator, image_counts = image_fixed_effect_components(path[use], delta[use], image_id[use])
        slope, low, high = bootstrap_slopes(numerator, denominator, rng)
        units[f"{context}_path_slope_bits_per_spike_per_arcmin"] = slope
        units[f"{context}_path_slope_ci_low"] = low
        units[f"{context}_path_slope_ci_high"] = high
        units[f"{context}_slope_ci_excludes_zero"] = (low > 0) | (high < 0)
        context_meta[context] = {
            "n_conditions": int(use.sum()), "n_images": int(len(image_counts)),
            "min_conditions_per_image": int(image_counts.min()), "max_conditions_per_image": int(image_counts.max()),
        }

    units.to_csv(OUT / "unit_path_effect_sf_tf_values.csv", index=False)

    # Exact leave-one-unit-out influence on Figure 4B's nonlinear pooled-spike estimand.
    drift_use = strong_image[image_id] & condition.context.eq("drift_only").to_numpy() & np.isfinite(path)
    influence_rows: list[dict[str, float | int | str]] = []
    baseline_info = baseline_ssi * baseline_spikes
    for group in GROUPS:
        group_ids = units.loc[units.sf_quartile.eq(group), "rr100_index"].to_numpy(int)

        def pooled_delta(selected: np.ndarray) -> np.ndarray:
            moving = np.asarray(moving_info[:, selected], float).sum(1) / np.maximum(
                np.asarray(moving_spikes[:, selected], float).sum(1), 1e-12
            )
            stabilized = baseline_info[:, selected].sum(1)[image_id] / np.maximum(
                baseline_spikes[:, selected].sum(1)[image_id], 1e-12
            )
            return moving - stabilized

        full = fixed_effect_slope(path[drift_use], pooled_delta(group_ids)[drift_use], image_id[drift_use])
        for unit in group_ids:
            remaining = group_ids[group_ids != unit]
            leave_one_out = fixed_effect_slope(
                path[drift_use], pooled_delta(remaining)[drift_use], image_id[drift_use]
            )
            influence_rows.append({
                "sf_quartile": group, "rr100_index": int(unit), "full_pooled_slope": full,
                "leave_one_out_slope": leave_one_out,
                "unit_influence_full_minus_leave_one_out": full - leave_one_out,
            })
    influence = pd.DataFrame(influence_rows)
    influence.to_csv(OUT / "panel_b_pooled_leave_one_unit_out_influence.csv", index=False)

    summary_rows = []
    for context in CONTEXTS:
        response = f"{context}_path_slope_bits_per_spike_per_arcmin"
        rho_sf, p_sf = stats.spearmanr(units.preferred_sf_cpd, units[response])
        rho_tf, p_tf = stats.spearmanr(units.preferred_tf_hz, units[response])
        row = {
            "context": context, "n_units": len(units),
            "spearman_rho_sf": rho_sf, "spearman_p_sf": p_sf,
            "spearman_rho_tf": rho_tf, "spearman_p_tf": p_tf,
            "n_unit_ci_excluding_zero": int(units[f"{context}_slope_ci_excludes_zero"].sum()),
        }
        row.update(standardized_multiple_regression(units, response))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "sf_tf_association_summary.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")
    for ax, context, letter in zip(axes[0], CONTEXTS, ("A", "B")):
        response = f"{context}_path_slope_bits_per_spike_per_arcmin"
        values = units[response].to_numpy(float) * 1e4
        limit = max(float(np.nanmax(np.abs(values))), 1e-8)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        sizes = 28 + 90 * np.sqrt(
            (units.mean_expected_spikes_moving + units.mean_expected_spikes_stabilized)
            / (units.mean_expected_spikes_moving + units.mean_expected_spikes_stabilized).max()
        )
        significant = units[f"{context}_slope_ci_excludes_zero"].to_numpy(bool)
        scatter = ax.scatter(
            units.preferred_sf_cpd, units.preferred_tf_hz, c=values, cmap=cmap, norm=norm, s=sizes,
            edgecolor=np.where(significant, "black", "white"), linewidth=np.where(significant, 1.35, 0.55), zorder=3,
        )
        for row in units.itertuples(index=False):
            ax.annotate(
                f"u{int(row.rr100_index):03d}", (row.preferred_sf_cpd, row.preferred_tf_hz),
                xytext=(3, 2), textcoords="offset points", fontsize=5.1, color="0.20",
            )
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks([1, 1.5, 2, 3, 4, 6], ["1", "1.5", "2", "3", "4", "6"])
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
        ax.set(xlabel="preferred SF (cycles/degree)", ylabel="preferred TF (Hz)")
        ax.set_title(f"{letter}  {CONTEXT_LABELS[context]} unit path effect", loc="left", weight="bold")
        cb = fig.colorbar(scatter, ax=ax, pad=0.01, shrink=0.78)
        cb.set_label("moving − stabilized path slope\n(×10⁻⁴ bits/spike/arcmin)")
        ax.text(0.01, 0.01, "black edge: image-bootstrap 95% CI excludes zero\nsize: expected-spike support", transform=ax.transAxes, fontsize=6.7)

    ax = axes[1, 0]
    drift = units.drift_only_path_slope_bits_per_spike_per_arcmin.to_numpy(float) * 1e4
    micro = units.microsaccade_path_slope_bits_per_spike_per_arcmin.to_numpy(float) * 1e4
    for group in GROUPS:
        take = units.sf_quartile.eq(group)
        ax.scatter(drift[take], micro[take], s=42, color=QCOLORS[group], edgecolor="white", lw=0.6, label=group.replace("sf_q", "Q"))
    low = float(min(drift.min(), micro.min())); high = float(max(drift.max(), micro.max()))
    ax.plot([low, high], [low, high], color="0.45", lw=0.9, ls="--")
    for row, x, y in zip(units.itertuples(index=False), drift, micro):
        ax.annotate(f"u{int(row.rr100_index):03d}", (x, y), xytext=(3, 2), textcoords="offset points", fontsize=5.1)
    rho, p = stats.spearmanr(drift, micro)
    ax.set(xlabel="drift-only unit slope (×10⁻⁴)", ylabel="microsaccade-containing unit slope (×10⁻⁴)")
    ax.set_title("C  Does unit ordering persist across trace contexts?", loc="left", weight="bold")
    ax.text(0.02, 0.96, f"Spearman ρ = {rho:.2f}, p = {p:.3g}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, ncol=4, loc="lower right")

    ax = axes[1, 1]
    y_positions = np.arange(len(units))
    ordered = units.sort_values("drift_only_path_slope_bits_per_spike_per_arcmin")
    effect = ordered.drift_only_path_slope_bits_per_spike_per_arcmin.to_numpy(float) * 1e4
    lo = ordered.drift_only_path_slope_ci_low.to_numpy(float) * 1e4
    hi = ordered.drift_only_path_slope_ci_high.to_numpy(float) * 1e4
    colors = ordered.sf_quartile.map(QCOLORS).to_numpy()
    ax.hlines(y_positions, lo, hi, color=colors, lw=1.0, alpha=0.65)
    ax.scatter(effect, y_positions, color=colors, s=24, edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(y_positions, [f"u{u:03d}" for u in ordered.rr100_index], fontsize=5.3)
    ax.set(xlabel="drift-only moving − stabilized path slope (×10⁻⁴ bits/spike/arcmin)")
    ax.set_title("D  Ranked unit effects with image-bootstrap intervals", loc="left", weight="bold")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    s_drift = summary.loc[summary.context.eq("drift_only")].iloc[0]
    fig.suptitle(
        "Validated RR100 unit path-length effects in preferred SF × TF space\n"
        f"strong-contour images · clean recorded history · matched stabilized baseline · n={len(units)} units\n"
        f"drift-only: ρSF={s_drift.spearman_rho_sf:.2f}, ρTF={s_drift.spearman_rho_tf:.2f}; "
        f"joint standardized βSF={s_drift.standardized_beta_log2_sf:.2f}, βTF={s_drift.standardized_beta_log2_tf:.2f}",
        fontsize=13, weight="bold",
    )
    stem = "rr100_unit_path_effect_sf_tf_plane"
    fig.savefig(OUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unit_path_effect_sf_tf_checkpoint_complete",
        "estimand": "per-unit slope of moving SSI minus matched-image stabilized SSI against calibrated scored path length, after image fixed-effect residualization",
        "scope": "strong-contour images (corrected orientation coherence >= 0.20), separated by corrected scored-window event context",
        "bootstrap": {"cluster": "image", "n": N_BOOTSTRAP, "seed": SEED},
        "n_units": int(len(units)), "contexts": context_meta,
        "sources": {"assembled": str(ASSEMBLED), "assignments": str(ASSIGNMENTS), "models": str(MODELS)},
        "outputs": {
            "png": str((OUT / f"{stem}.png").resolve()), "pdf": str((OUT / f"{stem}.pdf").resolve()),
            "unit_values": str((OUT / "unit_path_effect_sf_tf_values.csv").resolve()),
            "association_summary": str((OUT / "sf_tf_association_summary.csv").resolve()),
            "pooled_leave_one_unit_out_influence": str((OUT / "panel_b_pooled_leave_one_unit_out_influence.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

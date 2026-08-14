#!/usr/bin/env python3
"""Figure 02: one retinal spectrum routed through heterogeneous RR100 passbands."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/data"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/02_unit_filtering"


def select_units(table: pd.DataFrame) -> pd.DataFrame:
    candidates = table[table.routing_quality_pass].copy()
    used: set[int] = set()
    selected: list[pd.Series] = []

    def add(role: str, metric: str, largest: bool, criterion: str) -> None:
        frame = candidates[~candidates.rr100_index.isin(used)].dropna(subset=[metric])
        if frame.empty:
            return
        row = frame.loc[frame[metric].idxmax() if largest else frame[metric].idxmin()].copy()
        row["selection_role"] = role
        row["selection_metric"] = metric
        row["selection_value"] = float(row[metric])
        row["selection_criterion"] = criterion
        selected.append(row)
        used.add(int(row.rr100_index))

    # Use sampled in-support peaks for example roles. Parametric centers can
    # legitimately extrapolate outside the measured grid for edge-peaking
    # curves, but those extrapolations are not measured preferences.
    add("lowest-SF passband", "extended_sf_sampled_preferred_cpd", False, "minimum sampled SF peak inside the measured grid")
    add("highest-SF passband", "extended_sf_sampled_preferred_cpd", True, "maximum sampled SF peak inside the measured grid")
    add("highest-TF passband", "extended_tf_sampled_preferred_hz", True, "maximum sampled TF peak inside the measured grid among remaining units")
    interior = candidates[
        candidates.extended_sf_preferred_within_support.astype(bool)
        & candidates.extended_tf_preferred_within_support.astype(bool)
    ]
    if not interior.empty:
        original = candidates
        candidates = interior
        add("broad-TF passband", "extended_tf_fwhm_octaves", True, "maximum extended TF FWHM among units with in-support parametric centers")
        candidates = original
    else:
        add("broad-TF passband", "extended_tf_fwhm_octaves", True, "maximum extended TF FWHM among remaining units")
    return pd.DataFrame(selected)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "power_routing_joined_arrays.npz", allow_pickle=False) as data:
        units = np.asarray(data["rr100_index"], dtype=int)
        sf = np.asarray(data["sf_centers_cpd"], dtype=float)
        tf = np.asarray(data["tf_hz"], dtype=float)
        power = np.asarray(data["supported_retinal_power"], dtype=float)
        sensitivity = np.asarray(data["normalized_unit_sensitivity"], dtype=float)
        routed_power = np.asarray(data["routed_power"], dtype=float)
        routed_amp = np.asarray(data["routed_amplitude"], dtype=float)
        gain_amp = np.asarray(data["gain_weighted_routed_amplitude"], dtype=float)
        band_power = np.asarray(data["routed_band_power"], dtype=float)
        image_ids = np.asarray(data["image_index"], dtype=int)
        trace_ids = np.asarray(data["trace_index"], dtype=int)
    unit_table = pd.read_csv(DATA / "routing_unit_cohort.csv")
    condition_table = pd.read_csv(DATA / "routing_condition_table.csv")

    flattened = power.reshape(len(power), -1)
    normalized = flattened / np.maximum(flattened.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    entropy = -(normalized * np.log2(np.maximum(normalized, np.finfo(float).tiny))).sum(axis=1)
    amplitude = condition_table.global_power_amplitude.to_numpy(float)
    central = (amplitude >= np.quantile(amplitude, 0.25)) & (amplitude <= np.quantile(amplitude, 0.75))
    condition_index = int(np.flatnonzero(central)[np.argmax(entropy[central])])
    selected_condition = condition_table.iloc[[condition_index]].copy()
    selected_condition["selection_role"] = "broad-spectrum typical-amplitude input"
    selected_condition["selection_criterion"] = "maximum spectral entropy among middle 50% of global amplitudes"
    selected_condition["spectral_entropy_bits"] = entropy[condition_index]
    selected_condition.to_csv(OUT / "selected_input_condition.csv", index=False)

    selected = select_units(unit_table)
    selected.to_csv(OUT / "selected_filter_units.csv", index=False)
    position = {int(unit): idx for idx, unit in enumerate(units)}
    p = power[condition_index]

    fig = plt.figure(figsize=(14.5, 3.0 + 3.1 * len(selected)), constrained_layout=True)
    gs = fig.add_gridspec(len(selected) + 1, 4, height_ratios=[1.0] + [1.25] * len(selected))
    ax = fig.add_subplot(gs[0, 0])
    p_db = 10 * np.log10(np.maximum(p / max(float(p.max()), np.finfo(float).tiny), 1e-5))
    im = ax.pcolormesh(sf, tf, p_db, shading="nearest", cmap="magma", vmin=-50, vmax=0)
    ax.set_xscale("log")
    ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title="One observed FEM power map")
    fig.colorbar(im, ax=ax, label="within-map power (dB)")
    ax = fig.add_subplot(gs[0, 1:])
    ax.axis("off")
    ax.text(0.02, 0.72, r"Shape-only routing:  $R_{u,c}^{2}=\sum P_c(SF,TF)\,\widetilde H_u(SF,TF)^2$", fontsize=15, weight="bold")
    ax.text(0.02, 0.34, "The retinal movie is held fixed below. Only the independently measured unit passband changes.\n"
            "Map displays are normalized for visibility; annotated routing scalars use the unnormalized saved power.", fontsize=11)
    ax.text(0.98, 0.72, f"image {image_ids[condition_index]} · trace {trace_ids[condition_index]}", ha="right", fontsize=11)

    for row_index, row in enumerate(selected.itertuples(index=False), start=1):
        unit = int(row.rr100_index)
        unit_pos = position[unit]
        h2 = sensitivity[unit_pos] ** 2
        overlap = p * h2
        ax = fig.add_subplot(gs[row_index, 0])
        hm = ax.pcolormesh(sf, tf, h2, shading="nearest", cmap="magma", vmin=0, vmax=1)
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title=f"RR100 {unit}: {row.selection_role}\nnormalized sensitivity²")
        fig.colorbar(hm, ax=ax, label=r"$\widetilde H_u^2$")

        ax = fig.add_subplot(gs[row_index, 1])
        ov_db = 10 * np.log10(np.maximum(overlap / max(float(overlap.max()), np.finfo(float).tiny), 1e-5))
        om = ax.pcolormesh(sf, tf, ov_db, shading="nearest", cmap="magma", vmin=-50, vmax=0)
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title=r"Routed map: $P\times\widetilde H_u^2$")
        fig.colorbar(om, ax=ax, label="within-map routed power (dB)")

        ax = fig.add_subplot(gs[row_index, 2])
        values = band_power[condition_index, unit_pos]
        fractions = values / max(float(values.sum()), np.finfo(float).tiny)
        ax.bar(["≤32", "33–45", "46–56"], 100 * fractions, color=["#0072B2", "#E69F00", "#D55E00"])
        ax.set_ylim(0, 100)
        ax.set(xlabel="temporal band (Hz)", ylabel="routed variance fraction (%)", title="Where this unit's predicted drive comes from")

        ax = fig.add_subplot(gs[row_index, 3])
        ax.axis("off")
        sf_label = f"sampled SF peak = {row.extended_sf_sampled_preferred_cpd:.2f} cpd"
        tf_label = f"sampled TF peak = {row.extended_tf_sampled_preferred_hz:.1f} Hz"
        ax.text(0.02, 0.83, sf_label, fontsize=11)
        ax.text(0.02, 0.64, tf_label, fontsize=11)
        ax.text(0.02, 0.43, f"shape-only routed amplitude\nR = {routed_amp[condition_index, unit_pos]:.2e} a.u.", fontsize=11, weight="bold")
        ax.text(0.02, 0.18, f"including measured F0 gain\nD = {gain_amp[condition_index, unit_pos]:.2e} a.u.", fontsize=11)
    fig.suptitle("Figure 02 — The same FEM-created power is routed differently by heterogeneous unit passbands", fontsize=15, weight="bold")
    fig.savefig(OUT / "figure02_unit_specific_spectral_routing.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "figure02_unit_specific_spectral_routing.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "figure02_unit_filtering_complete",
        "n_available_quality_units": int(len(units)),
        "selected_condition_row": condition_index,
        "selected_units": selected.rr100_index.astype(int).tolist(),
        "visible_claim": "one retinal power map produces different routed maps and band contributions under different measured unit filters",
        "not_tested": "whether routed amplitude predicts the observed unit response or SSI",
        "normalization_guardrail": "H is shape-normalized for routing; native F0 gain is shown separately",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure 03: auditable successes and dissociations for routing vs response."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/data"
CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/03_response_examples"


def percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order / max(len(values) - 1, 1)


def select_examples(routing: np.ndarray, outcome: np.ndarray, units: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for u_pos, unit in enumerate(units):
        rp = percentile(routing[:, u_pos])
        yp = percentile(outcome[:, u_pos])
        for c in range(len(rp)):
            candidates.append({"condition_row": c, "unit_position": u_pos, "rr100_index": int(unit), "routing_percentile": rp[c], "response_percentile": yp[c]})
    frame = pd.DataFrame(candidates)
    used_conditions: set[int] = set()
    used_units: set[int] = set()

    def choose(role: str, mask: np.ndarray, objective: np.ndarray, criterion: str) -> None:
        available = frame[mask & ~frame.condition_row.isin(used_conditions) & ~frame.rr100_index.isin(used_units)].copy()
        if available.empty:
            available = frame[mask & ~frame.condition_row.isin(used_conditions)].copy()
        if available.empty:
            available = frame.copy()
        row = available.iloc[int(np.argmax(objective[available.index]))].to_dict()
        row.update(selection_role=role, selection_criterion=criterion)
        rows.append(row)
        used_conditions.add(int(row["condition_row"]))
        used_units.add(int(row["rr100_index"]))

    agreement = 0.5 * (frame.routing_percentile.to_numpy() + frame.response_percentile.to_numpy()) - np.abs(frame.routing_percentile - frame.response_percentile).to_numpy()
    choose("routing and response agree", (frame.routing_percentile >= 0.8).to_numpy() & (frame.response_percentile >= 0.8).to_numpy(), agreement,
           "both within-unit percentiles >=80%; maximize joint rank agreement")
    mismatch = frame.routing_percentile.to_numpy() - frame.response_percentile.to_numpy()
    choose("routing overpredicts", (frame.routing_percentile >= 0.8).to_numpy() & (frame.response_percentile <= 0.3).to_numpy(), mismatch,
           "routing >=80% and response <=30%; maximize percentile gap")
    converse = frame.response_percentile.to_numpy() - frame.routing_percentile.to_numpy()
    choose("response exceeds routing prediction", (frame.response_percentile >= 0.8).to_numpy() & (frame.routing_percentile <= 0.3).to_numpy(), converse,
           "response >=80% and routing <=30%; maximize percentile gap")
    return pd.DataFrame(rows)


def load_timecourses(round_id: int, image_id: int, trace_id: int, unit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    moving_path = CACHE / "moving" / f"round_{round_id:03d}" / f"image_{image_id:03d}.npz"
    baseline_path = CACHE / "baselines" / f"image_{image_id:03d}.npz"
    with np.load(moving_path, allow_pickle=False) as moving:
        trace_ids = moving["trace_index"].astype(int)
        positions = np.flatnonzero(trace_ids == trace_id)
        if len(positions) != 1:
            raise ValueError(f"Could not uniquely locate trace {trace_id} in {moving_path}")
        position = int(positions[0])
        moving_rate = moving["rate_timecourse_hz"][position, :, unit].astype(float)
        moving_ssi = moving["instantaneous_ssi_bits_per_spike"][position, :, unit].astype(float)
    with np.load(baseline_path, allow_pickle=False) as baseline:
        baseline_rate = baseline["rate_timecourse_hz"][:, unit].astype(float)
        baseline_ssi = baseline["instantaneous_ssi_bits_per_spike"][:, unit].astype(float)
    return moving_rate, baseline_rate, moving_ssi, baseline_ssi


def relative_db(values: np.ndarray) -> np.ndarray:
    maximum = max(float(np.nanmax(values)), np.finfo(float).tiny)
    return 10 * np.log10(np.maximum(values / maximum, 1e-5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "power_routing_joined_arrays.npz", allow_pickle=False) as data:
        d = {key: np.asarray(data[key]) for key in data.files}
    units = d["rr100_index"].astype(int)
    selections = select_examples(d["routed_amplitude"], d["temporal_rms_delta_from_stabilized_hz"], units)
    detail_rows: list[dict[str, object]] = []
    for row in selections.itertuples(index=False):
        c = int(row.condition_row)
        u_pos = int(row.unit_position)
        detail_rows.append(
            {
                **row._asdict(),
                "image_index": int(d["image_index"][c]),
                "trace_index": int(d["trace_index"][c]),
                "round_index": int(d["round_index"][c]),
                "routed_amplitude": float(d["routed_amplitude"][c, u_pos]),
                "global_power_amplitude": float(d["global_power_amplitude"][c]),
                "activation_rms_hz": float(d["temporal_rms_delta_from_stabilized_hz"][c, u_pos]),
                "delta_mean_rate_hz": float(d["delta_mean_rate_hz"][c, u_pos]),
                "delta_ssi_bits_per_spike": float(d["delta_ssi_bits_per_spike"][c, u_pos]),
            }
        )
    selected = pd.DataFrame(detail_rows)
    selected.to_csv(OUT / "selected_response_examples.csv", index=False)

    sf = d["sf_centers_cpd"].astype(float)
    tf = d["tf_hz"].astype(float)
    fig = plt.figure(figsize=(17, 3.0 + 3.2 * len(selected)), constrained_layout=True)
    gs = fig.add_gridspec(len(selected) + 1, 5, height_ratios=[0.65] + [1.0] * len(selected))
    header = fig.add_subplot(gs[0, :])
    header.axis("off")
    header.text(0.01, 0.72, "Prediction is now compared with the frozen model response", fontsize=16, weight="bold")
    header.text(0.01, 0.25, "Examples are selected by explicit within-unit ranks and include agreement plus both directions of dissociation.\n"
                "The spectral construction has no fitted access to the natural-movie response.", fontsize=11)

    for plot_row, row in enumerate(selected.itertuples(index=False), start=1):
        c = int(row.condition_row)
        u_pos = int(row.unit_position)
        unit = int(row.rr100_index)
        power = d["supported_retinal_power"][c]
        h2 = d["normalized_unit_sensitivity"][u_pos] ** 2
        overlap = power * h2
        moving_rate, baseline_rate, moving_ssi, baseline_ssi = load_timecourses(
            int(row.round_index), int(row.image_index), int(row.trace_index), unit
        )
        time_ms = np.arange(len(moving_rate)) / 120.0 * 1000.0

        ax = fig.add_subplot(gs[plot_row, 0])
        im = ax.pcolormesh(sf, tf, relative_db(power), shading="nearest", cmap="magma", vmin=-50, vmax=0)
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title=f"{row.selection_role}\ninput power")
        fig.colorbar(im, ax=ax, label="relative dB")

        ax = fig.add_subplot(gs[plot_row, 1])
        im = ax.pcolormesh(sf, tf, h2, shading="nearest", cmap="magma", vmin=0, vmax=1)
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title=f"RR100 {unit} filter²")
        fig.colorbar(im, ax=ax, label="normalized sensitivity²")

        ax = fig.add_subplot(gs[plot_row, 2])
        im = ax.pcolormesh(sf, tf, relative_db(overlap), shading="nearest", cmap="magma", vmin=-50, vmax=0)
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)", title=f"routed power\nR percentile {100*row.routing_percentile:.0f}%")
        fig.colorbar(im, ax=ax, label="relative dB")

        ax = fig.add_subplot(gs[plot_row, 3])
        ax.plot(time_ms, baseline_rate, color="0.35", lw=2, label="stabilized")
        ax.plot(time_ms, moving_rate, color="#D55E00", lw=2, label="FEM")
        ax.set(xlabel="scored time (ms)", ylabel="rate (Hz)", title=f"activation: RMS Δ={row.activation_rms_hz:.3f} Hz\nmean Δ={row.delta_mean_rate_hz:+.3f} Hz")
        ax.legend(frameon=False, fontsize=9)

        ax = fig.add_subplot(gs[plot_row, 4])
        ax.plot(time_ms, baseline_ssi, color="0.35", lw=2, label="stabilized")
        ax.plot(time_ms, moving_ssi, color="#0072B2", lw=2, label="FEM")
        ax.set(xlabel="scored time (ms)", ylabel="instantaneous SSI (bits/spike)", title=f"movie-level ΔSSI={row.delta_ssi_bits_per_spike:+.3f}\nresponse percentile {100*row.response_percentile:.0f}%")
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Figure 03 — Spectral routing can agree with the response, but the dissociations are equally important", fontsize=15, weight="bold")
    fig.savefig(OUT / "figure03_routing_response_examples.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "figure03_routing_response_examples.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "figure03_response_examples_complete",
        "n_available_quality_units": int(len(units)),
        "selected_roles": selected.selection_role.tolist(),
        "primary_response": "temporal RMS of FEM-minus-stabilized rate timecourses",
        "ssi_guardrail": "instantaneous SSI is shown diagnostically; annotations use separately computed movie-level SSI",
        "selection_guardrail": "selection criteria and values saved before population summarization",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

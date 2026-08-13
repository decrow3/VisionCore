#!/usr/bin/env python3
"""Checkpoint 22b: RR100 response maps for genuine versus synthetic history."""

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

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import load_dset
from declan.fig4_active_sensing.make_rr100_explicit_history_input_checkpoint import (
    IMAGE_INDEX,
    N_HISTORY,
    N_SCORE,
    SOURCE_RUN,
    TRACE_INDICES,
    explicit_segments,
    render_aligned,
)
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import (
    HALF_ASSIGNMENTS,
    MAPPING,
    corrected_patch,
    file_identity,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import RR100_MOVIE_MEDOID_VERSION
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
INPUT_CHECKPOINT = ROOT / "outputs/fig4_active_sensing/rr100_explicit_history_input_checkpoint_22a_v2"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_explicit_history_response_maps_checkpoint_22b_v1"
GROUPS = ("sf_low_half", "sf_high_half")
TRACE_PLOT = (TRACE_INDICES[0], TRACE_INDICES[-1])
DT = 1.0 / 120.0
EPS = 1e-10


def score_maps(scorer, view, stim) -> np.ndarray:
    normalized = (stim - 127.0) / 255.0
    full = scorer._compute_rate_map_batched(normalized)
    rr100 = apply_population_view(full, view).clamp_min(0.0)
    out = rr100.detach().cpu().numpy().astype(np.float32, copy=False)
    del normalized, full, rr100
    if scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    if out.shape[:2] != (N_SCORE, 100):
        raise ValueError(f"Unexpected response shape {out.shape}")
    return out


def map_metrics(maps: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(maps, dtype=np.float64).reshape(N_SCORE, 100, -1)
    rate = flat.mean(axis=2)
    gain = flat / np.maximum(rate[..., None], EPS)
    instantaneous_ssi = np.mean(gain * np.log2(np.maximum(gain, EPS)), axis=2)
    expected_t = rate * DT
    numerator = instantaneous_ssi * expected_t
    return {
        "instantaneous_ssi": instantaneous_ssi,
        "movie_ssi": numerator.sum(axis=0) / np.maximum(expected_t.sum(axis=0), EPS),
        "expected_spikes": expected_t.sum(axis=0),
        "mean_rate_hz": rate.mean(axis=0),
    }


def select_roles(unit_summary: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for group in GROUPS:
        sub = unit_summary[unit_summary.sf_outer_third.eq(group)].copy()
        used: set[int] = set()
        definitions = [
            ("largest_positive_ssi_history_effect", "mean_genuine_minus_synthetic_ssi", False),
            ("largest_negative_ssi_history_effect", "mean_genuine_minus_synthetic_ssi", True),
        ]
        for role, column, ascending in definitions:
            ordered = sub.sort_values(column, ascending=ascending)
            row = ordered.loc[~ordered.rr100_index.astype(int).isin(used)].iloc[0].copy()
            used.add(int(row.rr100_index))
            row["selection_role"] = f"{group}_{role}"
            row["selection_criterion"] = f"{'minimum' if ascending else 'maximum'} {column} within recorded-validated half"
            row["selection_criterion_value"] = float(row[column])
            chosen.append(row)
        target = float(sub.mean_absolute_response_map_difference_hz.median())
        ordered = sub.assign(
            distance_to_control_median=(sub.mean_absolute_response_map_difference_hz - target).abs()
        ).sort_values("distance_to_control_median")
        row = ordered.loc[~ordered.rr100_index.astype(int).isin(used)].iloc[0].copy()
        row["selection_role"] = f"{group}_median_map_change_control"
        row["selection_criterion"] = "closest to within-half median mean absolute response-map history difference"
        row["selection_criterion_value"] = float(row.mean_absolute_response_map_difference_hz)
        chosen.append(row)
    out = pd.DataFrame(chosen).reset_index(drop=True)
    out["selection_is_algorithmic"] = True
    return out


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SOURCE_RUN / "summary.json").read_text())
    source_rows = load_source_rows(Path(summary["source_csv"]))
    images = pd.read_csv(SOURCE_RUN / "image_feature_table.csv")
    traces = pd.read_csv(SOURCE_RUN / "trace_feature_table.csv")
    assignments = pd.read_csv(HALF_ASSIGNMENTS)
    dset_cache, canvas_cache = {}, {}

    image_row = images.loc[images.image_index.eq(IMAGE_INDEX)].iloc[0]
    image_source = source_row_by_id(source_rows, int(image_row.source_row))
    image_dset = load_dset(str(image_source.session), dset_cache)
    patch, patch_meta, _ = corrected_patch(image_source, image_dset, canvas_cache)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(MAPPING).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping.canonical_channel.to_numpy(int)):
        raise ValueError("RR100 mapping mismatch")
    scorer = CanonicalTwinScorer(device="cuda:0", batch_size=16, empty_cache_every_batch=True)

    stimuli, trace_info = {}, []
    for trace_index in TRACE_INDICES:
        row = traces.loc[traces.trace_index.eq(trace_index)].iloc[0]
        source = source_row_by_id(source_rows, int(row.trace_source_row))
        dset = load_dset(str(source.session), dset_cache)
        history, score, bounds = explicit_segments(source, dset)
        conditions = {
            "genuine": np.concatenate([history, score], axis=0),
            "synthetic": np.concatenate([score[:N_HISTORY], score], axis=0),
        }
        for condition, trace72 in conditions.items():
            aligned = render_aligned(scorer.common, patch, trace72, float(patch_meta["patch_ppd"]))
            stimuli[(int(trace_index), condition)] = aligned[N_HISTORY:].clone()
        trace_info.append({
            "trace_index": int(trace_index), "source_row": int(row.trace_source_row),
            "session": str(source.session),
            "scored_path_arcmin": float(np.linalg.norm(np.diff(score, axis=0), axis=1).sum() * 60.0),
            **bounds,
        })

    rows = []
    for trace_index in TRACE_INDICES:
        print(f"scoring trace {trace_index}: synthetic history", flush=True)
        synthetic_maps = score_maps(scorer, view, stimuli[(trace_index, "synthetic")])
        print(f"scoring trace {trace_index}: genuine history", flush=True)
        genuine_maps = score_maps(scorer, view, stimuli[(trace_index, "genuine")])
        sm, gm = map_metrics(synthetic_maps), map_metrics(genuine_maps)
        delta = genuine_maps - synthetic_maps
        for unit in range(100):
            rows.append({
                "trace_index": int(trace_index), "rr100_index": unit,
                "genuine_ssi_bits_per_spike": float(gm["movie_ssi"][unit]),
                "synthetic_ssi_bits_per_spike": float(sm["movie_ssi"][unit]),
                "genuine_minus_synthetic_ssi": float(gm["movie_ssi"][unit] - sm["movie_ssi"][unit]),
                "genuine_mean_rate_hz": float(gm["mean_rate_hz"][unit]),
                "synthetic_mean_rate_hz": float(sm["mean_rate_hz"][unit]),
                "genuine_minus_synthetic_mean_rate_hz": float(gm["mean_rate_hz"][unit] - sm["mean_rate_hz"][unit]),
                "genuine_expected_spikes": float(gm["expected_spikes"][unit]),
                "synthetic_expected_spikes": float(sm["expected_spikes"][unit]),
                "mean_absolute_response_map_difference_hz": float(np.mean(np.abs(delta[:, unit]))),
                "maximum_absolute_response_map_difference_hz": float(np.max(np.abs(delta[:, unit]))),
            })
        del synthetic_maps, genuine_maps, delta
    metrics = pd.DataFrame(rows).merge(
        assignments[["rr100_index", "sf_outer_third", "preferred_sf_cpd", "preferred_tf_hz", "recorded_sf_curve_r_full_support"]],
        on="rr100_index", how="left", validate="many_to_one",
    )
    metrics.to_csv(OUT / "all_unit_history_response_metrics.csv", index=False)
    unit_summary = metrics.groupby(
        ["rr100_index", "sf_outer_third", "preferred_sf_cpd", "preferred_tf_hz", "recorded_sf_curve_r_full_support"],
        as_index=False, dropna=False,
    ).agg(
        mean_genuine_minus_synthetic_ssi=("genuine_minus_synthetic_ssi", "mean"),
        mean_absolute_ssi_history_effect=("genuine_minus_synthetic_ssi", lambda x: float(np.mean(np.abs(x)))),
        mean_genuine_minus_synthetic_rate_hz=("genuine_minus_synthetic_mean_rate_hz", "mean"),
        mean_absolute_response_map_difference_hz=("mean_absolute_response_map_difference_hz", "mean"),
        maximum_absolute_response_map_difference_hz=("maximum_absolute_response_map_difference_hz", "max"),
    )
    unit_summary.to_csv(OUT / "unit_history_response_summary.csv", index=False)
    roles = select_roles(unit_summary[unit_summary.sf_outer_third.isin(GROUPS)])
    roles.to_csv(OUT / "selected_unit_roles.csv", index=False)
    selected = roles.rr100_index.to_numpy(int)

    selected_maps = {}
    selected_instantaneous = {}
    for trace_index in TRACE_PLOT:
        for condition in ("synthetic", "genuine"):
            print(f"rescoring selected-map cache trace {trace_index}: {condition}", flush=True)
            maps = score_maps(scorer, view, stimuli[(trace_index, condition)])[:, selected]
            selected_maps[(trace_index, condition)] = maps
            flat = maps.reshape(N_SCORE, len(selected), -1).astype(np.float64)
            rate = flat.mean(axis=2)
            gain = flat / np.maximum(rate[..., None], EPS)
            selected_instantaneous[(trace_index, condition)] = np.mean(
                gain * np.log2(np.maximum(gain, EPS)), axis=2
            ).astype(np.float32)

    fig, axes = plt.subplots(len(selected), 6, figsize=(15.2, 2.45 * len(selected)), constrained_layout=True)
    for row, role in roles.iterrows():
        for block, trace_index in enumerate(TRACE_PLOT):
            synthetic = selected_maps[(trace_index, "synthetic")][:, row]
            genuine = selected_maps[(trace_index, "genuine")][:, row]
            difference = genuine - synthetic
            frame = int(np.argmax(np.mean(np.abs(difference), axis=(1, 2))))
            rate_max = max(float(np.percentile(synthetic[frame], 99)), float(np.percentile(genuine[frame], 99)), 1e-5)
            diff_limit = max(float(np.percentile(np.abs(difference[frame]), 99)), 1e-5)
            start = block * 3
            rate_image = axes[row, start].imshow(synthetic[frame], cmap="magma", vmin=0, vmax=rate_max, interpolation="nearest")
            axes[row, start + 1].imshow(genuine[frame], cmap="magma", vmin=0, vmax=rate_max, interpolation="nearest")
            diff_image = axes[row, start + 2].imshow(
                difference[frame], cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit), interpolation="nearest",
            )
            ssi_s = selected_instantaneous[(trace_index, "synthetic")][frame, row]
            ssi_g = selected_instantaneous[(trace_index, "genuine")][frame, row]
            axes[row, start].set_title(f"trace {trace_index} · frame {frame}\nsynthetic · SSI {ssi_s:.3f}", fontsize=8)
            axes[row, start + 1].set_title(f"genuine · SSI {ssi_g:.3f}", fontsize=8)
            axes[row, start + 2].set_title(f"genuine − synthetic\nΔSSI {ssi_g - ssi_s:+.3f}", fontsize=8)
            fig.colorbar(rate_image, ax=axes[row, start:start + 2], shrink=0.60, pad=0.005, label="rate (Hz)")
            fig.colorbar(diff_image, ax=axes[row, start + 2], shrink=0.60, pad=0.005, label="Δ rate (Hz)")
        for col in range(6):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        label = str(role.sf_outer_third).replace("sf_", "").replace("_", " ")
        short_role = str(role.selection_role).replace(str(role.sf_outer_third) + "_", "").replace("_history_effect", "").replace("_", " ")
        axes[row, 0].set_ylabel(
            f"u{int(role.rr100_index):03d}\n{label}; {role.preferred_sf_cpd:.2f} cpd\n{short_role}", fontsize=8
        )
    fig.suptitle(
        "Checkpoint 22b — RR100 response maps under genuine versus synthetic prehistory\n"
        "algorithmic positive, negative, and median-change roles; per-unit/trace matched scales",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(OUT / "checkpoint_22b_explicit_history_response_maps.png", dpi=220)
    fig.savefig(OUT / "checkpoint_22b_explicit_history_response_maps.pdf")
    plt.close(fig)
    np.savez_compressed(
        OUT / "selected_unit_response_maps.npz",
        selected_rr100_indices=selected,
        **{f"trace_{trace:02d}_{condition}": maps for (trace, condition), maps in selected_maps.items()},
        **{f"trace_{trace:02d}_{condition}_instantaneous_ssi": values for (trace, condition), values in selected_instantaneous.items()},
    )
    pd.DataFrame(trace_info).to_csv(OUT / "selected_trace_contract.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "targeted_multi_map_checkpoint_complete_stop_before_drilldown_or_population_summary",
        "scope": "one strong-contour image x four history pairs x 100 RR100 units scored; six recorded-validated roles mapped for two trajectory extremes",
        "comparison": "genuine recorded 32-frame prehistory versus synthetic repeated prehistory; identical final 40 lag-zero retinal movies",
        "selection": "within each validated SF half: largest positive mean SSI effect, largest negative mean SSI effect, median response-map-change control",
        "map_scaling": "shared synthetic/genuine rate scale and symmetric difference scale per unit and trace",
        "input_checkpoint": file_identity(INPUT_CHECKPOINT / "manifest.json"),
        "assignments": file_identity(HALF_ASSIGNMENTS),
        "mapping": file_identity(MAPPING),
        "next_checkpoint": "human-selected or algorithmic full 40-frame drill-down with SSI/rate timecourses",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(roles[["rr100_index", "sf_outer_third", "selection_role", "selection_criterion_value", "mean_genuine_minus_synthetic_ssi", "mean_absolute_response_map_difference_hz"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

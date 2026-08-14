#!/usr/bin/env python3
"""Audit whether corrected production histories cross fixation boundaries."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
ASSEMBLED = CACHE / "assembled/rounds_000_011_n012_quartile_snapshot_v1"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
SOURCE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
EVENTS = ROOT / "outputs/fig4_active_sensing/backimage_240hz_timebase_checkpoint_25_v1/backimage_event_features_240hz.csv"
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_rounds000_011_v2/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_no_bottom_row_rounds000_011_v2_unit_assignments.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_pre_fixation_history_boundary_audit_v1"
GROUPS = ["sf_q1", "sf_q2", "sf_q3", "sf_q4"]
COLORS = {"sf_q1": "#0072B2", "sf_q2": "#009E73", "sf_q3": "#E69F00", "sf_q4": "#CC79A7"}


def resolve_boundaries(traces: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    source_lookup = source.reset_index(names="source_row")[[
        "source_row", "global_start", "local_start", "epoch_start_local", "epoch_stop_local", "phase", "samples_since_event"
    ]].rename(columns={column: f"source_{column}" for column in [
        "global_start", "local_start", "epoch_start_local", "epoch_stop_local", "phase", "samples_since_event"
    ]})
    out = traces.merge(source_lookup, on="source_row", how="left", validate="many_to_one")
    for column in ("global_start", "local_start", "epoch_start_local", "epoch_stop_local", "phase", "samples_since_event"):
        out[f"resolved_{column}"] = out[column].where(out[column].notna(), out[f"source_{column}"])
    trial_global_start = out.resolved_global_start - out.resolved_local_start
    out["trial_global_start"] = trial_global_start
    out["fixation_global_start"] = trial_global_start + out.resolved_epoch_start_local
    out["fixation_global_stop_exclusive"] = trial_global_start + out.resolved_epoch_stop_local
    pre_frames = []
    for row in out.itertuples(index=False):
        history = np.arange(int(row.corrected_history_global_start), int(row.corrected_history_global_stop_exclusive), 2)
        pre_frames.append(int(np.sum(history < row.fixation_global_start)))
    out["history_frames_before_fixation"] = pre_frames
    out["history_within_selected_fixation"] = (
        out.corrected_history_global_start.ge(out.fixation_global_start)
        & out.corrected_history_global_stop_exclusive.le(out.fixation_global_stop_exclusive)
    )
    out["score_within_selected_fixation"] = (
        out.corrected_scored_global_start.ge(out.fixation_global_start)
        & out.corrected_scored_global_stop_exclusive.le(out.fixation_global_stop_exclusive)
    )
    return out


def add_motion_and_event_metrics(traces: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    with np.load(CACHE / "input_cache/corrected_trace_segments.npz", allow_pickle=False) as archive:
        frozen_ids = np.asarray(archive["trace_index"], int)
        history = np.asarray(archive["history_xy_deg"], float)
        score = np.asarray(archive["score_xy_deg"], float)
    if not np.array_equal(frozen_ids, traces.trace_index.to_numpy(int)):
        raise ValueError("Frozen trace order changed")
    out = traces.copy()
    out["history_path_arcmin"] = np.linalg.norm(np.diff(history, axis=1), axis=2).sum(axis=1) * 60.0
    out["history_max_step_arcmin"] = np.linalg.norm(np.diff(history, axis=1), axis=2).max(axis=1) * 60.0
    out["score_path_recomputed_arcmin"] = np.linalg.norm(np.diff(score, axis=1), axis=2).sum(axis=1) * 60.0
    event_lookup = {
        (str(session), int(trial)): frame[["event_onset_sample", "event_offset_sample"]].to_numpy(int)
        for (session, trial), frame in events.groupby(["session", "trial_idx"], sort=False)
    }
    event_frames = []
    for row in out.itertuples(index=False):
        history_global = np.arange(int(row.corrected_history_global_start), int(row.corrected_history_global_stop_exclusive), 2)
        history_local = history_global - int(row.trial_global_start)
        overlap = np.zeros(len(history_local), bool)
        for onset, offset in event_lookup.get((str(row.session), int(row.trial_idx)), np.empty((0, 2), int)):
            overlap |= (history_local >= onset) & (history_local <= offset)
        event_frames.append(int(overlap.sum()))
    out["history_detected_event_frames"] = event_frames
    out["history_contains_detected_event"] = out.history_detected_event_frames.gt(0)
    return out


def residualized_slope(x: np.ndarray, y: np.ndarray, images: np.ndarray) -> float:
    xx = np.asarray(x, float) - pd.Series(x).groupby(images).transform("mean").to_numpy()
    yy = np.asarray(y, float) - pd.Series(y).groupby(images).transform("mean").to_numpy()
    return float(np.dot(xx, yy) / np.dot(xx, xx))


def sensitivity(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = pd.read_csv(ASSEMBLED / "condition_index.csv").merge(
        traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin", "history_within_selected_fixation"]],
        on="trace_index", validate="many_to_one",
    )
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index")
    assignments = pd.read_csv(ASSIGNMENTS)
    moving_ssi = np.load(ASSEMBLED / "moving_movie_ssi_bits_per_spike.npy", mmap_mode="r")
    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as archive:
        baseline_ssi = np.asarray(archive["movie_ssi_bits_per_spike"], float)
        baseline_spikes = np.asarray(archive["expected_spikes"], float)
    baseline_info = baseline_ssi * baseline_spikes
    image_ids = conditions.image_index.to_numpy(int)
    paths = conditions.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    strong = images.corrected_reconstruction_orientation_coherence.to_numpy(float)[image_ids] >= 0.20
    clean = conditions.history_within_selected_fixation.to_numpy(bool)
    rows = []
    unit_rows = []
    for scope, scope_mask in (("all_images", np.ones(len(conditions), bool)), ("strong_contours", strong)):
        for history_subset, history_mask in (("all_histories", np.ones(len(conditions), bool)), ("within_fixation_only", clean)):
            use = scope_mask & history_mask
            for group in GROUPS:
                units = assignments.loc[assignments.sf_quartile.eq(group), "rr100_index"].to_numpy(int)
                delta = np.asarray(moving_ssi[:, units], float) - baseline_ssi[image_ids][:, units]
                population = np.asarray(moving_info[:, units], float).sum(1) / np.maximum(
                    np.asarray(moving_spikes[:, units], float).sum(1), 1e-12
                ) - baseline_info[:, units].sum(1)[image_ids] / np.maximum(baseline_spikes[:, units].sum(1)[image_ids], 1e-12)
                unit_slopes = [residualized_slope(paths[use], delta[use, j], image_ids[use]) for j in range(len(units))]
                rows.append({
                    "scope": scope, "history_subset": history_subset, "sf_quartile": group,
                    "n_conditions": int(use.sum()),
                    "pooled_spike_path_slope": residualized_slope(paths[use], population[use], image_ids[use]),
                    "equal_unit_path_slope": residualized_slope(paths[use], delta[use].mean(1), image_ids[use]),
                    "median_unit_path_slope": float(np.median(unit_slopes)),
                })
            for unit in (54, 18):
                delta = np.asarray(moving_ssi[:, unit], float) - baseline_ssi[image_ids, unit]
                unit_rows.append({
                    "scope": scope, "history_subset": history_subset, "rr100_index": unit,
                    "n_conditions": int(use.sum()),
                    "unit_path_slope": residualized_slope(paths[use], delta[use], image_ids[use]),
                })
    return pd.DataFrame(rows), pd.DataFrame(unit_rows)


def completed_cache_exposure(affected: set[int]) -> dict[str, int]:
    completed_movies = 0
    affected_movies = 0
    for path in (CACHE / "moving").glob("round_*/image_*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            trace_ids = np.asarray(archive["trace_index"], int)
        completed_movies += len(trace_ids)
        affected_movies += int(np.isin(trace_ids, list(affected)).sum())
    return {"completed_movies_on_disk": completed_movies, "affected_completed_movies_on_disk": affected_movies}


def draw(traces: pd.DataFrame, sensitivity_table: pd.DataFrame) -> None:
    with np.load(CACHE / "input_cache/corrected_trace_segments.npz", allow_pickle=False) as archive:
        history = np.asarray(archive["history_xy_deg"], float)
        score = np.asarray(archive["score_xy_deg"], float)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for ax, trace_index in zip(axes[0, :2], (364, 574), strict=True):
        row = traces[traces.trace_index.eq(trace_index)].iloc[0]
        full = np.concatenate([history[trace_index], score[trace_index]], axis=0)
        steps = np.linalg.norm(np.diff(full, axis=0), axis=1) * 60.0
        n_pre = int(row.history_frames_before_fixation)
        ax.plot(np.arange(1, len(full)), steps, color="#333333", lw=1.4)
        if n_pre:
            ax.axvspan(0, n_pre, color="#D55E00", alpha=0.24, label="before fixation")
        ax.axvline(32, color="#0072B2", ls="--", lw=1.1, label="scored segment starts")
        ax.set(title=f"trace {trace_index}: {n_pre} history frames before fixation\n{int(row.history_detected_event_frames)} history frames overlap detected event",
               xlabel="32 history + 40 scored frames", ylabel="step (arcmin)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    counts = traces.history_frames_before_fixation.value_counts().sort_index()
    axes[0, 2].bar(counts.index.astype(str), counts.values, color=["#009E73" if value == 0 else "#D55E00" for value in counts.index])
    axes[0, 2].set(title="423/1,000 histories cross fixation onset", xlabel="120-Hz history frames before fixation", ylabel="traces")
    clean = traces.history_within_selected_fixation
    bins = np.geomspace(max(traces.history_max_step_arcmin.min(), 0.05), traces.history_max_step_arcmin.max() * 1.02, 35)
    axes[1, 0].hist(traces.loc[clean, "history_max_step_arcmin"], bins=bins, alpha=.65, label="within fixation", color="#009E73")
    axes[1, 0].hist(traces.loc[~clean, "history_max_step_arcmin"], bins=bins, alpha=.55, label="crosses onset", color="#D55E00")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(title="Boundary-crossing histories contain large steps", xlabel="maximum history step (arcmin, log scale)", ylabel="traces")
    axes[1, 0].legend(frameon=False)
    strong = sensitivity_table[sensitivity_table.scope.eq("strong_contours")]
    x = np.arange(4)
    for subset, marker, ls in (("all_histories", "o", "-"), ("within_fixation_only", "s", "--")):
        sub = strong[strong.history_subset.eq(subset)].set_index("sf_quartile").loc[GROUPS]
        axes[1, 1].plot(x, sub.pooled_spike_path_slope * 1e4, color="black" if subset == "all_histories" else "#D55E00", marker=marker, ls=ls, label=subset.replace("_", " "))
        axes[1, 2].plot(x, sub.equal_unit_path_slope * 1e4, color="black" if subset == "all_histories" else "#D55E00", marker=marker, ls=ls, label=subset.replace("_", " "))
    for ax, title in ((axes[1, 1], "Pooled-spike slope sensitivity"), (axes[1, 2], "Equal-unit slope sensitivity")):
        ax.axhline(0, color="0.55", lw=.8)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.set(title=title, xlabel="SF quartile", ylabel="path slope ×10⁻⁴")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Corrected production history audit: validity gate missed fixation boundaries", fontsize=15, weight="bold")
    fig.savefig(OUT / "pre_fixation_history_boundary_audit.png", dpi=210, facecolor="white")
    fig.savefig(OUT / "pre_fixation_history_boundary_audit.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv").sort_values("trace_index").reset_index(drop=True)
    source = pd.read_csv(SOURCE)
    events = pd.read_csv(EVENTS)
    traces = add_motion_and_event_metrics(resolve_boundaries(traces, source), events)
    traces.to_csv(OUT / "trace_history_boundary_audit.csv", index=False)
    affected = set(traces.loc[~traces.history_within_selected_fixation, "trace_index"].astype(int))
    sensitivity_table, unit_table = sensitivity(traces)
    sensitivity_table.to_csv(OUT / "quartile_clean_history_sensitivity.csv", index=False)
    unit_table.to_csv(OUT / "outlier_unit_clean_history_sensitivity.csv", index=False)
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    condition_exposure = condition.assign(history_boundary_affected=condition.trace_index.isin(affected))
    condition_exposure.to_csv(OUT / "snapshot_condition_history_exposure.csv", index=False)
    cache_exposure = completed_cache_exposure(affected)
    candidate = pd.read_csv(COHORT / "trace_candidate_pool_with_rejections.csv")
    candidate_epoch_start = candidate.global_start - candidate.local_start + candidate.epoch_start_local
    candidate_inside = candidate.corrected_history_global_start >= candidate_epoch_start
    candidate_eligible_inside = candidate_inside & candidate.prevalidation_eligible.fillna(False).astype(bool) & candidate.explicit_history_valid.fillna(False).astype(bool)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre_fixation_history_boundary_error_confirmed",
        "root_cause": "explicit_history_valid required same trial/DPI/finite samples but did not require all history frames to remain within epoch_start_local:epoch_stop_local",
        "cohort": {
            "n_traces": len(traces),
            "n_history_within_fixation": int(traces.history_within_selected_fixation.sum()),
            "n_history_crosses_fixation_start": int((~traces.history_within_selected_fixation).sum()),
            "n_history_contains_detected_event": int(traces.history_contains_detected_event.sum()),
            "n_scored_segment_within_fixation": int(traces.score_within_selected_fixation.sum()),
        },
        "snapshot": {
            "n_conditions": len(condition_exposure),
            "n_affected_conditions": int(condition_exposure.history_boundary_affected.sum()),
            "n_clean_conditions": int((~condition_exposure.history_boundary_affected).sum()),
        },
        "live_cache": cache_exposure,
        "full_100x1000_projection": {"affected_movies": len(affected) * 100, "total_movies": 100000},
        "replacement_feasibility": {
            "prevalidated_explicit_history_candidates_within_fixation": int(candidate_eligible_inside.sum()),
            "enough_for_1000_trace_rebuild": bool(candidate_eligible_inside.sum() >= 1000),
        },
        "interpretation": "clean-history-only filtering preserves the qualitative 12-round quartile slopes, but contaminated response movies do not satisfy the intended within-fixation prehistory contract",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    draw(traces, sensitivity_table)
    print(json.dumps(summary, indent=2), flush=True)
    print(sensitivity_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

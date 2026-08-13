#!/usr/bin/env python3
"""Audit RR100 identity, SF fits, and unit influence for outer-third SSI plots.

This is a cache-backed diagnostic.  It does not regenerate model responses and
does not overwrite the Figure 4 variants.  It verifies the unit identity chain,
computes exact leave-one-unit-out influence on the smallest-drift population
delta, and plots recorded versus parametric SF tuning for auditable roles.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information import (
    plot_backimage_real_trace_unit_first_and_population_schematics as schematic,
)
from declan.fig.ssi_figure_v2.compose_ssi_figure_v4_sf_outer_thirds import (
    GROUPS,
    prepare_outer_thirds,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles import (
    N_DRIFT_BINS,
    N_MICROSACCADE_BINS,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_MATRIX_DIR,
)
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_outer_thirds_identity_fit_audit_v1"
FIT_CSV = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
FIT_NPZ = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_model_arrays.npz"
MAPPING_CSV = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
RECORDED_CURVES_CSV = ROOT / (
    "outputs/redundancy_resolved_v1_twin/rr100_joint_f0_parametric_recorded_validation_v2/"
    "rr100_parametric_recorded_validation_curve_points.csv"
)
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_"
    "finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid"
)
COLORS = {
    "sf_bottom_third": "#2C7FB8",
    "sf_middle_third": "#8C8C8C",
    "sf_top_third": "#D95F0E",
    "invalid_model": "#D8D8D8",
}
LABELS = {
    "sf_bottom_third": "bottom SF third",
    "sf_middle_third": "middle SF third",
    "sf_top_third": "top SF third",
    "invalid_model": "invalid fit",
}
RELATIONS = {
    "strong_contours_no_osi": "strong contours",
    "contour_matched": "contour matched",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_audit(data: dict, fit: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    view = load_population_view(version_name=RR100_VERSION)
    reps = pd.DataFrame(view.meta["representatives"]).sort_values("rep_idx").reset_index(drop=True)
    mapping = pd.read_csv(MAPPING_CSV).sort_values("rr100_index").reset_index(drop=True)
    table = pd.DataFrame(
        {
            "rr100_index": mapping["rr100_index"].astype(int),
            "population_rep_idx": reps["rep_idx"].astype(int),
            "matrix_unit_index": data["unit"]["unit_index"].astype(int),
            "fit_rr100_index": fit.sort_values("rr100_index")["rr100_index"].astype(int).to_numpy(),
            "assignment_rr100_index": assignments.sort_values("rr100_index")["rr100_index"].astype(int).to_numpy(),
            "mapping_channel": mapping["canonical_channel"].astype(int),
            "population_selected_channel": reps["selected_channel"].astype(int),
            "mapping_session": mapping["session"].astype(str),
            "population_selected_session": reps["selected_session"].astype(str),
            "mapping_source_unit_index": mapping["source_unit_index"].astype(int),
            "population_source_unit_index": reps["selected_source_unit_index"].astype(int),
        }
    )
    table["index_chain_match"] = (
        table["rr100_index"].eq(table["population_rep_idx"])
        & table["rr100_index"].eq(table["matrix_unit_index"])
        & table["rr100_index"].eq(table["fit_rr100_index"])
        & table["rr100_index"].eq(table["assignment_rr100_index"])
    )
    table["channel_match"] = table["mapping_channel"].eq(table["population_selected_channel"])
    table["session_match"] = table["mapping_session"].eq(table["population_selected_session"])
    table["source_unit_match"] = table["mapping_source_unit_index"].eq(table["population_source_unit_index"])
    table["all_identity_fields_match"] = table[
        ["index_chain_match", "channel_match", "session_match", "source_unit_match"]
    ].all(axis=1)
    summary = {
        "rr100_version": RR100_VERSION,
        "n_units": int(len(table)),
        "n_all_identity_fields_match": int(table["all_identity_fields_match"].sum()),
        "n_index_chain_mismatch": int((~table["index_chain_match"]).sum()),
        "n_channel_mismatch": int((~table["channel_match"]).sum()),
        "n_session_mismatch": int((~table["session_match"]).sum()),
        "n_source_unit_mismatch": int((~table["source_unit_match"]).sum()),
        "population_membership_is_one_hot": bool(
            np.allclose(view.membership.sum(axis=1), 1.0)
            and np.allclose(view.membership.max(axis=1), 1.0)
        ),
        "passed": bool(table["all_identity_fields_match"].all()),
    }
    return table, summary


def endpoint_unit_influence(
    data: dict,
    unit_outer: pd.DataFrame,
    trace: pd.DataFrame,
    trace_bins: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
) -> pd.DataFrame:
    first_drift = trace_bins[trace_bins["context"].eq("drift_only")].sort_values("path_bin_order").iloc[0]
    trace_indices = trace[trace["path_bin"].eq(first_drift["path_bin"])]["trace_bank_index"].astype(int).to_numpy()
    rows_out: list[dict] = []
    for relation in RELATIONS:
        selections = schematic.unit_image_selection(
            unit_outer,
            data["image"],
            relation=relation,
            sf_groups=list(GROUPS),
            min_osi=0.05,
            match_max_deg=22.5,
            orthogonal_min_deg=67.5,
            image_axis_col="image_edge_axis_deg",
        )
        for group in GROUPS:
            records = []
            for unit_index, image_indices in selections[group].items():
                images = np.asarray(image_indices, dtype=int)
                baseline_rows = np.asarray([baseline_lookup[int(i)] for i in images], dtype=int)
                baseline_value = np.asarray(data["stabilized_ssi"][baseline_rows, unit_index], dtype=float)
                baseline_weight = np.asarray(data["stabilized_expected"][baseline_rows, unit_index], dtype=float)
                movie_rows = row_grid[np.ix_(images, trace_indices)]
                moving_value = np.asarray(data["ssi"][movie_rows, unit_index], dtype=float)
                moving_weight = np.asarray(data["expected"][movie_rows, unit_index], dtype=float)
                records.append(
                    {
                        "unit_index": int(unit_index),
                        "n_selected_images": int(images.size),
                        "moving_information_numerator": float(np.nansum(moving_value * moving_weight)),
                        "moving_expected_spikes": float(np.nansum(moving_weight)),
                        "baseline_information_numerator": float(np.nansum(baseline_value * baseline_weight)),
                        "baseline_expected_spikes": float(np.nansum(baseline_weight)),
                        "equal_unit_ssi_delta": float(np.nanmean(moving_value) - np.nanmean(baseline_value)),
                    }
                )
            frame = pd.DataFrame(records)
            m_num = float(frame["moving_information_numerator"].sum())
            m_den = float(frame["moving_expected_spikes"].sum())
            b_num = float(frame["baseline_information_numerator"].sum())
            b_den = float(frame["baseline_expected_spikes"].sum())
            full_delta = m_num / m_den - b_num / b_den
            frame["relation"] = relation
            frame["sf_outer_third"] = group
            frame["endpoint"] = "smallest_drift_bin"
            frame["path_median_arcmin"] = float(first_drift["median_path_arcmin"])
            frame["full_population_delta"] = full_delta
            frame["leave_one_out_population_delta"] = (
                (m_num - frame["moving_information_numerator"]) / (m_den - frame["moving_expected_spikes"])
                - (b_num - frame["baseline_information_numerator"]) / (b_den - frame["baseline_expected_spikes"])
            )
            frame["leave_one_out_influence_on_delta"] = (
                frame["full_population_delta"] - frame["leave_one_out_population_delta"]
            )
            rows_out.extend(frame.to_dict("records"))
    return pd.DataFrame(rows_out)


def choose_roles(influence: pd.DataFrame) -> pd.DataFrame:
    strong = influence[influence["relation"].eq("strong_contours_no_osi")].copy()
    bottom = strong[strong["sf_outer_third"].eq("sf_bottom_third")]
    top = strong[strong["sf_outer_third"].eq("sf_top_third")]
    choices = [
        ("bottom_negative_population_driver", bottom["leave_one_out_influence_on_delta"].idxmin(), "minimum leave-one-out influence"),
        ("bottom_positive_counterweight", bottom["leave_one_out_influence_on_delta"].idxmax(), "maximum leave-one-out influence"),
        ("bottom_boundary_control", bottom["preferred_sf_cpd"].idxmax(), "largest preferred SF retained in bottom third"),
        ("top_positive_unit_control", top["equal_unit_ssi_delta"].idxmax(), "largest positive equal-unit SSI delta"),
        ("top_negative_population_driver", top["leave_one_out_influence_on_delta"].idxmin(), "minimum leave-one-out influence in top third"),
    ]
    rows = []
    seen: set[int] = set()
    for role, idx, criterion in choices:
        row = strong.loc[idx].to_dict()
        unit_index = int(row["unit_index"])
        if unit_index in seen:
            continue
        seen.add(unit_index)
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_is_algorithmic"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def validation_gate_summary(influence: pd.DataFrame) -> pd.DataFrame:
    """Recompute the exact ratio after descriptive recorded-validation gates."""
    rows = []
    for (relation, group), sub in influence.groupby(["relation", "sf_outer_third"], sort=False):
        for threshold in (None, 0.5, 0.75):
            selected = sub if threshold is None else sub[sub["recorded_sf_curve_r_full_support"] >= threshold]
            moving = selected["moving_information_numerator"].sum() / selected["moving_expected_spikes"].sum()
            baseline = selected["baseline_information_numerator"].sum() / selected["baseline_expected_spikes"].sum()
            rows.append(
                {
                    "relation": relation,
                    "sf_outer_third": group,
                    "recorded_curve_r_min": threshold if threshold is not None else np.nan,
                    "gate_label": "all units" if threshold is None else f"recorded curve r >= {threshold:g}",
                    "n_units": int(len(selected)),
                    "population_delta_bits_per_spike": float(moving - baseline),
                    "descriptive_only_not_prespecified_exclusion": threshold is not None,
                }
            )
    return pd.DataFrame(rows)


def merge_features(
    frame: pd.DataFrame,
    fit: pd.DataFrame,
    old: pd.DataFrame,
    assignment_audit: pd.DataFrame,
) -> pd.DataFrame:
    fit_cols = [
        "rr100_index", "canonical_channel", "session", "source_unit_index", "preferred_sf_cpd",
        "recorded_sf_peak_cpd", "recorded_sf_curve_r_full_support", "recorded_sf_curve_nrmse_full_support",
        "sf_fit_r2", "joint_parametric_surface_r2", "sf_fit_support_min_cpd", "sf_fit_support_max_cpd",
    ]
    old_cols = [
        "unit_index", "dynamic_log_gaussian_marginal_sf_cpd", "dynamic_log_gaussian_marginal_r2",
        "static_peak_spatial_cpd_by_mean_rate", "dynamic_peak_spatial_cpd_by_amp",
    ]
    audit_cols = ["rr100_index", "sf_outer_third", "sf_outer_third_label"]
    out = frame.merge(fit[fit_cols], left_on="unit_index", right_on="rr100_index", validate="many_to_one")
    out = out.merge(old[old_cols], on="unit_index", validate="many_to_one")
    if "sf_outer_third" not in out.columns:
        out = out.merge(assignment_audit[audit_cols], on="rr100_index", validate="many_to_one")
    return out


def plot_overview(
    all_units: pd.DataFrame,
    influence: pd.DataFrame,
    roles: pd.DataFrame,
    identity_summary: dict,
    path: Path,
) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9})
    fig = plt.figure(figsize=(11.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.12])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    status = "PASS" if identity_summary["passed"] else "FAIL"
    ax0.text(0.02, 0.95, f"Identity chain: {status}", fontsize=15, weight="bold", va="top")
    ax0.text(
        0.02,
        0.78,
        "\n".join(
            [
                f"{identity_summary['n_all_identity_fields_match']} / {identity_summary['n_units']} units match",
                "RR index = population rep = matrix column = fit row",
                "canonical channel, session, and source unit all agree",
                f"one-hot medoid population: {identity_summary['population_membership_is_one_hot']}",
                "Conclusion: labels were not swapped by indexing.",
            ]
        ),
        va="top",
        linespacing=1.55,
    )

    role_units = set(roles["unit_index"].astype(int))
    ax1 = fig.add_subplot(gs[0, 1])
    for group, sub in all_units.groupby("sf_outer_third", sort=False):
        ax1.scatter(
            sub["dynamic_log_gaussian_marginal_sf_cpd"], sub["preferred_sf_cpd"],
            s=24, alpha=0.78, color=COLORS.get(group, "0.7"), label=LABELS.get(group, group),
            edgecolor="none",
        )
    for row in all_units[all_units["unit_index"].isin(role_units)].itertuples():
        ax1.annotate(f"u{row.unit_index:03d}", (row.dynamic_log_gaussian_marginal_sf_cpd, row.preferred_sf_cpd), xytext=(3, 3), textcoords="offset points")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("historical dynamic marginal SF (cpd)")
    ax1.set_ylabel("new parametric preferred SF (cpd)")
    ax1.set_title("New and historical SF ranks disagree")
    ax1.grid(alpha=0.18)
    ax1.legend(fontsize=7, frameon=False)

    ax2 = fig.add_subplot(gs[0, 2])
    for group, sub in all_units.groupby("sf_outer_third", sort=False):
        ax2.scatter(
            sub["recorded_sf_peak_cpd"], sub["preferred_sf_cpd"], s=24, alpha=0.78,
            color=COLORS.get(group, "0.7"), edgecolor="none",
        )
    line = np.asarray([1.0, 16.0])
    ax2.plot(line, line, color="0.35", lw=1, ls="--")
    for row in all_units[all_units["unit_index"].isin(role_units)].itertuples():
        ax2.annotate(f"u{row.unit_index:03d}", (row.recorded_sf_peak_cpd, row.preferred_sf_cpd), xytext=(3, 3), textcoords="offset points")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlim(0.85, 18)
    ax2.set_ylim(0.85, 13)
    ax2.set_xlabel("recorded SF peak (cpd)")
    ax2.set_ylabel("new parametric preferred SF (cpd)")
    ax2.set_title("Recorded peaks expose fit disagreements")
    ax2.grid(alpha=0.18)

    for col, relation in enumerate(RELATIONS):
        ax = fig.add_subplot(gs[1, col])
        sub = influence[influence["relation"].eq(relation)].copy()
        sub["label"] = sub["unit_index"].map(lambda value: f"u{int(value):03d}")
        sub = sub.reindex(sub["leave_one_out_influence_on_delta"].abs().sort_values(ascending=False).index).head(14)
        sub = sub.sort_values("leave_one_out_influence_on_delta")
        colors = [COLORS[g] for g in sub["sf_outer_third"]]
        ax.barh(sub["label"], sub["leave_one_out_influence_on_delta"], color=colors)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.set_xlabel("exact leave-one-out influence on\npopulation delta (bits/spike)")
        ax.set_title(f"{RELATIONS[relation]}: largest unit influences")
        ax.grid(axis="x", alpha=0.18)
    ax3 = fig.add_subplot(gs[1, 2])
    groups = ["sf_bottom_third", "sf_top_third"]
    values = [all_units.loc[all_units["sf_outer_third"].eq(g), "recorded_sf_curve_r_full_support"].dropna() for g in groups]
    ax3.boxplot(values, tick_labels=["bottom", "top"], showfliers=False)
    rng = np.random.default_rng(17)
    for i, (group, vals) in enumerate(zip(groups, values), start=1):
        ax3.scatter(i + rng.normal(0, 0.045, len(vals)), vals, s=15, alpha=0.65, color=COLORS[group])
    ax3.axhline(0.5, color="0.35", ls="--", lw=1)
    ax3.set_ylabel("recorded vs parametric SF curve Pearson r")
    ax3.set_title("Recorded validation quality is heterogeneous")
    ax3.grid(axis="y", alpha=0.18)
    fig.suptitle("RR100 outer-third Figure 4 audit — identity, tuning agreement, and SSI leverage", fontsize=14, weight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_role_tuning_sheet(
    roles: pd.DataFrame,
    curve_points: pd.DataFrame,
    fit_npz: np.lib.npyio.NpzFile,
    path: Path,
) -> None:
    n = len(roles)
    fig, axes = plt.subplots(n, 2, figsize=(11.5, 2.45 * n), squeeze=False, constrained_layout=True)
    sf_grid = fit_npz["sf_evaluation_grid_cpd"]
    curves = fit_npz["sf_factor_normalized_curves"]
    for row_idx, role in enumerate(roles.itertuples(index=False)):
        unit = int(role.unit_index)
        color = COLORS[str(role.sf_outer_third)]
        ax = axes[row_idx, 0]
        pts = curve_points[curve_points["rr100_index"].eq(unit)].sort_values("sf_cpd")
        ax.plot(sf_grid, curves[unit], color=color, lw=2.2, label="new parametric factor")
        ax.plot(pts["sf_cpd"], pts["recorded_range_normalized"], color="black", marker="o", lw=1.2, label="recorded tuning")
        ax.axvline(role.preferred_sf_cpd, color=color, ls="--", lw=1.2, label="new preference")
        ax.axvline(role.dynamic_log_gaussian_marginal_sf_cpd, color="#5E3C99", ls=":", lw=1.5, label="historical preference")
        ax.axvspan(role.sf_fit_support_min_cpd, role.sf_fit_support_max_cpd, color="0.5", alpha=0.08)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.01, 18)
        ax.set_ylim(-0.12, 1.12)
        ax.set_ylabel("range-normalized response")
        ax.set_title(
            f"u{unit:03d} — {role.selection_role}\n"
            f"new {role.preferred_sf_cpd:.2f} cpd; recorded peak {role.recorded_sf_peak_cpd:.2f}; "
            f"curve r={role.recorded_sf_curve_r_full_support:.2f}"
        )
        ax.grid(alpha=0.18)
        if row_idx == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)
        ax = axes[row_idx, 1]
        rel = influence_for_role(roles, unit)
        x = np.arange(len(rel))
        ax.bar(x, rel["leave_one_out_influence_on_delta"], color=[COLORS[g] for g in rel["sf_outer_third"]])
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_xticks(x, [RELATIONS[r] for r in rel["relation"]], rotation=0)
        ax.set_ylabel("LOO influence on smallest-drift delta")
        ax.set_title(
            f"SSI leverage; strong-contour equal-unit delta {role.equal_unit_ssi_delta:+.3f}\n"
            f"moving expected spikes {role.moving_expected_spikes:.1f}"
        )
        ax.grid(axis="y", alpha=0.18)
    axes[-1, 0].set_xlabel("spatial frequency (cpd); gray band = new fit support")
    fig.suptitle("Algorithmically selected units: recorded tuning beside SSI influence", fontsize=14, weight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def influence_for_role(roles: pd.DataFrame, unit_index: int) -> pd.DataFrame:
    # The full influence table is attached by main for compact plotting without global state.
    return influence_for_role.table[influence_for_role.table["unit_index"].eq(unit_index)].sort_values("relation")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = schematic.load_dataset(DEFAULT_MATRIX_DIR)
    fit = pd.read_csv(FIT_CSV).sort_values("rr100_index").reset_index(drop=True)
    assignments = pd.read_csv(DEFAULT_ASSIGNMENTS).sort_values("rr100_index").reset_index(drop=True)
    unit_outer, assignment_audit, boundaries = prepare_outer_thirds(data["unit"], assignments)
    identity_table, identity_summary = identity_audit(data, fit, assignments)
    if not identity_summary["passed"]:
        raise ValueError(f"RR100 identity audit failed: {identity_summary}")
    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"], n_drift_bins=N_DRIFT_BINS, n_microsaccade_bins=N_MICROSACCADE_BINS
    )
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])
    influence = endpoint_unit_influence(data, unit_outer, trace, trace_bins, row_grid, baseline_lookup)
    influence = merge_features(influence, fit, data["unit"], assignment_audit)
    roles = choose_roles(influence)
    roles = roles.sort_values("selection_role").reset_index(drop=True)

    all_units = assignment_audit.merge(fit, on="rr100_index", suffixes=("_assignment", ""), validate="one_to_one")
    all_units = all_units.merge(
        data["unit"][[
            "unit_index", "dynamic_log_gaussian_marginal_sf_cpd", "dynamic_log_gaussian_marginal_r2",
            "static_peak_spatial_cpd_by_mean_rate", "dynamic_peak_spatial_cpd_by_amp",
        ]],
        left_on="rr100_index", right_on="unit_index", validate="one_to_one",
    )
    quality_summary = (
        all_units[all_units["sf_outer_third"].isin(GROUPS)]
        .groupby("sf_outer_third", sort=False)
        .agg(
            n_units=("rr100_index", "size"),
            median_preferred_sf_cpd=("preferred_sf_cpd", "median"),
            median_recorded_peak_cpd=("recorded_sf_peak_cpd", "median"),
            median_recorded_curve_r=("recorded_sf_curve_r_full_support", "median"),
            n_recorded_curve_r_below_0p5=("recorded_sf_curve_r_full_support", lambda x: int((x < 0.5).sum())),
            median_sf_fit_r2=("sf_fit_r2", "median"),
        )
        .reset_index()
    )
    gate_summary = validation_gate_summary(influence)
    valid_units = all_units[all_units["model_valid"].fillna(False)].copy()
    rank_agreement = {
        "spearman_new_vs_historical_dynamic_marginal_sf": float(
            valid_units["preferred_sf_cpd"].corr(
                valid_units["dynamic_log_gaussian_marginal_sf_cpd"], method="spearman"
            )
        ),
        "spearman_new_vs_recorded_sf_peak": float(
            valid_units["preferred_sf_cpd"].corr(valid_units["recorded_sf_peak_cpd"], method="spearman")
        ),
    }

    identity_table.to_csv(OUT_DIR / "rr100_identity_chain.csv", index=False)
    influence.to_csv(OUT_DIR / "outer_thirds_smallest_drift_unit_influence.csv", index=False)
    roles.to_csv(OUT_DIR / "algorithmic_unit_roles.csv", index=False)
    quality_summary.to_csv(OUT_DIR / "outer_thirds_fit_quality_summary.csv", index=False)
    gate_summary.to_csv(OUT_DIR / "recorded_validation_gate_sensitivity.csv", index=False)
    assignment_audit.to_csv(OUT_DIR / "sf_outer_third_unit_assignments.csv", index=False)
    trace_bins.to_csv(OUT_DIR / "trace_path_bin_definitions.csv", index=False)

    overview = OUT_DIR / "rr100_outer_thirds_identity_fit_influence_audit.png"
    tuning_sheet = OUT_DIR / "rr100_outer_thirds_selected_unit_tuning_and_influence.png"
    plot_overview(all_units, influence, roles, identity_summary, overview)
    influence_for_role.table = influence
    curve_points = pd.read_csv(RECORDED_CURVES_CSV)
    with np.load(FIT_NPZ) as fit_npz:
        plot_role_tuning_sheet(roles, curve_points, fit_npz, tuning_sheet)

    strong = influence[influence["relation"].eq("strong_contours_no_osi")]
    driver = roles[roles["selection_role"].eq("bottom_negative_population_driver")].iloc[0]
    bottom_full = float(strong[strong["sf_outer_third"].eq("sf_bottom_third")]["full_population_delta"].iloc[0])
    bottom_without_driver = float(driver["leave_one_out_population_delta"])
    manifest = {
        "status": "checkpoint_identity_fit_influence_audit_complete",
        "purpose": "diagnose apparent low/high SF reversal in outer-third Figure 4 variant",
        "identity": identity_summary,
        "outer_third_boundaries": boundaries,
        "rank_agreement": rank_agreement,
        "strong_contour_bottom_third_smallest_drift_delta": bottom_full,
        "bottom_negative_driver_unit": int(driver["unit_index"]),
        "bottom_delta_without_negative_driver": bottom_without_driver,
        "interpretive_guardrail": "Unit roles are algorithmic diagnostics; no unit was excluded from the plotted Figure 4 variant.",
        "sources": {
            "matrix_dir": str(Path(DEFAULT_MATRIX_DIR).relative_to(ROOT)),
            "assignments_csv": str(Path(DEFAULT_ASSIGNMENTS).relative_to(ROOT)),
            "fit_csv": str(FIT_CSV.relative_to(ROOT)),
            "fit_npz": str(FIT_NPZ.relative_to(ROOT)),
            "mapping_csv": str(MAPPING_CSV.relative_to(ROOT)),
            "recorded_curve_points_csv": str(RECORDED_CURVES_CSV.relative_to(ROOT)),
        },
        "sha256": {
            "assignments_csv": sha256(Path(DEFAULT_ASSIGNMENTS)),
            "fit_csv": sha256(FIT_CSV),
            "fit_npz": sha256(FIT_NPZ),
            "mapping_csv": sha256(MAPPING_CSV),
        },
        "artifacts": {
            "overview": str(overview.relative_to(ROOT)),
            "selected_unit_tuning_sheet": str(tuning_sheet.relative_to(ROOT)),
            "identity_chain": str((OUT_DIR / "rr100_identity_chain.csv").relative_to(ROOT)),
            "unit_influence": str((OUT_DIR / "outer_thirds_smallest_drift_unit_influence.csv").relative_to(ROOT)),
            "algorithmic_roles": str((OUT_DIR / "algorithmic_unit_roles.csv").relative_to(ROOT)),
            "recorded_validation_gate_sensitivity": str(
                (OUT_DIR / "recorded_validation_gate_sensitivity.csv").relative_to(ROOT)
            ),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("\nFIT QUALITY")
    print(quality_summary.to_string(index=False))
    print("\nDESCRIPTIVE VALIDATION-GATE SENSITIVITY")
    print(gate_summary.to_string(index=False))
    print("\nROLES")
    print(roles[[
        "selection_role", "unit_index", "sf_outer_third", "preferred_sf_cpd", "recorded_sf_peak_cpd",
        "recorded_sf_curve_r_full_support", "dynamic_log_gaussian_marginal_sf_cpd",
        "equal_unit_ssi_delta", "leave_one_out_influence_on_delta",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

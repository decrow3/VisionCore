"""Stratify BackImage local-contour matching by trial display geometry.

This is a sensitivity analysis for the BackImage aspect-ratio audit.  It joins
the reviewed fixation-window table to the trial-level display reconstruction,
then asks two distinct questions within mutually exclusive display strata:

1. Is drift edge-parallel relative to zero (session-balanced bootstrap)?
2. Does the *paired local edge* outperform orientation marginals preserved by
   a within-session x phase pairing shuffle?

The second question is the direct check against a shared horizontal-orientation
bias masquerading as trial-specific local contour matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_WINDOWS = DEFAULT_ROOT / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
DEFAULT_TRIALS = DEFAULT_ROOT / "backimage_trial_scale_audit" / "backimage_trial_scale_audit.csv"
DEFAULT_OUT = DEFAULT_ROOT / "backimage_contour_matching_geometry_sensitivity"

STRATA = [
    "all_reviewed",
    "native_16x9_fullscreen",
    "stretched_4x3_fullscreen",
    "square_fullscreen",
    "reduced_size",
]
SUBSETS = {
    "all_windows": lambda d: np.ones(len(d), dtype=bool),
    "reliable_axes": lambda d: (d["image_orientation_coherence"] >= 0.20) & (d["anisotropy"] >= 0.20),
    "high_confidence": lambda d: (d["image_orientation_coherence"] >= 0.50) & (d["anisotropy"] >= 0.50),
}


def _display_stratum(df: pd.DataFrame) -> pd.Series:
    full = np.isclose(df["screen_area_fraction"].to_numpy(float), 1.0, atol=1e-6)
    source = df["source_aspect"].to_numpy(float)
    out = np.full(len(df), "unclassified", dtype=object)
    out[~full] = "reduced_size"
    out[full & np.isclose(source, 16.0 / 9.0, atol=1e-5)] = "native_16x9_fullscreen"
    out[full & np.isclose(source, 4.0 / 3.0, atol=1e-5)] = "stretched_4x3_fullscreen"
    out[full & np.isclose(source, 1.0, atol=1e-5)] = "square_fullscreen"
    return pd.Series(out, index=df.index, name="display_geometry_stratum")


def _derive_endpoints(df: pd.DataFrame, edge_deg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    edge = df["image_edge_axis_deg"].to_numpy(float) if edge_deg is None else np.asarray(edge_deg, dtype=float)
    drift = df["drift_orientation_deg"].to_numpy(float)
    cos2 = np.cos(2.0 * np.deg2rad(drift - edge))

    theta = np.deg2rad(edge)
    ct, st = np.cos(theta), np.sin(theta)
    cxx = df["cov_xx_deg2"].to_numpy(float)
    cxy = df["cov_xy_deg2"].to_numpy(float)
    cyy = df["cov_yy_deg2"].to_numpy(float)
    along_var = ct * ct * cxx + 2.0 * ct * st * cxy + st * st * cyy
    across_var = st * st * cxx - 2.0 * ct * st * cxy + ct * ct * cyy
    rms_delta_arcmin = 60.0 * (np.sqrt(np.maximum(along_var, 0.0)) - np.sqrt(np.maximum(across_var, 0.0)))
    return cos2, rms_delta_arcmin


def _session_values(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    table = pd.DataFrame({"session": sessions, "value": values}).dropna()
    return table.groupby("session", sort=True)["value"].mean().to_numpy(float)


def _bootstrap_session_mean(values: np.ndarray, sessions: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = _session_values(values, sessions)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    point = float(vals.mean())
    if vals.size < 2 or n_bootstrap <= 0:
        return point, np.nan, np.nan
    draws = rng.choice(vals, size=(n_bootstrap, vals.size), replace=True).mean(axis=1)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _aggregate_session_balanced(values: np.ndarray, session_codes: np.ndarray, n_sessions: int) -> float:
    finite = np.isfinite(values)
    sums = np.bincount(session_codes[finite], weights=values[finite], minlength=n_sessions)
    counts = np.bincount(session_codes[finite], minlength=n_sessions)
    means = np.divide(sums, counts, out=np.full(n_sessions, np.nan), where=counts > 0)
    return float(np.nanmean(means))


def _shuffle_null(sub: pd.DataFrame, rng: np.random.Generator, n_shuffles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge = sub["image_edge_axis_deg"].to_numpy(float)
    cell = sub["session"].astype(str) + "||" + sub["phase"].astype(str)
    group_indices = [np.flatnonzero(cell.to_numpy() == key) for key in sorted(cell.unique())]
    session_codes, unique_sessions = pd.factorize(sub["session"].astype(str), sort=True)
    null_cos = np.full(n_shuffles, np.nan)
    null_rms = np.full(n_shuffles, np.nan)
    null_cos_session = np.full((n_shuffles, len(unique_sessions)), np.nan)
    null_rms_session = np.full((n_shuffles, len(unique_sessions)), np.nan)
    for b in range(n_shuffles):
        shuffled = edge.copy()
        for idx in group_indices:
            if idx.size > 1:
                shuffled[idx] = edge[idx][rng.permutation(idx.size)]
        cos2, rms = _derive_endpoints(sub, shuffled)
        null_cos[b] = _aggregate_session_balanced(cos2, session_codes, len(unique_sessions))
        null_rms[b] = _aggregate_session_balanced(rms, session_codes, len(unique_sessions))
        for values, target in [(cos2, null_cos_session), (rms, null_rms_session)]:
            finite = np.isfinite(values)
            sums = np.bincount(session_codes[finite], weights=values[finite], minlength=len(unique_sessions))
            counts = np.bincount(session_codes[finite], minlength=len(unique_sessions))
            target[b] = np.divide(sums, counts, out=np.full(len(unique_sessions), np.nan), where=counts > 0)
    return null_cos, null_rms, null_cos_session, null_rms_session


def _row_for_metric(
    *, stratum: str, subset: str, metric: str, values: np.ndarray, sessions: np.ndarray,
    null: np.ndarray, null_session: np.ndarray, n_windows: int, n_trials: int,
    rng: np.random.Generator, n_bootstrap: int,
) -> dict[str, object]:
    point, lo, hi = _bootstrap_session_mean(values, sessions, rng, n_bootstrap)
    observed_session = _session_values(values, sessions)
    null_session_mean = np.nanmean(null_session, axis=0)
    session_effects = observed_session - null_session_mean
    if session_effects.size >= 2 and n_bootstrap > 0:
        effect_boot = rng.choice(session_effects, size=(n_bootstrap, session_effects.size), replace=True).mean(axis=1)
        effect_ci_low, effect_ci_high = np.quantile(effect_boot, [0.025, 0.975])
        effect_bootstrap_p_two_sided = min(1.0, 2.0 * min(np.mean(effect_boot <= 0), np.mean(effect_boot >= 0)))
    else:
        effect_ci_low = effect_ci_high = effect_bootstrap_p_two_sided = np.nan
    null = null[np.isfinite(null)]
    null_mean = float(null.mean()) if null.size else np.nan
    effect = point - null_mean
    # Directional test: is paired contour matching larger than shuffled pairing?
    p_greater = float((1 + np.count_nonzero(null >= point)) / (1 + null.size)) if null.size else np.nan
    return {
        "display_geometry_stratum": stratum,
        "analysis_subset": subset,
        "metric": metric,
        "n_windows": int(n_windows),
        "n_trials": int(n_trials),
        "n_sessions": int(pd.Series(sessions).nunique()),
        "session_balanced_observed": point,
        "session_bootstrap_ci_low": lo,
        "session_bootstrap_ci_high": hi,
        "shuffle_mean": null_mean,
        "shuffle_ci_low": float(np.quantile(null, 0.025)) if null.size else np.nan,
        "shuffle_ci_high": float(np.quantile(null, 0.975)) if null.size else np.nan,
        "observed_minus_shuffle": effect,
        "session_effect_bootstrap_ci_low": float(effect_ci_low),
        "session_effect_bootstrap_ci_high": float(effect_ci_high),
        "session_effect_bootstrap_p_two_sided": float(effect_bootstrap_p_two_sided),
        "shuffle_p_greater": p_greater,
    }


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "all_reviewed": "All",
        "native_16x9_fullscreen": "Native 16:9",
        "stretched_4x3_fullscreen": "Stretched 4:3",
        "square_fullscreen": "Square",
        "reduced_size": "Reduced",
    }
    subset_titles = {"all_windows": "All windows", "reliable_axes": "Reliable axes", "high_confidence": "High confidence"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), sharex=True)
    for col, subset in enumerate(SUBSETS):
        for row, metric in enumerate(["drift_edge_cos2", "rms_along_minus_across_arcmin"]):
            ax = axes[row, col]
            block = summary[(summary.analysis_subset == subset) & (summary.metric == metric)].set_index("display_geometry_stratum").reindex(STRATA)
            x = np.arange(len(STRATA))
            y = block["session_balanced_observed"].to_numpy(float)
            lo = block["session_bootstrap_ci_low"].to_numpy(float)
            hi = block["session_bootstrap_ci_high"].to_numpy(float)
            yerr = np.vstack([y - lo, hi - y])
            null = block["shuffle_mean"].to_numpy(float)
            ax.errorbar(x, y, yerr=yerr, fmt="o", color="#1261a0", capsize=3, label="paired local edge")
            ax.scatter(x, null, marker="x", s=55, color="#c44e52", label="pairing-shuffle mean")
            ax.axhline(0, color="0.45", lw=1)
            ax.set_xticks(x, [labels[s] for s in STRATA], rotation=28, ha="right")
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(subset_titles[subset])
                ax.set_ylabel("cos(2Δ), + = edge-parallel" if col == 0 else "")
            else:
                ax.set_ylabel("RMS along − across (arcmin)" if col == 0 else "")
            if col == 2 and row == 0:
                ax.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("BackImage contour matching by display geometry\nCIs bootstrap sessions; shuffle preserves edge/drift marginals within session × phase", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    windows = pd.read_csv(args.windows)
    trials = pd.read_csv(args.trials)
    keep = [
        "session", "trial_idx", "image_file", "source_aspect", "screen_area_fraction",
        "relative_horizontal_magnification", "nominal_size_deg",
    ]
    joined = windows.merge(trials[keep], on=["session", "trial_idx"], how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"] != "both", ["session", "trial_idx"]].drop_duplicates()
        raise RuntimeError(f"{len(missing)} reviewed trials did not match the audit table")
    joined = joined.drop(columns="_merge")
    joined["display_geometry_stratum"] = _display_stratum(joined)
    if joined["display_geometry_stratum"].eq("unclassified").any():
        bad = joined.loc[joined.display_geometry_stratum == "unclassified", ["source_aspect", "screen_area_fraction"]].drop_duplicates()
        raise RuntimeError(f"Unclassified display geometries:\n{bad}")
    joined["drift_edge_cos2_recomputed"], joined["rms_along_minus_across_arcmin"] = _derive_endpoints(joined)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out_dir / "reviewed_windows_with_display_geometry.csv", index=False)
    rows: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []
    null_arrays: dict[str, np.ndarray] = {}
    for stratum in STRATA:
        geometry = joined if stratum == "all_reviewed" else joined[joined.display_geometry_stratum == stratum]
        for subset, selector in SUBSETS.items():
            sub = geometry.loc[selector(geometry)].copy()
            required = ["image_edge_axis_deg", "drift_orientation_deg", "cov_xx_deg2", "cov_xy_deg2", "cov_yy_deg2"]
            sub = sub.dropna(subset=required)
            if sub.empty:
                continue
            observed_cos, observed_rms = _derive_endpoints(sub)
            null_cos, null_rms, null_cos_session, null_rms_session = _shuffle_null(sub, rng, args.n_shuffles)
            key = f"{stratum}__{subset}"
            null_arrays[key + "__drift_edge_cos2"] = null_cos
            null_arrays[key + "__rms_along_minus_across_arcmin"] = null_rms
            sessions = sub["session"].to_numpy()
            n_trials = sub[["session", "trial_idx"]].drop_duplicates().shape[0]
            session_names = np.asarray(sorted(sub["session"].astype(str).unique()))
            for metric, values, null_by_session in [
                ("drift_edge_cos2", observed_cos, null_cos_session),
                ("rms_along_minus_across_arcmin", observed_rms, null_rms_session),
            ]:
                observed_by_session = _session_values(values, sessions)
                shuffled_by_session = np.nanmean(null_by_session, axis=0)
                for session, observed_value, shuffled_value in zip(session_names, observed_by_session, shuffled_by_session, strict=True):
                    session_rows.append({
                        "display_geometry_stratum": stratum,
                        "analysis_subset": subset,
                        "metric": metric,
                        "session": session,
                        "observed_session_mean": observed_value,
                        "shuffle_session_mean": shuffled_value,
                        "observed_minus_shuffle": observed_value - shuffled_value,
                    })
            rows.append(_row_for_metric(
                stratum=stratum, subset=subset, metric="drift_edge_cos2", values=observed_cos,
                sessions=sessions, null=null_cos, null_session=null_cos_session, n_windows=len(sub), n_trials=n_trials,
                rng=rng, n_bootstrap=args.n_bootstrap,
            ))
            rows.append(_row_for_metric(
                stratum=stratum, subset=subset, metric="rms_along_minus_across_arcmin", values=observed_rms,
                sessions=sessions, null=null_rms, null_session=null_rms_session, n_windows=len(sub), n_trials=n_trials,
                rng=rng, n_bootstrap=args.n_bootstrap,
            ))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "geometry_stratified_contour_matching_summary.csv", index=False)
    pd.DataFrame(session_rows).to_csv(args.out_dir / "geometry_stratified_session_effects.csv", index=False)
    np.savez_compressed(args.out_dir / "pairing_shuffle_distributions.npz", **null_arrays)
    _plot(summary, args.out_dir / "geometry_stratified_contour_matching.png")

    metadata = {
        "windows_csv": str(args.windows),
        "trial_audit_csv": str(args.trials),
        "n_reviewed_windows": int(len(joined)),
        "n_sessions": int(joined.session.nunique()),
        "n_trials": int(joined[["session", "trial_idx"]].drop_duplicates().shape[0]),
        "stratum_window_counts": joined.display_geometry_stratum.value_counts().to_dict(),
        "stratum_trial_counts": joined.drop_duplicates(["session", "trial_idx"]).display_geometry_stratum.value_counts().to_dict(),
        "subsets": {"all_windows": "none", "reliable_axes": "coherence >= .20 and anisotropy >= .20", "high_confidence": "coherence >= .50 and anisotropy >= .50"},
        "n_bootstrap": int(args.n_bootstrap),
        "n_shuffles": int(args.n_shuffles),
        "seed": int(args.seed),
        "shuffle": "permute image_edge_axis_deg within session x phase, separately for each geometry stratum and confidence subset",
        "inference_unit": "session (mean within session, then equal weight across sessions)",
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

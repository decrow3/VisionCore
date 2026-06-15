"""Cheap posthoc audits for BackImage latent real-vs-random effects.

This script operates only on saved outputs from
``run_backimage_latent_information_screen``.  It does not rerun the twin.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta"
)
DEFAULT_SOURCE = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)


def _session_stats(values: np.ndarray, sessions: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float]:
    df = pd.DataFrame({"value": np.asarray(values, dtype=np.float64), "session": np.asarray(sessions)})
    df = df[np.isfinite(df["value"])]
    if df.empty:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_le_0": float("nan"),
            "p_ge_0": float("nan"),
            "fraction_positive_windows": float("nan"),
            "n_windows": 0,
            "n_sessions": 0,
        }
    session_mean = df.groupby("session")["value"].mean().to_numpy(dtype=np.float64)
    out = {
        "mean": float(np.mean(session_mean)),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_le_0": float("nan"),
        "p_ge_0": float("nan"),
        "fraction_positive_windows": float(np.mean(df["value"].to_numpy(dtype=np.float64) > 0.0)),
        "n_windows": int(df.shape[0]),
        "n_sessions": int(session_mean.size),
    }
    if session_mean.size > 1 and int(n_bootstrap) > 0:
        draws = session_mean[rng.integers(0, session_mean.size, size=(int(n_bootstrap), session_mean.size))]
        boot = np.mean(draws, axis=1)
        out.update(
            {
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "p_le_0": float((1.0 + np.count_nonzero(boot <= 0.0)) / (boot.size + 1.0)),
                "p_ge_0": float((1.0 + np.count_nonzero(boot >= 0.0)) / (boot.size + 1.0)),
            }
        )
    return out


def _candidate_pivot(block: pd.DataFrame) -> pd.DataFrame:
    pivot = block.pivot_table(
        index=["window_row", "window_id", "session"],
        columns="candidate",
        values="decode_score_neg_mse",
        aggfunc="mean",
    )
    random_cols = [col for col in pivot.columns if str(col).startswith("random_axis_")]
    if random_cols:
        pivot["random_axis_mean"] = pivot[random_cols].mean(axis=1)
    return pivot.reset_index()


def _real_random_delta(per_window: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    group_cols = [
        "motion_scale_id",
        "motion_scale_kind",
        "motion_scale_value",
        "motion_scale_label",
        "latent_name",
        "latent_family",
        "latent_scope",
        "observer",
        "pca_k",
    ]
    for key, block in per_window.groupby(group_cols, sort=True):
        pivot = _candidate_pivot(block)
        if "real_drift_axis" not in pivot.columns or "random_axis_mean" not in pivot.columns:
            continue
        out = pivot[["window_row", "window_id", "session"]].copy()
        out["real_minus_random"] = pivot["real_drift_axis"].to_numpy(dtype=np.float64) - pivot["random_axis_mean"].to_numpy(dtype=np.float64)
        for col, value in zip(group_cols, key, strict=True):
            out[col] = value
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _motion_scale_table(motion: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    source = motion[motion["candidate"].astype(str) != "static"].copy()
    dedup = source.drop_duplicates(["window_row", "motion_scale_id"]).copy()
    dedup = dedup.merge(
        windows[["window_row", "observed_rms_radius_deg"]],
        on="window_row",
        how="left",
    )
    observed = dedup["observed_rms_radius_deg"].to_numpy(dtype=np.float64)
    actual = dedup["rms_radius_deg"].to_numpy(dtype=np.float64)
    dedup["effective_observed_rms_scale"] = np.where(observed > 0.0, actual / observed, np.nan)
    return dedup


def _load_source_features(source_path: Path, windows: pd.DataFrame) -> pd.DataFrame:
    if not source_path.exists():
        return windows.copy()
    source = pd.read_csv(source_path)
    work = windows.copy()
    feature_cols = [
        "image_patch_rms_contrast",
        "image_patch_std",
        "image_gradient_energy",
        "image_edge_density",
        "image_high_freq_power_fraction",
        "image_spectrum_anisotropy",
        "regime",
    ]
    rows: list[dict[str, Any]] = []
    source_key = source.copy()
    for _, row in work.iterrows():
        mask = (
            (source_key["session"].astype(str) == str(row["session"]))
            & (source_key["trial_idx"].astype(int) == int(row["trial_idx"]))
            & np.isclose(source_key["drift_orientation_deg"].astype(float), float(row["real_drift_axis_deg"]), rtol=1e-7, atol=1e-7)
            & np.isclose(source_key["image_edge_axis_deg"].astype(float), float(row["edge_axis_deg"]), rtol=1e-7, atol=1e-7)
        )
        if "rms_radius_deg" in source_key.columns and pd.notna(row.get("observed_rms_radius_deg", np.nan)):
            mask &= np.isclose(source_key["rms_radius_deg"].astype(float), float(row["observed_rms_radius_deg"]), rtol=1e-7, atol=1e-7)
        match = source_key.loc[mask]
        extra = {"window_row": int(row["window_row"])}
        if not match.empty:
            first = match.iloc[0]
            for col in feature_cols:
                if col in first.index:
                    extra[col] = first[col]
        rows.append(extra)
    return work.merge(pd.DataFrame(rows), on="window_row", how="left")


def _effective_scale_audit(delta: pd.DataFrame, scale_meta: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    merged = delta.merge(
        scale_meta[
            [
                "window_row",
                "motion_scale_id",
                "rms_clipped_low",
                "rms_clipped_high",
                "raw_rms_radius_deg",
                "rms_radius_deg",
                "effective_observed_rms_scale",
            ]
        ],
        on=["window_row", "motion_scale_id"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    group_cols = ["latent_name", "observer", "pca_k", "motion_scale_id", "motion_scale_value", "motion_scale_label"]
    for key, block in merged.groupby(group_cols, sort=True):
        common = dict(zip(group_cols, key, strict=True))
        rows.append(
            {
                **common,
                "split": "all",
                "fraction_rms_clipped_high": float(np.mean(block["rms_clipped_high"].astype(bool))),
                "median_effective_observed_rms_scale": float(np.nanmedian(block["effective_observed_rms_scale"])),
                "median_actual_rms_radius_deg": float(np.nanmedian(block["rms_radius_deg"])),
                **{f"delta_{k}": v for k, v in _session_stats(block["real_minus_random"], block["session"], rng=rng, n_bootstrap=n_bootstrap).items()},
            }
        )
        for clipped_value, split in ((False, "unclipped"), (True, "high_clipped")):
            sub = block[block["rms_clipped_high"].astype(bool) == clipped_value]
            if sub.empty:
                continue
            rows.append(
                {
                    **common,
                    "split": split,
                    "fraction_rms_clipped_high": float(np.mean(sub["rms_clipped_high"].astype(bool))),
                    "median_effective_observed_rms_scale": float(np.nanmedian(sub["effective_observed_rms_scale"])),
                    "median_actual_rms_radius_deg": float(np.nanmedian(sub["rms_radius_deg"])),
                    **{f"delta_{k}": v for k, v in _session_stats(sub["real_minus_random"], sub["session"], rng=rng, n_bootstrap=n_bootstrap).items()},
                }
            )
    return pd.DataFrame(rows)


def _subsample_audit(delta: pd.DataFrame, *, rng: np.random.Generator, draws: int, sizes: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["latent_name", "observer", "pca_k", "motion_scale_id", "motion_scale_value", "motion_scale_label"]
    for key, block in delta.groupby(group_cols, sort=True):
        values = block["real_minus_random"].to_numpy(dtype=np.float64)
        sessions = block["session"].to_numpy()
        n = values.size
        full = pd.DataFrame({"value": values, "session": sessions}).groupby("session")["value"].mean().mean()
        for size in sizes:
            if size > n:
                continue
            means = np.empty(int(draws), dtype=np.float64)
            for j in range(int(draws)):
                idx = rng.choice(n, size=int(size), replace=False)
                sub = pd.DataFrame({"value": values[idx], "session": sessions[idx]})
                means[j] = sub.groupby("session")["value"].mean().mean()
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "subsample_n": int(size),
                    "full_session_mean": float(full),
                    "subsample_mean": float(np.mean(means)),
                    "subsample_sd": float(np.std(means)),
                    "subsample_p05": float(np.percentile(means, 5)),
                    "subsample_p50": float(np.percentile(means, 50)),
                    "subsample_p95": float(np.percentile(means, 95)),
                    "prob_positive": float(np.mean(means > 0.0)),
                    "prob_le_minus_2": float(np.mean(means <= -2.0)),
                    "prob_ge_plus_2": float(np.mean(means >= 2.0)),
                    "n_draws": int(draws),
                }
            )
    return pd.DataFrame(rows)


def _leave_session_out(delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["latent_name", "observer", "pca_k", "motion_scale_id", "motion_scale_value", "motion_scale_label"]
    for key, block in delta.groupby(group_cols, sort=True):
        sessions = sorted(block["session"].astype(str).unique())
        for sess in sessions:
            sub = block[block["session"].astype(str) != sess]
            if sub.empty:
                continue
            mean = sub.groupby("session")["real_minus_random"].mean().mean()
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "left_out_session": sess,
                    "leave_one_session_out_mean": float(mean),
                    "n_windows": int(sub.shape[0]),
                    "n_sessions": int(sub["session"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _regime_strata(delta: pd.DataFrame, windows_features: pd.DataFrame, scale_meta: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    merged = delta.merge(windows_features, on=["window_row", "window_id", "session"], how="left")
    merged = merged.merge(
        scale_meta[["window_row", "motion_scale_id", "rms_clipped_high", "effective_observed_rms_scale"]],
        on=["window_row", "motion_scale_id"],
        how="left",
    )
    available = {
        "image_orientation_coherence": "edge_coherence",
        "drift_anisotropy": "drift_anisotropy",
        "image_patch_rms_contrast": "rms_contrast",
        "image_edge_density": "edge_density",
        "image_high_freq_power_fraction": "high_freq_power",
        "drift_edge_cos2": "real_edge_alignment_cos2",
        "observed_rms_radius_deg": "observed_rms",
        "effective_observed_rms_scale": "effective_scale",
    }
    rows: list[dict[str, Any]] = []
    group_cols = ["latent_name", "observer", "pca_k", "motion_scale_id", "motion_scale_value", "motion_scale_label"]
    for key, block in merged.groupby(group_cols, sort=True):
        common = dict(zip(group_cols, key, strict=True))
        for col, label in available.items():
            if col not in block.columns:
                continue
            values = pd.to_numeric(block[col], errors="coerce")
            if values.notna().sum() < 12 or values.nunique(dropna=True) < 2:
                continue
            median = float(values.median())
            for side, mask in (
                ("low", values <= median),
                ("high", values > median),
            ):
                sub = block[mask.to_numpy()]
                if sub.empty:
                    continue
                stats = _session_stats(sub["real_minus_random"], sub["session"], rng=rng, n_bootstrap=n_bootstrap)
                rows.append(
                    {
                        **common,
                        "stratifier": label,
                        "source_column": col,
                        "split": side,
                        "threshold_median": median,
                        **{f"delta_{k}": v for k, v in stats.items()},
                    }
                )
        if "rms_clipped_high" in block.columns:
            for clipped, split in ((False, "unclipped"), (True, "high_clipped")):
                sub = block[block["rms_clipped_high"].astype(bool) == clipped]
                if sub.empty:
                    continue
                stats = _session_stats(sub["real_minus_random"], sub["session"], rng=rng, n_bootstrap=n_bootstrap)
                rows.append(
                    {
                        **common,
                        "stratifier": "clipping",
                        "source_column": "rms_clipped_high",
                        "split": split,
                        "threshold_median": float("nan"),
                        **{f"delta_{k}": v for k, v in stats.items()},
                    }
                )
    return pd.DataFrame(rows)


def _write_summary(run_dir: Path, effective: pd.DataFrame, subsample: pd.DataFrame, lso: pd.DataFrame, strata: pd.DataFrame) -> None:
    lines = [
        "# BackImage Real-vs-Random Audit",
        "",
        f"Run: `{run_dir}`",
        "",
        "## Effective Scale",
        "",
    ]
    target = effective[
        (effective["split"].isin(["all", "unclipped", "high_clipped"]))
        & (effective["latent_name"].isin(["gabor_local_field", "pyramid_local_field"]))
        & (effective["pca_k"].isin([4, 8]))
    ].copy()
    for _, row in target.sort_values(["latent_name", "pca_k", "motion_scale_value", "split"]).iterrows():
        if row["split"] == "all" or row["motion_scale_value"] in (0.25, 1.0):
            lines.append(
                f"- `{row['latent_name']}` k=`{int(row['pca_k'])}` scale `{row['motion_scale_label']}` `{row['split']}`: "
                f"real-random `{row['delta_mean']:+.3f}` [`{row['delta_ci_low']:+.3f}`, `{row['delta_ci_high']:+.3f}`], "
                f"n=`{int(row['delta_n_windows'])}`, clipped=`{row['fraction_rms_clipped_high']:.3f}`, "
                f"median effective scale=`{row['median_effective_observed_rms_scale']:.3f}`."
            )
    lines.extend(["", "## Subsampling", ""])
    sub_target = subsample[
        (subsample["latent_name"].isin(["gabor_local_field", "pyramid_local_field"]))
        & (subsample["pca_k"].isin([4, 8]))
        & (subsample["motion_scale_value"].isin([0.25, 1.0]))
    ].copy()
    for _, row in sub_target.sort_values(["latent_name", "pca_k", "motion_scale_value", "subsample_n"]).iterrows():
        lines.append(
            f"- `{row['latent_name']}` k=`{int(row['pca_k'])}` scale `{row['motion_scale_label']}` n=`{int(row['subsample_n'])}`: "
            f"p05/p50/p95 `{row['subsample_p05']:+.3f}`/`{row['subsample_p50']:+.3f}`/`{row['subsample_p95']:+.3f}`, "
            f"P(positive)=`{row['prob_positive']:.3f}`, P(<=-2)=`{row['prob_le_minus_2']:.3f}`."
        )
    lines.extend(["", "## Leave-Session-Out", ""])
    lso_summary = (
        lso.groupby(["latent_name", "observer", "pca_k", "motion_scale_id", "motion_scale_value", "motion_scale_label"])["leave_one_session_out_mean"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    lso_target = lso_summary[
        (lso_summary["latent_name"].isin(["gabor_local_field", "pyramid_local_field"]))
        & (lso_summary["pca_k"].isin([4, 8]))
        & (lso_summary["motion_scale_value"].isin([0.25, 1.0]))
    ]
    for _, row in lso_target.sort_values(["latent_name", "pca_k", "motion_scale_value"]).iterrows():
        lines.append(
            f"- `{row['latent_name']}` k=`{int(row['pca_k'])}` scale `{row['motion_scale_label']}`: "
            f"LSO min/median/max `{row['min']:+.3f}`/`{row['median']:+.3f}`/`{row['max']:+.3f}`."
        )
    lines.extend(["", "## Regime Strata", ""])
    if not strata.empty:
        candidates = strata[
            (strata["split"] == "high")
            & (strata["motion_scale_value"].isin([0.25, 1.0]))
            & (strata["pca_k"].isin([4, 8]))
        ].sort_values("delta_mean", ascending=False)
        for _, row in candidates.head(16).iterrows():
            lines.append(
                f"- `{row['latent_name']}` k=`{int(row['pca_k'])}` scale `{row['motion_scale_label']}` high `{row['stratifier']}`: "
                f"real-random `{row['delta_mean']:+.3f}` [`{row['delta_ci_low']:+.3f}`, `{row['delta_ci_high']:+.3f}`], "
                f"n=`{int(row['delta_n_windows'])}`."
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `posthoc_real_random_effective_scale_audit.csv`",
            "- `posthoc_real_random_subsample_audit.csv`",
            "- `posthoc_real_random_leave_session_out.csv`",
            "- `posthoc_real_random_regime_strata.csv`",
            "",
        ]
    )
    (run_dir / "posthoc_real_random_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def audit(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    rng = np.random.default_rng(int(args.seed))
    per_window = pd.read_csv(run_dir / "decode_score_by_window_candidate.csv")
    motion = pd.read_csv(run_dir / "candidate_motion_metadata.csv")
    windows = pd.read_csv(run_dir / "analysis_windows.csv")
    delta = _real_random_delta(per_window)
    scale_meta = _motion_scale_table(motion, windows)
    windows_features = _load_source_features(Path(args.source_windows), windows)

    effective = _effective_scale_audit(delta, scale_meta, rng=rng, n_bootstrap=int(args.n_bootstrap))
    subsample = _subsample_audit(delta, rng=rng, draws=int(args.n_subsample_draws), sizes=[int(v) for v in str(args.subsample_sizes).split(",") if v])
    lso = _leave_session_out(delta)
    strata = _regime_strata(delta, windows_features, scale_meta, rng=rng, n_bootstrap=int(args.n_bootstrap))

    effective.to_csv(run_dir / "posthoc_real_random_effective_scale_audit.csv", index=False)
    subsample.to_csv(run_dir / "posthoc_real_random_subsample_audit.csv", index=False)
    lso.to_csv(run_dir / "posthoc_real_random_leave_session_out.csv", index=False)
    strata.to_csv(run_dir / "posthoc_real_random_regime_strata.csv", index=False)
    _write_summary(run_dir, effective, subsample, lso, strata)

    payload = {
        "run_dir": str(run_dir),
        "source_windows": str(args.source_windows),
        "n_bootstrap": int(args.n_bootstrap),
        "n_subsample_draws": int(args.n_subsample_draws),
        "subsample_sizes": str(args.subsample_sizes),
        "seed": int(args.seed),
    }
    (run_dir / "posthoc_real_random_audit_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote BackImage real-vs-random audit to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-windows", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-subsample-draws", type=int, default=5000)
    parser.add_argument("--subsample-sizes", default="64,128")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    audit(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Feature-preservation decoder audit from cached BackImage endpoint responses."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

try:
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_latent_information_screen import (
        _choose_alpha,
        _extract_latents,
        _mean_r2,
        _split_outer,
        _standardize_train_test,
    )
    from .run_backimage_twin_drift_geometry import _clip_patch
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        _choose_alpha,
        _extract_latents,
        _mean_r2,
        _split_outer,
        _standardize_train_test,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch


DEFAULT_AUDIT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_twin_stability_metric_audit"
)
DEFAULT_OUT_DIR = DEFAULT_AUDIT_DIR / "endpoint_feature_preservation_static_decoder"


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _session_bootstrap_mean(values: np.ndarray, sessions: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    sessions = np.asarray(sessions)
    ok = np.isfinite(values) & pd.notna(sessions)
    values = values[ok]
    sessions = sessions[ok]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_windows": 0, "n_sessions": 0}
    unique = np.asarray(sorted(pd.unique(sessions)))
    session_means = pd.Series(values).groupby(pd.Series(sessions)).mean().reindex(unique).to_numpy(dtype=np.float64)
    mean = float(np.nanmean(session_means))
    if int(n_bootstrap) <= 0 or unique.size < 2:
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan"), "n_windows": int(values.size), "n_sessions": int(unique.size)}
    draws = rng.choice(session_means, size=(int(n_bootstrap), session_means.size), replace=True)
    boot = np.nanmean(draws, axis=1)
    return {
        "mean": mean,
        "ci_low": float(np.nanpercentile(boot, 2.5)),
        "ci_high": float(np.nanpercentile(boot, 97.5)),
        "n_windows": int(values.size),
        "n_sessions": int(unique.size),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 3:
        return float("nan")
    if np.nanstd(x[ok]) <= 1e-12 or np.nanstd(y[ok]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _demean(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return (series - series.groupby(pd.Series(sessions)).transform("mean")).to_numpy(dtype=np.float64)


def _bootstrap_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    within_session: bool,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    sub = df.dropna(subset=["session_id", x_col, y_col]).copy()
    sessions = np.asarray(sorted(sub["session_id"].unique()))
    if within_session:
        x = _demean(sub[x_col].to_numpy(dtype=np.float64), sub["session_id"].to_numpy())
        y = _demean(sub[y_col].to_numpy(dtype=np.float64), sub["session_id"].to_numpy())
        observed = _corr(x, y)
        session_values = sub["session_id"].to_numpy()
        pieces = [(x[session_values == sess], y[session_values == sess]) for sess in sessions]
    else:
        sess = sub.groupby("session_id")[[x_col, y_col]].mean()
        x = sess[x_col].to_numpy(dtype=np.float64)
        y = sess[y_col].to_numpy(dtype=np.float64)
        observed = _corr(x, y)
    if int(n_bootstrap) <= 0 or sessions.size < 2:
        return {"r": observed, "ci_low": float("nan"), "ci_high": float("nan")}
    vals = []
    for _ in range(int(n_bootstrap)):
        draw = rng.integers(0, sessions.size, size=sessions.size)
        if within_session:
            bx = np.concatenate([pieces[j][0] for j in draw])
            by = np.concatenate([pieces[j][1] for j in draw])
            vals.append(_corr(bx, by))
        else:
            vals.append(_corr(x[draw], y[draw]))
    arr = np.asarray(vals, dtype=np.float64)
    return {"r": observed, "ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))}


def _load_endpoint_table(audit_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    by_window = pd.read_csv(audit_dir / "twin_stability_metric_by_window.csv")
    endpoints = np.load(audit_dir / "twin_endpoint_tail_vectors.npz")
    endpoint_dict = {key: endpoints[key] for key in endpoints.files}
    metadata_path = audit_dir / "posthoc_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    order = pd.DataFrame({"window_row": endpoint_dict["window_row"].astype(int), "endpoint_order": np.arange(endpoint_dict["window_row"].shape[0])})
    rows = by_window.merge(order, on="window_row", how="inner", validate="one_to_one").sort_values("endpoint_order").reset_index(drop=True)
    return rows, endpoint_dict, metadata


def _compute_or_load_latents(rows: pd.DataFrame, out_dir: Path, metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache_path = out_dir / "feature_latents.npz"
    if cache_path.exists() and not bool(args.recompute_latents):
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}
    cfg = metadata.get("source_stability_metadata", {}).get("config", {})
    patch_size_px = int(cfg.get("patch_size_px", 540))
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    latents: dict[str, list[np.ndarray]] = {}
    wanted = set(_parse_list(args.latent_names))
    for idx, row in rows.iterrows():
        canvas_key = (str(row["session_id"]), int(row["image_id"]))
        if canvas_key not in canvas_cache:
            canvas_cache[canvas_key] = _backimage_canvas(str(row["session_id"]), int(row["image_id"]))
        canvas, ppd, screen_shape = canvas_cache[canvas_key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), patch_size_px)
        rec = _extract_latents(
            patch,
            latent_crop_px=int(args.latent_crop_px),
            center_crop_px=int(args.center_crop_px),
            local_field_grid=int(args.local_field_grid),
        )
        rec = {key: value for key, value in rec.items() if key in wanted}
        if set(rec) != wanted:
            missing = sorted(wanted.difference(rec))
            raise ValueError(f"Missing requested latents {missing}; available={sorted(rec)}")
        for key, value in rec.items():
            latents.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
        done = int(idx) + 1
        if done == 1 or done == rows.shape[0] or done % 32 == 0:
            print(f"[endpoint-feature-preservation] latents {done}/{rows.shape[0]}", flush=True)
    out = {key: np.vstack(values).astype(np.float32) for key, values in latents.items()}
    np.savez_compressed(cache_path, **out)
    return out


def _endpoint_response_sets(endpoints: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    return {
        "static": [np.asarray(endpoints["base"], dtype=np.float64)],
        "edge_parallel": [
            np.asarray(endpoints["parallel_plus"], dtype=np.float64),
            np.asarray(endpoints["parallel_minus"], dtype=np.float64),
        ],
        "edge_orthogonal": [
            np.asarray(endpoints["orthogonal_plus"], dtype=np.float64),
            np.asarray(endpoints["orthogonal_minus"], dtype=np.float64),
        ],
    }


def _fit_static_decoder_scores(
    base: np.ndarray,
    candidates: dict[str, list[np.ndarray]],
    latent: np.ndarray,
    sessions: np.ndarray,
    *,
    k: int,
    alphas: list[float],
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = np.asarray(base, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    sessions = np.asarray(sessions)
    n = base.shape[0]
    k_eff = int(min(int(k), latent.shape[1], max(1, n - 2)))
    per_candidate_endpoint_scores: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    per_candidate_endpoint_r2s: dict[str, list[float]] = {name: [] for name in candidates}
    chosen_alphas = []
    splits = _split_outer(sessions, int(outer_folds), int(seed))
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_static_test = _standardize_train_test(base[train_idx], base[test_idx])
        Z_train, Z_test_raw = _standardize_train_test(latent[train_idx], latent[test_idx])
        pca = PCA(n_components=k_eff, svd_solver="full")
        Y_train = pca.fit_transform(Z_train)
        Y_test = pca.transform(Z_test_raw)
        alpha = _choose_alpha(
            X_train,
            Y_train,
            sessions[train_idx],
            alphas=alphas,
            inner_folds=int(inner_folds),
            seed=int(seed) + fold + 1,
        )
        chosen_alphas.append(alpha)
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(X_train, Y_train)
        for candidate, endpoints_for_candidate in candidates.items():
            for endpoint_idx, endpoint in enumerate(endpoints_for_candidate):
                _, X_test = _standardize_train_test(base[train_idx], endpoint[test_idx])
                pred = model.predict(X_test)
                mse = np.mean((Y_test - pred) ** 2, axis=1)
                if len(per_candidate_endpoint_scores[candidate]) <= endpoint_idx:
                    per_candidate_endpoint_scores[candidate].append(np.full(n, np.nan, dtype=np.float64))
                per_candidate_endpoint_scores[candidate][endpoint_idx][test_idx] = -mse
                per_candidate_endpoint_r2s[candidate].append(_mean_r2(Y_test, pred))
    rows = []
    per_window = pd.DataFrame({"window_index": np.arange(n, dtype=int), "session_id": sessions})
    for candidate, score_arrays in per_candidate_endpoint_scores.items():
        stacked = np.vstack(score_arrays)
        per_window[f"{candidate}_score"] = np.nanmean(stacked, axis=0)
        rows.append(
            {
                "candidate": candidate,
                "mean_score_neg_mse": float(np.nanmean(per_window[f"{candidate}_score"])),
                "mean_outer_fold_r2": float(np.nanmean(per_candidate_endpoint_r2s[candidate])),
                "target_dim": int(k_eff),
                "chosen_alpha_median": float(np.nanmedian(chosen_alphas)) if chosen_alphas else float("nan"),
            }
        )
    return pd.DataFrame(rows), per_window


def _summarize_scores(per_window: pd.DataFrame, rows: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    for _, row in rows.iterrows():
        col = f"{row['candidate']}_score"
        stats = _session_bootstrap_mean(per_window[col].to_numpy(dtype=np.float64), per_window["session_id"].to_numpy(), rng=rng, n_bootstrap=n_bootstrap)
        candidate_rows.append({**row.to_dict(), **{f"session_{key}": value for key, value in stats.items()}})
    contrast_specs = [
        ("edge_parallel_minus_edge_orthogonal", "edge_parallel_score", "edge_orthogonal_score", "edge_parallel_preservation_minus_orthogonal"),
        ("edge_parallel_minus_static", "edge_parallel_score", "static_score", "edge_parallel_score_minus_static"),
        ("edge_orthogonal_minus_static", "edge_orthogonal_score", "static_score", "edge_orthogonal_score_minus_static"),
    ]
    contrast_rows = []
    for name, left, right, out_col in contrast_specs:
        if left not in per_window.columns or right not in per_window.columns:
            continue
        diff = per_window[left].to_numpy(dtype=np.float64) - per_window[right].to_numpy(dtype=np.float64)
        per_window[out_col] = diff
        stats = _session_bootstrap_mean(diff, per_window["session_id"].to_numpy(), rng=rng, n_bootstrap=n_bootstrap)
        contrast_rows.append({"contrast": name, "left_score": left, "right_score": right, "per_window_column": out_col, **stats})
    return pd.DataFrame(candidate_rows), pd.DataFrame(contrast_rows)


def _write_predictor_correlations(df: pd.DataFrame, out_dir: Path, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    predictors = [
        "pixel_stability_advantage",
        "pixel_relative_advantage",
        "raw_mse_stability_advantage",
        "response_norm_mse_stability_advantage",
        "per_rate_mse_stability_advantage",
        "diag_whitened_mse_stability_advantage",
        "full_cov_whitened_mse_stability_advantage",
        "drift_edge_align_signed",
        "image_orientation_coherence",
    ]
    outcome_cols = [
        "edge_parallel_preservation_minus_orthogonal",
        "edge_parallel_score_minus_static",
        "edge_orthogonal_score_minus_static",
    ]
    rows = []
    for outcome in outcome_cols:
        if outcome not in df.columns:
            continue
        for predictor in predictors:
            if predictor not in df.columns:
                continue
            for level, within in (("window_within_session", True), ("session_mean", False)):
                corr = _bootstrap_corr(df, predictor, outcome, within_session=within, rng=rng, n_bootstrap=n_bootstrap)
                rows.append({"outcome": outcome, "predictor": predictor, "level": level, **corr})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "feature_preservation_predictor_correlations.csv", index=False)
    return out


def _write_plots(out_dir: Path, contrast_summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    focus = contrast_summary.copy()
    if not focus.empty:
        fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=150)
        y = np.arange(focus.shape[0])
        ax.barh(y, focus["mean"], color="#4878a8")
        ax.errorbar(focus["mean"], y, xerr=[focus["mean"] - focus["ci_low"], focus["ci_high"] - focus["mean"]], fmt="none", color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(focus["contrast"], fontsize=8)
        ax.set_xlabel("session-mean score contrast")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_preservation_contrasts.png", dpi=150)
        plt.close(fig)
    cf = corr[
        (corr["level"] == "window_within_session")
        & (corr["outcome"] == "edge_parallel_preservation_minus_orthogonal")
    ].copy()
    if not cf.empty:
        cf = cf.sort_values("r")
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        y = np.arange(cf.shape[0])
        ax.barh(y, cf["r"], color="#6a8f5f")
        ax.errorbar(cf["r"], y, xerr=[cf["r"] - cf["ci_low"], cf["ci_high"] - cf["r"]], fmt="none", color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(cf["predictor"], fontsize=7)
        ax.set_xlabel("within-session correlation")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_preservation_predictor_correlations.png", dpi=150)
        plt.close(fig)


def _write_report(
    out_dir: Path,
    latent_name: str,
    candidate_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    corr: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        f"# Endpoint Feature-Preservation Decoder Audit: {latent_name}",
        "",
        "This cache-only pass trains a decoder from static endpoint twin responses to image-feature PCs, then tests that same decoder on static, edge-parallel, and edge-orthogonal endpoint responses.",
        "",
        "Important limitation: the saved endpoint cache does not contain real-drift-axis or random-axis endpoint responses, so `real - random` and `real - edge_parallel` preservation contrasts are not available in this pass.",
        "",
        "## Candidate Scores",
        "",
    ]
    for _, row in candidate_summary.iterrows():
        lines.append(
            f"- `{row['candidate']}`: score `{row['session_mean']:+.4f}` CI "
            f"`[{row['session_ci_low']:+.4f}, {row['session_ci_high']:+.4f}]`, "
            f"R2 `{row['mean_outer_fold_r2']:+.4f}`."
        )
    lines.extend(["", "## Preservation Contrasts", ""])
    for _, row in contrast_summary.iterrows():
        lines.append(
            f"- `{row['contrast']}`: `{row['mean']:+.4f}` CI "
            f"`[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]`."
        )
    focus = corr[(corr["level"] == "window_within_session") & (corr["outcome"] == "edge_parallel_preservation_minus_orthogonal")].copy()
    if not focus.empty:
        lines.extend(["", "## Predictors Of Edge-Parallel Preservation Advantage", ""])
        for _, row in focus.sort_values("r", ascending=False).iterrows():
            lines.append(f"- `{row['predictor']}`: r `{row['r']:+.3f}` CI `[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]`.")
    cfg = metadata.get("source_stability_metadata", {}).get("config", {})
    lines.extend(
        [
            "",
            "## Cache Provenance",
            "",
            f"- Endpoint displacement: `{cfg.get('displacement_deg', 'unknown')}` deg.",
            f"- Twin population in endpoint cache: `{cfg.get('twin_population_n', 'unknown')}` sampled units.",
            "",
        ]
    )
    (out_dir / f"{latent_name}_feature_preservation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    rows, endpoints, metadata = _load_endpoint_table(Path(args.audit_dir))
    latents = _compute_or_load_latents(rows, out_dir, metadata, args)
    response_sets = _endpoint_response_sets(endpoints)
    base = np.asarray(endpoints["base"], dtype=np.float64)
    sessions = rows["session_id"].to_numpy()
    pca_k_list = _parse_int_list(args.pca_k_list)
    if pca_k_list != [4]:
        print("[endpoint-feature-preservation] warning: report text is tuned for k=4; running requested k list", flush=True)
    all_candidate = []
    all_contrast = []
    all_window = []
    all_corr = []
    for latent_name, latent in latents.items():
        for k in pca_k_list:
            candidate_raw, per_window = _fit_static_decoder_scores(
                base,
                response_sets,
                latent,
                sessions,
                k=int(k),
                alphas=_parse_float_list(args.ridge_alphas),
                outer_folds=int(args.outer_folds),
                inner_folds=int(args.inner_folds),
                seed=int(args.seed),
            )
            candidate_summary, contrast_summary = _summarize_scores(per_window, candidate_raw, rng=rng, n_bootstrap=int(args.n_bootstrap))
            candidate_summary.insert(0, "pca_k", int(k))
            candidate_summary.insert(0, "latent_name", latent_name)
            contrast_summary.insert(0, "pca_k", int(k))
            contrast_summary.insert(0, "latent_name", latent_name)
            merged = pd.concat([rows.reset_index(drop=True), per_window.drop(columns=["session_id"]).reset_index(drop=True)], axis=1)
            merged.insert(0, "pca_k", int(k))
            merged.insert(0, "latent_name", latent_name)
            corr = _write_predictor_correlations(merged, out_dir, rng, int(args.n_bootstrap))
            corr.insert(0, "pca_k", int(k))
            corr.insert(0, "latent_name", latent_name)
            _write_report(out_dir, latent_name, candidate_summary, contrast_summary, corr, metadata)
            all_candidate.append(candidate_summary)
            all_contrast.append(contrast_summary)
            all_window.append(merged)
            all_corr.append(corr)
    candidate_df = pd.concat(all_candidate, ignore_index=True)
    contrast_df = pd.concat(all_contrast, ignore_index=True)
    window_df = pd.concat(all_window, ignore_index=True)
    corr_df = pd.concat(all_corr, ignore_index=True)
    candidate_df.to_csv(out_dir / "feature_preservation_candidate_scores.csv", index=False)
    contrast_df.to_csv(out_dir / "feature_preservation_contrasts.csv", index=False)
    window_df.to_csv(out_dir / "feature_preservation_by_window.csv", index=False)
    corr_df.to_csv(out_dir / "feature_preservation_predictor_correlations.csv", index=False)
    _write_plots(out_dir, contrast_df, corr_df)
    (out_dir / "posthoc_metadata.json").write_text(
        json.dumps(
            {
                "audit_dir": str(args.audit_dir),
                "out_dir": str(out_dir),
                "latent_names": _parse_list(args.latent_names),
                "pca_k_list": pca_k_list,
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "real_random_note": "Saved endpoint cache contains static, edge-parallel, and edge-orthogonal endpoints only; real/random require a cache extension.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote endpoint feature-preservation audit to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--pca-k-list", default="4")
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute-latents", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

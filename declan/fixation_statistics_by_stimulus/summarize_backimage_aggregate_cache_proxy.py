"""Cache-first aggregate proxy for BackImage FEM information analyses.

This posthoc uses saved ``latent_feature_arrays.npz`` and
``response_feature_arrays.npz`` from the local BackImage latent-information
screen. It does not replace a true aggregate motion-distribution run with
Brownian/OU/shuffled traces. Its purpose is to reuse existing canonical twin
responses for a cheap ensemble-level readout:

- shared/fixed-alpha feature decoding across cached candidate motions;
- session-bootstrap contrasts such as real minus random/edge/static;
- simple response covariance and delta-energy diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from .run_backimage_latent_information_screen import _cross_validated_decode
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _cross_validated_decode


DEFAULT_SOURCE_RUN = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta"
)
DEFAULT_OUT_DIR = DEFAULT_SOURCE_RUN / "aggregate_cache_proxy"


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in _parse_list(text)]


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in _parse_list(text)]


def _parse_response_key(key: str) -> tuple[str, str, str]:
    parts = key.split("__")
    if len(parts) != 3:
        raise ValueError(f"Unexpected response key {key!r}; expected scale__candidate__observer")
    return parts[0], parts[1], parts[2]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: np.asarray(loaded[key]) for key in loaded.files}


def _candidate_arrays(responses: dict[str, np.ndarray]) -> dict[tuple[str, str, str], np.ndarray]:
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for key, arr in responses.items():
        scale_id, candidate, observer = _parse_response_key(key)
        arrays[(scale_id, candidate, observer)] = np.asarray(arr)
    return arrays


def _filter_latents(latents: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    if not names or "all" in names:
        return latents
    out = {name: latents[name] for name in names if name in latents}
    missing = sorted(set(names) - set(out))
    if missing:
        raise ValueError(f"Requested latent names are missing from cache: {missing}")
    return out


def _filter_arrays(
    arrays: dict[tuple[str, str, str], np.ndarray],
    *,
    scale_ids: list[str],
    candidates: list[str],
    observers: list[str],
) -> dict[tuple[str, str, str], np.ndarray]:
    scale_filter = None if not scale_ids or "all" in scale_ids else set(scale_ids)
    candidate_filter = None if not candidates or "all" in candidates else set(candidates)
    observer_filter = None if not observers or "all" in observers else set(observers)
    wants_random_mean = candidate_filter is not None and "random_axis_mean" in candidate_filter
    out = {}
    for key, value in arrays.items():
        scale_id, candidate, observer = key
        if scale_filter is not None and scale_id not in scale_filter:
            continue
        if candidate_filter is not None and candidate not in candidate_filter:
            if not (wants_random_mean and candidate.startswith("random_axis_")):
                continue
        if observer_filter is not None and observer not in observer_filter:
            continue
        out[key] = value
    if not out:
        raise ValueError("No response arrays survived --scale-ids/--candidates/--observers filters")
    return out


def _validate_shapes(windows: pd.DataFrame, latents: dict[str, np.ndarray], arrays: dict[tuple[str, str, str], np.ndarray]) -> None:
    n = int(windows.shape[0])
    errors: list[str] = []
    for name, value in latents.items():
        if value.ndim != 2:
            errors.append(f"latent {name!r} has ndim={value.ndim}, expected 2")
        elif int(value.shape[0]) != n:
            errors.append(f"latent {name!r} has n={value.shape[0]}, expected {n}")
    for key, value in arrays.items():
        if value.ndim != 2:
            errors.append(f"response {key!r} has ndim={value.ndim}, expected 2")
        elif int(value.shape[0]) != n:
            errors.append(f"response {key!r} has n={value.shape[0]}, expected {n}")
    if errors:
        preview = "; ".join(errors[:6])
        suffix = "" if len(errors) <= 6 else f"; ... {len(errors) - 6} more"
        raise ValueError(f"Cache arrays are not aligned to analysis_windows.csv: {preview}{suffix}")


def _session_bootstrap(values: np.ndarray, sessions: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    sessions = np.asarray(sessions)
    ok = np.isfinite(values)
    values = values[ok]
    sessions = sessions[ok]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_sessions": 0}
    session_values = pd.DataFrame({"session": sessions, "value": values}).groupby("session")["value"].mean().to_numpy(dtype=np.float64)
    if session_values.size == 0:
        return {"mean": float(np.nanmean(values)), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(values.size), "n_sessions": 0}
    if int(n_bootstrap) <= 0 or session_values.size == 1:
        ci_low = ci_high = float(np.nanmean(session_values))
    else:
        draws = np.empty(int(n_bootstrap), dtype=np.float64)
        for i in range(int(n_bootstrap)):
            sample = rng.choice(session_values, size=session_values.size, replace=True)
            draws[i] = np.nanmean(sample)
        ci_low, ci_high = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "mean": float(np.nanmean(session_values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(values.size),
        "n_sessions": int(session_values.size),
    }


def _cov_trace(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return float("nan")
    centered = X - np.nanmean(X, axis=0, keepdims=True)
    return float(np.nansum(centered * centered) / max(1, X.shape[0] - 1))


def _participation_ratio(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return float("nan")
    centered = X - np.nanmean(X, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eig = (s * s) / max(1, X.shape[0] - 1)
    denom = float(np.sum(eig * eig))
    return float((np.sum(eig) ** 2) / denom) if denom > 1e-12 else float("nan")


def _response_covariance_summary(arrays: dict[tuple[str, str, str], np.ndarray], metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scale_id, candidate, observer), X in sorted(arrays.items()):
        static = arrays.get((scale_id, "static", observer))
        delta = X - static if static is not None else np.full_like(X, np.nan)
        rows.append(
            {
                "scale_id": scale_id,
                "candidate": candidate,
                "observer": observer,
                "n_windows": int(X.shape[0]),
                "n_units": int(X.shape[1]) if X.ndim == 2 else -1,
                "response_cov_trace": _cov_trace(X),
                "response_participation_ratio": _participation_ratio(X),
                "delta_from_static_cov_trace": _cov_trace(delta),
                "mean_delta_from_static_energy": float(np.nanmean(np.sum(delta.astype(np.float64) ** 2, axis=1))),
            }
        )

    # Random-axis response variability is the only cached multi-motion estimate
    # per image in the current local screen.
    random_by_scale: dict[tuple[str, str], list[np.ndarray]] = {}
    for (scale_id, candidate, observer), X in arrays.items():
        if candidate.startswith("random_axis_") and candidate != "random_axis_mean":
            random_by_scale.setdefault((scale_id, observer), []).append(X)
    for (scale_id, observer), values in sorted(random_by_scale.items()):
        stack = np.stack(values, axis=1).astype(np.float64)  # windows x axes x units
        per_window_motion_trace = []
        for i in range(stack.shape[0]):
            per_window_motion_trace.append(_cov_trace(stack[i]))
        stats = {
            "scale_id": scale_id,
            "candidate": "random_axis_within_window",
            "observer": observer,
            "n_windows": int(stack.shape[0]),
            "n_units": int(stack.shape[2]),
            "response_cov_trace": float("nan"),
            "response_participation_ratio": float("nan"),
            "delta_from_static_cov_trace": float("nan"),
            "mean_delta_from_static_energy": float("nan"),
            "within_image_motion_cov_trace_mean": float(np.nanmean(per_window_motion_trace)),
            "within_image_motion_cov_trace_median": float(np.nanmedian(per_window_motion_trace)),
        }
        rows.append(stats)
    out = pd.DataFrame(rows)
    scale_meta_cols = ["motion_scale_id", "motion_scale_label", "observed_rms_scale", "motion_scale_value"]
    if not metadata.empty and all(col in metadata.columns for col in scale_meta_cols):
        scale_meta = metadata[scale_meta_cols].drop_duplicates().rename(columns={"motion_scale_id": "scale_id"})
        out = out.merge(scale_meta, on="scale_id", how="left")
    return out


def _decode_from_cache(
    latents: dict[str, np.ndarray],
    arrays: dict[tuple[str, str, str], np.ndarray],
    windows: pd.DataFrame,
    *,
    pca_k_list: list[int],
    alphas: list[float],
    alpha_mode: str,
    fixed_alpha: float | None,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = windows["session"].astype(str).to_numpy()
    summary_rows: list[dict[str, Any]] = []
    per_window_cols = {
        "window_row": windows["window_row"].to_numpy() if "window_row" in windows.columns else np.arange(windows.shape[0]),
        "window_id": windows["window_id"].to_numpy() if "window_id" in windows.columns else np.arange(windows.shape[0]),
        "session": sessions,
    }
    per_window_frames: list[pd.DataFrame] = []
    total_jobs = len(latents) * len(arrays) * len(pca_k_list)
    done = 0
    print(f"[aggregate-cache-proxy] decode jobs {total_jobs}", flush=True)
    for latent_name, Z in sorted(latents.items()):
        for (scale_id, candidate, observer), X in sorted(arrays.items()):
            for k in pca_k_list:
                done += 1
                if done == 1 or done == total_jobs or done % 10 == 0:
                    print(
                        "[aggregate-cache-proxy] "
                        f"decode {done}/{total_jobs}: {latent_name} {scale_id} {candidate} {observer} k={k}",
                        flush=True,
                    )
                result = _cross_validated_decode(
                    X,
                    Z,
                    sessions,
                    k=int(k),
                    alphas=alphas,
                    alpha_mode=alpha_mode,
                    fixed_alpha=fixed_alpha,
                    outer_folds=int(outer_folds),
                    inner_folds=int(inner_folds),
                    seed=int(seed),
                )
                row = {
                    "latent_name": latent_name,
                    "scale_id": scale_id,
                    "candidate": candidate,
                    "observer": observer,
                    "pca_k": int(k),
                    "R2_z": float(result["r2"]),
                    "decode_score_neg_mse": float(result["mean_neg_mse"]),
                    "chosen_alpha_median": float(result["chosen_alpha_median"]),
                    "ridge_alpha_mode": str(result["ridge_alpha_mode"]),
                    "target_dim": int(result["target_dim"]),
                }
                summary_rows.append(row)
                per_window_frames.append(
                    pd.DataFrame(
                        {
                            **per_window_cols,
                            "latent_name": latent_name,
                            "scale_id": scale_id,
                            "candidate": candidate,
                            "observer": observer,
                            "pca_k": int(k),
                            "decode_score_neg_mse": np.asarray(result["per_window_score"], dtype=np.float64),
                        }
                    )
                )
    return pd.DataFrame(summary_rows), pd.concat(per_window_frames, ignore_index=True)


def _add_random_axis_mean_scores(decode: pd.DataFrame, per_window: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_decode = decode[decode["candidate"].astype(str).str.startswith("random_axis_")].copy()
    raw_decode = raw_decode[raw_decode["candidate"] != "random_axis_mean"]
    if raw_decode.empty:
        return decode, per_window

    decode_group_cols = ["latent_name", "scale_id", "observer", "pca_k"]
    numeric_mean_cols = ["R2_z", "decode_score_neg_mse"]
    numeric_median_cols = ["chosen_alpha_median"]
    random_rows = []
    for key, block in raw_decode.groupby(decode_group_cols, dropna=False):
        row = {
            "latent_name": key[0],
            "scale_id": key[1],
            "observer": key[2],
            "pca_k": int(key[3]),
            "candidate": "random_axis_mean",
            "ridge_alpha_mode": ",".join(sorted(block["ridge_alpha_mode"].astype(str).unique())),
            "target_dim": int(block["target_dim"].iloc[0]),
        }
        for col in numeric_mean_cols:
            row[col] = float(np.nanmean(block[col].to_numpy(dtype=np.float64)))
        for col in numeric_median_cols:
            row[col] = float(np.nanmedian(block[col].to_numpy(dtype=np.float64)))
        random_rows.append(row)
    decode_out = pd.concat([decode, pd.DataFrame(random_rows)], ignore_index=True)

    raw_per_window = per_window[per_window["candidate"].astype(str).str.startswith("random_axis_")].copy()
    raw_per_window = raw_per_window[raw_per_window["candidate"] != "random_axis_mean"]
    per_group_cols = ["window_row", "window_id", "session", "latent_name", "scale_id", "observer", "pca_k"]
    random_per = (
        raw_per_window.groupby(per_group_cols, dropna=False, as_index=False)["decode_score_neg_mse"]
        .mean()
        .assign(candidate="random_axis_mean")
    )
    per_window_out = pd.concat([per_window, random_per], ignore_index=True)
    return decode_out, per_window_out


def _decode_contrasts(per_window: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    specs = [
        ("real_minus_static", "real_drift_axis", "static"),
        ("real_minus_random", "real_drift_axis", "random_axis_mean"),
        ("real_minus_edge", "real_drift_axis", "edge"),
        ("edge_minus_random", "edge", "random_axis_mean"),
        ("edge_minus_orthogonal", "edge", "edge_orthogonal"),
    ]
    rows: list[dict[str, Any]] = []
    group_cols = ["latent_name", "scale_id", "observer", "pca_k"]
    for key, block in per_window.groupby(group_cols, dropna=False):
        pivot = block.pivot_table(index=["window_row", "session"], columns="candidate", values="decode_score_neg_mse", aggfunc="mean")
        for contrast, left, right in specs:
            if left not in pivot.columns or right not in pivot.columns:
                continue
            diff = pivot[left].to_numpy(dtype=np.float64) - pivot[right].to_numpy(dtype=np.float64)
            stats = _session_bootstrap(diff, pivot.index.get_level_values("session").to_numpy(), rng=rng, n_bootstrap=int(n_bootstrap))
            rows.append(
                {
                    "latent_name": key[0],
                    "scale_id": key[1],
                    "observer": key[2],
                    "pca_k": int(key[3]),
                    "contrast": contrast,
                    "left": left,
                    "right": right,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def _write_report(out_dir: Path, args: argparse.Namespace, decode: pd.DataFrame, contrasts: pd.DataFrame, cov: pd.DataFrame) -> None:
    focus = contrasts[
        contrasts["contrast"].isin(["real_minus_random", "real_minus_edge", "real_minus_static"])
        & contrasts["latent_name"].isin(["gabor_local_field", "pyramid_local_field"])
        & contrasts["pca_k"].isin([4, 8])
    ].copy()
    if not focus.empty:
        focus = focus.sort_values(["latent_name", "pca_k", "scale_id", "contrast"])
    if str(args.ridge_alpha_mode) == "fixed":
        alpha_line = f"- Ridge alpha mode: `fixed`; fixed alpha: `{args.fixed_ridge_alpha}`"
    else:
        alpha_line = f"- Ridge alpha mode: `{args.ridge_alpha_mode}`; fixed alpha is unused"
    lines = [
        "# BackImage Aggregate Cache Proxy",
        "",
        "This is a cache-only proxy, not the full aggregate FEM distribution run.",
        "It reuses saved latent and response summaries from the local BackImage",
        "latent-information screen to estimate ensemble-level decoding and",
        "response-covariance diagnostics.",
        "",
        "## Inputs",
        "",
        f"- Source run: `{args.source_run}`",
        alpha_line,
        f"- PCA k list: `{args.pca_k_list}`",
        "- `random_axis_mean` contrasts average decoded scores from raw random axes.",
        "",
        "## Headline Cached Contrasts",
        "",
    ]
    if focus.empty:
        lines.append("No headline contrasts were available.")
    else:
        for _, row in focus.iterrows():
            lines.append(
                f"- `{row['latent_name']}` k=`{int(row['pca_k'])}` `{row['scale_id']}` "
                f"`{row['contrast']}`: mean `{row['mean']:+.3f}`, "
                f"CI `[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]`, "
                f"n_sessions `{int(row['n_sessions'])}`"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "These cached candidates are fixed axis/scale summaries from the local",
        "`I_z` screen. They do not yet include Brownian, OU, phase-randomized,",
        "or unpaired empirical trace ensembles. Use this output to prioritize and",
        "debug the full aggregate runner, not as a final Figure-level test.",
        "",
        "## Files",
        "",
        "- `aggregate_cache_decode_summary.csv`",
        "- `aggregate_cache_decode_score_by_window.csv`",
        "- `aggregate_cache_decode_contrasts.csv`",
        "- `aggregate_cache_response_covariance_summary.csv`",
        "- `aggregate_cache_metadata.json`",
    ]
    (out_dir / "aggregate_cache_proxy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--latent-names", default="all")
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--candidates", default="static,real_drift_axis,edge,edge_orthogonal,random_axis_mean")
    parser.add_argument("--observers", default="pose_blind_delta_mean")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--ridge-alpha-mode", choices=["fixed", "nested_per_candidate"], default="fixed")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_run = Path(args.source_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(source_run / "analysis_windows.csv")
    metadata_path = source_run / "candidate_motion_metadata.csv"
    motion_metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    latents = _filter_latents(_load_npz(source_run / "latent_feature_arrays.npz"), _parse_list(args.latent_names))
    raw_responses = _load_npz(source_run / "response_feature_arrays.npz")
    arrays = _filter_arrays(
        _candidate_arrays(raw_responses),
        scale_ids=_parse_list(args.scale_ids),
        candidates=_parse_list(args.candidates),
        observers=_parse_list(args.observers),
    )
    _validate_shapes(windows, latents, arrays)
    rng = np.random.default_rng(int(args.seed))

    decode, per_window = _decode_from_cache(
        latents,
        arrays,
        windows,
        pca_k_list=_parse_int_list(args.pca_k_list),
        alphas=_parse_float_list(args.ridge_alphas),
        alpha_mode=str(args.ridge_alpha_mode),
        fixed_alpha=float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        seed=int(args.seed),
    )
    decode, per_window = _add_random_axis_mean_scores(decode, per_window)
    contrasts = _decode_contrasts(per_window, rng=rng, n_bootstrap=int(args.n_bootstrap))
    cov = _response_covariance_summary(arrays, motion_metadata)

    decode.to_csv(out_dir / "aggregate_cache_decode_summary.csv", index=False)
    per_window.to_csv(out_dir / "aggregate_cache_decode_score_by_window.csv", index=False)
    contrasts.to_csv(out_dir / "aggregate_cache_decode_contrasts.csv", index=False)
    cov.to_csv(out_dir / "aggregate_cache_response_covariance_summary.csv", index=False)
    metadata = {
        "source_run": str(source_run),
        "n_windows": int(windows.shape[0]),
        "n_latent_arrays": int(len(latents)),
        "n_response_arrays_raw": int(len(raw_responses)),
        "n_response_arrays_after_filter": int(len(arrays)),
        "pca_k_list": _parse_int_list(args.pca_k_list),
        "latent_names": _parse_list(args.latent_names),
        "scale_ids": _parse_list(args.scale_ids),
        "candidates": _parse_list(args.candidates),
        "observers": _parse_list(args.observers),
        "ridge_alpha_mode": str(args.ridge_alpha_mode),
        "fixed_ridge_alpha": float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
        "claim_boundary": "cache_only_fixed_axis_scale_proxy_not_full_aggregate_motion_distribution",
    }
    (out_dir / "aggregate_cache_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_report(out_dir, args, decode, contrasts, cov)
    print(f"[aggregate-cache-proxy] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()

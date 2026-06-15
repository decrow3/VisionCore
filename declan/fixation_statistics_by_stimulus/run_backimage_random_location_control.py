"""Compare BackImage fixation locations with random locations on the same image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    from .image_features import backimage_trial_geometry, local_backimage_features, screen_px_to_gaze_deg
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import backimage_trial_geometry, local_backimage_features, screen_px_to_gaze_deg


DEFAULT_INPUT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure/backimage_image_fem_windows.csv")
DEFAULT_OUT_DIR = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_random_location_control")
IMAGE_FEATURES = (
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_high_freq_power_fraction",
    "image_orientation_coherence",
    "image_spectrum_anisotropy",
    "image_edge_density",
    "image_power_0_2_cpd_fraction",
    "image_power_2_4_cpd_fraction",
    "image_power_4_8_cpd_fraction",
    "image_power_8plus_cpd_fraction",
)


def _random_gaze_for_trial(
    *,
    session_name: str,
    trial_idx: int,
    n: int,
    patch_radius_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    geom = backimage_trial_geometry(session_name, trial_idx)
    height, width = geom["screen_shape"]
    ppd = float(geom["ppd"])
    x0, y0, x1, y1 = geom["dest_rect"]
    inset = max(0, int(round(float(patch_radius_deg) * ppd)))
    lo_x, hi_x = x0 + inset, x1 - inset
    lo_y, hi_y = y0 + inset, y1 - inset
    if lo_x >= hi_x:
        lo_x, hi_x = x0, x1
    if lo_y >= hi_y:
        lo_y, hi_y = y0, y1
    xy_px = np.column_stack([
        rng.uniform(float(lo_x), float(hi_x), size=int(n)),
        rng.uniform(float(lo_y), float(hi_y), size=int(n)),
    ])
    return screen_px_to_gaze_deg(xy_px, ppd=ppd, screen_shape=(height, width))


def _sample_random_features(
    rows: pd.DataFrame,
    *,
    n_random_per_window: int,
    patch_radius_deg: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    grouped = rows.groupby(["session", "trial_idx"], sort=False)
    for (session_name, trial_idx), group in tqdm(grouped, desc="same-image random patches"):
        random_gaze = _random_gaze_for_trial(
            session_name=str(session_name),
            trial_idx=int(trial_idx),
            n=int(len(group) * n_random_per_window),
            patch_radius_deg=float(patch_radius_deg),
            rng=rng,
        )
        k = 0
        for window_id in group["window_id"].to_numpy(dtype=int):
            for draw_idx in range(int(n_random_per_window)):
                gaze = random_gaze[k]
                k += 1
                feats = local_backimage_features(
                    session_name=str(session_name),
                    trial_idx=int(trial_idx),
                    gaze_xy_deg=gaze,
                    patch_radius_deg=float(patch_radius_deg),
                )
                if not feats:
                    continue
                record = {
                    "window_id": int(window_id),
                    "session": str(session_name),
                    "trial_idx": int(trial_idx),
                    "draw_idx": int(draw_idx),
                    "random_x_deg": float(gaze[0]),
                    "random_y_deg": float(gaze[1]),
                }
                record.update({feature: feats.get(feature, np.nan) for feature in IMAGE_FEATURES})
                records.append(record)
    return pd.DataFrame(records)


def _restrict_to_top_trials(rows: pd.DataFrame, *, max_trials: int, max_per_session: int) -> pd.DataFrame:
    if int(max_trials) <= 0:
        return rows
    counts = (
        rows.groupby(["session", "trial_idx"], as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
        .sort_values(["n_windows", "session", "trial_idx"], ascending=[False, True, True])
    )
    selected = []
    per_session: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        session = str(row.session)
        if per_session.get(session, 0) >= int(max_per_session):
            continue
        selected.append((session, int(row.trial_idx)))
        per_session[session] = per_session.get(session, 0) + 1
        if len(selected) >= int(max_trials):
            break
    keep = pd.MultiIndex.from_frame(rows[["session", "trial_idx"]]).isin(pd.MultiIndex.from_tuples(selected, names=["session", "trial_idx"]))
    return rows.loc[keep].copy()


def _summarize_feature_control(observed: pd.DataFrame, random_rows: pd.DataFrame) -> pd.DataFrame:
    random_mean = random_rows.groupby("window_id", as_index=False)[list(IMAGE_FEATURES)].mean()
    merged = observed[["window_id", "session", *IMAGE_FEATURES]].merge(
        random_mean,
        on="window_id",
        suffixes=("_observed", "_random_mean"),
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for feature in IMAGE_FEATURES:
        obs = merged[f"{feature}_observed"].astype(float)
        rnd = merged[f"{feature}_random_mean"].astype(float)
        finite = np.isfinite(obs) & np.isfinite(rnd)
        sub = merged.loc[finite, ["session"]].copy()
        sub["delta"] = obs[finite].to_numpy() - rnd[finite].to_numpy()
        session_delta = sub.groupby("session")["delta"].mean()
        t_stat, p_value = stats.ttest_1samp(session_delta.to_numpy(dtype=float), popmean=0.0, nan_policy="omit")
        rows.append({
            "feature": feature,
            "n_windows": int(np.count_nonzero(finite)),
            "n_sessions": int(session_delta.shape[0]),
            "observed_mean": float(obs[finite].mean()),
            "random_mean": float(rnd[finite].mean()),
            "observed_minus_random": float(sub["delta"].mean()),
            "session_mean_delta": float(session_delta.mean()),
            "session_delta_sem": float(session_delta.sem()) if session_delta.shape[0] > 1 else np.nan,
            "session_t": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "session_p_two_sided": float(p_value) if np.isfinite(p_value) else np.nan,
            "fraction_windows_observed_gt_random": float((obs[finite].to_numpy() > rnd[finite].to_numpy()).mean()),
        })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--patch-radius-deg", type=float, default=1.0)
    parser.add_argument("--n-random-per-window", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--max-per-session", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(args.input)
    observed = observed.dropna(subset=["session", "trial_idx", *IMAGE_FEATURES]).copy()
    observed = _restrict_to_top_trials(
        observed,
        max_trials=int(args.max_trials),
        max_per_session=int(args.max_per_session),
    )
    observed = observed.reset_index(drop=True)
    observed["window_id"] = np.arange(len(observed), dtype=int)
    if int(args.max_windows) > 0 and len(observed) > int(args.max_windows):
        observed = (
            observed.sample(n=int(args.max_windows), random_state=int(args.seed), replace=False)
            .sort_values(["session", "trial_idx", "window_id"])
            .reset_index(drop=True)
        )
        observed["window_id"] = np.arange(len(observed), dtype=int)

    random_rows = _sample_random_features(
        observed,
        n_random_per_window=int(args.n_random_per_window),
        patch_radius_deg=float(args.patch_radius_deg),
        seed=int(args.seed),
    )
    summary = _summarize_feature_control(observed, random_rows)

    observed[["window_id", "session", "trial_idx", "mean_x_deg", "mean_y_deg", *IMAGE_FEATURES]].to_csv(
        out_dir / "observed_locations.csv",
        index=False,
    )
    random_rows.to_csv(out_dir / "same_image_random_locations.csv", index=False)
    summary.to_csv(out_dir / "same_image_random_feature_summary.csv", index=False)
    with (out_dir / "run_metadata.json").open("w") as f:
        json.dump({
            "input": str(args.input),
            "out_dir": str(out_dir),
            "patch_radius_deg": float(args.patch_radius_deg),
            "n_random_per_window": int(args.n_random_per_window),
            "max_windows": int(args.max_windows),
            "max_trials": int(args.max_trials),
            "max_per_session": int(args.max_per_session),
            "seed": int(args.seed),
            "n_observed_windows": int(observed.shape[0]),
            "n_random_samples": int(random_rows.shape[0]),
            "random_control": "uniform random patch centers within the same BackImage trial destination rectangle, inset by patch radius",
            "features": list(IMAGE_FEATURES),
        }, f, indent=2)
    print(f"Wrote random-location BackImage control to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

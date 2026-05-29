#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from eval.fixrsvp import get_dataset_from_config
from scripts.fixrsvp_eye_conventions import (
    DEFAULT_STORED_EYE_CONVENTION,
    dataset_eyepos_to_visual_deg,
)


DEFAULT_OUTPUT_DIR = Path("results/fixrsvp_eye_provenance")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _interp_xy(sample_times: np.ndarray, sample_xy: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    valid = np.isfinite(sample_times) & np.isfinite(sample_xy).all(axis=1)
    if int(valid.sum()) < 2:
        return np.full((target_times.shape[0], 2), np.nan, dtype=np.float64)
    order = np.argsort(sample_times[valid])
    t_src = sample_times[valid][order]
    xy_src = sample_xy[valid][order]
    return np.column_stack([
        np.interp(target_times, t_src, xy_src[:, 0], left=np.nan, right=np.nan),
        np.interp(target_times, t_src, xy_src[:, 1], left=np.nan, right=np.nan),
    ])


def _score_mapping(stored_xy: np.ndarray, ref_xy: np.ndarray) -> dict[str, float]:
    valid_mask = np.isfinite(stored_xy).all(axis=1) & np.isfinite(ref_xy).all(axis=1)
    score = {
        "n_valid": int(valid_mask.sum()),
        "corr_x": _safe_corr(stored_xy[:, 0], ref_xy[:, 0]),
        "corr_y": _safe_corr(stored_xy[:, 1], ref_xy[:, 1]),
        "mae_x": _safe_mae(stored_xy[:, 0], ref_xy[:, 0]),
        "mae_y": _safe_mae(stored_xy[:, 1], ref_xy[:, 1]),
        "mae_xy": _safe_mae(stored_xy, ref_xy),
    }
    corr_x = score["corr_x"]
    corr_y = score["corr_y"]
    if np.isfinite(corr_x) and np.isfinite(corr_y):
        score["corr_mean"] = float((corr_x + corr_y) / 2.0)
    else:
        score["corr_mean"] = float("nan")
    return score


def _trial_eye_smo(sess_trial: dict[str, Any], t_bins_trial: np.ndarray) -> np.ndarray:
    eye_smo = np.asarray(sess_trial.get("eyeSmo"), dtype=np.float64)
    if eye_smo.ndim != 2 or eye_smo.shape[1] < 3:
        return np.full((t_bins_trial.shape[0], 2), np.nan, dtype=np.float64)
    sample_times = float(sess_trial["START_EPHYS"]) + eye_smo[:, 0]
    sample_xy = eye_smo[:, 1:3]
    return _interp_xy(sample_times, sample_xy, t_bins_trial)


def _mapping_variants(ref_xy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": ref_xy,
        "flip_y": ref_xy * np.array([1.0, -1.0], dtype=np.float64),
        "swap_xy": ref_xy[:, ::-1],
        "swap_xy_flip_y": ref_xy[:, ::-1] * np.array([1.0, -1.0], dtype=np.float64),
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmedian(np.asarray(values, dtype=np.float64)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(subject: str, session_date: str, dataset_config_path: str) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, float]]]:
    dataset, _ = get_dataset_from_config(subject, session_date, dataset_config_path)
    dset_idx = int(dataset.inds[:, 0].unique().item())
    dset = dataset.dsets[dset_idx]
    sess = dset.metadata["sess"]
    dset_path = Path(sess.sess_dir) / "datasets" / "fixrsvp.dset"

    trial_inds = dset["trial_inds"].detach().cpu().numpy().astype(int)
    t_bins = dset["t_bins"].detach().cpu().numpy().astype(np.float64)
    stored_xy = dataset_eyepos_to_visual_deg(dset["eyepos"], stored_convention=DEFAULT_STORED_EYE_CONVENTION).astype(np.float64)

    rows: list[dict[str, Any]] = []
    summary_values: dict[str, dict[str, list[float]]] = {}

    for trial in np.unique(trial_inds):
        mask = trial_inds == int(trial)
        if int(mask.sum()) < 4:
            continue
        stored_trial = stored_xy[mask]
        t_bins_trial = t_bins[mask]
        ref_eye_smo = _trial_eye_smo(sess.exp["D"][int(trial)], t_bins_trial)
        for mapping_name, ref_xy in _mapping_variants(ref_eye_smo).items():
            score = _score_mapping(stored_trial, ref_xy)
            row = {
                "trial": int(trial),
                "mapping": mapping_name,
                "n_samples": int(mask.sum()),
                **score,
            }
            rows.append(row)
            mapping_summary = summary_values.setdefault(mapping_name, {})
            for key, value in score.items():
                mapping_summary.setdefault(key, []).append(value)

    summary: dict[str, dict[str, float]] = {}
    for mapping_name, mapping_scores in summary_values.items():
        summary[mapping_name] = {key: _median(values) for key, values in mapping_scores.items()}

    ranked = sorted(
        summary.items(),
        key=lambda item: (
            -np.inf if not np.isfinite(item[1].get("corr_mean", np.nan)) else item[1]["corr_mean"],
            np.inf if not np.isfinite(item[1].get("mae_xy", np.nan)) else -item[1]["mae_xy"],
        ),
        reverse=True,
    )
    best_mapping = ranked[0][0] if ranked else "none"
    best_summary = summary.get(best_mapping, {})
    identity_summary = summary.get("identity", {})
    flip_y_summary = summary.get("flip_y", {})

    loader_source = inspect.getsourcefile(sess.get_dataset)
    lines = [
        f"fixRSVP eye provenance report for {subject}_{session_date}",
        "",
        "dataset provenance:",
        f"- session loader: {loader_source}",
        f"- loader behavior: YatesV1Session.get_dataset() directly loads {dset_path} and does not transform eyepos after load.",
        f"- stored convention helper default: {DEFAULT_STORED_EYE_CONVENTION}",
        "",
        "eyeSmo alignment summary:",
        f"- best mapping by median corr/mae across trials: {best_mapping}",
        f"- identity median corr_x={identity_summary.get('corr_x', float('nan')):.4f}, corr_y={identity_summary.get('corr_y', float('nan')):.4f}, mae_xy={identity_summary.get('mae_xy', float('nan')):.4f}",
        f"- flip_y median corr_x={flip_y_summary.get('corr_x', float('nan')):.4f}, corr_y={flip_y_summary.get('corr_y', float('nan')):.4f}, mae_xy={flip_y_summary.get('mae_xy', float('nan')):.4f}",
        f"- winning median corr_mean={best_summary.get('corr_mean', float('nan')):.4f}, mae_xy={best_summary.get('mae_xy', float('nan')):.4f}",
        "",
        "interpretation:",
        "- If identity beats flip_y, stored dset['eyepos'] matches the sign of raw session eyeSmo y rather than a post-load negation.",
        "- This report establishes the immediate upstream trace that stored eyepos follows. It does not by itself prove whether eyeSmo y is visual-field up or image-row down before dataset serialization.",
    ]
    return "\n".join(lines) + "\n", rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace fixRSVP eyepos provenance against raw session eyeSmo traces.")
    parser.add_argument("--subject", default="Allen")
    parser.add_argument("--session-date", default="2022-02-16")
    parser.add_argument("--dataset-config-path", default="experiments/dataset_configs/multi_basic_120_long_legacy.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_tag = f"{args.subject}_{args.session_date}"
    report_text, rows, summary = build_report(args.subject, args.session_date, args.dataset_config_path)

    report_path = args.output_dir / f"{session_tag}_eye_provenance_report.txt"
    csv_path = args.output_dir / f"{session_tag}_eye_smo_alignment.csv"
    summary_path = args.output_dir / f"{session_tag}_eye_smo_alignment_summary.csv"

    report_path.write_text(report_text)
    _write_csv(csv_path, rows)
    summary_rows = [{"mapping": mapping, **metrics} for mapping, metrics in summary.items()]
    _write_csv(summary_path, summary_rows)

    print(report_text, end="")
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

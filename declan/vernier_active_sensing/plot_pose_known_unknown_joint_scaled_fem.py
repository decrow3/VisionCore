#!/usr/bin/env python3
"""Plot Vernier pose-known, pose-unknown, and sign-readout diagnostics.

The continuous quadratic joint observer is intentionally not used here: its
60-frame Vernier diagnostic can collapse to a one-label solution.  This script
therefore uses a conservative trace-disjoint response-feature decoder as the
sign-readout sanity curve, plotted against the Fisher proxies on a shared
static-normalized axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .joint_observer import THETA_MINUS, THETA_PLUS
from .run_continuous_quadratic_joint_observer import _cache_counts, _load_rate_pose_caches


DEFAULT_CONDITIONS = ("static_center", "scaled_real_0.5", "real_fem", "scaled_real_1.5")
DEFAULT_LABELS = {
    "static_center": "Static",
    "scaled_real_0.5": "FEM 0.5x",
    "real_fem": "FEM 1x",
    "scaled_real_1.5": "FEM 1.5x",
}
KNOWN_READOUT = "pose_aware_diagonal_poisson"
UNKNOWN_READOUT = "pose_blind_diagonal_count_plus_marginal"


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _ridge_dual_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    train = (x_train - mean[None, :]) / scale[None, :]
    test = (x_test - mean[None, :]) / scale[None, :]
    kernel = train @ train.T
    weights = np.linalg.solve(kernel + float(alpha) * np.eye(kernel.shape[0]), y_train)
    return test @ train.T @ weights


def _response_features(
    counts_by_label: dict[str, np.ndarray],
    mode: str,
    *,
    prefix_frames: int = 0,
    unit_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    plus = np.asarray(counts_by_label[THETA_PLUS], dtype=np.float64)
    minus = np.asarray(counts_by_label[THETA_MINUS], dtype=np.float64)
    if int(prefix_frames) > 0:
        plus = plus[:, : int(prefix_frames), :]
        minus = minus[:, : int(prefix_frames), :]
    stacked = np.concatenate([plus, minus], axis=0)
    if mode == "sum_time":
        features = np.sum(stacked, axis=1)
    if mode == "mean_time":
        features = np.mean(stacked, axis=1)
    elif mode == "flat":
        features = stacked.reshape(stacked.shape[0], -1)
    elif mode == "early32_flat":
        early = stacked[:, : min(32, stacked.shape[1])]
        features = early.reshape(early.shape[0], -1)
    elif mode not in {"sum_time", "mean_time"}:
        raise ValueError(f"Unsupported feature mode {mode!r}")
    if unit_indices is not None and mode in {"sum_time", "mean_time"}:
        features = features[:, np.asarray(unit_indices, dtype=np.int64)]
    return features


def _condition_family(conditions: list[str], train_policy: str, test_condition: str) -> list[str]:
    policy = str(train_policy)
    if policy == "same_condition":
        return [str(test_condition)]
    if policy == "fem_family":
        family = [condition for condition in conditions if condition != "static_center"]
        if not family:
            raise ValueError("joint train policy fem_family requires at least one non-static condition")
        return family
    if policy == "all_conditions":
        return list(conditions)
    raise ValueError(f"Unsupported joint train policy {train_policy!r}")


def _select_top_units(
    *,
    counts_by_condition: dict[str, dict[str, np.ndarray]],
    train_conditions: list[str],
    heldout_trace: int,
    prefix_frames: int,
    topk: int,
) -> tuple[int, ...] | None:
    if int(topk) <= 0:
        return None
    scores: list[np.ndarray] = []
    for condition in train_conditions:
        counts = counts_by_condition[condition]
        plus = np.asarray(counts[THETA_PLUS], dtype=np.float64)
        minus = np.asarray(counts[THETA_MINUS], dtype=np.float64)
        if int(prefix_frames) > 0:
            plus = plus[:, : int(prefix_frames), :]
            minus = minus[:, : int(prefix_frames), :]
        plus_sum = np.sum(plus, axis=1)
        minus_sum = np.sum(minus, axis=1)
        trace_count = int(plus_sum.shape[0])
        train = np.asarray([idx for idx in range(trace_count) if idx != int(heldout_trace)], dtype=np.int64)
        delta = np.mean(plus_sum[train], axis=0) - np.mean(minus_sum[train], axis=0)
        noise = 0.5 * (np.mean(plus_sum[train], axis=0) + np.mean(minus_sum[train], axis=0))
        scores.append(np.abs(delta) / np.sqrt(np.maximum(noise, 1e-9)))
    unit_score = np.mean(scores, axis=0)
    selected = np.argsort(unit_score)[::-1][: min(int(topk), unit_score.size)]
    return tuple(int(idx) for idx in selected.tolist())


def _joint_feature_decoder_rows(
    *,
    source_dir: Path,
    conditions: list[str],
    fd_step_arcmin: float,
    inference_mode: str,
    feature_mode: str,
    alpha: float,
    bin_seconds: float,
    train_policy: str,
    prefix_frames: int,
    topk_units: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    caches = _load_rate_pose_caches(source_dir)
    counts_by_condition = {
        condition: _cache_counts(
            caches[(condition, float(fd_step_arcmin), inference_mode)],
            bin_seconds=float(bin_seconds),
            max_timebins=0,
        )
        for condition in conditions
    }
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        key = (condition, float(fd_step_arcmin), inference_mode)
        if key not in caches:
            raise FileNotFoundError(f"Missing cache for {key}")
        counts = counts_by_condition[condition]
        n_traces = int(counts[THETA_PLUS].shape[0])
        y = np.concatenate([np.ones(n_traces), -np.ones(n_traces)])
        trace_index = np.concatenate([np.arange(n_traces), np.arange(n_traces)])
        true_label = np.asarray([THETA_PLUS] * n_traces + [THETA_MINUS] * n_traces)
        scores = np.empty_like(y, dtype=np.float64)
        train_conditions = _condition_family(conditions, str(train_policy), condition)
        for heldout_trace in range(n_traces):
            unit_indices = _select_top_units(
                counts_by_condition=counts_by_condition,
                train_conditions=train_conditions,
                heldout_trace=int(heldout_trace),
                prefix_frames=int(prefix_frames),
                topk=int(topk_units),
            )
            train_features: list[np.ndarray] = []
            train_labels: list[np.ndarray] = []
            for train_condition in train_conditions:
                train_counts = counts_by_condition[train_condition]
                features = _response_features(
                    train_counts,
                    mode=feature_mode,
                    prefix_frames=int(prefix_frames),
                    unit_indices=unit_indices,
                )
                n_train_traces = int(train_counts[THETA_PLUS].shape[0])
                train_trace_index = np.concatenate([np.arange(n_train_traces), np.arange(n_train_traces)])
                train = train_trace_index != int(heldout_trace)
                train_features.append(features[train])
                train_labels.append(np.concatenate([np.ones(n_train_traces), -np.ones(n_train_traces)])[train])
            test_features = _response_features(
                counts,
                mode=feature_mode,
                prefix_frames=int(prefix_frames),
                unit_indices=unit_indices,
            )
            test = trace_index == int(heldout_trace)
            scores[test] = _ridge_dual_scores(
                np.concatenate(train_features, axis=0),
                np.concatenate(train_labels, axis=0),
                test_features[test],
                alpha=float(alpha),
            )
        pred_label = np.where(scores >= 0.0, THETA_PLUS, THETA_MINUS)
        correct = pred_label == true_label
        for row_index in range(features.shape[0]):
            trial_rows.append(
                {
                    "condition": condition,
                    "fd_step_arcmin": float(fd_step_arcmin),
                    "inference_mode": inference_mode,
                    "feature_mode": feature_mode,
                    "ridge_alpha": float(alpha),
                    "joint_train_policy": str(train_policy),
                    "prefix_frames": int(prefix_frames),
                    "topk_units": int(topk_units),
                    "trace_index": int(trace_index[row_index]),
                    "true_label": str(true_label[row_index]),
                    "score_plus_minus": float(scores[row_index]),
                    "pred_label": str(pred_label[row_index]),
                    "correct": bool(correct[row_index]),
                }
            )
        summary_rows.append(
            {
                "readout": "pose_robust_decoder",
                "condition": condition,
                "fd_step_arcmin": float(fd_step_arcmin),
                "inference_mode": inference_mode,
                "n": int(correct.size),
                "n_traces": n_traces,
                "feature_mode": feature_mode,
                "ridge_alpha": float(alpha),
                "joint_train_policy": str(train_policy),
                "prefix_frames": int(prefix_frames),
                "topk_units": int(topk_units),
                "accuracy": float(np.mean(correct)),
                "mean_plus_score": float(np.mean(scores[true_label == THETA_PLUS])),
                "mean_minus_score": float(np.mean(scores[true_label == THETA_MINUS])),
            }
        )
    return trial_rows, summary_rows


def _fisher_rows(
    *,
    source_dir: Path,
    conditions: list[str],
    fd_step_arcmin: float,
    inference_mode: str,
) -> pd.DataFrame:
    path = source_dir / "condition_reliability_summary.csv"
    df = pd.read_csv(path)
    out = df[
        df["condition"].isin(conditions)
        & np.isclose(df["fd_step_arcmin"].astype(float), float(fd_step_arcmin))
        & df["inference_mode"].astype(str).eq(inference_mode)
        & df["readout"].isin([KNOWN_READOUT, UNKNOWN_READOUT])
    ].copy()
    missing = {
        (readout, condition)
        for readout in [KNOWN_READOUT, UNKNOWN_READOUT]
        for condition in conditions
    }.difference(set(zip(out["readout"], out["condition"], strict=False)))
    if missing:
        raise ValueError(f"Missing Fisher rows: {sorted(missing)}")
    return out


def build_plot(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _parse_csv(args.conditions)
    order = {condition: idx for idx, condition in enumerate(conditions)}
    labels = {condition: DEFAULT_LABELS.get(condition, condition) for condition in conditions}

    fisher = _fisher_rows(
        source_dir=source_dir,
        conditions=conditions,
        fd_step_arcmin=float(args.fd_step_arcmin),
        inference_mode=str(args.inference_mode),
    )
    fisher["condition_order"] = fisher["condition"].map(order)
    fisher["condition_label"] = fisher["condition"].map(labels)
    static_by_readout = (
        fisher[fisher["condition"].eq(args.static_condition)]
        .set_index("readout")["mean_final_fisher"]
        .to_dict()
    )
    fisher["relative_metric_static1"] = fisher.apply(
        lambda row: float(row["mean_final_fisher"]) / float(static_by_readout[str(row["readout"])]),
        axis=1,
    )
    fisher.to_csv(out_dir / "pose_known_unknown_fisher_scaled_fem.csv", index=False)

    joint_trials, joint_summary_rows = _joint_feature_decoder_rows(
        source_dir=source_dir,
        conditions=conditions,
        fd_step_arcmin=float(args.fd_step_arcmin),
        inference_mode=str(args.inference_mode),
        feature_mode=str(args.joint_feature_mode),
        alpha=float(args.joint_ridge_alpha),
        bin_seconds=float(args.bin_seconds),
        train_policy=str(args.joint_train_policy),
        prefix_frames=int(args.joint_prefix_frames),
        topk_units=int(args.joint_topk_units),
    )
    joint_trials_df = pd.DataFrame(joint_trials)
    joint_summary = pd.DataFrame(joint_summary_rows)
    joint_summary["condition_order"] = joint_summary["condition"].map(order)
    joint_summary["condition_label"] = joint_summary["condition"].map(labels)
    static_acc = float(
        joint_summary.loc[joint_summary["condition"].eq(args.static_condition), "accuracy"].iloc[0]
    )
    joint_summary["relative_metric_static1"] = joint_summary["accuracy"] / static_acc
    joint_trials_df.to_csv(out_dir / "pose_robust_decoder_trials.csv", index=False)
    joint_summary.to_csv(out_dir / "pose_robust_decoder_summary.csv", index=False)

    combined = pd.concat(
        [
            fisher.assign(
                plotted_readout=fisher["readout"].map(
                    {
                        KNOWN_READOUT: "Pose-known Fisher",
                        UNKNOWN_READOUT: "Pose-unknown Fisher",
                    }
                ),
                raw_metric=fisher["mean_final_fisher"],
            )[
                [
                    "plotted_readout",
                    "condition",
                    "condition_label",
                    "condition_order",
                    "relative_metric_static1",
                    "raw_metric",
                ]
            ],
            joint_summary.assign(
                plotted_readout="Pose-robust decoder",
                raw_metric=joint_summary["accuracy"],
            )[
                [
                    "plotted_readout",
                    "condition",
                    "condition_label",
                    "condition_order",
                    "relative_metric_static1",
                    "raw_metric",
                ]
            ],
        ],
        ignore_index=True,
    )
    combined.to_csv(out_dir / "pose_known_unknown_pose_robust_decoder_same_axes_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 170,
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    styles = {
        "Pose-known Fisher": ("#2C7FB8", "o", "-"),
        "Pose-unknown Fisher": ("#7B3294", "o", "-"),
        "Pose-robust decoder": ("#D95F02", "s", "--"),
    }
    for readout, group in combined.groupby("plotted_readout", sort=False):
        group = group.sort_values("condition_order")
        color, marker, linestyle = styles[str(readout)]
        ax.plot(
            group["condition_order"],
            group["relative_metric_static1"],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            label=str(readout),
        )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([labels[condition] for condition in conditions], rotation=20, ha="right")
    ax.set_ylabel("Relative readout metric (static = 1)")
    ax.set_title("Vernier scaled FEM: Fisher bounds and pose-robust decoder")
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.025,
        (
            "Fisher curves use mean final Fisher. Pose-robust decoder curve is trace-disjoint ridge accuracy "
            f"on {args.joint_feature_mode} response features, normalized by static accuracy. "
            f"Train policy: {args.joint_train_policy}. No eye trace is estimated."
        ),
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "pose_known_unknown_pose_robust_decoder_same_axes.png", bbox_inches="tight")
    fig.savefig(out_dir / "pose_known_unknown_pose_robust_decoder_same_axes.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/vernier_joint_geometry_enumerated_gpu0_fixed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/vernier_pose_known_unknown_joint_scaled_fem_v1"))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--static-condition", default="static_center")
    parser.add_argument("--fd-step-arcmin", type=float, default=0.5)
    parser.add_argument("--inference-mode", default="framewise")
    parser.add_argument("--joint-feature-mode", choices=("sum_time", "mean_time", "flat", "early32_flat"), default="sum_time")
    parser.add_argument("--joint-ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--joint-train-policy",
        choices=("same_condition", "fem_family", "all_conditions"),
        default="same_condition",
    )
    parser.add_argument("--joint-prefix-frames", type=int, default=0)
    parser.add_argument("--joint-topk-units", type=int, default=0)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    return parser.parse_args()


def main() -> None:
    build_plot(parse_args())


if __name__ == "__main__":
    main()

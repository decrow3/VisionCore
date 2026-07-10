"""Sweep 2x self-consistency settings for the candidate-free hidden-joint observer.

This is a controlled recovery diagnostic.  It keeps the synthetic response
source fixed as ``F(z_true, tau_true)`` and varies only inference/prior balance
knobs for the hidden-joint observer.  The promotion target for each setting is:

    known_tau_forward_model > hidden_joint_forward_model > zero_tau_forward_model

at 2x motion scale under pooled held-out R2_cv.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer import (
    _clean_axis,
    _configure_matplotlib,
    _parse_str_list,
)
from declan.figure4_active_sensing_atlas.scripts.run_candidate_free_self_consistency_control import (
    DEFAULT_OUT_DIR as SELF_CONSISTENCY_OUT_DIR,
    build as build_self_consistency,
    build_parser as build_self_consistency_parser,
)


DEFAULT_OUT_DIR = SELF_CONSISTENCY_OUT_DIR.parent / "candidate_free_2x_self_consistency_sweep"


def _parse_float_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower() in {"auto", "none", "nan"}:
            values.append(None)
        else:
            values.append(float(item))
    return values


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _slug_float(value: float | None) -> str:
    if value is None:
        return "auto"
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _parse_args() -> argparse.Namespace:
    parser = build_self_consistency_parser()
    parser.description = __doc__
    parser.set_defaults(
        out_dir=DEFAULT_OUT_DIR,
        scales="2.0",
        n_bootstrap=0,
        progress_every=0,
    )
    parser.add_argument("--joint-iterations-list", default="6,12")
    parser.add_argument("--brownian-cov-scale-list", default="0.5,1.0,2.0")
    parser.add_argument("--forward-z-prior-precision-list", default="0.1,1.0")
    parser.add_argument("--process-var-list", default="0.001,0.01")
    parser.add_argument("--observation-var-list", default="auto")
    parser.add_argument("--sweep-max-runs", type=int, default=0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-setting gate tables instead of rerunning them.",
    )
    return parser.parse_args()


def _setting_slug(
    *,
    iterations: int,
    brownian_cov_scale: float,
    z_prior_precision: float,
    process_var: float,
    observation_var: float | None,
) -> str:
    return (
        f"iter{int(iterations)}"
        f"_bcov{_slug_float(brownian_cov_scale)}"
        f"_zprior{_slug_float(z_prior_precision)}"
        f"_pvar{_slug_float(process_var)}"
        f"_ovar{_slug_float(observation_var)}"
    )


def _read_gate(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    gate_path = path / "candidate_free_self_consistency_gate_table.csv"
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    gate = pd.read_csv(gate_path)
    all_rows = gate[gate["group_kind"].astype(str).eq("all")]
    if all_rows.empty:
        raise ValueError(f"{gate_path} lacks all-row")
    return all_rows.iloc[0].to_dict(), gate


def _plot_summary(summary: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    _configure_matplotlib()
    finite = summary[np.isfinite(summary["joint_minus_zero"].to_numpy(dtype=np.float64))].copy()
    finite = finite.sort_values("joint_minus_zero", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.32 * len(finite))))
    colors = np.where(finite["known_gt_joint_gt_zero"].astype(bool), "#047857", "#9ca3af")
    labels = finite["setting"].astype(str).tolist()
    ax.barh(labels, finite["joint_minus_zero"], color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("2x self-consistency: joint - zero R2_cv")
    ax.set_title("Candidate-free hidden-joint 2x sweep")
    _clean_axis(ax)
    fig.tight_layout()
    png = out_dir / "candidate_free_2x_self_consistency_sweep.png"
    pdf = out_dir / "candidate_free_2x_self_consistency_sweep.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _write_readme(out_dir: Path, summary: pd.DataFrame, png: Path, pdf: Path) -> None:
    passed = summary[summary["known_gt_joint_gt_zero"].astype(bool)].copy()
    best = summary.sort_values("joint_minus_zero", ascending=False).iloc[0]
    lines = [
        "# Candidate-free 2x self-consistency sweep",
        "",
        "Synthetic response source: `F(z_true, tau_true)` from the source-disjoint compact observation model.",
        "Scale: `2.0` only.",
        "",
        f"Settings tested: {int(len(summary))}",
        f"Settings passing `known > joint > zero`: {int(len(passed))}",
        "",
        "Best setting by `joint - zero`:",
        f"- setting: `{best['setting']}`",
        f"- known: {float(best['S_known']):.6g}",
        f"- joint: {float(best['S_joint']):.6g}",
        f"- zero: {float(best['S_zero']):.6g}",
        f"- joint - zero: {float(best['joint_minus_zero']):.6g}",
        f"- known - joint: {float(best['known_minus_joint']):.6g}",
        "",
        "Outputs:",
        f"- `{png.name}`",
        f"- `{pdf.name}`",
        "- `candidate_free_2x_self_consistency_sweep_summary.csv`",
        "- `candidate_free_2x_self_consistency_sweep_group_rows.csv`",
        "",
    ]
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations_list = _parse_int_list(str(args.joint_iterations_list))
    brownian_list = [float(value) for value in _parse_float_list(str(args.brownian_cov_scale_list)) if value is not None]
    zprior_list = [
        float(value)
        for value in _parse_float_list(str(args.forward_z_prior_precision_list))
        if value is not None
    ]
    process_list = [float(value) for value in _parse_float_list(str(args.process_var_list)) if value is not None]
    observation_list = _parse_float_list(str(args.observation_var_list))
    rows: list[dict[str, Any]] = []
    group_rows: list[pd.DataFrame] = []
    run_count = 0
    for iterations in iterations_list:
        for brownian_cov_scale in brownian_list:
            for z_prior_precision in zprior_list:
                for process_var in process_list:
                    for observation_var in observation_list:
                        if int(args.sweep_max_runs) > 0 and run_count >= int(args.sweep_max_runs):
                            continue
                        run_count += 1
                        setting = _setting_slug(
                            iterations=int(iterations),
                            brownian_cov_scale=float(brownian_cov_scale),
                            z_prior_precision=float(z_prior_precision),
                            process_var=float(process_var),
                            observation_var=observation_var,
                        )
                        setting_dir = out_dir / setting
                        run_args = copy.deepcopy(args)
                        run_args.out_dir = setting_dir
                        run_args.scales = "2.0"
                        run_args.forward_model_joint_iterations = int(iterations)
                        run_args.brownian_cov_scale = float(brownian_cov_scale)
                        run_args.brownian_cov_scale_by_scale = ""
                        run_args.forward_model_z_prior_precision = float(z_prior_precision)
                        run_args.process_var = float(process_var)
                        run_args.observation_var = observation_var
                        run_args.observer_modes = (
                            "response_only,pose_known_forward_model,hidden_joint_forward_model,zero_tau_forward_model"
                        )
                        run_args.n_bootstrap = 0
                        # Avoid nested sweep options leaking into self-consistency manifests.
                        if setting_dir.joinpath("candidate_free_self_consistency_gate_table.csv").exists() and bool(
                            args.skip_existing
                        ):
                            pass
                        else:
                            build_self_consistency(run_args)
                        try:
                            all_row, gate = _read_gate(setting_dir)
                            gate = gate.copy()
                            gate["setting"] = setting
                            group_rows.append(gate)
                            rows.append(
                                {
                                    "setting": setting,
                                    "setting_dir": str(setting_dir),
                                    "forward_model_joint_iterations": int(iterations),
                                    "brownian_cov_scale": float(brownian_cov_scale),
                                    "forward_model_z_prior_precision": float(z_prior_precision),
                                    "process_var": float(process_var),
                                    "observation_var": (
                                        "auto" if observation_var is None else float(observation_var)
                                    ),
                                    "S_known": float(all_row["S_known"]),
                                    "S_joint": float(all_row["S_joint"]),
                                    "S_zero": float(all_row["S_zero"]),
                                    "S_response": float(all_row["S_response"]),
                                    "known_minus_zero": float(all_row["known_minus_zero"]),
                                    "joint_minus_zero": float(all_row["joint_minus_zero"]),
                                    "known_minus_joint": float(all_row["known_minus_joint"]),
                                    "known_gt_joint_gt_zero": bool(all_row["known_gt_joint_gt_zero"]),
                                    "known_gt_zero": bool(all_row["known_gt_zero"]),
                                    "joint_gt_zero": bool(all_row["joint_gt_zero"]),
                                }
                            )
                        except Exception as exc:  # pragma: no cover - diagnostic path
                            rows.append(
                                {
                                    "setting": setting,
                                    "setting_dir": str(setting_dir),
                                    "forward_model_joint_iterations": int(iterations),
                                    "brownian_cov_scale": float(brownian_cov_scale),
                                    "forward_model_z_prior_precision": float(z_prior_precision),
                                    "process_var": float(process_var),
                                    "observation_var": (
                                        "auto" if observation_var is None else float(observation_var)
                                    ),
                                    "error": str(exc),
                                }
                            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No sweep settings were run")
    summary = summary.sort_values("joint_minus_zero", ascending=False, na_position="last").reset_index(drop=True)
    summary.to_csv(out_dir / "candidate_free_2x_self_consistency_sweep_summary.csv", index=False)
    if group_rows:
        pd.concat(group_rows, ignore_index=True, sort=False).to_csv(
            out_dir / "candidate_free_2x_self_consistency_sweep_group_rows.csv",
            index=False,
        )
    png, pdf = _plot_summary(summary, out_dir)
    _write_readme(out_dir, summary, png, pdf)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(_parse_args())


if __name__ == "__main__":
    main()

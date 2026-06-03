from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(VISIONCORE_ROOT))
    from declan.twin_covariance_structure.covariance_core import (  # type: ignore
        compute_cfem_for_image,
        eigensystem,
        participation_ratio,
    )
    from declan.twin_covariance_structure.generate_control_response_caches import (  # type: ignore
        _compute_rates_for_control,
        _load_eye_traces,
        _load_model_and_readout,
    )
    from declan.twin_covariance_structure.run_twin_covariance_structure import _load_rates_array  # type: ignore
else:
    from .covariance_core import compute_cfem_for_image, eigensystem, participation_ratio
    from .generate_control_response_caches import _compute_rates_for_control, _load_eye_traces, _load_model_and_readout
    from .run_twin_covariance_structure import _load_rates_array


DEFAULT_CONDITIONS = ("real", "x_only", "y_only", "line_random_angle")
DEFAULT_TRACE_COUNTS = (40, 80, 160, 320)
EPS = 1e-12


def _parse_csv_ints(text: str) -> list[int]:
    return [int(float(x)) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_label(n_traces: int, total: int) -> str:
    return "full" if n_traces >= total else str(int(n_traces))


def _compute_condition_metrics(rates: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    C, _ = compute_cfem_for_image(rates, return_per_t=False)
    evals, _ = eigensystem(C)
    tr = float(np.sum(np.clip(evals, 0.0, None)))
    frac_top2 = float(np.sum(evals[:2]) / (tr + EPS)) if evals.size >= 2 else float("nan")
    return (
        {
            "pr": float(participation_ratio(evals)),
            "frac_top2": frac_top2,
            "cfem_trace": tr,
        },
        np.asarray(evals, dtype=np.float64),
    )


def _trace_geometry_metrics(traces: np.ndarray, durations: np.ndarray) -> dict[str, float]:
    rms_radius_vals: list[float] = []
    x_sd_vals: list[float] = []
    y_sd_vals: list[float] = []
    path_length_vals: list[float] = []

    for i in range(len(durations)):
        t = int(durations[i])
        eye_trace = np.asarray(traces[i, :t], dtype=np.float64)
        if eye_trace.size == 0:
            continue
        mean_xy = np.mean(eye_trace, axis=0)
        centered = eye_trace - mean_xy[None, :]
        rms_radius_vals.append(float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))))
        x_sd_vals.append(float(np.std(eye_trace[:, 0], ddof=0)))
        y_sd_vals.append(float(np.std(eye_trace[:, 1], ddof=0)))
        if eye_trace.shape[0] >= 2:
            step_lengths = np.sqrt(np.sum(np.diff(eye_trace, axis=0) ** 2, axis=1))
            path_length_vals.append(float(np.sum(step_lengths)))
        else:
            path_length_vals.append(0.0)

    return {
        "mean_rms_eye_radius": float(np.mean(rms_radius_vals)) if rms_radius_vals else float("nan"),
        "mean_x_sd": float(np.mean(x_sd_vals)) if x_sd_vals else float("nan"),
        "mean_y_sd": float(np.mean(y_sd_vals)) if y_sd_vals else float("nan"),
        "mean_path_length": float(np.mean(path_length_vals)) if path_length_vals else float("nan"),
        "sd_rms_eye_radius": float(np.std(rms_radius_vals, ddof=0)) if rms_radius_vals else float("nan"),
        "sd_x_sd": float(np.std(x_sd_vals, ddof=0)) if x_sd_vals else float("nan"),
        "sd_y_sd": float(np.std(y_sd_vals, ddof=0)) if y_sd_vals else float("nan"),
        "sd_path_length": float(np.std(path_length_vals, ddof=0)) if path_length_vals else float("nan"),
        "n_traces_selected": int(len(rms_radius_vals)),
    }


def _mean_series(rows: list[dict[str, object]], condition: str, metric: str, counts: list[int]) -> list[float]:
    out = []
    for n in counts:
        vals = [float(r[metric]) for r in rows if str(r["condition"]) == condition and int(r["n_traces"]) == int(n)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def _plot_metric_vs_count(rows: list[dict[str, object]], counts: list[int], out_path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(counts), dtype=np.float64)
    xticklabels = [_count_label(n, max(counts)) for n in counts]
    for condition in DEFAULT_CONDITIONS:
        ax.plot(x, _mean_series(rows, condition, metric, counts), marker="o", linewidth=2, label=condition)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("n_traces")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_eigenspectra_vs_count(
    spectrum_rows: list[dict[str, object]],
    counts: list[int],
    out_path: Path,
    max_rank: int,
) -> None:
    ncols = 2
    nrows = int(np.ceil(len(counts) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(10.5, 4.0 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)

    for panel_idx, n in enumerate(counts):
        ax = axs.ravel()[panel_idx]
        ax.set_visible(True)
        for condition in DEFAULT_CONDITIONS:
            cond_rows = [
                r for r in spectrum_rows
                if str(r["condition"]) == condition and int(r["n_traces"]) == int(n) and int(r["eigen_index"]) <= int(max_rank)
            ]
            if not cond_rows:
                continue
            by_rank: dict[int, list[float]] = {}
            for row in cond_rows:
                by_rank.setdefault(int(row["eigen_index"]), []).append(float(row["eigenvalue"]))
            ranks = sorted(by_rank)
            vals = [float(np.mean(by_rank[k])) for k in ranks]
            ax.plot(ranks, vals, marker="o", linewidth=1.8, label=condition)
        ax.set_title(f"n_traces = {_count_label(int(n), max(counts))}")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Mean eigenvalue")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_control_audit_payload() -> dict[str, object]:
    return {
        "controls": {
            "x_only": {
                "construction": "exact per-trace projection of the real trace: x(t) is unchanged, y(t) is fixed to that trace's mean y",
                "exact_projection_of_real_traces": True,
                "uses_real_x_trajectory": True,
                "uses_real_y_trajectory": False,
                "temporal_order_preserved": True,
                "x_mean_preserved": True,
                "y_mean_preserved": True,
            },
            "y_only": {
                "construction": "exact per-trace projection of the real trace: y(t) is unchanged, x(t) is fixed to that trace's mean x",
                "exact_projection_of_real_traces": True,
                "uses_real_x_trajectory": False,
                "uses_real_y_trajectory": True,
                "temporal_order_preserved": True,
                "x_mean_preserved": True,
                "y_mean_preserved": True,
            },
            "line_random_angle": {
                "construction": "exact per-trace projection of the real trace onto a single random axis through the trace mean; one angle is sampled per cache and applied to all traces in that cache",
                "exact_projection_of_real_traces": True,
                "uses_real_x_trajectory": False,
                "uses_real_y_trajectory": False,
                "temporal_order_preserved": True,
                "x_mean_preserved": True,
                "y_mean_preserved": True,
            },
        }
    }


def _build_recorded_vs_twin_table(
    recorded_path: Path,
    real_rates_by_count: dict[int, dict[int, np.ndarray]],
    count_rows: list[dict[str, object]],
    n_unit_subsamples: int,
    seed: int,
) -> list[dict[str, object]]:
    with recorded_path.open() as f:
        recorded_rows = list(csv.DictReader(f))

    out: list[dict[str, object]] = []
    count_list = sorted(real_rates_by_count)
    rng = np.random.default_rng(seed)
    for session_idx, rec in enumerate(recorded_rows):
        recorded_pr = float(rec["b_emp_participation_ratio"])
        recorded_top2 = float(rec["b_emp_top2_fraction"])
        recorded_units = int(float(rec["n_units_primary"]))
        for n in count_list:
            full_unit_pr_vals = [
                float(r["pr"])
                for r in count_rows
                if int(r["n_traces"]) == int(n) and str(r["condition"]) == "real"
            ]
            full_unit_top2_vals = [
                float(r["frac_top2"])
                for r in count_rows
                if int(r["n_traces"]) == int(n) and str(r["condition"]) == "real"
            ]

            subsampled_pr: list[float] = []
            available_units = next(iter(real_rates_by_count[n].values())).shape[2]
            unit_count = min(recorded_units, available_units)
            for rep in range(n_unit_subsamples):
                rep_seed = int(rng.integers(0, 2**31 - 1)) + rep + (1000 * session_idx)
                rep_rng = np.random.default_rng(rep_seed)
                unit_idx = rep_rng.choice(available_units, size=unit_count, replace=False)
                rep_vals = []
                for orientation, rates in real_rates_by_count[n].items():
                    rep_metrics, _ = _compute_condition_metrics(rates[:, :, unit_idx])
                    rep_vals.append(rep_metrics["pr"])
                subsampled_pr.append(float(np.mean(rep_vals)))

            out.append(
                {
                    "recorded_session": rec["session"],
                    "recorded_pr": recorded_pr,
                    "recorded_frac_top2": recorded_top2,
                    "recorded_n_units": recorded_units,
                    "twin_n_traces": int(n),
                    "twin_mean_pr_full_units": float(np.mean(full_unit_pr_vals)) if full_unit_pr_vals else float("nan"),
                    "twin_mean_frac_top2_full_units": float(np.mean(full_unit_top2_vals)) if full_unit_top2_vals else float("nan"),
                    "twin_unit_matched_pr_mean": float(np.mean(subsampled_pr)) if subsampled_pr else float("nan"),
                    "twin_unit_matched_pr_sd": float(np.std(subsampled_pr, ddof=0)) if subsampled_pr else float("nan"),
                    "twin_n_units_available": int(available_units),
                    "unit_count_match_mode": "random_subsample_without_replacement",
                    "eye_cloud_match": "yes_same_real_trace_source",
                    "count_window_match": "no_recorded_uses_empirical_bins_not_trace_count",
                    "stimulus_match": "no_recorded_phase1_is_fixrsvp_natural_images",
                    "model_pr_comparable": False,
                    "comparability_note": (
                        "Treat as a qualitative anchor only: eye traces are matched in source/scale and unit count is matched by subsampling, "
                        "but the recorded result is empirical fixRSVP covariance and the twin result is deterministic eoptotype covariance."
                    ),
                }
            )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run A2 control audit and trace-count sweep")
    p.add_argument("--logmar", type=float, default=-0.30)
    p.add_argument("--orientations", type=str, default="0,90,180,270")
    p.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    p.add_argument("--n-traces-list", type=str, default=",".join(str(x) for x in DEFAULT_TRACE_COUNTS))
    p.add_argument("--include-full", action="store_true")
    p.add_argument("--eye-traces-path", type=Path, default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz")
    p.add_argument("--recorded-summary", type=Path, default=VISIONCORE_ROOT / "outputs" / "phase1_fem_covariance" / "covariance_geometry" / "covariance_geometry_session_metrics.csv")
    p.add_argument("--out-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "twin_covariance_structure" / "a2_audit")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hires-threshold", type=float, default=0.35)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--n-unit-subsamples", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    orientations = _parse_csv_ints(args.orientations)
    conditions = _parse_csv_strings(args.conditions)
    if conditions != list(DEFAULT_CONDITIONS):
        raise ValueError("A2 audit currently expects conditions real,x_only,y_only,line_random_angle")

    full_traces, full_durations = _load_eye_traces(args.eye_traces_path)
    total_traces = int(len(full_traces))
    trace_counts = _parse_csv_ints(args.n_traces_list)
    if args.include_full and total_traces not in trace_counts:
        trace_counts.append(total_traces)
    trace_counts = sorted(set(min(int(n), total_traces) for n in trace_counts if int(n) > 0))
    if not trace_counts:
        raise ValueError("No valid trace counts specified")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    perm = np.random.default_rng(int(args.selection_seed)).permutation(total_traces)
    model, readout = _load_model_and_readout(args.device)

    count_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    eye_geometry_rows: list[dict[str, object]] = []
    real_rates_by_count: dict[int, dict[int, np.ndarray]] = {}

    for n in trace_counts:
        selected_idx = np.asarray(perm[:n], dtype=np.int32)
        traces = full_traces[selected_idx]
        durations = full_durations[selected_idx]
        label = _count_label(int(n), total_traces)
        rates_dir = out_dir / f"rates_{label}"
        real_rates_by_count[int(n)] = {}

        eye_geometry_rows.append(
            {
                "n_traces": int(n),
                "n_traces_label": label,
                **_trace_geometry_metrics(traces, durations),
            }
        )

        for orientation in orientations:
            for control_idx, condition in enumerate(conditions):
                random_seed = int(args.selection_seed) + (1000 * int(orientation)) + (100000 * int(round((float(args.logmar) + 10.0) * 100))) + control_idx
                path = _compute_rates_for_control(
                    model=model,
                    readout=readout,
                    rates_dir=rates_dir,
                    logmar=float(args.logmar),
                    orientation=int(orientation),
                    control_type=condition,
                    traces=traces,
                    durations=durations,
                    hires_threshold=float(args.hires_threshold),
                    batch_size=int(args.batch_size),
                    spatial_collapse="max",
                    force=bool(args.force),
                    random_seed=random_seed,
                    file_tag="",
                    source_trace_count=total_traces,
                    selected_trace_indices=selected_idx,
                    selected_trace_indices_identical_across_conditions=True,
                )
                rates = _load_rates_array(path, max_trials=int(n))
                metrics, evals = _compute_condition_metrics(rates)
                count_rows.append(
                    {
                        "n_traces": int(n),
                        "n_traces_label": label,
                        "orientation": int(orientation),
                        "condition": condition,
                        "pr": metrics["pr"],
                        "frac_top2": metrics["frac_top2"],
                        "cfem_trace": metrics["cfem_trace"],
                        "rates_path": str(path),
                    }
                )
                for eigen_index, eigenvalue in enumerate(evals, start=1):
                    spectrum_rows.append(
                        {
                            "n_traces": int(n),
                            "n_traces_label": label,
                            "orientation": int(orientation),
                            "condition": condition,
                            "eigen_index": int(eigen_index),
                            "eigenvalue": float(eigenvalue),
                        }
                    )
                if condition == "real":
                    real_rates_by_count[int(n)][int(orientation)] = rates

    _write_csv(out_dir / "a2_tracecount_metrics.csv", count_rows)
    _write_csv(out_dir / "a2_tracecount_eigenspectra.csv", spectrum_rows)
    _write_csv(out_dir / "a2_tracecount_eye_geometry.csv", eye_geometry_rows)
    (out_dir / "a2_control_construction_audit.json").write_text(
        json.dumps(
            {
                **_build_control_audit_payload(),
                "selection": {
                    "mode": "nested_prefix_single_permutation",
                    "selection_seed": int(args.selection_seed),
                    "selected_trace_indices_identical_across_conditions": True,
                    "trace_counts": trace_counts,
                },
            },
            indent=2,
        )
        + "\n"
    )

    recorded_table = _build_recorded_vs_twin_table(
        args.recorded_summary,
        real_rates_by_count,
        count_rows,
        n_unit_subsamples=int(args.n_unit_subsamples),
        seed=int(args.selection_seed),
    )
    _write_csv(out_dir / "recorded_vs_twin_pr_comparison.csv", recorded_table)

    _plot_metric_vs_count(count_rows, trace_counts, figures_dir / "a2_pr_vs_n_traces.png", "pr", "Participation Ratio")
    _plot_metric_vs_count(
        count_rows,
        trace_counts,
        figures_dir / "a2_frac_top2_vs_n_traces.png",
        "frac_top2",
        "Top-2 Variance Fraction",
    )
    _plot_eigenspectra_vs_count(
        spectrum_rows,
        trace_counts,
        figures_dir / "a2_eigenspectra_vs_n_traces.png",
        max_rank=20,
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "trace_counts": trace_counts,
                "n_metric_rows": len(count_rows),
                "n_spectrum_rows": len(spectrum_rows),
                "n_eye_geometry_rows": len(eye_geometry_rows),
                "n_recorded_bridge_rows": len(recorded_table),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
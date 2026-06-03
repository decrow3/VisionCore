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
        top_subspace,
    )
    from declan.twin_covariance_structure.generate_control_response_caches import (  # type: ignore
        _compute_rates_for_control,
        _load_eye_traces,
        _load_model_and_readout,
    )
    from declan.twin_covariance_structure.run_twin_covariance_structure import _load_rates_array  # type: ignore
    from declan.twin_covariance_structure.subspace_metrics import subspace_overlap  # type: ignore
else:
    from .covariance_core import compute_cfem_for_image, eigensystem, top_subspace
    from .generate_control_response_caches import _compute_rates_for_control, _load_eye_traces, _load_model_and_readout
    from .run_twin_covariance_structure import _load_rates_array
    from .subspace_metrics import subspace_overlap


DEFAULT_TRACE_COUNTS = (40, 80, 160, 320)
DEFAULT_A3_K_LIST = (1, 2, 3)
EPS = 1e-12


def _parse_csv_ints(text: str) -> list[int]:
    return [int(float(x)) for x in str(text).split(",") if str(x).strip()]


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


def _is_orientation_proxy(image_ids: list[str]) -> bool:
    return all(x.startswith("ori") for x in image_ids) and len(image_ids) <= 4


def _split_half_overlap(rates: np.ndarray, k: int, rng: np.random.Generator) -> float:
    n_eye = int(rates.shape[0])
    if n_eye < 2:
        return float("nan")
    idx = rng.permutation(n_eye)
    mid = max(1, n_eye // 2)
    a = rates[idx[:mid]]
    b = rates[idx[mid:]]
    if b.shape[0] < 1:
        b = rates[idx[:mid]]

    C_a, _ = compute_cfem_for_image(a, return_per_t=False)
    C_b, _ = compute_cfem_for_image(b, return_per_t=False)
    _, evecs_a = eigensystem(C_a)
    _, evecs_b = eigensystem(C_b)
    U_a = top_subspace(evecs_a, k)
    U_b = top_subspace(evecs_b, k)
    return float(subspace_overlap(U_a, U_b))


def _plot_delta_vs_count(metrics_rows: list[dict[str, object]], counts: list[int], k_list: list[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(counts), dtype=np.float64)
    xlabels = [str(c) for c in counts]

    for k in k_list:
        vals = []
        for n in counts:
            match = [
                float(r["delta_within_minus_cross"])
                for r in metrics_rows
                if int(r["n_traces"]) == int(n) and int(r["k_a3"]) == int(k)
            ]
            vals.append(float(np.mean(match)) if match else float("nan"))
        ax.plot(x, vals, marker="o", linewidth=2.0, label=f"k={k}")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("n_traces")
    ax.set_ylabel("within - cross overlap")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_within_cross_vs_count(
    metrics_rows: list[dict[str, object]],
    counts: list[int],
    k_list: list[int],
    out_path: Path,
) -> None:
    ncols = 2
    nrows = int(np.ceil(len(k_list) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(11.0, 4.2 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)

    x = np.arange(len(counts), dtype=np.float64)
    xlabels = [str(c) for c in counts]

    for i, k in enumerate(k_list):
        ax = axs.ravel()[i]
        ax.set_visible(True)
        within = []
        cross = []
        for n in counts:
            row = next(
                (
                    r
                    for r in metrics_rows
                    if int(r["n_traces"]) == int(n) and int(r["k_a3"]) == int(k)
                ),
                None,
            )
            within.append(float(row["within_mean"]) if row is not None else float("nan"))
            cross.append(float(row["cross_mean"]) if row is not None else float("nan"))
        ax.plot(x, within, marker="o", linewidth=2.0, label="within split-half")
        ax.plot(x, cross, marker="o", linewidth=2.0, label="cross-image")
        ax.set_title(f"k={k}")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("n_traces")
        ax.set_ylabel("overlap")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmaps(
    matrix_rows: list[dict[str, object]],
    image_ids: list[str],
    selected_counts: list[int],
    k_for_plot: int,
    out_path: Path,
) -> None:
    ncols = len(selected_counts)
    fig, axs = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.4), squeeze=False)

    for col, n in enumerate(selected_counts):
        ax = axs[0, col]
        mat = np.full((len(image_ids), len(image_ids)), np.nan, dtype=np.float64)
        for row in matrix_rows:
            if int(row["n_traces"]) != int(n) or int(row["k_a3"]) != int(k_for_plot):
                continue
            i = image_ids.index(str(row["image_i"]))
            j = image_ids.index(str(row["image_j"]))
            mat[i, j] = float(row["overlap"])

        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(image_ids)))
        ax.set_yticks(np.arange(len(image_ids)))
        ax.set_xticklabels(image_ids, rotation=30, ha="right")
        ax.set_yticklabels(image_ids)
        ax.set_title(f"n_traces={n}, k={k_for_plot}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run A3 image-specificity trace-count audit")
    p.add_argument("--logmar", type=float, default=-0.30)
    p.add_argument("--orientations", type=str, default="0,90,180,270")
    p.add_argument("--condition", type=str, default="real")
    p.add_argument("--n-traces-list", type=str, default=",".join(str(x) for x in DEFAULT_TRACE_COUNTS))
    p.add_argument("--include-full", action="store_true")
    p.add_argument("--a3-k-list", type=str, default=",".join(str(x) for x in DEFAULT_A3_K_LIST))
    p.add_argument("--split-repeats", type=int, default=64)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--allow-orientation-proxy", action="store_true")
    p.add_argument(
        "--eye-traces-path",
        type=Path,
        default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hires-threshold", type=float, default=0.35)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=VISIONCORE_ROOT / "outputs" / "twin_covariance_structure" / "a3_audit",
    )
    p.add_argument("--plot-k", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    orientations = _parse_csv_ints(args.orientations)
    if not orientations:
        raise ValueError("No orientations provided")

    full_traces, full_durations = _load_eye_traces(args.eye_traces_path)
    total_traces = int(len(full_traces))
    trace_counts = _parse_csv_ints(args.n_traces_list)
    if args.include_full and total_traces not in trace_counts:
        trace_counts.append(total_traces)
    trace_counts = sorted(set(min(int(n), total_traces) for n in trace_counts if int(n) > 0))
    if not trace_counts:
        raise ValueError("No valid trace counts specified")

    a3_k_list = sorted(set(int(x) for x in _parse_csv_ints(args.a3_k_list) if int(x) > 0))
    if not a3_k_list:
        raise ValueError("No valid a3-k values specified")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    perm = np.random.default_rng(int(args.selection_seed)).permutation(total_traces)
    model, readout = _load_model_and_readout(args.device)

    metrics_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []

    image_ids = [f"ori{int(o)}" for o in orientations]
    orientation_proxy = _is_orientation_proxy(image_ids)
    plot_k = int(args.plot_k) if args.plot_k is not None else int(a3_k_list[min(1, len(a3_k_list) - 1)])

    for n in trace_counts:
        selected_idx = np.asarray(perm[:n], dtype=np.int32)
        traces = full_traces[selected_idx]
        durations = full_durations[selected_idx]
        label = _count_label(int(n), total_traces)
        rates_dir = out_dir / f"rates_{label}"

        rates_by_image: dict[str, np.ndarray] = {}
        evecs_by_image: dict[str, np.ndarray] = {}

        for orientation in orientations:
            random_seed = int(args.selection_seed) + (1000 * int(orientation)) + (100000 * int(round((float(args.logmar) + 10.0) * 100)))
            path = _compute_rates_for_control(
                model=model,
                readout=readout,
                rates_dir=rates_dir,
                logmar=float(args.logmar),
                orientation=int(orientation),
                control_type=str(args.condition),
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
            image_id = f"ori{int(orientation)}"
            rates_by_image[image_id] = rates
            C, _ = compute_cfem_for_image(rates, return_per_t=False)
            _, evecs = eigensystem(C)
            evecs_by_image[image_id] = evecs

        for k in a3_k_list:
            U_by_image = {image_id: top_subspace(evecs_by_image[image_id], int(k)) for image_id in image_ids}
            overlap_matrix = np.zeros((len(image_ids), len(image_ids)), dtype=np.float64)

            for i, image_i in enumerate(image_ids):
                for j, image_j in enumerate(image_ids):
                    ov = float(subspace_overlap(U_by_image[image_i], U_by_image[image_j]))
                    overlap_matrix[i, j] = ov
                    matrix_rows.append(
                        {
                            "n_traces": int(n),
                            "n_traces_label": label,
                            "k_a3": int(k),
                            "image_i": image_i,
                            "image_j": image_j,
                            "overlap": ov,
                            "pair_type": "diag" if i == j else "offdiag",
                        }
                    )

            cross_vals = [float(overlap_matrix[i, j]) for i in range(len(image_ids)) for j in range(len(image_ids)) if i != j]
            within_vals = []
            for image_id in image_ids:
                rates = rates_by_image[image_id]
                for rep in range(int(args.split_repeats)):
                    rep_seed = int(args.selection_seed) + (10000 * int(n)) + (1000 * int(k)) + rep
                    rep_rng = np.random.default_rng(rep_seed)
                    ov = _split_half_overlap(rates, int(k), rep_rng)
                    within_vals.append(ov)
                    split_rows.append(
                        {
                            "n_traces": int(n),
                            "n_traces_label": label,
                            "k_a3": int(k),
                            "image_id": image_id,
                            "repeat_idx": int(rep),
                            "within_overlap": float(ov),
                        }
                    )

            within_arr = np.asarray(within_vals, dtype=np.float64)
            cross_arr = np.asarray(cross_vals, dtype=np.float64)
            delta = float(np.nanmean(within_arr) - np.nanmean(cross_arr)) if within_arr.size and cross_arr.size else float("nan")

            can_claim = within_arr.size > 0 and cross_arr.size > 0 and (args.allow_orientation_proxy or (not orientation_proxy))
            metrics_rows.append(
                {
                    "n_traces": int(n),
                    "n_traces_label": label,
                    "k_a3": int(k),
                    "condition": str(args.condition),
                    "n_images": int(len(image_ids)),
                    "split_repeats": int(args.split_repeats),
                    "orientation_proxy": bool(orientation_proxy),
                    "a3_ran_for_claim": bool(can_claim),
                    "within_mean": float(np.nanmean(within_arr)) if within_arr.size else float("nan"),
                    "within_sd": float(np.nanstd(within_arr, ddof=0)) if within_arr.size else float("nan"),
                    "cross_mean": float(np.nanmean(cross_arr)) if cross_arr.size else float("nan"),
                    "cross_sd": float(np.nanstd(cross_arr, ddof=0)) if cross_arr.size else float("nan"),
                    "delta_within_minus_cross": delta,
                }
            )

    _write_csv(out_dir / "a3_tracecount_metrics.csv", metrics_rows)
    _write_csv(out_dir / "a3_splithalf_repeats.csv", split_rows)
    _write_csv(out_dir / "a3_overlap_matrix_long.csv", matrix_rows)

    _plot_delta_vs_count(metrics_rows, trace_counts, a3_k_list, figures_dir / "a3_delta_vs_n_traces.png")
    _plot_within_cross_vs_count(metrics_rows, trace_counts, a3_k_list, figures_dir / "a3_within_cross_vs_n_traces.png")

    selected_counts = []
    if trace_counts:
        selected_counts = sorted(set([trace_counts[0], trace_counts[len(trace_counts) // 2], trace_counts[-1]]))
    if selected_counts:
        _plot_heatmaps(
            matrix_rows,
            image_ids,
            selected_counts,
            int(plot_k),
            figures_dir / "a3_overlap_heatmaps_selected_counts.png",
        )

    payload = {
        "selection": {
            "mode": "nested_prefix_single_permutation",
            "selection_seed": int(args.selection_seed),
            "trace_counts": trace_counts,
            "selected_trace_indices_identical_across_conditions": True,
        },
        "a3_claim_guard": {
            "orientation_proxy": bool(orientation_proxy),
            "allow_orientation_proxy": bool(args.allow_orientation_proxy),
            "reason_when_not_allowed": (
                "current_stimulus_set_is_orientation_proxy_only_use_diverse_images_or_pass_allow_orientation_proxy"
                if orientation_proxy and not args.allow_orientation_proxy
                else ""
            ),
        },
        "config": {
            "logmar": float(args.logmar),
            "orientations": orientations,
            "condition": str(args.condition),
            "a3_k_list": a3_k_list,
            "split_repeats": int(args.split_repeats),
            "batch_size": int(args.batch_size),
            "device": args.device,
        },
    }
    (out_dir / "a3_audit_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "trace_counts": trace_counts,
                "a3_k_list": a3_k_list,
                "n_metric_rows": len(metrics_rows),
                "n_splithalf_rows": len(split_rows),
                "n_matrix_rows": len(matrix_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

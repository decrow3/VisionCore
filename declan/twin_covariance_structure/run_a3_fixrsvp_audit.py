from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import dill

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]

from eval.fixrsvp import get_fixrsvp_data

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(VISIONCORE_ROOT))
    from declan.twin_covariance_structure.covariance_core import compute_signal_covariance, eigensystem, top_subspace  # type: ignore
    from declan.twin_covariance_structure.subspace_metrics import subspace_overlap  # type: ignore
else:
    from .covariance_core import compute_signal_covariance, eigensystem, top_subspace
    from .subspace_metrics import subspace_overlap


DEFAULT_SAMPLE_COUNTS = (40, 80, 160, 320)
DEFAULT_A3_K_LIST = (1, 2, 3)


def _parse_csv_ints(text: str) -> list[int]:
    return [int(float(x)) for x in str(text).split(",") if str(x).strip()]


def _top_subspace_from_samples(samples: np.ndarray, k: int) -> np.ndarray:
    """Compute top-k population subspace via sample-space SVD (fast when samples << units)."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D samples array, got shape {x.shape}")
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k_eff = max(1, min(int(k), vt.shape[0]))
    return vt[:k_eff].T


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_label(n_samples: int, total: int) -> str:
    return "full" if n_samples >= total else str(int(n_samples))


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
                if int(r["n_samples"]) == int(n) and int(r["k_a3"]) == int(k)
            ]
            vals.append(float(np.mean(match)) if match else float("nan"))
        ax.plot(x, vals, marker="o", linewidth=2.0, label=f"k={k}")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("n_samples per image")
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
            row = next((r for r in metrics_rows if int(r["n_samples"]) == int(n) and int(r["k_a3"]) == int(k)), None)
            within.append(float(row["within_mean"]) if row is not None else float("nan"))
            cross.append(float(row["cross_mean"]) if row is not None else float("nan"))
        ax.plot(x, within, marker="o", linewidth=2.0, label="within split-half")
        ax.plot(x, cross, marker="o", linewidth=2.0, label="cross-image")
        ax.set_title(f"k={k}")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("n_samples per image")
        ax.set_ylabel("overlap")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmaps(
    matrix_rows: list[dict[str, object]],
    image_ids: list[int],
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
            if int(row["n_samples"]) != int(n) or int(row["k_a3"]) != int(k_for_plot):
                continue
            i = image_ids.index(int(row["image_i"]))
            j = image_ids.index(int(row["image_j"]))
            mat[i, j] = float(row["overlap"])

        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(image_ids)))
        ax.set_yticks(np.arange(len(image_ids)))
        ax.set_xticklabels([str(i) for i in image_ids], rotation=30, ha="right")
        ax.set_yticklabels([str(i) for i in image_ids])
        ax.set_title(f"n_samples={n}, k={k_for_plot}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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

    U_a = _top_subspace_from_samples(a, int(k))
    U_b = _top_subspace_from_samples(b, int(k))
    return float(subspace_overlap(U_a, U_b))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run A3 image-specificity audit on fixRSVP stimuli")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--dataset-configs-path", type=Path, required=True)
    p.add_argument("--session-name", type=str, default=None)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--sample-counts", type=str, default=",".join(str(x) for x in DEFAULT_SAMPLE_COUNTS))
    p.add_argument("--include-full", action="store_true")
    p.add_argument("--a3-k-list", type=str, default=",".join(str(x) for x in DEFAULT_A3_K_LIST))
    p.add_argument("--split-repeats", type=int, default=64)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--source", choices=("recorded", "twin", "both"), default="recorded")
    p.add_argument("--model-type", type=str, default="learned_res_ddp_bs256_ds30_lr1e-3_wd1e-4_corelrscale0.5_warmup5")
    p.add_argument("--model-index", type=int, default=None)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--dataset-idx", type=int, default=None)
    p.add_argument("--model-device", type=str, default="cuda")
    p.add_argument("--predict-batch-size", type=int, default=512)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=VISIONCORE_ROOT / "outputs" / "twin_covariance_structure" / "a3_fixrsvp_audit",
    )
    p.add_argument("--use-cached-data", action="store_true", default=True)
    return p


def _predict_twin_rates(
    *,
    data: dict[str, object],
    subject: str,
    date: str,
    dataset_configs_path: str,
    model_type: str | None,
    model_index: int | None,
    checkpoint_path: str | None,
    dataset_idx: int | None,
    model_device: str,
    predict_batch_size: int,
) -> tuple[np.ndarray, int]:
    import sys

    scripts_dir = VISIONCORE_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from scripts.utils import get_model_and_dataset_configs
    from scripts.spatial_info import embed_time_lags, get_spatial_readout
    def _compute_trial_rates_fast(model_wrapper, readout_module, stim_lagged: torch.Tensor, batch_size: int) -> np.ndarray:
        """Fast trial rate computation without per-batch CUDA cache flushes."""
        device = next(model_wrapper.model.parameters()).device
        model_wrapper.model.eval()
        readout_module.eval()
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for t_start in range(0, int(stim_lagged.shape[0]), int(batch_size)):
                t_end = min(t_start + int(batch_size), int(stim_lagged.shape[0]))
                x_batch = stim_lagged[t_start:t_end].to(device)
                feats = model_wrapper.model.core_forward(x_batch, None)
                feats_last = feats[:, :, -1]
                y = readout_module(feats_last)
                rates_spatial = model_wrapper.model.activation(y)
                rates = rates_spatial.amax(dim=(-2, -1))
                chunks.append(rates.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)

    model, _ = get_model_and_dataset_configs(mode="standard")
    model = model.to(model_device)
    model.model.eval()

    session_name = f"{subject}_{date}"
    resolved_dataset_idx = dataset_idx
    if resolved_dataset_idx is None:
        if hasattr(model, "names") and session_name in model.names:
            resolved_dataset_idx = int(model.names.index(session_name))
        else:
            raise ValueError("Could not auto-resolve dataset_idx from model.names; pass --dataset-idx explicitly for twin source.")

    stim = np.asarray(data["stim"], dtype=np.float32)
    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)
    base_valid = np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)
    if not np.any(base_valid):
        raise ValueError("No valid fixRSVP bins available for twin prediction")

    outputs_path = VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl"
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing readout source for twin predictions: {outputs_path}")
    with outputs_path.open("rb") as handle:
        outputs = dill.load(handle)
    readout = get_spatial_readout(model, outputs).to(model_device)
    readout.eval()

    n_lags = 32
    rates = np.full((image_ids.shape[0], image_ids.shape[1], int(readout.n_units)), np.nan, dtype=np.float64)

    for trial_idx in range(stim.shape[0]):
        movie = torch.as_tensor(stim[trial_idx], dtype=torch.float32)
        if movie.dim() == 4 and movie.shape[1] == 1:
            movie = movie[:, 0]
        if movie.dim() != 3:
            continue
        if movie.shape[0] < n_lags:
            continue
        lagged = embed_time_lags(movie, n_lags=n_lags)
        trial_rates = _compute_trial_rates_fast(
            model,
            readout,
            lagged,
            batch_size=int(predict_batch_size),
        )
        t_start = n_lags - 1
        t_end = min(image_ids.shape[1], t_start + trial_rates.shape[0])
        if t_end > t_start:
            rates[trial_idx, t_start:t_end] = trial_rates[: (t_end - t_start)]
        if (trial_idx + 1) % 8 == 0 or (trial_idx + 1) == stim.shape[0]:
            print(f"twin prediction progress: {trial_idx + 1}/{stim.shape[0]} trials", flush=True)

    rates = np.where(base_valid[:, :, None], rates, np.nan)
    return rates, int(resolved_dataset_idx)


def _harmonize_fixrsvp_arrays(data: dict[str, object]) -> dict[str, object]:
    robs = np.asarray(data["robs"]) if "robs" in data else None
    eyepos = np.asarray(data["eyepos"]) if "eyepos" in data else None
    image_ids = np.asarray(data["image_ids"]) if "image_ids" in data else None
    stim = np.asarray(data["stim"]) if "stim" in data and data["stim"] is not None else None

    nt_candidates = []
    t_candidates = []
    for arr in (robs, eyepos, image_ids, stim):
        if arr is None or arr.ndim < 2:
            continue
        nt_candidates.append(int(arr.shape[0]))
        t_candidates.append(int(arr.shape[1]))

    if not nt_candidates or not t_candidates:
        return data

    nt = min(nt_candidates)
    tt = min(t_candidates)

    out = dict(data)
    if robs is not None and robs.ndim >= 2:
        out["robs"] = robs[:nt, :tt, ...]
    if eyepos is not None and eyepos.ndim >= 2:
        out["eyepos"] = eyepos[:nt, :tt, ...]
    if image_ids is not None and image_ids.ndim >= 2:
        out["image_ids"] = image_ids[:nt, :tt, ...]
    if stim is not None and stim.ndim >= 2:
        out["stim"] = stim[:nt, :tt, ...]
    return out


def _run_one_source(args: argparse.Namespace, data: dict[str, object], source: str) -> dict[str, object]:
    if source == "recorded":
        source_rates = np.asarray(data["robs"], dtype=np.float64)
        resolved_dataset_idx = None
    elif source == "twin":
        source_rates, resolved_dataset_idx = _predict_twin_rates(
            data=data,
            subject=args.subject,
            date=args.date,
            dataset_configs_path=str(args.dataset_configs_path),
            model_type=args.model_type,
            model_index=args.model_index,
            checkpoint_path=args.checkpoint_path,
            dataset_idx=args.dataset_idx,
            model_device=str(args.model_device),
            predict_batch_size=int(args.predict_batch_size),
        )
    else:
        raise ValueError(f"Unknown source={source}")

    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)

    valid = np.isfinite(source_rates).all(axis=2) & np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)
    support_rows: list[dict[str, object]] = []
    image_rates: dict[int, np.ndarray] = {}

    for img_id in sorted(int(x) for x in np.unique(image_ids[valid])):
        mask = valid & (image_ids == img_id)
        rates = source_rates[mask].astype(np.float64)
        support_rows.append({
            "source": source,
            "image_id": int(img_id),
            "n_samples": int(rates.shape[0]),
        })
        if rates.shape[0] >= int(args.min_samples):
            image_rates[int(img_id)] = rates

    sample_counts = _parse_csv_ints(args.sample_counts)
    if args.include_full and image_rates:
        full_min = min(r.shape[0] for r in image_rates.values())
        if full_min not in sample_counts:
            sample_counts.append(full_min)
    max_supported = max((r.shape[0] for r in image_rates.values()), default=0)
    sample_counts = sorted(set(min(int(n), max_supported) for n in sample_counts if int(n) > 0))
    sample_counts = [n for n in sample_counts if n > 0]
    if not sample_counts:
        raise ValueError(f"No valid sample counts specified for source={source}")

    a3_k_list = sorted(set(int(x) for x in _parse_csv_ints(args.a3_k_list) if int(x) > 0))
    if not a3_k_list:
        raise ValueError("No valid a3-k values specified")

    session_name = args.session_name or f"{args.subject}_{args.date}_fixrsvp_a3"
    out_dir = Path(args.out_dir) / session_name / f"source_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_image_orders: dict[int, np.ndarray] = {}
    for img_id, rates in image_rates.items():
        rng = np.random.default_rng(int(args.selection_seed) + (10007 * int(img_id)))
        per_image_orders[int(img_id)] = rng.permutation(rates.shape[0])

    metrics_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []

    image_ids_sorted = sorted(image_rates.keys())
    plot_k = int(a3_k_list[min(1, len(a3_k_list) - 1)])

    for n in sample_counts:
        selected_image_ids = [img_id for img_id in image_ids_sorted if image_rates[img_id].shape[0] >= int(n)]
        if len(selected_image_ids) < 2:
            continue

        selected_rates_by_image: dict[int, np.ndarray] = {}

        for img_id in selected_image_ids:
            rates = image_rates[img_id]
            order = per_image_orders[img_id]
            selected = rates[order[: int(n)]]
            selected_rates_by_image[img_id] = selected

        for k in a3_k_list:
            U_by_image = {img_id: _top_subspace_from_samples(selected_rates_by_image[img_id], int(k)) for img_id in selected_image_ids}
            overlap_matrix = np.zeros((len(selected_image_ids), len(selected_image_ids)), dtype=np.float64)

            for i, image_i in enumerate(selected_image_ids):
                for j, image_j in enumerate(selected_image_ids):
                    ov = float(subspace_overlap(U_by_image[image_i], U_by_image[image_j]))
                    overlap_matrix[i, j] = ov
                    matrix_rows.append(
                        {
                            "source": source,
                            "n_samples": int(n),
                            "n_samples_label": _count_label(int(n), max(sample_counts)),
                            "k_a3": int(k),
                            "image_i": int(image_i),
                            "image_j": int(image_j),
                            "overlap": ov,
                            "pair_type": "diag" if i == j else "offdiag",
                        }
                    )

            cross_vals = [float(overlap_matrix[i, j]) for i in range(len(selected_image_ids)) for j in range(len(selected_image_ids)) if i != j]
            within_vals = []
            for image_id in selected_image_ids:
                rates = selected_rates_by_image[image_id]
                for rep in range(int(args.split_repeats)):
                    rep_seed = int(args.selection_seed) + (10000 * int(n)) + (1000 * int(k)) + rep + int(image_id)
                    rep_rng = np.random.default_rng(rep_seed)
                    ov = _split_half_overlap(rates, int(k), rep_rng)
                    within_vals.append(ov)
                    split_rows.append(
                        {
                            "source": source,
                            "n_samples": int(n),
                            "n_samples_label": _count_label(int(n), max(sample_counts)),
                            "k_a3": int(k),
                            "image_id": int(image_id),
                            "repeat_idx": int(rep),
                            "within_overlap": float(ov),
                        }
                    )

            within_arr = np.asarray(within_vals, dtype=np.float64)
            cross_arr = np.asarray(cross_vals, dtype=np.float64)
            metrics_rows.append(
                {
                    "source": source,
                    "session": session_name,
                    "subject": args.subject,
                    "date": args.date,
                    "n_samples": int(n),
                    "n_samples_label": _count_label(int(n), max(sample_counts)),
                    "k_a3": int(k),
                    "n_images": int(len(selected_image_ids)),
                    "min_image_support": int(min(image_rates[img_id].shape[0] for img_id in selected_image_ids)),
                    "split_repeats": int(args.split_repeats),
                    "a3_ran_for_claim": bool(within_arr.size > 0 and cross_arr.size > 0),
                    "within_mean": float(np.nanmean(within_arr)) if within_arr.size else float("nan"),
                    "within_sd": float(np.nanstd(within_arr, ddof=0)) if within_arr.size else float("nan"),
                    "cross_mean": float(np.nanmean(cross_arr)) if cross_arr.size else float("nan"),
                    "cross_sd": float(np.nanstd(cross_arr, ddof=0)) if cross_arr.size else float("nan"),
                    "delta_within_minus_cross": (
                        float(np.nanmean(within_arr) - np.nanmean(cross_arr)) if within_arr.size and cross_arr.size else float("nan")
                    ),
                }
            )

    _write_csv(out_dir / "a3_image_support.csv", support_rows)
    _write_csv(out_dir / "a3_tracecount_metrics.csv", metrics_rows)
    _write_csv(out_dir / "a3_splithalf_repeats.csv", split_rows)
    _write_csv(out_dir / "a3_overlap_matrix_long.csv", matrix_rows)

    if metrics_rows:
        _plot_delta_vs_count(metrics_rows, sample_counts, a3_k_list, figures_dir / "a3_delta_vs_n_samples.png")
        _plot_within_cross_vs_count(metrics_rows, sample_counts, a3_k_list, figures_dir / "a3_within_cross_vs_n_samples.png")

    selected_counts = []
    if sample_counts:
        selected_counts = sorted(set([sample_counts[0], sample_counts[len(sample_counts) // 2], sample_counts[-1]]))
    if selected_counts and image_ids_sorted:
        _plot_heatmaps(
            matrix_rows,
            image_ids_sorted,
            selected_counts,
            int(plot_k),
            figures_dir / "a3_overlap_heatmaps_selected_counts.png",
        )

    payload = {
        "session": {
            "session_name": session_name,
            "subject": args.subject,
            "date": args.date,
            "dataset_configs_path": str(args.dataset_configs_path),
        },
        "source": source,
        "model": {
            "model_type": args.model_type,
            "model_index": args.model_index,
            "checkpoint_path": args.checkpoint_path,
            "dataset_idx": resolved_dataset_idx,
            "model_device": args.model_device,
        },
        "selection": {
            "mode": "nested_prefix_per_image",
            "selection_seed": int(args.selection_seed),
            "sample_counts": sample_counts,
        },
        "a3_claim_guard": {
            "orientation_proxy": False,
            "allow_orientation_proxy": True,
            "reason_when_not_allowed": "",
        },
        "config": {
            "min_samples": int(args.min_samples),
            "a3_k_list": a3_k_list,
            "split_repeats": int(args.split_repeats),
        },
    }
    (out_dir / "a3_audit_metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return {
        "source": source,
        "out_dir": str(out_dir),
        "sample_counts": sample_counts,
        "a3_k_list": a3_k_list,
        "n_image_support_rows": len(support_rows),
        "n_metric_rows": len(metrics_rows),
        "n_splithalf_rows": len(split_rows),
        "n_matrix_rows": len(matrix_rows),
    }


def main() -> None:
    args = build_parser().parse_args()
    data = get_fixrsvp_data(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=str(args.dataset_configs_path),
        use_cached_data=bool(args.use_cached_data),
    )
    data = _harmonize_fixrsvp_arrays(data)

    sources = ["recorded", "twin"] if args.source == "both" else [str(args.source)]
    session_name = args.session_name or f"{args.subject}_{args.date}_fixrsvp_a3"
    summaries = []
    for source in sources:
        try:
            summaries.append(_run_one_source(args, data, source))
        except Exception as exc:
            out_dir = Path(args.out_dir) / session_name / f"source_{source}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "a3_run_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            summaries.append(
                {
                    "source": source,
                    "out_dir": str(out_dir),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
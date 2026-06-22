from __future__ import annotations

import argparse
import json
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .image_features import _backimage_canvas, gaze_deg_to_screen_px
from .run_backimage_latent_information_screen import (
    _central_crop,
    _resize_to_square,
    _standardize_uint_like,
    _zscore_image,
)
from .run_backimage_twin_drift_geometry import _clip_patch

from jake.twininfo.retinal_examples import (
    _copy_pyramid_coeffs,
    _padded_even_patch,
    _patch_to_tensor,
    _reconstruct_pyramid_patch,
    _steerable_pyramid,
)


DEFAULT_RUN_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_aggregate_fem_information_pose_unaware_production_n384_empirical_k8_seed0"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _standardize_train(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (arr - mu) / sd, mu, sd


def _pca_reconstruct_features(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    standardized, mu, sd = _standardize_train(features)
    k_eff = int(min(int(k), standardized.shape[0] - 1, standardized.shape[1]))
    pca = PCA(n_components=k_eff, svd_solver="full")
    scores = pca.fit_transform(standardized)
    recon = pca.inverse_transform(scores) * sd + mu
    return recon.astype(np.float64), pca.explained_variance_ratio_


def _expand_block_means(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.float64)
    grid = int(blocks.shape[0])
    rows = np.array_split(np.arange(int(shape[0])), grid)
    cols = np.array_split(np.arange(int(shape[1])), grid)
    out = np.zeros(shape, dtype=np.float32)
    for r, rr in enumerate(rows):
        for c, cc in enumerate(cols):
            out[np.ix_(rr, cc)] = float(blocks[r, c])
    return out


def _feature_vector_to_coeffs(feature: np.ndarray, template_coeffs: OrderedDict, *, grid: int) -> OrderedDict:
    import torch

    vec = np.asarray(feature, dtype=np.float64)
    out = OrderedDict()
    pos = 0
    for key, value in template_coeffs.items():
        squeezed = value.detach().cpu().squeeze().numpy()
        if not np.iscomplexobj(squeezed):
            out[key] = value * 0
            continue
        if squeezed.ndim == 2:
            squeezed = squeezed[None, :, :]
        arr = np.zeros(squeezed.shape, dtype=np.complex64)
        for orient_idx in range(arr.shape[0]):
            real_blocks = vec[pos : pos + grid * grid].reshape(grid, grid)
            pos += grid * grid
            imag_blocks = vec[pos : pos + grid * grid].reshape(grid, grid)
            pos += grid * grid
            pos += grid * grid  # magnitude is a target feature but not an invertible coefficient.
            real = _expand_block_means(real_blocks, arr.shape[-2:])
            imag = _expand_block_means(imag_blocks, arr.shape[-2:])
            arr[orient_idx] = real + 1j * imag
        tensor = torch.as_tensor(arr, dtype=value.dtype, device=value.device).reshape(value.shape)
        out[key] = tensor
    if pos != vec.size:
        raise ValueError(f"Consumed {pos} feature values but vector has {vec.size}")
    return out


def _display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    arr = arr - np.nanmedian(arr)
    scale = np.nanpercentile(np.abs(arr), 98)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.nanstd(arr)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip(arr / scale, -1.0, 1.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    den = float(np.sqrt(np.nansum(x * x) * np.nansum(y * y)))
    if den <= 1e-12 or not np.isfinite(den):
        return float("nan")
    return float(np.nansum(x * y) / den)


def _image_for_row(row: pd.Series, *, patch_size_px: int, latent_crop_px: int) -> np.ndarray:
    canvas, ppd, screen_shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center_px = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=ppd,
        screen_shape=screen_shape,
    )
    patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(patch_size_px))
    image = _standardize_uint_like(patch)
    crop = _central_crop(image, int(latent_crop_px))
    return _zscore_image(_resize_to_square(crop, size_px=128)).astype(np.float32)


def _select_indices(n_images: int, *, explicit: list[int] | None, n_select: int, seed: int) -> np.ndarray:
    if explicit:
        idx = np.asarray(explicit, dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_images):
            raise ValueError(f"image indices must be in [0, {n_images - 1}]")
        return idx
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(np.arange(n_images), size=min(int(n_select), n_images), replace=False))


def build_reconstruction_sheet(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "target_reconstruction_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_json(run_dir / "run_metadata.json")
    cfg = metadata["config"]
    latent_name = str(args.latent_name)
    pca_k = int(args.pca_k)
    grid = int(cfg.get("local_field_grid", 8))
    patch_size_px = int(cfg.get("patch_size_px", 540))
    latent_crop_px = int(cfg.get("latent_crop_px", 151))

    latent_npz = np.load(run_dir / "latent_feature_arrays.npz")
    if latent_name not in latent_npz:
        raise KeyError(f"{latent_name!r} not found in {run_dir / 'latent_feature_arrays.npz'}")
    features = np.asarray(latent_npz[latent_name], dtype=np.float64)
    pca_features, explained = _pca_reconstruct_features(features, pca_k)

    analysis = pd.read_csv(run_dir / "analysis_images.csv")
    input_path = Path(cfg["input"])
    source = pd.read_csv(input_path)
    source["source_row"] = np.arange(source.shape[0], dtype=int)
    rows = analysis[["image_index", "source_row"]].merge(source, on="source_row", how="left", validate="one_to_one")
    if rows[["session", "trial_idx", "mean_x_deg", "mean_y_deg"]].isna().any().any():
        raise ValueError("Could not recover source image rows for every analysis image.")

    image_indices = _select_indices(
        features.shape[0],
        explicit=_parse_int_list(args.image_indices) if args.image_indices else None,
        n_select=int(args.n_images),
        seed=int(args.seed),
    )

    template = np.zeros((128, 128), dtype=np.float32)
    patch, padding = _padded_even_patch(template)
    pyr = _steerable_pyramid(patch.shape, height=4, order=3)
    template_coeffs = pyr(_patch_to_tensor(patch))

    records: list[dict[str, Any]] = []
    plot_rows = []
    for image_index in image_indices:
        row = rows.loc[rows["image_index"] == int(image_index)].iloc[0]
        original = _image_for_row(row, patch_size_px=patch_size_px, latent_crop_px=latent_crop_px)
        original_patch, original_padding = _padded_even_patch(original)
        original_coeffs = pyr(_patch_to_tensor(original_patch))
        exact = _reconstruct_pyramid_patch(pyr, _copy_pyramid_coeffs(original_coeffs), original_padding)
        target_coeffs = _feature_vector_to_coeffs(features[int(image_index)], template_coeffs, grid=grid)
        target = _reconstruct_pyramid_patch(pyr, target_coeffs, padding)
        pca_coeffs = _feature_vector_to_coeffs(pca_features[int(image_index)], template_coeffs, grid=grid)
        pca_target = _reconstruct_pyramid_patch(pyr, pca_coeffs, padding)

        plot_rows.append((int(image_index), original, exact, target, pca_target))
        records.append(
            {
                "image_index": int(image_index),
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "source_corr_exact_full_pyramid": _corr(original, exact),
                "source_corr_blockmean_target": _corr(original, target),
                "source_corr_pca_target": _corr(original, pca_target),
                "target_corr_blockmean_vs_pca": _corr(target, pca_target),
            }
        )

    metrics = pd.DataFrame.from_records(records)
    metrics_path = out_dir / "pyramid_local_field_reconstruction_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    n_rows = len(plot_rows)
    col_titles = [
        "original crop",
        "full pyramid recon",
        f"{latent_name}\nblock means",
        f"{latent_name}\nPCA{k_eff_label(pca_k)} target",
    ]
    fig, axes = plt.subplots(n_rows, 4, figsize=(9.2, max(2.0, 1.85 * n_rows)), squeeze=False)
    for row_idx, (image_index, original, exact, target, pca_target) in enumerate(plot_rows):
        for col_idx, image in enumerate((original, exact, target, pca_target)):
            ax = axes[row_idx, col_idx]
            ax.imshow(_display_image(image), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"img {image_index}", fontsize=8)
    explained_text = f"PCA{k_eff_label(pca_k)} explained variance: {float(np.sum(explained[:pca_k])):.3f}"
    fig.suptitle("Fig 4B target reconstructions from retained pyramid-local-field features", fontsize=11)
    footer = textwrap.fill(
        "Block-mean/PCA images are approximate inversions: residual highpass/lowpass are omitted "
        "and magnitude channels are non-invertible. "
        + explained_text,
        width=150,
    )
    fig.text(
        0.5,
        0.028,
        footer,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.965))
    png_path = out_dir / "pyramid_local_field_target_reconstruction_sheet.png"
    pdf_path = out_dir / "pyramid_local_field_target_reconstruction_sheet.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "latent_name": latent_name,
        "pca_k": pca_k,
        "pca_explained_variance": float(np.sum(explained[:pca_k])),
        "image_indices": [int(v) for v in image_indices],
        "metrics_csv": str(metrics_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "note": (
            "Block-mean and PCA reconstructions approximate the analysis target, not a full image inverse. "
            "The target omits real-valued highpass/lowpass residuals and stores block means plus magnitude summaries."
        ),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def k_eff_label(k: int) -> str:
    return str(int(k))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--latent-name", default="pyramid_local_field")
    parser.add_argument("--pca-k", type=int, default=16)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--image-indices", default="")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    build_reconstruction_sheet(args)


if __name__ == "__main__":
    main()

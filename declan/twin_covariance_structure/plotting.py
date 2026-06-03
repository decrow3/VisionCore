from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_a1_signal_alignment(rows: list[dict], out_stem: Path) -> None:
    if not rows:
        return
    ks = sorted({int(r["k"]) for r in rows})
    overlap = [np.mean([r["overlap_fem_signal"] for r in rows if int(r["k"]) == k]) for k in ks]
    fem_by_signal = [np.mean([r["fem_variance_captured_by_signal"] for r in rows if int(r["k"]) == k]) for k in ks]
    signal_by_fem = [np.mean([r["signal_variance_captured_by_fem"] for r in rows if int(r["k"]) == k]) for k in ks]
    null_mean = [np.mean([r["null_mean"] for r in rows if int(r["k"]) == k]) for k in ks]

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.ravel()
    axs[0].plot(ks, overlap, marker="o", label="real")
    axs[0].plot(ks, null_mean, marker="s", linestyle="--", label="null")
    axs[0].set_title("Subspace overlap")
    axs[0].set_xlabel("k")
    axs[0].set_ylabel("overlap")
    axs[0].legend(frameon=False)

    axs[1].plot(ks, fem_by_signal, marker="o")
    axs[1].set_title("FEM variance captured by signal")
    axs[1].set_xlabel("k")
    axs[1].set_ylabel("fraction")

    axs[2].plot(ks, signal_by_fem, marker="o")
    axs[2].set_title("Signal variance captured by FEM")
    axs[2].set_xlabel("k")
    axs[2].set_ylabel("fraction")

    pr = [float(r["cfem_pr"]) for r in rows if int(r["k"]) == ks[0]]
    al = [float(r["overlap_fem_signal"]) for r in rows if int(r["k"]) == ks[0]]
    axs[3].scatter(pr, al, s=20, alpha=0.8)
    axs[3].set_title("Per-image PR vs alignment")
    axs[3].set_xlabel("PR")
    axs[3].set_ylabel("overlap (k=1)")

    fig.suptitle("A1 Signal alignment")
    fig.tight_layout()
    _save_all(fig, out_stem)


def plot_a2_rank_mechanism(rows: list[dict], out_stem: Path) -> None:
    if not rows:
        return
    conditions = sorted({r["condition"] for r in rows})
    pr = [np.mean([rr["pr"] for rr in rows if rr["condition"] == c]) for c in conditions]
    top2 = [np.mean([rr["frac_top2"] for rr in rows if rr["condition"] == c]) for c in conditions]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].bar(np.arange(len(conditions)), pr, color="#4c78a8")
    axs[0].set_xticks(np.arange(len(conditions)))
    axs[0].set_xticklabels(conditions, rotation=40, ha="right")
    axs[0].set_ylabel("PR")
    axs[0].set_title("Participation ratio by condition")

    axs[1].bar(np.arange(len(conditions)), top2, color="#f58518")
    axs[1].set_xticks(np.arange(len(conditions)))
    axs[1].set_xticklabels(conditions, rotation=40, ha="right")
    axs[1].set_ylabel("Top-2 variance fraction")
    axs[1].set_title("Top-2 variance fraction by condition")

    fig.suptitle("A2 Rank mechanism")
    fig.tight_layout()
    _save_all(fig, out_stem)


def plot_a3_image_specificity(overlap_matrix: np.ndarray, within_vals: np.ndarray, cross_vals: np.ndarray, out_stem: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axs[0].imshow(overlap_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axs[0].set_title("Cross-image FEM overlap")
    axs[0].set_xlabel("image")
    axs[0].set_ylabel("image")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    bins = np.linspace(0.0, 1.0, 20)
    axs[1].hist(within_vals, bins=bins, alpha=0.6, label="within split-half")
    axs[1].hist(cross_vals, bins=bins, alpha=0.6, label="cross-image")
    axs[1].set_xlim(0.0, 1.0)
    axs[1].set_title("Within vs cross overlap")
    axs[1].set_xlabel("overlap")
    axs[1].legend(frameon=False)

    fig.suptitle("A3 Image specificity")
    fig.tight_layout()
    _save_all(fig, out_stem)


def plot_a4_tangent_alignment(rows: list[dict], out_stem: Path) -> None:
    if not rows:
        return
    ori = [r["image_id"] for r in rows]
    ov = [r["tangent_overlap"] for r in rows]
    cap = [r["fem_variance_captured_by_tangent"] for r in rows]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].bar(np.arange(len(ori)), ov, color="#54a24b")
    axs[0].set_xticks(np.arange(len(ori)))
    axs[0].set_xticklabels(ori)
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_title("Tangent overlap")

    axs[1].bar(np.arange(len(ori)), cap, color="#e45756")
    axs[1].set_xticks(np.arange(len(ori)))
    axs[1].set_xticklabels(ori)
    axs[1].set_ylim(0.0, 1.0)
    axs[1].set_title("FEM variance captured by tangent")

    fig.suptitle("A4 Translation tangent alignment")
    fig.tight_layout()
    _save_all(fig, out_stem)


def plot_a5_occupancy(rows: list[dict], out_stem: Path) -> None:
    if not rows:
        return
    labels = [r["comparison"] for r in rows]
    overlaps = [r["subspace_overlap"] for r in rows]
    pr_delta = [r["abs_pr_delta"] for r in rows]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].bar(np.arange(len(labels)), overlaps, color="#72b7b2")
    axs[0].set_xticks(np.arange(len(labels)))
    axs[0].set_xticklabels(labels, rotation=30, ha="right")
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_title("Real-vs-control overlap")

    axs[1].bar(np.arange(len(labels)), pr_delta, color="#b279a2")
    axs[1].set_xticks(np.arange(len(labels)))
    axs[1].set_xticklabels(labels, rotation=30, ha="right")
    axs[1].set_title("|PR real - PR control|")

    fig.suptitle("A5 Occupancy vs dynamics")
    fig.tight_layout()
    _save_all(fig, out_stem)


def plot_a6_single_unit(rows: list[dict], curve: np.ndarray, out_stem: Path) -> None:
    if not rows:
        return
    gain = np.array([r["gain_mag"] for r in rows], dtype=np.float64)
    diag = np.array([r["diag_cfem"] for r in rows], dtype=np.float64)
    u1 = np.array([abs(r["u1_loading"]) for r in rows], dtype=np.float64)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].scatter(gain, diag, s=10, alpha=0.6)
    axs[0].set_xlabel("unit translation gain")
    axs[0].set_ylabel("diag(C_FEM)")
    axs[0].set_title("Gain vs unit FEM variance")

    axs[1].scatter(gain, u1, s=10, alpha=0.6)
    axs[1].set_xlabel("unit translation gain")
    axs[1].set_ylabel("|u1 loading|")
    axs[1].set_title("Gain vs top eigen loading")

    axs[2].plot(np.arange(1, len(curve) + 1), curve)
    axs[2].set_ylim(0.0, 1.0)
    axs[2].set_xlabel("units sorted by gain")
    axs[2].set_ylabel("cumulative fraction")
    axs[2].set_title("Cumulative diag(C_FEM)")

    fig.suptitle("A6 Single-unit to population bridge")
    fig.tight_layout()
    _save_all(fig, out_stem)

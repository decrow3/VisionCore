from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry_utils import (
    ORIENTATIONS,
    compute_signal_covariance,
    compute_translation_mimicry,
    compute_alpha,
    dump_json,
    find_jacobian_bundle,
    format_logmar,
    load_eoptotype_jacobian,
    load_eoptotype_trial_means,
    mean_offdiag,
    maybe_git_commit,
    ordered_matrix_from_rows,
    orthonormal_basis,
    symmetrize_ordered_matrix,
    zscore_trial_means_and_jacobians,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATES_DIR = REPO_ROOT / "scripts" / "temporal_decoding" / "data" / "rates"
DEFAULT_JACOBIAN_DIR = REPO_ROOT / "declan" / "jacobian_results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "declan" / "results" / "translation_mimicry"


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(x)) for x in str(text).split(",") if str(x).strip())


def _parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _safe_logmar_tag(logmar: float) -> str:
    return format_logmar(logmar).replace("-", "m").replace(".", "p")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _class_means(trial_means_by_ori: dict[int, np.ndarray], orientations: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {ori: np.asarray(trial_means_by_ori[ori], dtype=np.float64).mean(axis=0) for ori in orientations}


def _compute_condition_rows(
    *,
    logmar: float,
    condition: str,
    normalization: str,
    trial_means_by_ori: dict[int, np.ndarray],
    jacobians_by_ori: dict[int, np.ndarray],
    orientations: tuple[int, ...],
    ridge_scale: float,
    arcmin_limits: tuple[float, ...],
    svd_eps: float,
    n_constrained_angles: int,
) -> tuple[list[dict], dict[str, float]]:
    class_means = _class_means(trial_means_by_ori, orientations)
    signal_cov = compute_signal_covariance(np.stack([class_means[ori] for ori in orientations], axis=0))
    pooled_J = np.concatenate([jacobians_by_ori[ori] for ori in orientations], axis=1)
    pooled_U, _ = orthonormal_basis(pooled_J, svd_eps=svd_eps)

    alpha_by_ori = []
    for ori in orientations:
        U, _ = orthonormal_basis(jacobians_by_ori[ori], svd_eps=svd_eps)
        alpha_by_ori.append(compute_alpha(U, signal_cov))

    rows: list[dict] = []
    for a in orientations:
        for b in orientations:
            if a == b:
                continue
            metrics = compute_translation_mimicry(
                class_means[a],
                class_means[b],
                jacobians_by_ori[a],
                ridge_scale=ridge_scale,
                arcmin_limits=arcmin_limits,
                svd_eps=svd_eps,
                n_constrained_angles=n_constrained_angles,
            )
            row = {
                "logmar": float(logmar),
                "condition": condition,
                "normalization": normalization,
                "orientation_a": int(a),
                "orientation_b": int(b),
                **metrics,
            }
            rows.append(row)

    summary = {
        "alpha_pooled": compute_alpha(pooled_U, signal_cov),
        "alpha_orientation_mean": float(np.mean(alpha_by_ori)),
        "signal_trace": float(np.trace(signal_cov)),
        "mean_pairwise_separation": float(
            np.mean(
                [
                    np.linalg.norm(class_means[b] - class_means[a])
                    for a in orientations
                    for b in orientations
                    if a != b
                ]
            )
        ),
    }
    return rows, summary


def _plot_mimicry_vs_logmar(
    output_dir: Path,
    summaries: list[dict],
    conditions: list[str],
    plateau_start: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#1b5e20", "#1565c0", "#c62828", "#6a1b9a"]
    for idx, condition in enumerate(conditions):
        points = [row for row in summaries if row["condition"] == condition and row["normalization"] == "raw"]
        points.sort(key=lambda row: row["logmar"])
        if not points:
            continue
        x = np.array([row["logmar"] for row in points])
        y = np.array([row["mean_mimicry_unconstrained"] for row in points])
        y_c = np.array([row["mean_mimicry_constrained_1p0_arcmin"] for row in points])
        color = colors[idx % len(colors)]
        ax.plot(x, y, marker="o", color=color, label=f"{condition} projection")
        ax.plot(x, y_c, marker="s", linestyle="--", color=color, alpha=0.7, label=f"{condition} 1 arcmin")
    ax.axvspan(plateau_start, max([row["logmar"] for row in summaries]) + 1e-6, color="#d0d0d0", alpha=0.25)
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("Mean pairwise mimicry")
    ax.set_title("Translation mimicry vs LogMAR")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "translation_mimicry_vs_logmar.png", dpi=200)
    plt.close(fig)


def _plot_matrices(
    output_dir: Path,
    matrices: dict[tuple[str, str, float], np.ndarray],
    logmars: list[float],
    conditions: list[str],
    orientations: tuple[int, ...],
) -> None:
    vmin = min(np.nanmin(mat) for key, mat in matrices.items() if key[1] == "raw")
    vmax = max(np.nanmax(mat) for key, mat in matrices.items() if key[1] == "raw")
    for condition in conditions:
        fig, axes = plt.subplots(
            1,
            len(logmars),
            figsize=(4.0 * len(logmars), 3.6),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, logmar in zip(axes[0], logmars):
            mat = matrices[(condition, "raw", logmar)]
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{condition} lm={logmar:+.2f}")
            ax.set_xticks(range(len(orientations)), orientations)
            ax.set_yticks(range(len(orientations)), orientations)
            ax.set_xlabel("target b")
            ax.set_ylabel("source a")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="mimicry")
        fig.savefig(output_dir / f"translation_mimicry_matrix_{condition}.png", dpi=200)
        plt.close(fig)


def _plot_translation_vectors(
    output_dir: Path,
    rows: list[dict],
    logmars: list[float],
    conditions: list[str],
) -> None:
    for condition in conditions:
        for logmar in logmars:
            subset = [r for r in rows if r["condition"] == condition and r["normalization"] == "raw" and r["logmar"] == logmar]
            if not subset:
                continue
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            colors = {0: "#1b9e77", 90: "#d95f02", 180: "#7570b3", 270: "#e7298a"}
            for row in subset:
                ax.arrow(
                    0.0,
                    0.0,
                    row["dx_star_deg"] * 60.0,
                    row["dy_star_deg"] * 60.0,
                    length_includes_head=True,
                    head_width=0.08,
                    head_length=0.10,
                    alpha=0.75,
                    color=colors[int(row["orientation_b"])],
                )
            for radius, ls in [(1.0, "--"), (2.0, ":")]:
                circ = plt.Circle((0.0, 0.0), radius, fill=False, linestyle=ls, color="#777777")
                ax.add_patch(circ)
            ax.axhline(0.0, color="#cccccc", linewidth=1.0)
            ax.axvline(0.0, color="#cccccc", linewidth=1.0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("dx* (arcmin)")
            ax.set_ylabel("dy* (arcmin)")
            ax.set_title(f"Optimal translations: {condition} lm={logmar:+.2f}")
            lim = max(2.2, max(abs(r["dx_star_deg"] * 60.0) for r in subset), max(abs(r["dy_star_deg"] * 60.0) for r in subset))
            ax.set_xlim(-1.1 * lim, 1.1 * lim)
            ax.set_ylim(-1.1 * lim, 1.1 * lim)
            fig.tight_layout()
            fig.savefig(output_dir / f"translation_vectors_{condition}_lm{_safe_logmar_tag(logmar)}.png", dpi=200)
            plt.close(fig)


def _plot_mimicry_vs_separation(
    output_dir: Path,
    summaries: list[dict],
    conditions: list[str],
) -> None:
    if len(conditions) < 2:
        return
    raw = [row for row in summaries if row["normalization"] == "raw"]
    by_key = {(row["condition"], row["logmar"]): row for row in raw}
    shared_logmars = sorted(set(row["logmar"] for row in raw if row["condition"] == conditions[0]) & set(row["logmar"] for row in raw if row["condition"] == conditions[1]))
    if not shared_logmars:
        return
    x = []
    y = []
    labels = []
    for logmar in shared_logmars:
        first = by_key[(conditions[0], logmar)]
        second = by_key[(conditions[1], logmar)]
        x.append(first["mean_mimicry_unconstrained"])
        y.append(first["mean_pairwise_separation"] - second["mean_pairwise_separation"])
        labels.append(f"{logmar:+.2f}")
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(x, y, color="#1565c0")
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi, yi, label, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(f"Mean mimicry ({conditions[0]})")
    ax.set_ylabel(f"Mean pairwise separation delta: {conditions[0]} - {conditions[1]}")
    ax.set_title("Mimicry predicts class-separation change")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "translation_mimicry_vs_separation.png", dpi=200)
    plt.close(fig)


def _build_pairwise_summary(rows: list[dict], orientations: tuple[int, ...]) -> list[dict]:
    grouped: dict[tuple[str, str, float], dict[tuple[int, int], float]] = {}
    for row in rows:
        key = (row["condition"], row["normalization"], float(row["logmar"]))
        grouped.setdefault(key, {})[(int(row["orientation_a"]), int(row["orientation_b"]))] = float(row["mimicry_unconstrained"])

    out: list[dict] = []
    for (condition, normalization, logmar), values in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        for i, a in enumerate(orientations):
            for b in orientations[i + 1 :]:
                sym_mean = 0.5 * (values[(a, b)] + values[(b, a)])
                sym_max = max(values[(a, b)], values[(b, a)])
                sym_min = min(values[(a, b)], values[(b, a)])
                out.append(
                    {
                        "logmar": logmar,
                        "condition": condition,
                        "normalization": normalization,
                        "pair": f"{a}-{b}",
                        "pair_label": f"{a}\u2194{b}",
                        "mimicry_sym_mean": sym_mean,
                        "mimicry_sym_max": sym_max,
                        "mimicry_sym_min": sym_min,
                        "mimicry_ab": values[(a, b)],
                        "mimicry_ba": values[(b, a)],
                    }
                )
    return out


def _plot_pairwise_lines(output_dir: Path, pair_rows: list[dict], conditions: list[str]) -> None:
    pair_order = ["0-180", "0-90", "0-270", "90-180", "90-270", "180-270"]
    colors = {
        "0-180": "#d95f02",
        "0-90": "#1b9e77",
        "0-270": "#7570b3",
        "90-180": "#e7298a",
        "90-270": "#66a61e",
        "180-270": "#e6ab02",
    }
    for normalization in ["raw", "zscore"]:
        fig, axes = plt.subplots(1, len(conditions), figsize=(5.0 * len(conditions), 4.2), squeeze=False, constrained_layout=True)
        for ax, condition in zip(axes[0], conditions):
            subset = [row for row in pair_rows if row["condition"] == condition and row["normalization"] == normalization]
            for pair in pair_order:
                pts = [row for row in subset if row["pair"] == pair]
                pts.sort(key=lambda row: row["logmar"])
                if not pts:
                    continue
                x = np.array([row["logmar"] for row in pts], dtype=np.float64)
                y = np.array([row["mimicry_sym_mean"] for row in pts], dtype=np.float64)
                ax.plot(x, y, marker="o", color=colors[pair], label=pts[0]["pair_label"])
            ax.set_title(f"{condition} ({normalization})")
            ax.set_xlabel("LogMAR")
            ax.set_ylabel("Symmetrized pairwise mimicry")
            ax.grid(alpha=0.2)
        axes[0, -1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.savefig(output_dir / f"translation_mimicry_pairwise_{normalization}.png", dpi=220)
        plt.close(fig)


def _plot_raw_vs_zscore(output_dir: Path, summaries: list[dict], conditions: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    colors = {"real": "#1565c0", "stabilized": "#1b5e20"}

    for condition in conditions:
        raw = sorted([row for row in summaries if row["condition"] == condition and row["normalization"] == "raw"], key=lambda row: row["logmar"])
        zscore = sorted([row for row in summaries if row["condition"] == condition and row["normalization"] == "zscore"], key=lambda row: row["logmar"])
        if not raw or not zscore:
            continue
        x = np.array([row["logmar"] for row in raw], dtype=np.float64)
        y_raw = np.array([row["mean_mimicry_unconstrained"] for row in raw], dtype=np.float64)
        y_z = np.array([row["mean_mimicry_unconstrained"] for row in zscore], dtype=np.float64)
        axes[0].plot(x, y_raw, marker="o", color=colors[condition], label=f"{condition} raw")
        axes[0].plot(x, y_z, marker="s", linestyle="--", color=colors[condition], alpha=0.8, label=f"{condition} z-score")
        axes[1].scatter(y_raw, y_z, color=colors[condition], label=condition)
        for row_raw, row_z in zip(raw, zscore):
            axes[1].text(row_raw["mean_mimicry_unconstrained"], row_z["mean_mimicry_unconstrained"], f"{row_raw['logmar']:+.2f}", fontsize=7)

    axes[0].set_title("Raw vs normalized mean mimicry")
    axes[0].set_xlabel("LogMAR")
    axes[0].set_ylabel("Mean pairwise mimicry")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].plot([0, 1], [0, 1], transform=axes[1].transAxes, color="#bbbbbb", linestyle=":")
    axes[1].set_title("Raw vs normalized sensitivity")
    axes[1].set_xlabel("Raw mean mimicry")
    axes[1].set_ylabel("Z-scored mean mimicry")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(output_dir / "translation_mimicry_raw_vs_zscore.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Translation mimicry analysis from cached E-optotype rates and Jacobians")
    parser.add_argument("--logmars", type=str, default="-0.20,-0.25,-0.30,-0.35")
    parser.add_argument("--conditions", type=str, default="real,stabilized")
    parser.add_argument("--orientations", type=str, default="0,90,180,270")
    parser.add_argument("--rates_dir", type=Path, default=DEFAULT_RATES_DIR)
    parser.add_argument("--jacobian_dir", type=Path, default=DEFAULT_JACOBIAN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jacobian_kind", choices=("int", "eff", "point"), default="int")
    parser.add_argument("--ridge_scale", type=float, default=1e-6)
    parser.add_argument("--arcmin_limits", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--svd_eps", type=float, default=1e-9)
    parser.add_argument("--n_constrained_angles", type=int, default=720)
    parser.add_argument("--plateau_start", type=float, default=-0.40)
    args = parser.parse_args()

    logmars = _parse_csv_floats(args.logmars)
    conditions = _parse_csv_strings(args.conditions)
    orientations = _parse_csv_ints(args.orientations)
    arcmin_limits = tuple(_parse_csv_floats(args.arcmin_limits))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    summary_rows: list[dict] = []
    matrices: dict[tuple[str, str, float], np.ndarray] = {}
    npz_payload: dict[str, np.ndarray] = {
        "logmars": np.asarray(logmars, dtype=np.float64),
        "orientations": np.asarray(orientations, dtype=np.int64),
        "arcmin_limits": np.asarray(arcmin_limits, dtype=np.float64),
        "conditions": np.asarray(conditions, dtype=object),
    }

    n_cond = len(conditions)
    n_log = len(logmars)
    n_ori = len(orientations)
    mimicry_raw = np.full((n_log, n_cond, n_ori, n_ori), np.nan, dtype=np.float64)
    mimicry_z = np.full_like(mimicry_raw, np.nan)
    residual_raw = np.full_like(mimicry_raw, np.nan)
    residual_z = np.full_like(mimicry_raw, np.nan)
    dx_raw = np.full_like(mimicry_raw, np.nan)
    dy_raw = np.full_like(mimicry_raw, np.nan)
    mag_raw = np.full_like(mimicry_raw, np.nan)
    cond_gap_raw = np.full_like(mimicry_raw, np.nan)
    mimicry_constrained_raw = np.full((n_log, n_cond, len(arcmin_limits), n_ori, n_ori), np.nan, dtype=np.float64)
    identity_norm_raw = np.full_like(mimicry_raw, np.nan)
    alpha_pooled_raw = np.full((n_log, n_cond), np.nan, dtype=np.float64)
    alpha_orientation_mean_raw = np.full((n_log, n_cond), np.nan, dtype=np.float64)
    signal_trace_raw = np.full((n_log, n_cond), np.nan, dtype=np.float64)
    separation_raw = np.full((n_log, n_cond), np.nan, dtype=np.float64)

    for li, logmar in enumerate(logmars):
        jacobians_raw, jacobian_path = load_eoptotype_jacobian(logmar, args.jacobian_dir, jacobian_kind=args.jacobian_kind, orientations=orientations)
        for ci, condition in enumerate(conditions):
            trial_means_raw = load_eoptotype_trial_means(logmar, condition, args.rates_dir, orientations=orientations)
            trial_means_z, jacobians_z, _, _ = zscore_trial_means_and_jacobians(trial_means_raw, jacobians_raw)

            for normalization, trial_means_by_ori, jacobians_by_ori in [
                ("raw", trial_means_raw, jacobians_raw),
                ("zscore", trial_means_z, jacobians_z),
            ]:
                condition_rows, summary = _compute_condition_rows(
                    logmar=logmar,
                    condition=condition,
                    normalization=normalization,
                    trial_means_by_ori=trial_means_by_ori,
                    jacobians_by_ori=jacobians_by_ori,
                    orientations=orientations,
                    ridge_scale=args.ridge_scale,
                    arcmin_limits=arcmin_limits,
                    svd_eps=args.svd_eps,
                    n_constrained_angles=args.n_constrained_angles,
                )
                for row in condition_rows:
                    row["jacobian_path"] = jacobian_path.name
                rows.extend(condition_rows)
                mat = ordered_matrix_from_rows(condition_rows, "mimicry_unconstrained", orientations)
                matrices[(condition, normalization, logmar)] = mat

                summary_row = {
                    "logmar": float(logmar),
                    "condition": condition,
                    "normalization": normalization,
                    "mean_mimicry_unconstrained": mean_offdiag(mat),
                    "mean_residual_unconstrained": mean_offdiag(ordered_matrix_from_rows(condition_rows, "residual_unconstrained", orientations)),
                    "mean_mimicry_constrained_1p0_arcmin": mean_offdiag(ordered_matrix_from_rows(condition_rows, "mimicry_constrained_1p0_arcmin", orientations)),
                    "mean_projection_ls_gap": mean_offdiag(ordered_matrix_from_rows(condition_rows, "projection_ls_gap", orientations)),
                    **summary,
                    "jacobian_path": jacobian_path.name,
                }
                summary_rows.append(summary_row)

                if normalization == "raw":
                    mimicry_raw[li, ci] = mat
                    residual_raw[li, ci] = ordered_matrix_from_rows(condition_rows, "residual_unconstrained", orientations)
                    dx_raw[li, ci] = ordered_matrix_from_rows(condition_rows, "dx_star_deg", orientations)
                    dy_raw[li, ci] = ordered_matrix_from_rows(condition_rows, "dy_star_deg", orientations)
                    mag_raw[li, ci] = ordered_matrix_from_rows(condition_rows, "translation_mag_deg", orientations)
                    cond_gap_raw[li, ci] = ordered_matrix_from_rows(condition_rows, "jacobian_condition_number", orientations)
                    identity_norm_raw[li, ci] = ordered_matrix_from_rows(condition_rows, "identity_norm", orientations)
                    alpha_pooled_raw[li, ci] = summary["alpha_pooled"]
                    alpha_orientation_mean_raw[li, ci] = summary["alpha_orientation_mean"]
                    signal_trace_raw[li, ci] = summary["signal_trace"]
                    separation_raw[li, ci] = summary["mean_pairwise_separation"]
                    for ai, limit in enumerate(arcmin_limits):
                        key = f"mimicry_constrained_{str(limit).replace('.', 'p')}_arcmin"
                        mimicry_constrained_raw[li, ci, ai] = ordered_matrix_from_rows(condition_rows, key, orientations)
                else:
                    mimicry_z[li, ci] = mat
                    residual_z[li, ci] = ordered_matrix_from_rows(condition_rows, "residual_unconstrained", orientations)

    npz_payload.update(
        {
            "mimicry_raw": mimicry_raw,
            "mimicry_zscore": mimicry_z,
            "residual_raw": residual_raw,
            "residual_zscore": residual_z,
            "dx_star_deg_raw": dx_raw,
            "dy_star_deg_raw": dy_raw,
            "translation_mag_deg_raw": mag_raw,
            "mimicry_constrained_raw": mimicry_constrained_raw,
            "identity_norm_raw": identity_norm_raw,
            "jacobian_condition_number_raw": cond_gap_raw,
            "alpha_pooled_raw": alpha_pooled_raw,
            "alpha_orientation_mean_raw": alpha_orientation_mean_raw,
            "signal_trace_raw": signal_trace_raw,
            "mean_pairwise_separation_raw": separation_raw,
        }
    )

    _write_csv(args.output_dir / "translation_mimicry_summary.csv", rows)
    _write_csv(args.output_dir / "translation_mimicry_overview.csv", summary_rows)
    pairwise_rows = _build_pairwise_summary(rows, orientations)
    _write_csv(args.output_dir / "translation_mimicry_pairwise_summary.csv", pairwise_rows)
    np.savez_compressed(args.output_dir / "translation_mimicry_by_logmar.npz", **npz_payload)

    config = {
        "script": "declan/translation_mimicry.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": maybe_git_commit(REPO_ROOT),
        "rates_dir": str(args.rates_dir),
        "jacobian_dir": str(args.jacobian_dir),
        "output_dir": str(args.output_dir),
        "logmars": logmars,
        "conditions": conditions,
        "orientations": list(orientations),
        "jacobian_kind": args.jacobian_kind,
        "ridge_scale": args.ridge_scale,
        "arcmin_limits": list(arcmin_limits),
        "svd_eps": args.svd_eps,
        "n_constrained_angles": args.n_constrained_angles,
        "plateau_start": args.plateau_start,
        "model_checkpoint": "learned_resnet_none_convgru_gaussian epoch 147",
        "retina_ppd": 37.50476617,
        "world_ppd": 120.0,
        "normalizations": ["raw", "zscore"],
    }
    dump_json(args.output_dir / "translation_mimicry_config.json", config)

    _plot_mimicry_vs_logmar(args.output_dir, summary_rows, conditions, args.plateau_start)
    _plot_matrices(args.output_dir, matrices, logmars, conditions, orientations)
    _plot_translation_vectors(args.output_dir, rows, logmars, conditions)
    _plot_mimicry_vs_separation(args.output_dir, summary_rows, conditions)
    _plot_pairwise_lines(args.output_dir, pairwise_rows, conditions)
    _plot_raw_vs_zscore(args.output_dir, summary_rows, conditions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_DIR = ROOT / "scripts" / "temporal_decoding"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEMPORAL_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_DIR))

from geometry_utils import (  # noqa: E402
    ellipse_from_covariance,
    format_logmar,
    load_eoptotype_jacobian,
    load_eoptotype_trial_means,
    maybe_git_commit,
    orthonormal_basis,
)
from scripts.temporal_decoding.stimulus_hires import hires_counterfactual_stim, RETINA_PPD  # noqa: E402


DEFAULT_RATES_DIR = ROOT / "scripts" / "temporal_decoding" / "data" / "rates"
DEFAULT_JACOBIAN_DIR = ROOT / "declan" / "jacobian_results"
DEFAULT_MIMICRY_DIR = ROOT / "declan" / "results" / "translation_mimicry_primary"
DEFAULT_OUTPUT_DIR = ROOT / "declan" / "results" / "jacobian_identity_geometry"
EPS = 1e-12


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _rate_cache_path(rates_dir: Path, logmar: float, orientation: int, condition: str) -> Path:
    lm = format_logmar(logmar)
    hires = rates_dir / f"rates_hires_lm{lm}_ori{orientation}_{condition}.npz"
    lores = rates_dir / f"rates_lm{lm}_ori{orientation}_{condition}.npz"
    if hires.exists():
        return hires
    if lores.exists():
        return lores
    raise FileNotFoundError(f"Missing cached rates for lm={lm} ori={orientation} cond={condition}")


def _load_trial_rate_matrix(rates_dir: Path, logmar: float, orientation: int, condition: str, trial_idx: int) -> np.ndarray:
    data = np.load(_rate_cache_path(rates_dir, logmar, orientation, condition), allow_pickle=True)
    rates = np.asarray(data["rates"], dtype=np.float64)
    lengths = np.asarray(data["lengths"], dtype=np.int64)
    return rates[trial_idx, : lengths[trial_idx]]


def _load_test3_bundle(jacobian_dir: Path, logmar: float) -> np.lib.npyio.NpzFile:
    path = jacobian_dir / f"test3_lm{format_logmar(logmar)}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _load_mimicry_overview(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _panel_a_images(ax, logmar: float, orientation: int, epsilon_deg: float) -> None:
    zero_trace = np.zeros((1, 2), dtype=np.float32)
    offsets = [(0.0, 0.0), (epsilon_deg, 0.0), (0.0, epsilon_deg)]
    labels = ["center", "+x", "+y"]
    images = []
    for dx, dy in offsets:
        stim = hires_counterfactual_stim(
            float(orientation),
            float(logmar),
            zero_trace,
            condition="real",
            center_offset_deg=(float(dx), float(dy)),
            device="cpu",
        )
        images.append(np.asarray(stim[0, 0, -1], dtype=np.float32))
    strip = np.concatenate(images, axis=1)
    ax.imshow(strip, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("A  Retinal image translations", loc="left", fontweight="bold")
    ax.axis("off")
    w = images[0].shape[1]
    for idx, label in enumerate(labels):
        ax.text((idx + 0.5) * w, 6, label, color="#ffffff", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(0.02, 0.96, r"$\partial r/\partial x$ and $\partial r/\partial y$ probe tiny translations", transform=ax.transAxes, color="#f5f5f5", va="top", fontsize=8)


def _panel_b_tangent_plane(ax, mu: np.ndarray, J: np.ndarray) -> None:
    U, _ = orthonormal_basis(J)
    if U.shape[1] == 0:
        raise ValueError("Jacobian basis is empty")
    coords = U.T @ J
    delta_grid = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    ) / RETINA_PPD
    cloud = np.stack([U.T @ (mu + J @ d - mu) for d in delta_grid], axis=0)
    ax.scatter(cloud[:, 0], cloud[:, 1], color="#90caf9", s=28, edgecolor="#1565c0", linewidth=0.4)
    ax.arrow(0.0, 0.0, coords[0, 0], coords[1, 0], color="#d95f02", width=0.0, head_width=0.004, length_includes_head=True)
    ax.arrow(0.0, 0.0, coords[0, 1], coords[1, 1], color="#1b9e77", width=0.0, head_width=0.004, length_includes_head=True)
    ax.text(coords[0, 0], coords[1, 0], r"$J_x$", color="#d95f02", fontsize=9)
    ax.text(coords[0, 1], coords[1, 1], r"$J_y$", color="#1b9e77", fontsize=9)
    ax.scatter([0.0], [0.0], color="#111111", s=25)
    ax.set_title("B  Translation tangent plane", loc="left", fontweight="bold")
    ax.set_xlabel("U_J coord 1")
    ax.set_ylabel("U_J coord 2")
    ax.axhline(0.0, color="#dddddd", lw=1.0)
    ax.axvline(0.0, color="#dddddd", lw=1.0)
    ax.set_aspect("equal", adjustable="box")


def _panel_c_trajectory(ax, trial_rates: np.ndarray, mu: np.ndarray, J: np.ndarray) -> None:
    U, _ = orthonormal_basis(J)
    centered = trial_rates - mu[None, :]
    coords = centered @ U
    t = np.linspace(0.0, 1.0, coords.shape[0])
    for i in range(coords.shape[0] - 1):
        ax.plot(coords[i : i + 2, 0], coords[i : i + 2, 1], color=plt.cm.viridis(t[i]), lw=1.6)
    ax.scatter(coords[0, 0], coords[0, 1], color="#111111", s=18, label="start")
    ax.scatter(coords[-1, 0], coords[-1, 1], color="#c62828", s=18, label="end")
    ax.set_title("C  FEM trajectory in J coordinates", loc="left", fontweight="bold")
    ax.set_xlabel("U_J coord 1")
    ax.set_ylabel("U_J coord 2")
    ax.legend(frameon=False, fontsize=7)
    ax.set_aspect("equal", adjustable="box")


def _panel_d_covariance(ax, C_fem: np.ndarray, J: np.ndarray, sigma_trial: np.ndarray) -> None:
    U, _ = orthonormal_basis(J)
    C_pred = J @ sigma_trial @ J.T
    C_emp_2d = U.T @ C_fem @ U
    C_pred_2d = U.T @ C_pred @ U
    emp = ellipse_from_covariance(C_emp_2d)
    pred = ellipse_from_covariance(C_pred_2d)
    ax.plot(emp[:, 0], emp[:, 1], color="#1565c0", lw=2.0, label="empirical C_FEM")
    ax.plot(pred[:, 0], pred[:, 1], color="#d95f02", lw=2.0, linestyle="--", label=r"$J\Sigma_{eye}J^T$")
    ax.fill(emp[:, 0], emp[:, 1], color="#1565c0", alpha=0.08)
    ax.fill(pred[:, 0], pred[:, 1], color="#d95f02", alpha=0.08)
    ax.set_title("D  Empirical vs predicted covariance", loc="left", fontweight="bold")
    ax.set_xlabel("U_J coord 1")
    ax.set_ylabel("U_J coord 2")
    ax.legend(frameon=False, fontsize=7)
    ax.set_aspect("equal", adjustable="box")


def _panel_e_bridge(ax) -> None:
    ax.axis("off")
    ax.set_facecolor("#f7f4ef")
    text = (
        "E  Empirical bridge\n\n"
        "No matched real-data / model-neuron mapping is\n"
        "available for the E-optotype Jacobian analysis.\n\n"
        "Use the existing real-data covariance decomposition\n"
        "as the empirical motivation panel, rather than an\n"
        "ill-posed projection into the model J-subspace."
    )
    ax.text(0.03, 0.97, text, va="top", ha="left", fontsize=9.5, fontweight="bold")


def _panel_f_mimicry(ax, rows: list[dict], conditions: list[str], plateau_start: float) -> None:
    colors = {"real": "#1565c0", "stabilized": "#1b5e20"}
    for condition in conditions:
        pts = [r for r in rows if r["condition"] == condition and r["normalization"] == "raw"]
        pts.sort(key=lambda r: float(r["logmar"]))
        x = np.array([float(r["logmar"]) for r in pts], dtype=np.float64)
        y = np.array([float(r["mean_mimicry_unconstrained"]) for r in pts], dtype=np.float64)
        ax.plot(x, y, marker="o", color=colors.get(condition, "#444444"), label=condition)
    ax.axvspan(plateau_start, x.max() + 1e-6, color="#d0d0d0", alpha=0.25)
    ax.set_title("F  Translation mimicry across LogMAR", loc="left", fontweight="bold")
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("Mean pairwise mimicry")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Central Jacobian identity-geometry figure")
    parser.add_argument("--logmar", type=float, default=-0.20)
    parser.add_argument("--orientation", type=int, default=0)
    parser.add_argument("--trial_idx", type=int, default=0)
    parser.add_argument("--condition", type=str, default="real")
    parser.add_argument("--jacobian_kind", choices=("int", "eff", "point"), default="int")
    parser.add_argument("--rates_dir", type=Path, default=DEFAULT_RATES_DIR)
    parser.add_argument("--jacobian_dir", type=Path, default=DEFAULT_JACOBIAN_DIR)
    parser.add_argument("--mimicry_dir", type=Path, default=DEFAULT_MIMICRY_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon_arcmin", type=float, default=1.0)
    parser.add_argument("--plateau_start", type=float, default=-0.40)
    args = parser.parse_args()

    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epsilon_deg = args.epsilon_arcmin / 60.0

    jacobians, jacobian_path = load_eoptotype_jacobian(args.logmar, args.jacobian_dir, jacobian_kind=args.jacobian_kind)
    J = jacobians[int(args.orientation)]
    trial_means = load_eoptotype_trial_means(args.logmar, args.condition, args.rates_dir)
    mu = np.asarray(trial_means[int(args.orientation)].mean(axis=0), dtype=np.float64)
    bundle = _load_test3_bundle(args.jacobian_dir, args.logmar)
    C_fem = np.asarray(bundle[f"C_FEM_ori{int(args.orientation)}"], dtype=np.float64)
    sigma_trial = np.asarray(bundle["sigma_trial"], dtype=np.float64)
    trial_rates = _load_trial_rate_matrix(args.rates_dir, args.logmar, args.orientation, args.condition, args.trial_idx)
    mimicry_rows = _load_mimicry_overview(args.mimicry_dir / "translation_mimicry_overview.csv")

    fig = plt.figure(figsize=(13.8, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    _panel_a_images(axA, args.logmar, args.orientation, epsilon_deg)
    _panel_b_tangent_plane(axB, mu, J)
    _panel_c_trajectory(axC, trial_rates, mu, J)
    _panel_d_covariance(axD, C_fem, J, sigma_trial)
    _panel_e_bridge(axE)
    _panel_f_mimicry(axF, mimicry_rows, ["real", "stabilized"], args.plateau_start)

    title = (
        "Jacobian Identity/Transformation Geometry\n"
        f"lm={args.logmar:+.2f}  ori={args.orientation}  cond={args.condition}  J={args.jacobian_kind}"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    png_path = args.output_dir / "jacobian_identity_geometry.png"
    pdf_path = args.output_dir / "jacobian_identity_geometry.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    config = {
        "script": "declan/figure_jacobian_identity_geometry.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": maybe_git_commit(ROOT),
        "logmar": args.logmar,
        "orientation": args.orientation,
        "trial_idx": args.trial_idx,
        "condition": args.condition,
        "jacobian_kind": args.jacobian_kind,
        "jacobian_bundle": jacobian_path.name,
        "epsilon_arcmin": args.epsilon_arcmin,
        "mimicry_overview": str(args.mimicry_dir / "translation_mimicry_overview.csv"),
    }
    (args.output_dir / "jacobian_identity_geometry_config.json").write_text(
        __import__("json").dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
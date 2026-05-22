"""Integration-time diagnostics with alternative feature constructions.

This script is meant to answer a narrow question:

  Does accuracy improve with integration window W under FEM when we use an
  accumulation-aligned representation (time-mean of the last W frames), rather
  than flatten(W×N)+PCA?

It intentionally loads cached rate matrices directly from .npz files to avoid
importing the full simulation stack (which may require external deps).

Usage examples:
  /home/declan/VisionCore/.venv/bin/python scripts/temporal_decoding/integration_time_controls.py \
    --logmar -0.20 --prefix rates_hires_lm --windows 1,6,24,60

  /home/declan/VisionCore/.venv/bin/python scripts/temporal_decoding/integration_time_controls.py \
    --logmar 0.40 --prefix rates_lm --windows 1,6,24,60
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_WINDOWS = (1, 3, 6, 12, 24, 36, 48, 60)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_CONDITIONS = ("real", "stabilized")


def _format_logmar_for_filename(logmar: float) -> str:
    # Filenames look like: rates_lm0.40_... or rates_hires_lm-0.20_...
    # i.e. no leading '+' for positive values.
    return f"{logmar:.2f}"


def _load_rates_npz(path: Path) -> list[np.ndarray]:
    """Load rates saved by scripts/temporal_decoding/rate_computation.save_rates()."""
    d = np.load(path, allow_pickle=True)
    rates_padded = d["rates"]  # (M, T_max, N) float32, NaN padded
    lengths = d["lengths"].astype(int)  # (M,)
    return [rates_padded[i, : lengths[i]] for i in range(rates_padded.shape[0])]


def _window_last_frames(r: np.ndarray, window: int) -> np.ndarray:
    """Take the last `window` frames, padding by repeating the first frame."""
    if window <= 0:
        raise ValueError("window must be positive")
    T = int(r.shape[0])
    if T >= window:
        return r[-window:]
    pad = np.repeat(r[[0]], window - T, axis=0)
    return np.concatenate([pad, r], axis=0)


def build_rates_by_condition(
    rates_dir: Path,
    prefix: str,
    logmar: float,
    orientations: Iterable[int] = DEFAULT_ORIENTATIONS,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Load cached rate lists into the format expected by the decoding scripts."""
    rates_by_condition: dict[str, dict[str, list[np.ndarray]]] = {}
    lm_str = _format_logmar_for_filename(float(logmar))

    for cond in conditions:
        by_stim: dict[str, list[np.ndarray]] = {}
        for ori in orientations:
            fname = f"{prefix}{lm_str}_ori{int(ori)}_{cond}.npz"
            path = rates_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing cached rates: {path}")
            by_stim[f"ori{int(ori)}"] = _load_rates_npz(path)
        rates_by_condition[cond] = by_stim

    return rates_by_condition


def decode_causal_window_mean(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
    n_splits: int = 5,
    C_logistic: float = 1.0,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Grouped-CV decoding using only the time-mean over the last W frames."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)
    if K < 2:
        raise ValueError("Need at least 2 classes")

    # Equalize number of trials across stimuli
    M_min = min(len(rates_by_stim[sid]) for sid in stim_ids)
    X_list, y_list = [], []
    for label, sid in enumerate(stim_ids):
        feats = []
        for r in rates_by_stim[sid][:M_min]:
            r_win = _window_last_frames(np.asarray(r), window)
            feats.append(r_win.mean(axis=0))  # (N,)
        X_list.append(np.stack(feats, axis=0))
        y_list.append(np.full(M_min, label, dtype=int))

    X_all = np.concatenate(X_list, axis=0)  # (K*M_min, N)
    y_all = np.concatenate(y_list, axis=0)
    groups_all = np.tile(np.arange(M_min, dtype=int), K)

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []

    for train_idx, test_idx in gkf.split(X_all, y_all, groups=groups_all):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])

        clf = LogisticRegression(
            C=C_logistic,
            max_iter=2000,
            solver="lbfgs",
            random_state=rng_seed,
        )
        clf.fit(X_tr, y_all[train_idx])
        fold_accs.append(float(clf.score(X_te, y_all[test_idx])))

    fold_accs_arr = np.asarray(fold_accs, dtype=float)
    return float(fold_accs_arr.mean()), float(fold_accs_arr.std()), fold_accs_arr


@dataclass(frozen=True)
class CurveResult:
    windows: list[int]
    mean_acc: np.ndarray
    std_acc: np.ndarray


def compute_curves(
    rates_by_condition: dict[str, dict[str, list[np.ndarray]]],
    windows: Iterable[int],
    n_splits: int = 5,
    n_pca_flat: int = 30,
    n_components_C: int = 30,
    C_logistic: float = 1.0,
    decoders: Iterable[str] = ("flat_pca", "time_mean", "ladder_A", "ladder_C"),
) -> dict[str, dict[str, CurveResult]]:
    """Compute curves for multiple decoders under each condition."""
    from integration_time import decode_causal_window as decode_flat_pca
    from decoding import run_decoding_ladder

    windows = [int(w) for w in windows]
    decoders = tuple(decoders)
    out: dict[str, dict[str, CurveResult]] = {}

    for cond, rates_by_stim in rates_by_condition.items():
        by_decoder: dict[str, CurveResult] = {}

        # 1) Existing: flatten(W*N) + PCA(n_pca_flat)
        if "flat_pca" in decoders:
            m, s = [], []
            for W in windows:
                mean_acc, std_acc, _fold = decode_flat_pca(
                    rates_by_stim,
                    window=W,
                    n_splits=n_splits,
                    n_pca=n_pca_flat,
                    C_logistic=C_logistic,
                )
                m.append(mean_acc)
                s.append(std_acc)
            by_decoder["flat_pca"] = CurveResult(windows, np.asarray(m), np.asarray(s))

        # 2) Control: time-mean over last W frames (no PCA)
        if "time_mean" in decoders:
            m, s = [], []
            for W in windows:
                mean_acc, std_acc, _fold = decode_causal_window_mean(
                    rates_by_stim,
                    window=W,
                    n_splits=n_splits,
                    C_logistic=C_logistic,
                )
                m.append(mean_acc)
                s.append(std_acc)
            by_decoder["time_mean"] = CurveResult(windows, np.asarray(m), np.asarray(s))

        # 3) Ladder (A,C) on windowed trajectories, so C>=A by construction
        #    This is a supervised alternative to unsupervised PCA on flattened windows.
        if ("ladder_A" in decoders) or ("ladder_C" in decoders):
            mA, sA, mC, sC = [], [], [], []
            for W in windows:
                rates_win: dict[str, list[np.ndarray]] = {}
                for sid, r_list in rates_by_stim.items():
                    r_win_list = [_window_last_frames(np.asarray(r), W) for r in r_list]
                    rates_win[sid] = r_win_list

                ladder = run_decoding_ladder(
                    rates_win,
                    models=["A", "C"],
                    n_splits=n_splits,
                    C_logistic=C_logistic,
                    n_components_C=n_components_C,
                    verbose=False,
                )
                mA.append(ladder["A"]["mean_acc"])
                sA.append(ladder["A"]["std_acc"])
                mC.append(ladder["C"]["mean_acc"])
                sC.append(ladder["C"]["std_acc"])

            if "ladder_A" in decoders:
                by_decoder["ladder_A"] = CurveResult(windows, np.asarray(mA), np.asarray(sA))
            if "ladder_C" in decoders:
                by_decoder["ladder_C"] = CurveResult(windows, np.asarray(mC), np.asarray(sC))

        out[cond] = by_decoder

    return out


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return list(DEFAULT_WINDOWS)
    return [int(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates_dir", type=str, default="scripts/temporal_decoding/data/rates")
    parser.add_argument(
        "--prefix",
        type=str,
        default="rates_lm",
        help="Filename prefix including trailing 'lm' (e.g. 'rates_lm' or 'rates_hires_lm').",
    )
    parser.add_argument("--logmar", type=float, required=True)
    parser.add_argument("--windows", type=str, default="1,6,24,60")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument(
        "--decoders",
        type=str,
        default="flat_pca,time_mean,ladder_A,ladder_C",
        help="Comma-separated subset: flat_pca,time_mean,ladder_A,ladder_C",
    )
    parser.add_argument("--n_pca_flat", type=int, default=30)
    parser.add_argument("--n_components_C", type=int, default=30)
    parser.add_argument("--C_logistic", type=float, default=1.0)
    args = parser.parse_args()

    # sklearn PCA can emit RuntimeWarnings in degenerate folds; suppress to keep output readable.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"sklearn\\.decomposition\\._pca",
    )

    rates_dir = Path(args.rates_dir)
    windows = _parse_int_list(args.windows)

    rates_by_condition = build_rates_by_condition(
        rates_dir=rates_dir,
        prefix=str(args.prefix),
        logmar=float(args.logmar),
    )

    curves = compute_curves(
        rates_by_condition=rates_by_condition,
        windows=windows,
        n_splits=int(args.n_splits),
        n_pca_flat=int(args.n_pca_flat),
        n_components_C=int(args.n_components_C),
        C_logistic=float(args.C_logistic),
        decoders=[s.strip() for s in str(args.decoders).split(",") if s.strip()],
    )

    print(f"\nIntegration-time controls @ LogMAR={float(args.logmar):+.2f} (prefix={args.prefix})")
    for cond, dd in curves.items():
        print(f"\nCondition: {cond}")
        for name, res in dd.items():
            mean = np.round(res.mean_acc, 3)
            std = np.round(res.std_acc, 3)
            print(f"  {name:>9s}: " + "  ".join(f"{w:>3d}:{m:.3f}±{s:.3f}" for w, m, s in zip(res.windows, mean, std)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from VisionCore.paths import VISIONCORE_ROOT

ROOT = VISIONCORE_ROOT
TD_RATES_DIR = ROOT / "scripts" / "temporal_decoding" / "data" / "rates"
JPF_BASE = ROOT / "outputs" / "jacobian_predictive_framework"
ORIS = (0, 90, 180, 270)
PAIRS = tuple((a, b) for i, a in enumerate(ORIS) for b in ORIS[i + 1 :])


def _pair_label(a: int, b: int) -> str:
    return f"{int(a)}_vs_{int(b)}"


def _load_rate_file(path: Path) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    rates = np.asarray(d["rates"], dtype=np.float64)  # (M, T_max, N)
    lengths = np.asarray(d["lengths"], dtype=int)
    # Match Model A: mean(axis=1) over the full trial duration.
    # With padded arrays, use true lengths per trial to avoid pad bias.
    M, _, N = rates.shape
    out = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        L = max(1, int(lengths[i]))
        out[i] = rates[i, :L].mean(axis=0)
    return out


def _decode_pairwise_grouped(Xa: np.ndarray, Xb: np.ndarray) -> float:
    n = min(Xa.shape[0], Xb.shape[0])
    if n < 4:
        return float("nan")
    Xa = Xa[:n]
    Xb = Xb[:n]
    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)], axis=0)
    groups = np.concatenate([np.arange(n, dtype=int), np.arange(n, dtype=int)], axis=0)

    n_splits = min(5, n)
    if n_splits < 2:
        return float("nan")

    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full_like(y, fill_value=-1)
    for tr, te in gkf.split(X, y, groups=groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
        clf.fit(Xtr, y[tr])
        preds[te] = clf.predict(Xte)
    return float(np.mean(preds == y))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Build Keystone cache from mono temporal-decoding rate files.")
    p.add_argument("--logmar-values", nargs="+", type=float, required=True)
    p.add_argument("--out-run-dir", type=str, default="active_sensing_efficiency_mono_recovered_20260601")
    p.add_argument("--rates-dir", type=str, default=str(TD_RATES_DIR))
    args = p.parse_args()

    rates_dir = Path(args.rates_dir)
    out_dir = JPF_BASE / str(args.out_run_dir)
    eid = out_dir / "eoptotype_identity"
    eid.mkdir(parents=True, exist_ok=True)

    # Build Keystone-compatible response-vector NPZ.
    all_vecs: list[np.ndarray] = []
    all_cond: list[str] = []
    all_lm: list[float] = []
    all_ori: list[int] = []
    all_tid: list[int] = []
    all_rr: list[int] = []
    all_fk: list[str] = []
    all_tb: list[int] = []
    all_nu: list[int] = []

    modelA: dict[tuple[str, float, int], np.ndarray] = {}

    for lm in [float(v) for v in args.logmar_values]:
        lm_tag = f"{lm:.2f}"
        for ori in ORIS:
            for cond in ("real", "stabilized"):
                path = rates_dir / f"rates_hires_lm{lm_tag}_ori{ori}_{cond}.npz"
                if not path.exists():
                    raise FileNotFoundError(f"Missing rate file: {path}")
                X = _load_rate_file(path)  # (M, N)
                modelA[(cond, round(lm, 4), int(ori))] = X

                M, N = X.shape
                for t in range(M):
                    all_vecs.append(X[t].astype(np.float32, copy=False))
                    all_cond.append(cond)
                    all_lm.append(float(lm))
                    all_ori.append(int(ori))
                    all_tid.append(int(t))
                    all_rr.append(-1)
                    all_fk.append(f"mono_feat_{cond}_{lm_tag}_{ori}_{t}")
                    all_tb.append(60)
                    all_nu.append(int(N))

    vec_arr = np.stack(all_vecs, axis=0)
    np.savez_compressed(
        eid / "eoptotype_response_vectors.npz",
        trial_index=np.asarray(all_tid, dtype=np.int32),
        condition=np.asarray(all_cond, dtype=np.str_),
        logmar=np.asarray(all_lm, dtype=np.float32),
        orientation=np.asarray(all_ori, dtype=np.int16),
        random_repeat=np.asarray(all_rr, dtype=np.int16),
        feature_key=np.asarray(all_fk, dtype=np.str_),
        n_time_bins=np.asarray(all_tb, dtype=np.int16),
        n_units=np.asarray(all_nu, dtype=np.int16),
        vectors__spatial_avg_time_mean=vec_arr,
    )

    # Build D1 integration-window sweep CSV (pairwise grouped-CV, window=60).
    rows: list[dict[str, object]] = []
    for lm in [float(v) for v in args.logmar_values]:
        lk = round(lm, 4)
        real_acc: dict[str, float] = {}
        stab_acc: dict[str, float] = {}
        for a, b in PAIRS:
            pair = _pair_label(a, b)
            acc_r = _decode_pairwise_grouped(modelA[("real", lk, a)], modelA[("real", lk, b)])
            acc_s = _decode_pairwise_grouped(modelA[("stabilized", lk, a)], modelA[("stabilized", lk, b)])
            real_acc[pair] = acc_r
            stab_acc[pair] = acc_s

        for pair in sorted(real_acc):
            delta = real_acc[pair] - stab_acc[pair] if np.isfinite(real_acc[pair]) and np.isfinite(stab_acc[pair]) else float("nan")
            rows.append(
                {
                    "condition": "real",
                    "logmar": float(lm),
                    "orientation_pair": pair,
                    "readout_type": "linear",
                    "feature_representation": "spatial_avg_time_mean",
                    "integration_window": 60,
                    "d1_time_mean_accuracy": real_acc[pair],
                    "real_minus_stabilized_d1_time_mean_accuracy": delta,
                }
            )
            rows.append(
                {
                    "condition": "stabilized",
                    "logmar": float(lm),
                    "orientation_pair": pair,
                    "readout_type": "linear",
                    "feature_representation": "spatial_avg_time_mean",
                    "integration_window": 60,
                    "d1_time_mean_accuracy": stab_acc[pair],
                    "real_minus_stabilized_d1_time_mean_accuracy": float("nan"),
                }
            )

    _write_csv(
        out_dir / "eoptotype_D1_integration_window_sweep.csv",
        rows,
        [
            "condition",
            "logmar",
            "orientation_pair",
            "readout_type",
            "feature_representation",
            "integration_window",
            "d1_time_mean_accuracy",
            "real_minus_stabilized_d1_time_mean_accuracy",
        ],
    )

    print(f"Wrote response vectors: {eid / 'eoptotype_response_vectors.npz'}")
    print(f"Wrote D1 sweep: {out_dir / 'eoptotype_D1_integration_window_sweep.csv'}")
    print(f"N rows: {vec_arr.shape[0]} | N units: {vec_arr.shape[1]}")


if __name__ == "__main__":
    main()

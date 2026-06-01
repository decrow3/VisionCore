#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from VisionCore.paths import VISIONCORE_ROOT

ROOT = VISIONCORE_ROOT
JPF_BASE = ROOT / "outputs" / "jacobian_predictive_framework"
ORIENTATIONS = (0, 90, 180, 270)
PAIRS = tuple((a, b) for i, a in enumerate(ORIENTATIONS) for b in ORIENTATIONS[i + 1 :])


DEFAULT_RVEC_DIRS = (
    "active_sensing_efficiency_e1_d1_reconciliation_20260601",
    "active_sensing_efficiency_e1_dense_missing_20260601",
)
DEFAULT_D1_SWEEP_PATHS = (
    JPF_BASE / "active_sensing_efficiency_e1_d1_reconciliation_20260601" / "eoptotype_D1_integration_window_sweep.csv",
    JPF_BASE / "active_sensing_efficiency_e1_dense_missing_20260601" / "eoptotype_D1_integration_window_sweep.csv",
)
DEFAULT_OUT_DIR = JPF_BASE / "keystone_v3_core7_readout_validation_20260601"
DEFAULT_LOGMAR_GRID = (-0.35, -0.325, -0.30, -0.275, -0.25, -0.225, -0.20)
DEFAULT_CONDITIONS = ("real", "stabilized")
DEFAULT_WINDOW = 60


def _fmt_lm(v: float) -> str:
    return f"{float(v):.3f}"


def _f(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _i(x: object) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _ids_hash(ids: list[int]) -> str:
    payload = ",".join(str(v) for v in ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    if a.shape != b.shape:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    c = np.corrcoef(a, b)[0, 1]
    return float(c)


def _decode_grouped_logistic(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, int, str]:
    unique_groups = np.unique(groups)
    n_splits = max(2, min(5, int(unique_groups.size)))
    if unique_groups.size < 2 or X.shape[0] < 4:
        return float("nan"), int(n_splits), ""

    splitter = GroupKFold(n_splits=n_splits)
    preds = np.full_like(y, fill_value=-1)
    fold_sigs: list[str] = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
        clf.fit(X_train, y[train_idx])
        preds[test_idx] = clf.predict(X_test)
        test_groups = sorted(int(g) for g in np.unique(groups[test_idx]))
        fold_sigs.append(_ids_hash(test_groups))

    acc = float(np.mean(preds == y))
    fold_sig = hashlib.sha1("|".join(fold_sigs).encode("utf-8")).hexdigest()
    return acc, int(n_splits), fold_sig


def _load_d1_sweep_accuracy(
    sweep_paths: tuple[Path, ...],
    logmar_grid: tuple[float, ...],
    conditions: tuple[str, ...],
    window: int,
) -> dict[tuple[str, float, str], float]:
    out: dict[tuple[str, float, str], float] = {}
    for p in sweep_paths:
        if not p.exists():
            continue
        rows = _read_csv(p)
        for r in rows:
            cond = str(r.get("condition", ""))
            if cond not in conditions:
                continue
            if str(r.get("feature_representation", "")) != "spatial_avg_time_mean":
                continue
            if str(r.get("readout_type", "")) != "linear":
                continue
            if _i(r.get("integration_window")) != int(window):
                continue
            lm = _f(r.get("logmar"))
            if not any(abs(lm - x) < 0.005 for x in logmar_grid):
                continue
            pair = str(r.get("orientation_pair", ""))
            out[(cond, round(lm, 4), pair)] = _f(r.get("d1_time_mean_accuracy"))
    return out


def _pair_label(a: int, b: int) -> str:
    return f"{int(a)}_vs_{int(b)}"


def _load_records_for_source(source_dir: str) -> list[dict[str, object]]:
    identity_dir = JPF_BASE / source_dir / "eoptotype_identity"
    trial_csv = identity_dir / "eoptotype_identity_trial_metrics.csv"
    features_npz = identity_dir / "eoptotype_identity_features.npz"
    rvec_npz = identity_dir / "eoptotype_response_vectors.npz"
    if not (trial_csv.exists() and features_npz.exists() and rvec_npz.exists()):
        return []

    trial_rows = _read_csv(trial_csv)
    features = np.load(features_npz, allow_pickle=False)
    rvec = np.load(rvec_npz, allow_pickle=False)

    # Mirror _aggregate_eoptotype_trial_rows key order to recover agg_feat index mapping.
    ordered_keys: list[tuple[str, float, int, int]] = []
    seen: set[tuple[str, float, int, int]] = set()
    for row in trial_rows:
        key = (
            str(row.get("condition", "")),
            round(_f(row.get("logmar")), 4),
            _i(row.get("orientation")),
            _i(row.get("trial_index")),
        )
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)

    d1_by_key: dict[tuple[str, float, int, int], np.ndarray] = {}
    for idx, key in enumerate(ordered_keys):
        agg_fkey = f"agg_feat_{idx:06d}"
        d1_key = f"{agg_fkey}_d1_time_mean_w60"
        if d1_key in features:
            d1_by_key[key] = np.asarray(features[d1_key], dtype=np.float64)

    conds = rvec["condition"].tolist()
    lms = rvec["logmar"].astype(float).tolist()
    oris = rvec["orientation"].astype(int).tolist()
    tids = rvec["trial_index"].astype(int).tolist()
    vecs = rvec["vectors__spatial_avg_time_mean"]

    val_by_key: dict[tuple[str, float, int, int], np.ndarray] = {}
    for idx in range(len(conds)):
        key = (str(conds[idx]), round(float(lms[idx]), 4), int(oris[idx]), int(tids[idx]))
        if key not in val_by_key:
            val_by_key[key] = np.asarray(vecs[idx], dtype=np.float64)

    out: list[dict[str, object]] = []
    for key, d1_vec in d1_by_key.items():
        if key not in val_by_key:
            continue
        cond, lm, ori, tid = key
        out.append(
            {
                "source_dir": source_dir,
                "condition": cond,
                "logmar": lm,
                "orientation": ori,
                "trial_id": tid,
                "feature_key": f"{cond}|{lm}|{ori}|{tid}",
                "d1_vec": d1_vec,
                "val_vec": val_by_key[key],
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Audit D1 feature/readout path versus cached validation vectors.")
    p.add_argument("--rvec-dirs", nargs="+", default=list(DEFAULT_RVEC_DIRS))
    p.add_argument("--d1-sweep-paths", nargs="+", default=[str(x) for x in DEFAULT_D1_SWEEP_PATHS])
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--logmar-grid", nargs="+", type=float, default=list(DEFAULT_LOGMAR_GRID))
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for source_dir in tuple(str(x) for x in args.rvec_dirs):
        records.extend(_load_records_for_source(source_dir))

    logmar_grid = tuple(float(x) for x in args.logmar_grid)
    conditions = tuple(str(x) for x in args.conditions)

    records = [
        r
        for r in records
        if r["condition"] in conditions and any(abs(float(r["logmar"]) - lm) < 0.005 for lm in logmar_grid)
    ]

    by_key: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for r in records:
        by_key[(str(r["condition"]), float(r["logmar"]), int(r["orientation"]))].append(r)

    # 1/2/3: feature names, row counts, trial IDs
    audit_rows: list[dict[str, object]] = []
    for cond in conditions:
        for lm in logmar_grid:
            lm_key = round(float(lm), 4)
            n_rows_validation_4way = 0
            for ori in ORIENTATIONS:
                grp = by_key.get((cond, lm_key, ori), [])
                trial_ids = sorted(int(x["trial_id"]) for x in grp)
                n_rows_validation_4way += len(grp)
                audit_rows.append(
                    {
                        "row_type": "orientation",
                        "logmar": _fmt_lm(lm),
                        "condition": cond,
                        "orientation": int(ori),
                        "feature_name_d1": f"d1_time_mean_w{int(args.window)}",
                        "feature_name_validation": "vectors__spatial_avg_time_mean",
                        "feature_name_armA": "vectors__spatial_avg_time_mean",
                        "n_trials_orientation": len(grp),
                        "trial_ids": ",".join(str(v) for v in trial_ids),
                        "trial_ids_hash": _ids_hash(trial_ids),
                        "n_rows_validation_4way": "",
                        "orientation_pair": "",
                        "n_common_trial_ids": "",
                        "common_trial_ids": "",
                        "common_trial_ids_hash": "",
                        "n_rows_d1_pair": "",
                        "n_rows_validation_pair": "",
                        "d1_validation_trial_ids_exact_match": "",
                    }
                )

            for a, b in PAIRS:
                ga = by_key.get((cond, lm_key, a), [])
                gb = by_key.get((cond, lm_key, b), [])
                ids_a = sorted({int(x["trial_id"]) for x in ga})
                ids_b = sorted({int(x["trial_id"]) for x in gb})
                common = sorted(set(ids_a) & set(ids_b))
                audit_rows.append(
                    {
                        "row_type": "pair",
                        "logmar": _fmt_lm(lm),
                        "condition": cond,
                        "orientation": "",
                        "feature_name_d1": f"d1_time_mean_w{int(args.window)}",
                        "feature_name_validation": "vectors__spatial_avg_time_mean",
                        "feature_name_armA": "vectors__spatial_avg_time_mean",
                        "n_trials_orientation": "",
                        "trial_ids": "",
                        "trial_ids_hash": "",
                        "n_rows_validation_4way": int(n_rows_validation_4way),
                        "orientation_pair": _pair_label(a, b),
                        "n_common_trial_ids": len(common),
                        "common_trial_ids": ",".join(str(v) for v in common),
                        "common_trial_ids_hash": _ids_hash(common),
                        "n_rows_d1_pair": int(2 * len(common)),
                        "n_rows_validation_pair": int(2 * len(common)),
                        "d1_validation_trial_ids_exact_match": int(set(common) == (set(ids_a) & set(ids_b))),
                    }
                )

    # 4: feature vector equality checks
    eq_rows: list[dict[str, object]] = []
    for cond in conditions:
        for lm in logmar_grid:
            lm_key = round(float(lm), 4)
            for ori in ORIENTATIONS:
                grp = by_key.get((cond, lm_key, ori), [])
                if not grp:
                    continue
                trial_diffs: list[np.ndarray] = []
                for rec in sorted(grp, key=lambda z: int(z["trial_id"])):
                    d1_vec = np.asarray(rec["d1_vec"], dtype=np.float64)
                    val_vec = np.asarray(rec["val_vec"], dtype=np.float64)
                    if d1_vec.shape != val_vec.shape:
                        continue
                    diff = np.abs(d1_vec - val_vec)
                    trial_diffs.append(diff)
                    eq_rows.append(
                        {
                            "row_type": "trial",
                            "source_dir": rec["source_dir"],
                            "logmar": _fmt_lm(lm),
                            "condition": cond,
                            "orientation": int(ori),
                            "trial_id": int(rec["trial_id"]),
                            "vector_dim": int(d1_vec.size),
                            "max_abs_diff": float(np.max(diff)),
                            "mean_abs_diff": float(np.mean(diff)),
                            "correlation": _safe_corr(d1_vec, val_vec),
                        }
                    )
                if trial_diffs:
                    all_diffs = np.concatenate([d.reshape(-1) for d in trial_diffs], axis=0)
                    d1_stack = np.stack([np.asarray(r["d1_vec"], dtype=np.float64) for r in grp], axis=0)
                    val_stack = np.stack([np.asarray(r["val_vec"], dtype=np.float64) for r in grp], axis=0)
                    eq_rows.append(
                        {
                            "row_type": "aggregate",
                            "source_dir": "mixed",
                            "logmar": _fmt_lm(lm),
                            "condition": cond,
                            "orientation": int(ori),
                            "trial_id": "",
                            "vector_dim": int(d1_stack.shape[1]),
                            "max_abs_diff": float(np.max(all_diffs)),
                            "mean_abs_diff": float(np.mean(all_diffs)),
                            "correlation": _safe_corr(d1_stack.reshape(-1), val_stack.reshape(-1)),
                        }
                    )

    # 5: pairwise D1 reproduction with exact D1 design and same common trial IDs
    d1_lookup = _load_d1_sweep_accuracy(
        tuple(Path(x) for x in args.d1_sweep_paths),
        logmar_grid,
        conditions,
        int(args.window),
    )
    pair_rows: list[dict[str, object]] = []
    for cond in conditions:
        for lm in logmar_grid:
            lm_key = round(float(lm), 4)
            by_ori_trial: dict[int, dict[int, dict[str, object]]] = {}
            for ori in ORIENTATIONS:
                recs = by_key.get((cond, lm_key, ori), [])
                by_ori_trial[ori] = {int(r["trial_id"]): r for r in recs}

            for a, b in PAIRS:
                ids_a = set(by_ori_trial.get(a, {}).keys())
                ids_b = set(by_ori_trial.get(b, {}).keys())
                common = sorted(ids_a & ids_b)
                if len(common) < 4:
                    pair_rows.append(
                        {
                            "logmar": _fmt_lm(lm),
                            "condition": cond,
                            "orientation_pair": _pair_label(a, b),
                            "n_common_trial_ids": len(common),
                            "common_trial_ids_hash": _ids_hash(common),
                            "common_trial_ids": ",".join(str(v) for v in common),
                            "d1_csv_accuracy_w60": d1_lookup.get((cond, lm_key, _pair_label(a, b)), float("nan")),
                            "reproduced_accuracy_d1_features": float("nan"),
                            "reproduced_accuracy_cached_vectors": float("nan"),
                            "abs_diff_d1csv_vs_reproduced": float("nan"),
                            "n_splits": 0,
                            "fold_signature": "",
                            "task_equivalent_pairwise_design": 0,
                        }
                    )
                    continue

                Xa_d1 = np.stack([np.asarray(by_ori_trial[a][tid]["d1_vec"], dtype=np.float64) for tid in common], axis=0)
                Xb_d1 = np.stack([np.asarray(by_ori_trial[b][tid]["d1_vec"], dtype=np.float64) for tid in common], axis=0)
                Xa_val = np.stack([np.asarray(by_ori_trial[a][tid]["val_vec"], dtype=np.float64) for tid in common], axis=0)
                Xb_val = np.stack([np.asarray(by_ori_trial[b][tid]["val_vec"], dtype=np.float64) for tid in common], axis=0)

                y = np.concatenate([np.zeros(len(common), dtype=np.int64), np.ones(len(common), dtype=np.int64)], axis=0)
                groups = np.concatenate([np.asarray(common, dtype=np.int64), np.asarray(common, dtype=np.int64)], axis=0)

                X_d1 = np.concatenate([Xa_d1, Xb_d1], axis=0)
                X_val = np.concatenate([Xa_val, Xb_val], axis=0)
                acc_d1, n_splits, fold_sig = _decode_grouped_logistic(X_d1, y, groups)
                acc_val, _, _ = _decode_grouped_logistic(X_val, y, groups)
                d1_csv_acc = d1_lookup.get((cond, lm_key, _pair_label(a, b)), float("nan"))

                pair_rows.append(
                    {
                        "logmar": _fmt_lm(lm),
                        "condition": cond,
                        "orientation_pair": _pair_label(a, b),
                        "n_common_trial_ids": len(common),
                        "common_trial_ids_hash": _ids_hash(common),
                        "common_trial_ids": ",".join(str(v) for v in common),
                        "d1_csv_accuracy_w60": d1_csv_acc,
                        "reproduced_accuracy_d1_features": acc_d1,
                        "reproduced_accuracy_cached_vectors": acc_val,
                        "abs_diff_d1csv_vs_reproduced": abs(d1_csv_acc - acc_d1) if np.isfinite(d1_csv_acc) and np.isfinite(acc_d1) else float("nan"),
                        "n_splits": int(n_splits),
                        "fold_signature": fold_sig,
                        "task_equivalent_pairwise_design": 1,
                    }
                )

    d1_audit_csv = out_dir / "d1_input_reproduction_audit.csv"
    pair_csv = out_dir / "pairwise_d1_reproduction_validation.csv"
    eq_csv = out_dir / "feature_vector_equality_check.csv"
    readme = out_dir / "cached_vector_D1_mismatch_readme.md"

    _write_csv(d1_audit_csv, audit_rows)
    _write_csv(pair_csv, pair_rows)
    _write_csv(eq_csv, eq_rows)

    # Readme summary
    def _nanmean(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    pair_diffs = [_f(r.get("abs_diff_d1csv_vs_reproduced")) for r in pair_rows]
    acc_d1_feat = [_f(r.get("reproduced_accuracy_d1_features")) for r in pair_rows]
    acc_cached = [_f(r.get("reproduced_accuracy_cached_vectors")) for r in pair_rows]
    agg_rows = [r for r in eq_rows if str(r.get("row_type")) == "aggregate"]
    eq_mae = _nanmean([_f(r.get("mean_abs_diff")) for r in agg_rows])
    eq_corr = _nanmean([_f(r.get("correlation")) for r in agg_rows])

    lines = [
        "# Cached Vector vs D1 Input Reproduction Audit",
        "",
        "Geometry interpretation remains paused.",
        "Current state label: cached_vector_D1_mismatch.",
        "",
        "## Feature Path",
        "",
        f"- D1 decoder feature: d1_time_mean_w{int(args.window)}.",
        "- Validation feature (step2b) and Arm A vector input: vectors__spatial_avg_time_mean.",
        "- Pairwise reproduction was run on the exact D1 pair design (same orientation pair, same common trial IDs, same grouped-CV structure).",
        "",
        "## Reproduction Summary",
        "",
        f"- Mean |D1 CSV - reproduced D1-feature accuracy|: {_nanmean(pair_diffs):.6f}.",
        f"- Mean reproduced accuracy with D1 features: {_nanmean(acc_d1_feat):.6f}.",
        f"- Mean reproduced accuracy with cached validation vectors on the same pair design: {_nanmean(acc_cached):.6f}.",
        "",
        "## Feature Equality (D1 vs cached validation vectors)",
        "",
        f"- Mean absolute difference across aggregate cells: {eq_mae:.6f}.",
        f"- Mean correlation across aggregate cells: {eq_corr:.6f}.",
        "",
        "## Output Files",
        "",
        "- d1_input_reproduction_audit.csv",
        "- pairwise_d1_reproduction_validation.csv",
        "- feature_vector_equality_check.csv",
        "- cached_vector_D1_mismatch_readme.md",
        "",
    ]
    readme.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {d1_audit_csv}")
    print(f"Wrote: {pair_csv}")
    print(f"Wrote: {eq_csv}")
    print(f"Wrote: {readme}")


if __name__ == "__main__":
    main()

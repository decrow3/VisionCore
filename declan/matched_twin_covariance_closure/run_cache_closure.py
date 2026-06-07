from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import dill
import numpy as np

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_FIG3_CACHE = VISIONCORE_ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"
DEFAULT_FIG2_CACHE = VISIONCORE_ROOT / "outputs" / "cache" / "fig2_decomposition_ryan.pkl"
DEFAULT_OUT = VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return dill.load(handle)


def _sym(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (a + a.T)


def _eigh_desc(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(_sym(c))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _psd_clip(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = _eigh_desc(c)
    clipped = np.maximum(vals, 0.0)
    return _sym((vecs * clipped[None, :]) @ vecs.T), vals


def _orth(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.size == 0 or x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(x)
    keep = np.abs(np.diag(r)) > eps
    return q[:, keep]


def _projection_complement(n: int, modes: np.ndarray) -> np.ndarray:
    u = _orth(modes)
    if u.shape[1] == 0:
        return np.eye(n, dtype=np.float64)
    return np.eye(n, dtype=np.float64) - u @ u.T


def _apply_projection_to_cov(c: np.ndarray, p: np.ndarray) -> np.ndarray:
    return _sym(p @ _sym(c) @ p)


def _capture(c: np.ndarray, u: np.ndarray, eps: float = 1e-12) -> float:
    c = _sym(c)
    u = _orth(u)
    tr = float(np.trace(c))
    if u.shape[1] == 0 or not np.isfinite(tr) or abs(tr) < eps:
        return float("nan")
    return float(np.trace(u.T @ c @ u) / (tr + eps))


def _cov_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.all(np.isfinite(x), axis=1)]
    if x.shape[0] < 3:
        return np.full((x.shape[1], x.shape[1]), np.nan)
    x = x - np.mean(x, axis=0, keepdims=True)
    return _sym((x.T @ x) / max(x.shape[0] - 1, 1))


def _nan_time_residual(r: np.ndarray, dfs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=np.float64)
    valid = np.asarray(dfs) != 0
    valid = valid & np.isfinite(r)
    counts = np.sum(valid, axis=0, keepdims=True)
    sums = np.sum(np.where(valid, r, 0.0), axis=0, keepdims=True)
    time_mean = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=time_mean, where=counts > 0)
    masked = np.where(valid, r, np.nan)
    resid = masked - time_mean
    row_valid = np.all(np.isfinite(resid), axis=2)
    return resid, row_valid


def _flatten_valid(r: np.ndarray, row_valid: np.ndarray) -> np.ndarray:
    flat = np.asarray(r, dtype=np.float64).reshape(-1, r.shape[-1])
    mask = np.asarray(row_valid, dtype=bool).reshape(-1)
    x = flat[mask]
    return x[np.all(np.isfinite(x), axis=1)]


def _fit_eye_regression_basis(r_resid: np.ndarray, eye: np.ndarray, row_valid: np.ndarray) -> dict[str, Any]:
    x = _flatten_valid(r_resid, row_valid)
    e = np.asarray(eye, dtype=np.float64).reshape(-1, 2)[np.asarray(row_valid, dtype=bool).reshape(-1)]
    ok = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(e), axis=1)
    x = x[ok]
    e = e[ok]
    if x.shape[0] < 50:
        return {
            "basis_matrix": np.full((r_resid.shape[-1], 2), np.nan),
            "covariance": np.full((r_resid.shape[-1], r_resid.shape[-1]), np.nan),
            "n_samples": int(x.shape[0]),
            "status": "too_few_samples",
        }
    e_mean = np.mean(e, axis=0, keepdims=True)
    e_c = e - e_mean
    b_t, _, _, _ = np.linalg.lstsq(e_c, x, rcond=None)
    b = b_t.T
    sigma_e = np.cov(e.T)
    return {
        "basis_matrix": b,
        "covariance": _sym(b @ sigma_e @ b.T),
        "n_samples": int(x.shape[0]),
        "status": "ok",
    }


def _unit_mask_intersection(fig2_mask: np.ndarray, fig3_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fig2_pos = {int(u): i for i, u in enumerate(np.asarray(fig2_mask).tolist())}
    fig3_pos = {int(u): i for i, u in enumerate(np.asarray(fig3_mask).tolist())}
    common = np.array(sorted(set(fig2_pos).intersection(fig3_pos)), dtype=int)
    idx2 = np.array([fig2_pos[int(u)] for u in common], dtype=int)
    idx3 = np.array([fig3_pos[int(u)] for u in common], dtype=int)
    return common, idx2, idx3


def _projection_modes(kind: str, target_cov: np.ndarray) -> np.ndarray:
    n = target_cov.shape[0]
    modes: list[np.ndarray] = []
    if kind in {"global_rate", "global_rate+target_pc1"}:
        modes.append(np.ones(n, dtype=np.float64))
    if kind in {"target_pc1", "global_rate+target_pc1"}:
        vals, vecs = _eigh_desc(target_cov)
        if vals.size and np.isfinite(vals[0]):
            modes.append(vecs[:, 0])
    if not modes:
        return np.zeros((n, 0), dtype=np.float64)
    return np.stack(modes, axis=1)


def _basis_from_cov_or_matrix(source: str, cov: np.ndarray | None, mat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if source.endswith("_cov"):
        if cov is None:
            raise ValueError(f"{source} requires covariance")
        vals, vecs = _eigh_desc(cov)
        return vals, vecs
    if mat is None:
        raise ValueError(f"{source} requires basis matrix")
    vals, vecs = _eigh_desc(mat @ mat.T)
    return vals, vecs


def _null_captures(
    *,
    rng: np.random.Generator,
    target: np.ndarray,
    basis_vecs: np.ndarray,
    source_matrix: np.ndarray | None,
    k: int,
    n_nulls: int,
) -> dict[str, Any]:
    n = target.shape[0]
    random_vals = []
    unit_vals = []
    for _ in range(n_nulls):
        q, _ = np.linalg.qr(rng.standard_normal((n, max(k, 1))))
        random_vals.append(_capture(target, q[:, :k]))

        perm = rng.permutation(n)
        if source_matrix is not None:
            shuffled = np.asarray(source_matrix)[perm, :]
            _, u_shuf = _basis_from_cov_or_matrix("matrix", None, shuffled)
        else:
            u_shuf = np.asarray(basis_vecs)[perm, :]
        unit_vals.append(_capture(target, u_shuf[:, :k]))

    def pack(vals: list[float], prefix: str) -> dict[str, Any]:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_median": float("nan"),
                f"{prefix}_ci_low": float("nan"),
                f"{prefix}_ci_high": float("nan"),
            }
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_ci_low": float(np.percentile(arr, 2.5)),
            f"{prefix}_ci_high": float(np.percentile(arr, 97.5)),
        }

    out = {}
    out.update(pack(random_vals, "random_subspace_null"))
    out.update(pack(unit_vals, "unit_shuffle_null"))
    return out


def _fig2_by_session(fig2_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["session"]): row for row in fig2_rows}


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _safe_sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def summarize_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    group_keys = ("target_variant", "projection_control", "basis_source", "k")
    for row in metric_rows:
        if row.get("row_status") != "ok":
            continue
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        vals = {
            "capture": np.array([r["capture"] for r in rows], dtype=np.float64),
            "effect_minus_unit_shuffle_median": np.array(
                [r["effect_minus_unit_shuffle_median"] for r in rows], dtype=np.float64
            ),
            "effect_minus_random_subspace_median": np.array(
                [r["effect_minus_random_subspace_median"] for r in rows], dtype=np.float64
            ),
            "target_trace": np.array([r["target_trace"] for r in rows], dtype=np.float64),
            "n_common_units": np.array([r["n_common_units"] for r in rows], dtype=np.float64),
        }
        summary = dict(zip(group_keys, key, strict=True))
        summary.update(
            {
                "n_sessions": int(len(rows)),
                "capture_mean": _safe_mean(vals["capture"]),
                "capture_sem": _safe_sem(vals["capture"]),
                "capture_ci_low": _safe_percentile(vals["capture"], 2.5),
                "capture_ci_high": _safe_percentile(vals["capture"], 97.5),
                "effect_unit_mean": _safe_mean(vals["effect_minus_unit_shuffle_median"]),
                "effect_unit_sem": _safe_sem(vals["effect_minus_unit_shuffle_median"]),
                "effect_unit_ci_low": _safe_percentile(vals["effect_minus_unit_shuffle_median"], 2.5),
                "effect_unit_ci_high": _safe_percentile(vals["effect_minus_unit_shuffle_median"], 97.5),
                "effect_random_mean": _safe_mean(vals["effect_minus_random_subspace_median"]),
                "effect_random_sem": _safe_sem(vals["effect_minus_random_subspace_median"]),
                "target_trace_mean": _safe_mean(vals["target_trace"]),
                "n_common_units_mean": _safe_mean(vals["n_common_units"]),
            }
        )
        summary_rows.append(summary)
    return summary_rows


def build_inventory(fig3_rows: list[dict[str, Any]], fig2_rows: list[dict[str, Any]], window_idx: int) -> list[dict[str, Any]]:
    fig2 = _fig2_by_session(fig2_rows)
    rows: list[dict[str, Any]] = []
    for sr in fig3_rows:
        session = str(sr["session"])
        f2 = fig2.get(session)
        status = "ok"
        n_common = 0
        if f2 is None:
            status = "missing_fig2_session"
        elif window_idx >= len(f2.get("mats", [])):
            status = "missing_fig2_window"
        else:
            common, _, _ = _unit_mask_intersection(f2["neuron_mask"], sr["neuron_mask"])
            n_common = int(common.size)
            if n_common < 3:
                status = "too_few_common_units"
        rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "fig3_n_units": int(len(sr.get("neuron_mask", []))),
                "fig2_n_units": int(len(f2.get("neuron_mask", []))) if f2 is not None else 0,
                "n_common_units": n_common,
                "fig3_n_trials": int(sr.get("n_trials", 0)),
                "fig3_n_time": int(sr.get("n_time", 0)),
                "status": status,
            }
        )
    return rows


def run_analysis(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig3_cache = Path(args.fig3_cache)
    fig2_cache = Path(args.fig2_cache)
    manifest = {
        "fig3_cache": str(fig3_cache),
        "fig2_cache": str(fig2_cache),
        "window_idx": int(args.window_idx),
        "basis_sources": args.basis_sources,
        "k_list": args.k_list,
        "projection_controls": args.projection_controls,
        "target_variants": args.target_variants,
        "n_nulls": int(args.n_nulls),
        "seed": int(args.seed),
        "analysis_note": (
            "Cache-based pilot. eye_regression uses fitted-twin rhat residuals "
            "regressed on measured eye position in matched recorded units; "
            "model_residual_cov uses covariance of fitted-twin PSTH residuals."
        ),
    }

    if not fig3_cache.exists() or not fig2_cache.exists():
        manifest["status"] = "missing_input_cache"
        manifest["fig3_exists"] = fig3_cache.exists()
        manifest["fig2_exists"] = fig2_cache.exists()
        _write_json(out_dir / "run_manifest.json", manifest)
        return

    fig3_rows = _load_pickle(fig3_cache)
    fig2_rows = _load_pickle(fig2_cache)
    inventory = build_inventory(fig3_rows, fig2_rows, int(args.window_idx))
    _write_csv(out_dir / "session_inventory.csv", inventory)

    fig2 = _fig2_by_session(fig2_rows)
    k_list = [int(x) for x in str(args.k_list).split(",") if str(x).strip()]
    basis_sources = [x.strip() for x in str(args.basis_sources).split(",") if x.strip()]
    projection_controls = [x.strip() for x in str(args.projection_controls).split(",") if x.strip()]
    target_variants = [x.strip() for x in str(args.target_variants).split(",") if x.strip()]
    rng = np.random.default_rng(int(args.seed))
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    requested_sessions = [x.strip() for x in str(args.sessions).split(",") if x.strip()]
    allowed = set(requested_sessions) if requested_sessions else None

    for sr in fig3_rows:
        session = str(sr["session"])
        if allowed is not None and session not in allowed:
            continue
        inv = next((r for r in inventory if r["session"] == session), None)
        if inv is None or inv["status"] != "ok":
            continue
        f2 = fig2[session]
        common, idx2, idx3 = _unit_mask_intersection(f2["neuron_mask"], sr["neuron_mask"])
        mats = f2["mats"][int(args.window_idx)]
        target_full = _sym(np.asarray(mats["FEM"], dtype=np.float64)[np.ix_(idx2, idx2)])
        unit_keep = np.isfinite(np.diag(target_full))
        if int(np.sum(unit_keep)) < 3:
            continue
        common = common[unit_keep]
        idx2 = idx2[unit_keep]
        idx3 = idx3[unit_keep]
        target_raw = _sym(np.asarray(mats["FEM"], dtype=np.float64)[np.ix_(idx2, idx2)])
        if not np.isfinite(target_raw).all():
            finite_rows = np.isfinite(target_raw).all(axis=0) & np.isfinite(target_raw).all(axis=1)
            if int(np.sum(finite_rows)) < 3:
                continue
            common = common[finite_rows]
            idx2 = idx2[finite_rows]
            idx3 = idx3[finite_rows]
            target_raw = _sym(np.asarray(mats["FEM"], dtype=np.float64)[np.ix_(idx2, idx2)])
        target_trace_raw = float(np.trace(target_raw))
        target_psd, target_raw_eigs = _psd_clip(target_raw)
        target_trace_psd = float(np.trace(target_psd))

        rhat = np.asarray(sr["rhat_used"], dtype=np.float64)[:, :, idx3]
        dfs = np.asarray(sr["dfs_used"])[:, :, idx3]
        eye = np.asarray(sr["eyepos_used"], dtype=np.float64)
        resid, row_valid_units = _nan_time_residual(rhat, dfs)
        row_valid = row_valid_units & np.isfinite(eye).all(axis=-1)
        x_model = _flatten_valid(resid, row_valid)
        model_resid_cov = _cov_rows(x_model)
        eye_fit = _fit_eye_regression_basis(resid, eye, row_valid)

        source_payload = {
            "model_residual_cov": {"cov": model_resid_cov, "mat": None, "status": "ok"},
            "eye_regression_matrix": {"cov": eye_fit["covariance"], "mat": eye_fit["basis_matrix"], "status": eye_fit["status"]},
            "eye_regression_cov": {"cov": eye_fit["covariance"], "mat": None, "status": eye_fit["status"]},
        }

        summary_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "n_common_units": int(common.size),
                "n_common_units_before_finite_filter": int(target_full.shape[0]),
                "n_units_dropped_nonfinite_target": int(target_full.shape[0] - common.size),
                "target_trace_raw": target_trace_raw,
                "target_trace_psd": target_trace_psd,
                "target_min_eigenvalue_raw": float(np.min(target_raw_eigs)) if target_raw_eigs.size else float("nan"),
                "target_negative_eigenvalue_mass_raw": float(np.sum(np.abs(target_raw_eigs[target_raw_eigs < 0.0]))),
                "model_residual_samples": int(x_model.shape[0]),
                "eye_regression_samples": int(eye_fit["n_samples"]),
                "eye_regression_status": eye_fit["status"],
            }
        )

        targets = {
            "raw": target_raw,
            "psd": target_psd,
        }

        for target_variant in target_variants:
            target_base = targets.get(target_variant)
            if target_base is None:
                continue
            for projection_control in projection_controls:
                modes = _projection_modes(projection_control, target_base)
                p = _projection_complement(common.size, modes)
                target = _apply_projection_to_cov(target_base, p)
                for source in basis_sources:
                    payload = source_payload.get(source)
                    if payload is None:
                        continue
                    cov = payload["cov"]
                    mat = payload["mat"]
                    if cov is not None:
                        cov = _apply_projection_to_cov(cov, p)
                    if mat is not None:
                        mat = p @ np.asarray(mat, dtype=np.float64)
                    if cov is None and mat is None:
                        continue
                    if cov is not None and not np.isfinite(cov).all():
                        status = "invalid_source_covariance"
                        vals = np.array([])
                        vecs = np.zeros((common.size, 0))
                    elif mat is not None and not np.isfinite(mat).all():
                        status = "invalid_source_matrix"
                        vals = np.array([])
                        vecs = np.zeros((common.size, 0))
                    else:
                        vals, vecs = _basis_from_cov_or_matrix(source, cov, mat)
                        status = str(payload["status"])

                    rank = int(np.sum(np.maximum(vals, 0.0) > max(np.max(vals) if vals.size else 0.0, 1.0) * 1e-10))
                    for k in k_list:
                        row = {
                            "session": session,
                            "subject": sr.get("subject", ""),
                            "window_idx": int(args.window_idx),
                            "target_variant": target_variant,
                            "projection_control": projection_control,
                            "basis_source": source,
                            "basis_status": status,
                            "n_common_units": int(common.size),
                            "basis_rank": rank,
                            "k": int(k),
                            "target_trace": float(np.trace(target)),
                            "target_trace_raw": target_trace_raw,
                            "target_trace_psd": target_trace_psd,
                        }
                        if status != "ok" or k > max(rank, 0):
                            row.update(
                                {
                                    "capture": float("nan"),
                                    "effect_minus_unit_shuffle_median": float("nan"),
                                    "effect_minus_random_subspace_median": float("nan"),
                                    "row_status": "not_evaluable",
                                }
                            )
                            metric_rows.append(row)
                            continue
                        cap = _capture(target, vecs[:, :k])
                        nulls = _null_captures(
                            rng=rng,
                            target=target,
                            basis_vecs=vecs,
                            source_matrix=mat,
                            k=int(k),
                            n_nulls=int(args.n_nulls),
                        )
                        row.update(nulls)
                        row.update(
                            {
                                "capture": cap,
                                "effect_minus_unit_shuffle_median": cap - row["unit_shuffle_null_median"],
                                "effect_minus_random_subspace_median": cap - row["random_subspace_null_median"],
                                "row_status": "ok",
                            }
                        )
                        metric_rows.append(row)

    _write_csv(out_dir / "closure_session_summary.csv", summary_rows)
    _write_csv(out_dir / "closure_capture_metrics.csv", metric_rows)
    _write_csv(out_dir / "closure_metric_summary.csv", summarize_metrics(metric_rows))
    manifest["status"] = "ok"
    manifest["n_sessions_inventory"] = len(inventory)
    manifest["n_metric_rows"] = len(metric_rows)
    _write_json(out_dir / "run_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matched fitted-twin covariance closure from Ryan Fig4/Fig2 caches")
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window-idx", type=int, default=1)
    p.add_argument("--sessions", type=str, default="")
    p.add_argument("--basis-sources", type=str, default="eye_regression_matrix,eye_regression_cov,model_residual_cov")
    p.add_argument("--projection-controls", type=str, default="none,global_rate,target_pc1,global_rate+target_pc1")
    p.add_argument("--target-variants", type=str, default="raw,psd")
    p.add_argument("--k-list", type=str, default="1,2,3,5,10,20")
    p.add_argument("--n-nulls", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()

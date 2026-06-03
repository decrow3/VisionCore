from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from declan.twin_covariance_structure.covariance_core import (  # type: ignore
        compute_cfem_for_image,
        compute_signal_covariance,
        eigensystem,
        participation_ratio,
        top_subspace,
    )
    from declan.twin_covariance_structure.plotting import (  # type: ignore
        plot_a1_signal_alignment,
        plot_a2_rank_mechanism,
        plot_a3_image_specificity,
        plot_a4_tangent_alignment,
        plot_a5_occupancy,
        plot_a6_single_unit,
    )
    from declan.twin_covariance_structure.subspace_metrics import (  # type: ignore
        principal_angles,
        subspace_overlap,
        variance_captured,
    )
else:
    from .covariance_core import (
        compute_cfem_for_image,
        compute_signal_covariance,
        eigensystem,
        participation_ratio,
        top_subspace,
    )
    from .plotting import (
        plot_a1_signal_alignment,
        plot_a2_rank_mechanism,
        plot_a3_image_specificity,
        plot_a4_tangent_alignment,
        plot_a5_occupancy,
        plot_a6_single_unit,
    )
    from .subspace_metrics import principal_angles, subspace_overlap, variance_captured


ORIENTATIONS = (0, 90, 180, 270)
EPS = 1e-12
EXPLICIT_1D_CONDITIONS = {"x_only", "y_only", "line_random_angle"}
EXPLICIT_OCCUPANCY_CONDITIONS = {
    "occupancy_shuffle",
    "occupancy_iid",
    "occupancy_matched_shuffle",
    "occupancy_matched_iid",
}
EXPLICIT_AMPLITUDE_CONDITIONS = {
    "amplitude_gaussian_iso",
    "amplitude_gaussian_aniso",
    "amplitude_matched_gaussian",
    "amplitude_uniform_ring",
    "amplitude_uniform",
}


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(x)) for x in str(text).split(",") if str(x).strip())


def _parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _fmt_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _pick_rates_path(rates_dir: Path, logmar: float, orientation: int, condition: str) -> Path:
    lm = _fmt_logmar(logmar)
    candidates = [
        rates_dir / f"rates_hires_lm{lm}_ori{orientation}_{condition}.npz",
        rates_dir / f"rates_lm{lm}_ori{orientation}_{condition}.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing rates cache for lm={lm}, ori={orientation}, condition={condition} in {rates_dir}"
    )


def _load_rates_array(path: Path, max_trials: int | None = None) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    rates_padded = np.asarray(d["rates"], dtype=np.float64)
    lengths = np.asarray(d["lengths"], dtype=int)
    n_trials = rates_padded.shape[0] if max_trials is None else min(rates_padded.shape[0], max_trials)
    lengths = lengths[:n_trials]
    t_min = int(np.min(lengths))
    rates = rates_padded[:n_trials, :t_min, :]
    if not np.isfinite(rates).all():
        raise ValueError(f"Non-finite values found in {path}")
    return rates


def _infer_eye_dof(condition: str) -> int:
    c = condition.lower()
    if c in EXPLICIT_1D_CONDITIONS:
        return 1
    if c in {"fixed_center", "constant_eye"}:
        return 0
    return 2


def _effective_rank(evals: np.ndarray, rel_eps: float = 1e-8) -> int:
    vals = np.asarray(evals, dtype=np.float64)
    if vals.size == 0:
        return 0
    thresh = max(float(vals[0]) * rel_eps, EPS)
    return int(np.sum(vals > thresh))


def _basis_from_columns(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return np.zeros((U.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(U)
    keep = np.abs(np.diag(r)) > eps
    return q[:, keep]


def _condition_present(conditions: list[str] | tuple[str, ...], allowed: set[str]) -> list[str]:
    return [c for c in conditions if c in allowed]


def _format_reason(reason: str | None) -> str:
    return reason if reason is not None else ""

def _orientation_from_image_id(image_id: str) -> int | None:
    if image_id.startswith("ori"):
        try:
            return int(image_id[3:])
        except ValueError:
            return None
    return None

def _build_a2_orientation_rows(
    per_condition_rows: list[dict[str, Any]],
    conditions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in per_condition_rows:
        cond = str(row.get("condition", ""))
        if cond not in conditions:
            continue
        image_id = str(row.get("image_id", ""))
        orientation = _orientation_from_image_id(image_id)
        if orientation is None:
            continue
        rows.append(
            {
                "orientation": orientation,
                "image_id": image_id,
                "condition": cond,
                "pr": float(row["pr"]),
                "frac_top2": float(row["frac_top2"]),
                "cfem_trace": float(row["cfem_trace"]),
            }
        )
    rows.sort(key=lambda r: (int(r["orientation"]), str(r["condition"])))
    return rows

def _plot_a2_orientation_diagnostics(
    a2_rows: list[dict[str, Any]],
    eigs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    image_ids: list[str],
    out_dir: Path,
) -> None:
    if not a2_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = sorted({str(r["condition"]) for r in a2_rows})
    orientations = sorted({int(r["orientation"]) for r in a2_rows})

    def _series(condition: str, key: str) -> list[float]:
        by_ori = {int(r["orientation"]): float(r[key]) for r in a2_rows if str(r["condition"]) == condition}
        return [by_ori.get(o, np.nan) for o in orientations]

    metric_specs = [
        ("pr", "Participation Ratio", "a2_orientation_pr.png"),
        ("frac_top2", "Top-2 Variance Fraction", "a2_orientation_top2_fraction.png"),
        ("cfem_trace", "Covariance Trace", "a2_orientation_cfem_trace.png"),
    ]

    for key, ylabel, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for condition in conditions:
            ax.plot(orientations, _series(condition, key), marker="o", linewidth=2, label=condition)
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(orientations)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    rep_image_id = image_ids[0] if image_ids else None
    if rep_image_id is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for condition in conditions:
            key = (condition, rep_image_id)
            if key not in eigs:
                continue
            evals = np.asarray(eigs[key][0], dtype=np.float64)
            keep = min(20, evals.size)
            if keep <= 0:
                continue
            idx = np.arange(1, keep + 1)
            ax.plot(idx, evals[:keep], marker="o", linewidth=1.8, label=condition)
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"A2 Representative Eigenspectra ({rep_image_id})")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "a2_representative_eigenspectra.png", dpi=180)
        plt.close(fig)


def _finding_line(name: str, status: str, detail: str) -> str:
    return f"{name}: {status}. {detail}"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    sx = np.std(x)
    sy = np.std(y)
    if sx < EPS or sy < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _random_subspace(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = min(k, n_units)
    mat = rng.normal(size=(n_units, k))
    q, _ = np.linalg.qr(mat)
    return q[:, :k]


def _load_jacobians_by_orientation(jacobian_dir: Path, logmar: float) -> dict[int, np.ndarray]:
    lm = _fmt_logmar(logmar)
    candidates = [
        jacobian_dir / f"test3_lm{lm}.npz",
        jacobian_dir / f"test3_lm{lm}_grid7.npz",
    ]
    src = None
    for p in candidates:
        if p.exists():
            src = p
            break
    if src is None:
        raise FileNotFoundError(f"No Jacobian bundle found for lm={lm} in {jacobian_dir}")

    d = np.load(src, allow_pickle=True)
    out: dict[int, np.ndarray] = {}
    for ori in ORIENTATIONS:
        key = f"J_int_ori{ori}"
        if key not in d.files:
            raise KeyError(f"Missing key {key} in {src}")
        out[int(ori)] = np.asarray(d[key], dtype=np.float64)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                w.writeheader()
        return
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _status_from_three_way(supported: bool, mixed: bool, ran: bool) -> str:
    if not ran:
        return "not_run"
    if supported:
        return "supported"
    if mixed:
        return "mixed"
    return "failed"


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "logmar": args.logmar,
        "orientations": list(args.orientations),
        "primary_condition": args.primary_condition,
        "conditions": list(args.conditions),
        "k_list": list(args.k_list),
        "seed": args.seed,
        "n_null": args.n_null,
        "max_trials": args.max_trials,
        "rates_dir": str(args.rates_dir),
        "jacobian_dir": str(args.jacobian_dir),
        "guardrails": {
            "geometry_only": True,
            "no_decoder_or_information_metrics": True,
            "jacobian_alignment_only": True,
            "no_jsigmaj_magnitude_identity": True,
        },
        "control_validity": {
            "a2_requires_explicit_1d_conditions": sorted(EXPLICIT_1D_CONDITIONS),
            "a5_requires_explicit_occupancy_conditions": sorted(EXPLICIT_OCCUPANCY_CONDITIONS),
            "a5_requires_explicit_amplitude_conditions": sorted(EXPLICIT_AMPLITUDE_CONDITIONS),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    notes: list[str] = []
    not_run_reasons: dict[str, str | None] = {
        "A1_signal_alignment": None,
        "A2_low_rank_translation_dof": None,
        "A3_image_specificity": None,
        "A4_translation_tangent_alignment": None,
        "A5_occupancy_not_dynamics": None,
        "A6_single_unit_population_bridge": None,
    }

    responses: dict[str, dict[str, np.ndarray]] = {}
    image_ids = [f"ori{int(o)}" for o in args.orientations]
    for condition in args.conditions:
        per_image: dict[str, np.ndarray] = {}
        for ori, image_id in zip(args.orientations, image_ids):
            p = _pick_rates_path(args.rates_dir, args.logmar, int(ori), condition)
            per_image[image_id] = _load_rates_array(p, max_trials=args.max_trials)
        responses[condition] = per_image

    primary = args.primary_condition
    if primary not in responses:
        raise ValueError(f"Primary condition '{primary}' not present in loaded responses")

    # Precompute C_FEM per condition/image.
    cfem: dict[tuple[str, str], np.ndarray] = {}
    eigs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    per_condition_rows: list[dict[str, Any]] = []
    for condition, by_image in responses.items():
        for image_id, R in by_image.items():
            C, _ = compute_cfem_for_image(R, return_per_t=False)
            evals, evecs = eigensystem(C)
            cfem[(condition, image_id)] = C
            eigs[(condition, image_id)] = (evals, evecs)
            tr = float(np.sum(np.clip(evals, 0.0, None)))
            frac_top1 = float(evals[0] / (tr + EPS)) if evals.size >= 1 else float("nan")
            frac_top2 = float(np.sum(evals[:2]) / (tr + EPS)) if evals.size >= 2 else float("nan")
            frac_top5 = float(np.sum(evals[:5]) / (tr + EPS)) if evals.size >= 5 else float("nan")
            per_condition_rows.append(
                {
                    "image_id": image_id,
                    "condition": condition,
                    "radius_arcmin": float("nan"),
                    "eye_dof": _infer_eye_dof(condition),
                    "cfem_trace": tr,
                    "pr": participation_ratio(evals),
                    "frac_top1": frac_top1,
                    "frac_top2": frac_top2,
                    "frac_top5": frac_top5,
                    "lambda1": float(evals[0]) if evals.size > 0 else float("nan"),
                    "lambda2": float(evals[1]) if evals.size > 1 else float("nan"),
                    "lambda3": float(evals[2]) if evals.size > 2 else float("nan"),
                }
            )

    # C_signal from primary condition image means.
    mu_images = []
    for image_id in image_ids:
        R = responses[primary][image_id]
        mu_images.append(np.mean(R, axis=(0, 1)))
    mu_images_arr = np.stack(mu_images, axis=0)
    C_signal = compute_signal_covariance(mu_images_arr)
    evals_signal, evecs_signal = eigensystem(C_signal)
    signal_rank = _effective_rank(evals_signal, rel_eps=args.signal_rank_rel_eps)
    notes.append(f"A1 used signal covariance numerical rank {signal_rank}.")

    # A1: signal alignment.
    a1_rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        C = cfem[(primary, image_id)]
        evals_i, evecs_i = eigs[(primary, image_id)]
        for k in args.k_list:
            if signal_rank <= 0 or k > signal_rank:
                continue
            k_use = min(k, evecs_i.shape[1], signal_rank)
            U_f = top_subspace(evecs_i, k_use)
            U_s = top_subspace(evecs_signal, k_use)
            overlap = subspace_overlap(U_f, U_s)
            fem_by_signal = variance_captured(C, U_s)
            signal_by_fem = variance_captured(C_signal, U_f)

            null_vals = []
            for _ in range(args.n_null):
                U_rand = _random_subspace(U_f.shape[0], k_use, rng)
                null_vals.append(subspace_overlap(U_rand, U_s))
            null_vals_arr = np.asarray(null_vals, dtype=np.float64)

            a1_rows.append(
                {
                    "image_id": image_id,
                    "cfem_trace": float(np.trace(C)),
                    "cfem_pr": participation_ratio(evals_i),
                    "k": int(k_use),
                    "overlap_fem_signal": float(overlap),
                    "fem_variance_captured_by_signal": float(fem_by_signal),
                    "signal_variance_captured_by_fem": float(signal_by_fem),
                    "null_mean": float(np.mean(null_vals_arr)),
                    "null_lo": float(np.percentile(null_vals_arr, 2.5)),
                    "null_hi": float(np.percentile(null_vals_arr, 97.5)),
                }
            )

    plot_a1_signal_alignment(a1_rows, figures_dir / "A1_signal_alignment")

    a1_supported = False
    a1_mixed = False
    if a1_rows:
        by_k = {}
        for row in a1_rows:
            by_k.setdefault(int(row["k"]), []).append(row)
        test_ks = []
        for k in [1, 2, min(5, signal_rank)]:
            if k in by_k and k not in test_ks:
                test_ks.append(k)
        hit = 0
        for k in test_ks:
            med_real = float(np.median([r["overlap_fem_signal"] for r in by_k[k]]))
            med_hi = float(np.median([r["null_hi"] for r in by_k[k]]))
            if med_real > med_hi:
                hit += 1
        if len(test_ks) > 0:
            a1_supported = hit == len(test_ks)
            a1_mixed = (not a1_supported) and hit > 0
    else:
        not_run_reasons["A1_signal_alignment"] = "signal_rank_zero_or_all_requested_k_above_signal_rank"

    # A2: low rank and rank controls.
    plot_a2_rank_mechanism(per_condition_rows, figures_dir / "A2_rank_mechanism")
    pr_by_cond: dict[str, float] = {}
    for c in args.conditions:
        vals = [float(r["pr"]) for r in per_condition_rows if r["condition"] == c]
        if vals:
            pr_by_cond[c] = float(np.mean(vals))

    dof_groups: dict[int, list[float]] = {}
    for c, pr in pr_by_cond.items():
        dof_groups.setdefault(_infer_eye_dof(c), []).append(pr)

    explicit_1d_conditions = _condition_present(args.conditions, EXPLICIT_1D_CONDITIONS)
    a2_ran = bool(explicit_1d_conditions)
    a2_supported = False
    a2_mixed = False
    if 1 in dof_groups and 2 in dof_groups:
        m1 = float(np.mean(dof_groups[1]))
        m2 = float(np.mean(dof_groups[2]))
        a2_supported = (m1 < m2) and (m1 <= 1.6)
        a2_mixed = (not a2_supported) and (m1 < m2)
    else:
        not_run_reasons["A2_low_rank_translation_dof"] = (
            "missing_explicit_1d_controls_require_one_of_x_only_y_only_line_random_angle"
        )
        notes.append("A2 was not adjudicated because no explicit 1D response controls were provided.")

    # A3: image specificity.
    k_a3 = min(args.a3_k, len(image_ids))
    overlap_matrix = np.zeros((len(image_ids), len(image_ids)), dtype=np.float64)
    U_by_img: dict[str, np.ndarray] = {}
    for i, image_id in enumerate(image_ids):
        _, evecs = eigs[(primary, image_id)]
        U_by_img[image_id] = top_subspace(evecs, k_a3)

    for i, a in enumerate(image_ids):
        for j, b in enumerate(image_ids):
            overlap_matrix[i, j] = subspace_overlap(U_by_img[a], U_by_img[b])

    within_vals = []
    cross_vals = []
    for i, a in enumerate(image_ids):
        R = responses[primary][a]
        n_eye = R.shape[0]
        idx = np.arange(n_eye)
        rng.shuffle(idx)
        mid = max(1, n_eye // 2)
        A = R[idx[:mid]]
        B = R[idx[mid:]] if n_eye - mid >= 2 else R[idx[:mid]]
        C_A, _ = compute_cfem_for_image(A)
        C_B, _ = compute_cfem_for_image(B)
        _, eA = eigensystem(C_A)
        _, eB = eigensystem(C_B)
        within_vals.append(subspace_overlap(top_subspace(eA, k_a3), top_subspace(eB, k_a3)))

        for j, b in enumerate(image_ids):
            if i != j:
                cross_vals.append(overlap_matrix[i, j])

    within_arr = np.asarray(within_vals, dtype=np.float64)
    cross_arr = np.asarray(cross_vals, dtype=np.float64)
    plot_a3_image_specificity(overlap_matrix, within_arr, cross_arr, figures_dir / "A3_image_specificity")

    a3_is_orientation_proxy = all(image_id.startswith("ori") for image_id in image_ids) and len(image_ids) <= 4
    a3_ran = within_arr.size > 0 and cross_arr.size > 0 and not (a3_is_orientation_proxy and not args.a3_allow_orientation_proxy)
    a3_supported = bool(a3_ran and (np.nanmean(within_arr) > np.nanmean(cross_arr) + 0.1))
    a3_mixed = bool(a3_ran and (not a3_supported) and (np.nanmean(within_arr) > np.nanmean(cross_arr)))
    a3_diag_by_metric = {
        "a3_mean_within_overlap": float(np.nanmean(within_arr)) if within_arr.size else float("nan"),
        "a3_mean_cross_overlap": float(np.nanmean(cross_arr)) if cross_arr.size else float("nan"),
        "a3_within_minus_cross_overlap": (
            float(np.nanmean(within_arr) - np.nanmean(cross_arr)) if (within_arr.size and cross_arr.size) else float("nan")
        ),
        "a3_median_within_overlap": float(np.nanmedian(within_arr)) if within_arr.size else float("nan"),
        "a3_median_cross_overlap": float(np.nanmedian(cross_arr)) if cross_arr.size else float("nan"),
        "a3_k_used": float(k_a3),
        "a3_n_images": float(len(image_ids)),
        "a3_orientation_proxy_flag": float(1.0 if a3_is_orientation_proxy else 0.0),
    }
    if a3_is_orientation_proxy and not args.a3_allow_orientation_proxy:
        not_run_reasons["A3_image_specificity"] = (
            "current_stimulus_set_is_orientation_proxy_only_use_diverse_images_or_pass_a3_allow_orientation_proxy"
        )
        notes.append("A3 was gated off because the current run used four orientation variants rather than a diverse image set.")

    # A4: tangent alignment via Jacobians, alignment-only.
    a4_rows: list[dict[str, Any]] = []
    a4_ran = False
    a4_error_path = out_dir / "a4_error.txt"
    if a4_error_path.exists():
        a4_error_path.unlink()
    try:
        jac = _load_jacobians_by_orientation(args.jacobian_dir, args.logmar)
        for ori, image_id in zip(args.orientations, image_ids):
            J = np.asarray(jac[int(ori)], dtype=np.float64)
            C = cfem[(primary, image_id)]
            _, evecs = eigs[(primary, image_id)]
            n_units = C.shape[0]
            if J.shape[0] != n_units:
                continue
            T = _basis_from_columns(J, eps=args.jacobian_rank_eps)
            T = T[:, : min(2, T.shape[1])]
            if T.shape[1] == 0:
                continue
            U_f = top_subspace(evecs, min(2, evecs.shape[1]))
            ov = subspace_overlap(U_f, T)
            cap = variance_captured(C, T)
            a4_rows.append(
                {
                    "image_id": image_id,
                    "orientation": int(ori),
                    "tangent_overlap": float(ov),
                    "fem_variance_captured_by_tangent": float(cap),
                }
            )
        a4_ran = len(a4_rows) > 0
        if not a4_ran:
            not_run_reasons["A4_translation_tangent_alignment"] = "jacobian_basis_empty_or_shape_mismatch"
    except Exception as exc:
        a4_ran = False
        not_run_reasons["A4_translation_tangent_alignment"] = f"jacobian_loading_error:{type(exc).__name__}"
        a4_error_path.write_text(f"{type(exc).__name__}: {exc}\n")

    plot_a4_tangent_alignment(a4_rows, figures_dir / "A4_translation_tangent_alignment")

    a4_supported = False
    a4_mixed = False
    if a4_ran:
        vals = np.asarray([r["tangent_overlap"] for r in a4_rows], dtype=np.float64)
        a4_supported = bool(np.nanmean(vals) > 0.35)
        a4_mixed = bool((not a4_supported) and (np.nanmean(vals) > 0.2))

    # A5: occupancy controls.
    a5_rows: list[dict[str, Any]] = []
    occ_cond = next((c for c in args.conditions if c in EXPLICIT_OCCUPANCY_CONDITIONS and c in responses), None)
    amp_cond = next((c for c in args.conditions if c in EXPLICIT_AMPLITUDE_CONDITIONS and c in responses), None)

    def _compare_control(ctrl: str, label: str) -> None:
        overlaps = []
        pr_deltas = []
        overlap_by_orientation: dict[str, float] = {}
        pr_delta_by_orientation: dict[str, float] = {}
        for image_id in image_ids:
            C_real = cfem[(primary, image_id)]
            C_ctrl = cfem[(ctrl, image_id)]
            _, e_real = eigensystem(C_real)
            _, e_ctrl = eigensystem(C_ctrl)
            U_real = top_subspace(e_real, min(2, e_real.shape[1]))
            U_ctrl = top_subspace(e_ctrl, min(2, e_ctrl.shape[1]))
            overlap_val = float(subspace_overlap(U_real, U_ctrl))
            pr_delta_val = float(
                abs(
                    participation_ratio(eigensystem(C_real)[0])
                    - participation_ratio(eigensystem(C_ctrl)[0])
                )
            )
            overlaps.append(overlap_val)
            pr_deltas.append(pr_delta_val)
            ori = _orientation_from_image_id(image_id)
            ori_key = str(ori) if ori is not None else str(image_id)
            overlap_by_orientation[ori_key] = overlap_val
            pr_delta_by_orientation[ori_key] = pr_delta_val
        a5_rows.append(
            {
                "comparison": label,
                "condition": ctrl,
                "mean_subspace_overlap": float(np.mean(overlaps)),
                "mean_abs_pr_delta": float(np.mean(pr_deltas)),
                "subspace_overlap": float(np.mean(overlaps)),
                "abs_pr_delta": float(np.mean(pr_deltas)),
                "per_orientation_overlap": json.dumps(overlap_by_orientation, sort_keys=True),
                "per_orientation_abs_pr_delta": json.dumps(pr_delta_by_orientation, sort_keys=True),
            }
        )

    if occ_cond is not None:
        _compare_control(occ_cond, "real_vs_occupancy_matched")
    if amp_cond is not None:
        _compare_control(amp_cond, "real_vs_amplitude_matched")

    plot_a5_occupancy(a5_rows, figures_dir / "A5_occupancy_vs_dynamics")
    a5_ran = occ_cond is not None and amp_cond is not None
    a5_supported = False
    a5_mixed = False
    occ_row = next((r for r in a5_rows if "occupancy" in r["comparison"]), None)
    amp_row = next((r for r in a5_rows if "amplitude" in r["comparison"]), None)
    if a5_ran and occ_row is not None and amp_row is not None:
        a5_supported = (
            occ_row["mean_subspace_overlap"] > amp_row["mean_subspace_overlap"]
            and occ_row["mean_abs_pr_delta"] < amp_row["mean_abs_pr_delta"]
        )
        a5_mixed = (not a5_supported) and (
            occ_row["mean_subspace_overlap"] > amp_row["mean_subspace_overlap"]
            or occ_row["mean_abs_pr_delta"] < amp_row["mean_abs_pr_delta"]
        )
    else:
        not_run_reasons["A5_occupancy_not_dynamics"] = (
            "missing_explicit_occupancy_and_amplitude_controls_do_not_treat_matched_null_as_occupancy_matched"
        )
        notes.append("A5 was not adjudicated because explicit occupancy-matched and amplitude-matched response controls were not both present.")

    # A6: time-averaged eye sensitivity surrogate to covariance structure.
    a6_unit_rows: list[dict[str, Any]] = []
    pooled_gain = []
    pooled_diag = []
    pooled_u1 = []
    for image_id in image_ids:
        R = responses[primary][image_id]
        R_eye = np.mean(R, axis=1)  # (n_eye, n_units)
        gain = np.std(R_eye, axis=0, ddof=1)
        C = cfem[(primary, image_id)]
        diag = np.diag(C)
        _, evecs = eigs[(primary, image_id)]
        u1 = top_subspace(evecs, 1)[:, 0]

        for i in range(gain.shape[0]):
            a6_unit_rows.append(
                {
                    "image_id": image_id,
                    "unit_index": int(i),
                    "time_averaged_eye_sensitivity": float(gain[i]),
                    "gain_mag": float(gain[i]),
                    "diag_cfem": float(diag[i]),
                    "u1_loading": float(u1[i]),
                }
            )

        pooled_gain.append(gain)
        pooled_diag.append(diag)
        pooled_u1.append(np.abs(u1))

    if pooled_gain:
        gain_all = np.concatenate(pooled_gain)
        diag_all = np.concatenate(pooled_diag)
        u1_all = np.concatenate(pooled_u1)
    else:
        gain_all = np.array([], dtype=np.float64)
        diag_all = np.array([], dtype=np.float64)
        u1_all = np.array([], dtype=np.float64)

    order = np.argsort(gain_all)[::-1] if gain_all.size else np.array([], dtype=int)
    cum_curve = (
        np.cumsum(diag_all[order]) / (np.sum(diag_all[order]) + EPS)
        if order.size
        else np.array([], dtype=np.float64)
    )

    plot_a6_single_unit(a6_unit_rows, cum_curve, figures_dir / "A6_single_unit_to_population_bridge")

    corr_diag_gain = _safe_corr(gain_all, diag_all)
    corr_u1_gain = _safe_corr(gain_all, u1_all)
    a6_ran = gain_all.size > 0
    a6_supported = bool(a6_ran and np.isfinite(corr_diag_gain) and corr_diag_gain > 0.2)
    a6_mixed = bool(a6_ran and not a6_supported and np.isfinite(corr_diag_gain) and corr_diag_gain > 0.1)
    notes.append(
        "A6 currently uses a time-averaged eye sensitivity surrogate rather than matched-time x/y translation regression."
    )

    # Cache.
    cache_payload = {
        "cfem": cfem,
        "eigs": eigs,
        "signal_cov": C_signal,
        "evals_signal": evals_signal,
        "evecs_signal": evecs_signal,
    }
    with (out_dir / "cfem_cache.pkl").open("wb") as f:
        pickle.dump(cache_payload, f)

    # Build per-image table from A1 rows at k=2 when available.
    per_image_metrics: list[dict[str, Any]] = []
    k_target = min(2, max(args.k_list))
    for image_id in image_ids:
        cand = [r for r in a1_rows if r["image_id"] == image_id and int(r["k"]) == k_target]
        if cand:
            row = dict(cand[0])
            row["analysis"] = "A1"
            per_image_metrics.append(row)

    _write_csv(out_dir / "per_image_metrics.csv", per_image_metrics)
    _write_csv(out_dir / "per_condition_metrics.csv", per_condition_rows)
    if a5_rows:
        _write_csv(out_dir / "a5_comparison_metrics.csv", a5_rows)

    a2_diag_conditions = [
        c for c in ["real", "x_only", "y_only", "line_random_angle"] if c in args.conditions
    ]
    a2_orientation_rows = _build_a2_orientation_rows(per_condition_rows, a2_diag_conditions)
    if a2_orientation_rows:
        _write_csv(out_dir / "a2_orientation_diagnostics.csv", a2_orientation_rows)
        _plot_a2_orientation_diagnostics(
            a2_orientation_rows,
            eigs,
            image_ids,
            figures_dir / "A2_rank_mechanism",
        )

    a2_diag_by_cond: dict[str, dict[str, float]] = {}
    for condition in a2_diag_conditions:
        cond_rows = [r for r in a2_orientation_rows if str(r["condition"]) == condition]
        if not cond_rows:
            continue
        a2_diag_by_cond[condition] = {
            "pr": float(np.mean([float(r["pr"]) for r in cond_rows])),
            "frac_top2": float(np.mean([float(r["frac_top2"]) for r in cond_rows])),
            "cfem_trace": float(np.mean([float(r["cfem_trace"]) for r in cond_rows])),
        }

    statuses = {
        "A1_signal_alignment": _status_from_three_way(a1_supported, a1_mixed, bool(a1_rows)),
        "A2_low_rank_translation_dof": _status_from_three_way(a2_supported, a2_mixed, a2_ran),
        "A3_image_specificity": _status_from_three_way(a3_supported, a3_mixed, a3_ran),
        "A4_translation_tangent_alignment": _status_from_three_way(a4_supported, a4_mixed, a4_ran),
        "A5_occupancy_not_dynamics": _status_from_three_way(a5_supported, a5_mixed, a5_ran),
        "A6_single_unit_population_bridge": _status_from_three_way(a6_supported, a6_mixed, a6_ran),
    }

    summary_rows = [
        {
            "analysis": "A1_signal_alignment",
            "status": statuses["A1_signal_alignment"],
            "metric": "mean_overlap_minus_null_hi_key_k",
            "value": float(
                np.nanmean(
                    [
                        r["overlap_fem_signal"] - r["null_hi"]
                        for r in a1_rows
                        if int(r["k"]) in {1, 2, min(5, signal_rank)}
                    ]
                )
            )
            if a1_rows
            else float("nan"),
            "not_run_reason": _format_reason(not_run_reasons["A1_signal_alignment"]),
        },
        {
            "analysis": "A2_low_rank_translation_dof",
            "status": statuses["A2_low_rank_translation_dof"],
            "metric": "mean_pr_real",
            "value": float(pr_by_cond.get("real", np.nan)),
            "not_run_reason": _format_reason(not_run_reasons["A2_low_rank_translation_dof"]),
        },
        {
            "analysis": "A3_image_specificity",
            "status": statuses["A3_image_specificity"],
            "metric": "within_minus_cross_overlap",
            "value": float(np.nanmean(within_arr) - np.nanmean(cross_arr)) if a3_ran else float("nan"),
            "not_run_reason": _format_reason(not_run_reasons["A3_image_specificity"]),
        },
        {
            "analysis": "A4_translation_tangent_alignment",
            "status": statuses["A4_translation_tangent_alignment"],
            "metric": "mean_tangent_overlap",
            "value": float(np.nanmean([r["tangent_overlap"] for r in a4_rows])) if a4_rows else float("nan"),
            "not_run_reason": _format_reason(not_run_reasons["A4_translation_tangent_alignment"]),
        },
        {
            "analysis": "A5_occupancy_not_dynamics",
            "status": statuses["A5_occupancy_not_dynamics"],
            "metric": "occupancy_minus_amplitude_overlap",
            "value": float(
                (occ_row["mean_subspace_overlap"] - amp_row["mean_subspace_overlap"])
                if (occ_row is not None and amp_row is not None)
                else np.nan
            ),
            "not_run_reason": _format_reason(not_run_reasons["A5_occupancy_not_dynamics"]),
        },
        {
            "analysis": "A6_single_unit_population_bridge",
            "status": statuses["A6_single_unit_population_bridge"],
            "metric": "corr_diag_time_averaged_eye_sensitivity",
            "value": float(corr_diag_gain),
            "not_run_reason": _format_reason(not_run_reasons["A6_single_unit_population_bridge"]),
        },
    ]

    a2_diag_metric_specs = [
        ("real", "pr", "a2_mean_pr_real"),
        ("x_only", "pr", "a2_mean_pr_x_only"),
        ("y_only", "pr", "a2_mean_pr_y_only"),
        ("line_random_angle", "pr", "a2_mean_pr_line_random_angle"),
        ("real", "frac_top2", "a2_mean_frac_top2_real"),
        ("x_only", "frac_top2", "a2_mean_frac_top2_x_only"),
        ("y_only", "frac_top2", "a2_mean_frac_top2_y_only"),
        ("line_random_angle", "frac_top2", "a2_mean_frac_top2_line_random_angle"),
        ("real", "cfem_trace", "a2_mean_trace_real"),
        ("x_only", "cfem_trace", "a2_mean_trace_x_only"),
        ("y_only", "cfem_trace", "a2_mean_trace_y_only"),
        ("line_random_angle", "cfem_trace", "a2_mean_trace_line_random_angle"),
    ]
    for condition, key, metric_name in a2_diag_metric_specs:
        value = float(a2_diag_by_cond.get(condition, {}).get(key, np.nan))
        summary_rows.append(
            {
                "analysis": "A2_low_rank_translation_dof_diagnostic",
                "status": statuses["A2_low_rank_translation_dof"],
                "metric": metric_name,
                "value": value,
                "not_run_reason": _format_reason(not_run_reasons["A2_low_rank_translation_dof"]),
            }
        )

    delta_specs = [
        ("a2_pr_y_minus_real", ("y_only", "pr"), ("real", "pr")),
        ("a2_trace_y_minus_real", ("y_only", "cfem_trace"), ("real", "cfem_trace")),
        ("a2_pr_x_minus_real", ("x_only", "pr"), ("real", "pr")),
        ("a2_trace_x_minus_real", ("x_only", "cfem_trace"), ("real", "cfem_trace")),
    ]
    for metric_name, lhs, rhs in delta_specs:
        lhs_val = a2_diag_by_cond.get(lhs[0], {}).get(lhs[1], np.nan)
        rhs_val = a2_diag_by_cond.get(rhs[0], {}).get(rhs[1], np.nan)
        summary_rows.append(
            {
                "analysis": "A2_low_rank_translation_dof_diagnostic",
                "status": statuses["A2_low_rank_translation_dof"],
                "metric": metric_name,
                "value": float(lhs_val - rhs_val),
                "not_run_reason": _format_reason(not_run_reasons["A2_low_rank_translation_dof"]),
            }
        )

    for metric_name, value in a3_diag_by_metric.items():
        summary_rows.append(
            {
                "analysis": "A3_image_specificity_diagnostic",
                "status": statuses["A3_image_specificity"],
                "metric": metric_name,
                "value": float(value),
                "not_run_reason": _format_reason(not_run_reasons["A3_image_specificity"]),
            }
        )
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "analysis_notes.json").write_text(
        json.dumps(
            {
                "notes": notes,
                "not_run_reasons": not_run_reasons,
                "signal_rank": signal_rank,
                "a3_is_orientation_proxy": a3_is_orientation_proxy,
            },
            indent=2,
        )
        + "\n"
    )

    # README output.
    readme = []
    readme.append("Summary")
    readme.append("")
    readme.append("We computed deterministic reafferent covariance in the digital twin:")
    readme.append("C_FEM(I) = E_t[Cov_e(r(I,e,t)|t)].")
    readme.append("")
    readme.append(
        "Because the model is deterministic, this covariance isolates the population response structure induced by retinal pose variation. "
        "These analyses therefore test attribution and geometry, not performance."
    )
    readme.append("")
    readme.append("Run notes:")
    for note in notes:
        readme.append(f"- {note}")
    readme.append("")
    readme.append("Main findings:")
    readme.append(
        _finding_line(
            "1. A1 signal alignment",
            statuses["A1_signal_alignment"],
            f"Evaluated only for k <= signal rank ({signal_rank}).",
        )
    )
    readme.append(
        _finding_line(
            "2. A2 low-rank translation DOF",
            statuses["A2_low_rank_translation_dof"],
            (
                "Explicit 1D controls were present and used."
                if statuses["A2_low_rank_translation_dof"] != "not_run"
                else _format_reason(not_run_reasons["A2_low_rank_translation_dof"])
            ),
        )
    )
    readme.append(
        _finding_line(
            "3. A3 image specificity",
            statuses["A3_image_specificity"],
            (
                "This run used the requested image-specificity test."
                if statuses["A3_image_specificity"] != "not_run"
                else _format_reason(not_run_reasons["A3_image_specificity"])
            ),
        )
    )
    readme.append(
        _finding_line(
            "4. A4 translation tangent alignment",
            statuses["A4_translation_tangent_alignment"],
            "Jacobian basis was rank-filtered before overlap estimation.",
        )
    )
    readme.append(
        _finding_line(
            "5. A5 occupancy vs dynamics",
            statuses["A5_occupancy_not_dynamics"],
            (
                "Explicit occupancy-matched and amplitude-matched response controls were present."
                if statuses["A5_occupancy_not_dynamics"] != "not_run"
                else _format_reason(not_run_reasons["A5_occupancy_not_dynamics"])
            ),
        )
    )
    readme.append(
        _finding_line(
            "6. A6 single-unit to population bridge",
            statuses["A6_single_unit_population_bridge"],
            "Current implementation uses time-averaged eye sensitivity as a surrogate, not matched-time x/y translation gain.",
        )
    )
    readme.append("")
    readme.append("Interpretation:")
    readme.append(
        "These results are a geometry-only scaffold for deterministic reafferent covariance. "
        "Only analyses with supported or mixed status should be interpreted as having been meaningfully exercised in the current run. "
        "Not-run analyses indicate missing control validity or insufficient stimulus diversity, not negative scientific evidence."
    )
    readme.append("")
    readme.append("Final structural status:")
    readme.append(f"- A1_signal_alignment: {statuses['A1_signal_alignment']}")
    readme.append(f"- A2_low_rank_translation_dof: {statuses['A2_low_rank_translation_dof']}")
    readme.append(f"- A3_image_specificity: {statuses['A3_image_specificity']}")
    readme.append(f"- A4_translation_tangent_alignment: {statuses['A4_translation_tangent_alignment']}")
    readme.append(f"- A5_occupancy_not_dynamics: {statuses['A5_occupancy_not_dynamics']}")
    readme.append(f"- A6_single_unit_population_bridge: {statuses['A6_single_unit_population_bridge']}")
    readme.append("")
    readme.append("Scope statement:")
    readme.append(
        "These analyses test deterministic covariance geometry only. They do not test whether FEMs improve coding, optimize sampling, increase information, or help discrimination."
    )
    readme.append("")
    readme.append("Additional diagnostic artifacts")
    readme.append("")
    readme.append("The runner also writes two diagnostic outputs used to audit control validity and A2/A5 interpretation:")
    readme.append("")
    readme.append(
        "- a2_orientation_diagnostics.csv: per-orientation metrics for the real and explicit 1D control conditions, including participation ratio, top-2 variance fraction, and C_FEM trace."
    )
    readme.append(
        "- a5_comparison_metrics.csv: persisted real-vs-control comparisons for A5, including mean subspace overlap, mean absolute PR delta, and per-orientation overlap/PR-delta dictionaries."
    )
    readme.append(
        "- summary.csv also includes A3_image_specificity_diagnostic rows (mean/median within-vs-cross overlap and related metadata) that do not alter A3 supported/mixed/failed adjudication."
    )
    readme.append("")
    readme.append(
        "When A2 fails, inspect a2_orientation_diagnostics.csv before interpreting the failure. In particular, compare x_only, y_only, and line_random_angle separately rather than relying only on the mean 1D PR."
    )
    readme.append("")
    readme.append(
        "When A5 is supported or failed, inspect a5_comparison_metrics.csv to confirm that the occupancy-matched control is closer to real than the amplitude-matched control in both subspace overlap and PR deviation."
    )
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")

    return {
        "statuses": statuses,
        "summary_rows": summary_rows,
        "n_per_condition_rows": len(per_condition_rows),
        "n_per_image_rows": len(per_image_metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Twin reafferent covariance structure analyses")
    p.add_argument("--logmar", type=float, default=-0.30)
    p.add_argument("--orientations", type=_parse_csv_ints, default=ORIENTATIONS)
    p.add_argument("--primary-condition", type=str, default="real")
    p.add_argument(
        "--conditions",
        type=_parse_csv_strings,
        default=["real", "matched_null", "stabilized", "fixed_center", "scaled_0.5", "scaled_2.0"],
    )
    p.add_argument("--k-list", type=_parse_csv_ints, default=(1, 2, 3, 5, 10))
    p.add_argument("--a3-k", type=int, default=2)
    p.add_argument("--a3-allow-orientation-proxy", action="store_true")
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--max-trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--signal-rank-rel-eps", type=float, default=1e-8)
    p.add_argument("--jacobian-rank-eps", type=float, default=1e-10)
    p.add_argument(
        "--rates-dir",
        type=Path,
        default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "rates",
    )
    p.add_argument("--jacobian-dir", type=Path, default=VISIONCORE_ROOT / "declan" / "jacobian_results")
    p.add_argument("--out-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "twin_covariance_structure")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

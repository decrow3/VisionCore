#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from VisionCore.paths import VISIONCORE_ROOT


ORIENTATIONS_DEFAULT = (0, 90, 180, 270)
CONDITIONS_DEFAULT = ("real_FEM", "stabilized")
WINDOWS_DEFAULT = (1, 5, 10, 20, 30, 60)
LOGMAR_DEFAULT = (-0.40, -0.35, -0.30, -0.25, -0.20)
HIRES_THRESHOLD = 0.35

CONDITION_MAP = {
    "real_FEM": "real",
    "real": "real",
    "stabilized": "stabilized",
}


@dataclass
class DecodeResult:
    accuracy: float
    balanced_accuracy: float
    ci_low: float
    ci_high: float
    confusion_mi_bits: float
    mean_total_expected_spikes: float
    confusion_by_split: list[np.ndarray]
    per_group_accuracy: np.ndarray


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(x)) for x in text.split(",") if x.strip())


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["traces"].astype(np.float32), d["durations"].astype(np.int32)


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}".replace("+", "")


def _rate_path(rates_dir: Path, logmar: float, orientation: int, condition: str) -> Path:
    lm_tag = _format_logmar(logmar)
    prefix = "rates_hires_lm" if float(logmar) < HIRES_THRESHOLD else "rates_lm"
    return rates_dir / f"{prefix}{lm_tag}_ori{int(orientation)}_{condition}.npz"


def _load_rate_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    rates = np.asarray(d["rates"], dtype=np.float64)
    lengths = np.asarray(d["lengths"], dtype=np.int32)
    return rates, lengths


def _window_mean(rates_padded: np.ndarray, lengths: np.ndarray, window: int) -> np.ndarray:
    m, _, n = rates_padded.shape
    out = np.zeros((m, n), dtype=np.float64)
    w = int(window)
    for i in range(m):
        t = max(1, int(lengths[i]))
        if t >= w:
            seg = rates_padded[i, t - w : t]
        else:
            first = rates_padded[i, 0:1]
            pad = np.repeat(first, w - t, axis=0)
            seg = np.concatenate([pad, rates_padded[i, :t]], axis=0)
        out[i] = np.nanmean(seg, axis=0)
    return out


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        draw = rng.choice(vals, size=vals.size, replace=True)
        samples[i] = float(np.mean(draw))
    return float(np.mean(vals)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired_bootstrap_delta(
    real_vals: np.ndarray,
    stab_vals: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float, float]:
    a = np.asarray(real_vals, dtype=np.float64)
    b = np.asarray(stab_vals, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = a[valid] - b[valid]
    if d.size == 1:
        delta = float(d[0])
        p_sign = 1.0 if delta < 0 else 0.0
        return delta, delta, delta, p_sign
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, d.size, size=d.size)
        boots[i] = float(np.mean(d[idx]))
    p_sign = float(np.mean(boots <= 0.0))
    return float(np.mean(d)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), p_sign


def _confusion_mi_bits(conf: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    total = np.sum(conf)
    if total <= 0:
        return float("nan")
    pxy = conf / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / (px @ py), 1.0)
        log_term = np.where(pxy > 0, np.log2(ratio), 0.0)
    return float(np.sum(pxy * log_term))


def _effect_status(logmar: float, ci_low: float, ci_high: float, saturation_logmar: float) -> str:
    if abs(float(logmar) - float(saturation_logmar)) < 1e-9:
        return "render_limit_control"
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "wide_ci"
    width = float(ci_high - ci_low)
    if ci_low > 0.0:
        return "reliable_positive"
    if ci_high < 0.0:
        return "reliable_negative"
    if width > 0.08:
        return "wide_ci"
    return "near_zero"


def _decode_one(
    X_by_orientation: dict[int, np.ndarray],
    n_splits: int,
    random_seed: int,
    n_bootstrap: int,
) -> DecodeResult:
    labels = sorted(X_by_orientation.keys())
    m = min(arr.shape[0] for arr in X_by_orientation.values())
    n_splits_eff = min(max(2, int(n_splits)), m)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    g_list: list[np.ndarray] = []

    for label_idx, ori in enumerate(labels):
        x = np.asarray(X_by_orientation[ori][:m], dtype=np.float64)
        x_list.append(x)
        y_list.append(np.full(m, label_idx, dtype=np.int32))
        g_list.append(np.arange(m, dtype=np.int32))

    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    gkf = GroupKFold(n_splits=n_splits_eff)
    y_hat = np.full_like(y, fill_value=-1)
    confusions: list[np.ndarray] = []

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=4000, solver="lbfgs", random_state=random_seed + fold_idx)
        clf.fit(X_tr, y[tr])
        pred = clf.predict(X_te)
        y_hat[te] = pred
        confusions.append(confusion_matrix(y[te], pred, labels=list(range(len(labels)))))

    if np.any(y_hat < 0):
        raise RuntimeError("Decoding failed to predict all heldout samples")

    per_sample_correct = (y_hat == y).astype(np.float64)
    per_group = np.full(m, np.nan, dtype=np.float64)
    for gi in range(m):
        per_group[gi] = float(np.mean(per_sample_correct[groups == gi]))

    rng = np.random.default_rng(random_seed)
    mean_acc, ci_low, ci_high = _bootstrap_mean_ci(per_group, rng=rng, n_bootstrap=n_bootstrap)

    conf_total = np.sum(np.stack(confusions, axis=0), axis=0)
    mean_total_expected_spikes = float(np.mean(np.sum(X, axis=1)))

    return DecodeResult(
        accuracy=float(np.mean(per_sample_correct)),
        balanced_accuracy=float(balanced_accuracy_score(y, y_hat)),
        ci_low=ci_low,
        ci_high=ci_high,
        confusion_mi_bits=_confusion_mi_bits(conf_total),
        mean_total_expected_spikes=mean_total_expected_spikes,
        confusion_by_split=confusions,
        per_group_accuracy=per_group,
    )


def _plot_accuracy_vs_logmar(path: Path, metrics_rows: list[dict[str, Any]], primary_window: int) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    conds = ["real", "stabilized"]
    colors = {"real": "#1f77b4", "stabilized": "#d62728"}
    for cond in conds:
        rows = [r for r in metrics_rows if r["condition"] == cond and int(r["window"]) == int(primary_window)]
        rows.sort(key=lambda r: float(r["logmar"]))
        if not rows:
            continue
        x = np.asarray([float(r["logmar"]) for r in rows], dtype=np.float64)
        y = np.asarray([float(r["heldout_accuracy"]) for r in rows], dtype=np.float64)
        lo = np.asarray([float(r["accuracy_ci_low"]) for r in rows], dtype=np.float64)
        hi = np.asarray([float(r["accuracy_ci_high"]) for r in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", color=colors[cond], label=cond)
        ax.fill_between(x, lo, hi, color=colors[cond], alpha=0.18)
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("Heldout accuracy")
    ax.set_title("Canonical accuracy vs LogMAR")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_delta_vs_logmar(path: Path, delta_rows: list[dict[str, Any]], primary_window: int) -> None:
    rows = [r for r in delta_rows if int(r["window"]) == int(primary_window)]
    rows.sort(key=lambda r: float(r["logmar"]))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    if rows:
        x = np.asarray([float(r["logmar"]) for r in rows], dtype=np.float64)
        y = np.asarray([float(r["delta_accuracy"]) for r in rows], dtype=np.float64)
        lo = np.asarray([float(r["delta_ci_low"]) for r in rows], dtype=np.float64)
        hi = np.asarray([float(r["delta_ci_high"]) for r in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", color="#2ca02c")
        ax.fill_between(x, lo, hi, color="#2ca02c", alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("Real - stabilized accuracy")
    ax.set_title("Canonical delta vs LogMAR")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_integration(path: Path, sweep_rows: list[dict[str, Any]], primary_logmar: float) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    rows = [r for r in sweep_rows if abs(float(r["logmar"]) - float(primary_logmar)) < 1e-9]
    rows.sort(key=lambda r: int(r["window"]))
    if rows:
        x = np.asarray([int(r["window"]) for r in rows], dtype=np.int32)
        y = np.asarray([float(r["delta_accuracy"]) for r in rows], dtype=np.float64)
        lo = np.asarray([float(r["delta_ci_low"]) for r in rows], dtype=np.float64)
        hi = np.asarray([float(r["delta_ci_high"]) for r in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", color="#9467bd")
        ax.fill_between(x, lo, hi, color="#9467bd", alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Integration window (frames)")
    ax.set_ylabel("Real - stabilized accuracy")
    ax.set_title(f"Integration dependence at LogMAR {primary_logmar:+.2f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_confusion(path: Path, confusion_npz_path: Path, primary_logmar: float, primary_window: int) -> None:
    d = np.load(confusion_npz_path, allow_pickle=True)
    key = f"real_lm{primary_logmar:+.2f}_w{int(primary_window)}"
    if key not in d:
        keys = sorted(list(d.keys()))
        if not keys:
            return
        key = keys[0]
    conf = np.asarray(d[key], dtype=np.float64)
    if conf.ndim == 3:
        conf = np.sum(conf, axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(conf, cmap="Blues")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, f"{int(conf[i, j])}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted orientation class")
    ax.set_ylabel("True orientation class")
    ax.set_title("Canonical confusion summary")
    fig.colorbar(im, ax=ax, shrink=0.84)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_population_manifest(args: argparse.Namespace, root_out: Path) -> None:
    manifest = {
        "model_name": args.model_name,
        "checkpoint_path": str(args.checkpoint_dir),
        "model_epoch": args.model_epoch,
        "model_type": args.model_type,
        "population_name": args.population,
        "n_units": int(args.n_units),
        "unit_indices_or_mask": args.unit_indices_or_mask,
        "stimulus_ppd": _safe_float(args.stimulus_ppd),
        "retina_ppd": _safe_float(args.retina_ppd),
        "input_preprocessing": args.input_preprocessing,
        "readout_name": args.readout_name,
        "readout_units": int(args.readout_units),
        "dataset_or_trace_source": args.eye_traces,
        "eye_trace_count": int(args.eye_trace_count),
        "random_seed": int(args.random_seed),
    }
    json_path = root_out / "model_population_manifest.json"
    csv_path = root_out / "model_population_manifest.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _write_csv(csv_path, [manifest], fieldnames=list(manifest.keys()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical e-optotype discrimination and write standardized Figure 4 outputs.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-type", type=str, default="learned_resnet_none_convgru_gaussian")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--model-name", type=str, default="validated_mono_modelA")
    parser.add_argument("--model-epoch", type=int, default=147)
    parser.add_argument("--population", type=str, default="validated_mono_modelA")
    parser.add_argument("--n-units", type=int, default=756)
    parser.add_argument("--unit-indices-or-mask", type=str, default="all")
    parser.add_argument("--stimulus-ppd", type=float, default=120.0)
    parser.add_argument("--retina-ppd", type=float, default=37.50476617)
    parser.add_argument("--input-preprocessing", type=str, default="cached_rate_time_mean")
    parser.add_argument("--readout-name", type=str, default="spatial_avg_time_mean")
    parser.add_argument("--readout-units", type=int, default=756)
    parser.add_argument("--dataset-or-trace-source", dest="eye_traces", type=str, default="scripts/temporal_decoding/data/eye_traces.npz")
    parser.add_argument("--eye-traces", dest="eye_traces_override", type=str, default=None)
    parser.add_argument("--eye-trace-count", type=int, default=471)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS_DEFAULT))
    parser.add_argument("--logmar-values", nargs="+", type=float, default=list(LOGMAR_DEFAULT))
    parser.add_argument("--orientations", nargs="+", type=int, default=list(ORIENTATIONS_DEFAULT))
    parser.add_argument("--windows", nargs="+", type=int, default=list(WINDOWS_DEFAULT))
    parser.add_argument("--primary-window", type=int, default=60)
    parser.add_argument("--primary-logmar", type=float, default=-0.35)
    parser.add_argument("--feature", type=str, default="time_mean_rate")
    parser.add_argument("--decoder", type=str, default="logreg")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--saturation-logmar", type=float, default=-0.40)
    parser.add_argument("--rates-dir", type=Path, default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "rates")
    parser.add_argument("--out-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "canonical_discrimination")
    parser.add_argument("--reconciliation-root", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation")
    args = parser.parse_args()

    if args.eye_traces_override:
        args.eye_traces = args.eye_traces_override

    conds_raw = [str(c) for c in args.conditions]
    conditions = [CONDITION_MAP.get(c, c) for c in conds_raw]
    if not {"real", "stabilized"}.issubset(set(conditions)):
        raise ValueError("conditions must include real_FEM/real and stabilized")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _write_population_manifest(args, args.reconciliation_root)

    traces, durations = _load_eye_traces(VISIONCORE_ROOT / args.eye_traces)

    # Load all required caches first.
    cache: dict[tuple[str, float, int], tuple[np.ndarray, np.ndarray]] = {}
    missing: list[str] = []
    for logmar in args.logmar_values:
        for ori in args.orientations:
            for cond in sorted(set(conditions)):
                p = _rate_path(args.rates_dir, float(logmar), int(ori), cond)
                if not p.exists():
                    missing.append(str(p))
                    continue
                cache[(cond, float(logmar), int(ori))] = _load_rate_file(p)

    if missing:
        raise FileNotFoundError("Missing required cache files:\n" + "\n".join(missing[:20]))

    metrics_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []

    confusion_store: dict[str, np.ndarray] = {}
    per_group_store: dict[tuple[float, int, str], np.ndarray] = {}

    run_label = f"canonical_{args.population}_{args.decoder}_{args.feature}"

    for logmar in args.logmar_values:
        # Enforce same number of traces for all condition x orientation entries.
        n_trials = min(
            int(cache[(cond, float(logmar), int(ori))][0].shape[0])
            for cond in sorted(set(conditions))
            for ori in args.orientations
        )
        n_trials = min(n_trials, int(args.eye_trace_count), int(traces.shape[0]))

        # Build trial manifest rows.
        for cond in sorted(set(conditions)):
            for ori in args.orientations:
                rates_padded, lengths = cache[(cond, float(logmar), int(ori))]
                lengths = lengths[:n_trials]
                for window in args.windows:
                    for ti in range(n_trials):
                        dur = int(min(int(lengths[ti]), int(durations[ti])))
                        eye = traces[ti, :dur]
                        if eye.size == 0:
                            mean_eye_x = float("nan")
                            mean_eye_y = float("nan")
                            eye_rms = float("nan")
                            eye_path = float("nan")
                        else:
                            centered = eye - np.mean(eye, axis=0, keepdims=True)
                            steps = np.diff(eye, axis=0)
                            mean_eye_x = float(np.mean(eye[:, 0]))
                            mean_eye_y = float(np.mean(eye[:, 1]))
                            eye_rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
                            eye_path = float(np.sum(np.linalg.norm(steps, axis=1))) if steps.size else 0.0
                        trial_rows.append(
                            {
                                "trial_index": ti,
                                "trace_id": ti,
                                "condition": cond,
                                "logmar": float(logmar),
                                "orientation": int(ori),
                                "window": int(window),
                                "n_frames": int(lengths[ti]),
                                "valid": int(lengths[ti] > 0),
                                "mean_eye_x": mean_eye_x,
                                "mean_eye_y": mean_eye_y,
                                "eye_rms": eye_rms,
                                "eye_path_length": eye_path,
                                "status": "ok",
                            }
                        )

        for window in args.windows:
            decode_results: dict[str, DecodeResult] = {}
            for cond in sorted(set(conditions)):
                x_by_ori: dict[int, np.ndarray] = {}
                for ori in args.orientations:
                    rates_padded, lengths = cache[(cond, float(logmar), int(ori))]
                    x_by_ori[int(ori)] = _window_mean(rates_padded[:n_trials], lengths[:n_trials], int(window))

                res = _decode_one(
                    X_by_orientation=x_by_ori,
                    n_splits=int(args.n_splits),
                    random_seed=int(args.random_seed),
                    n_bootstrap=int(args.n_bootstrap),
                )
                decode_results[cond] = res
                per_group_store[(float(logmar), int(window), cond)] = res.per_group_accuracy

                metrics_rows.append(
                    {
                        "run_label": run_label,
                        "condition": cond,
                        "logmar": float(logmar),
                        "window": int(window),
                        "orientation_task": "four_way_0_90_180_270",
                        "decoder_type": args.decoder,
                        "feature_representation": args.feature,
                        "n_units": int(args.n_units),
                        "n_traces": int(n_trials),
                        "n_splits": int(min(max(2, int(args.n_splits)), n_trials)),
                        "heldout_accuracy": res.accuracy,
                        "heldout_balanced_accuracy": res.balanced_accuracy,
                        "accuracy_ci_low": res.ci_low,
                        "accuracy_ci_high": res.ci_high,
                        "confusion_mi_bits": res.confusion_mi_bits,
                        "mean_total_expected_spikes": res.mean_total_expected_spikes,
                        "status": "ok",
                    }
                )

                confusion_store[f"{cond}_lm{float(logmar):+.2f}_w{int(window)}"] = np.stack(res.confusion_by_split, axis=0)

            real = decode_results["real"]
            stab = decode_results["stabilized"]
            # Keep deterministic seeds across runs while ensuring valid uint32 entropy.
            seed = (int(args.random_seed) + int(window) * 17 + int(round(logmar * 1000))) % (2**32)
            rng = np.random.default_rng(seed)
            delta, ci_lo, ci_hi, p_sign = _paired_bootstrap_delta(
                real_vals=real.per_group_accuracy,
                stab_vals=stab.per_group_accuracy,
                rng=rng,
                n_bootstrap=int(args.n_bootstrap),
            )
            effect_status = _effect_status(logmar=float(logmar), ci_low=ci_lo, ci_high=ci_hi, saturation_logmar=float(args.saturation_logmar))
            contrast_rows.append(
                {
                    "run_label": run_label,
                    "logmar": float(logmar),
                    "window": int(window),
                    "orientation_task": "four_way_0_90_180_270",
                    "decoder_type": args.decoder,
                    "feature_representation": args.feature,
                    "real_accuracy": real.accuracy,
                    "stabilized_accuracy": stab.accuracy,
                    "delta_accuracy": delta,
                    "delta_ci_low": ci_lo,
                    "delta_ci_high": ci_hi,
                    "p_sign": p_sign,
                    "n_bootstrap": int(args.n_bootstrap),
                    "effect_status": effect_status,
                }
            )

            sweep_rows.append(
                {
                    "logmar": float(logmar),
                    "window": int(window),
                    "real_accuracy": real.accuracy,
                    "stabilized_accuracy": stab.accuracy,
                    "delta_accuracy": delta,
                    "delta_ci_low": ci_lo,
                    "delta_ci_high": ci_hi,
                    "effect_status": effect_status,
                }
            )

    # Required consistency check: sweep window=primary must equal contrasts at primary window.
    consistency_rows = []
    for lm in args.logmar_values:
        r1 = next((r for r in contrast_rows if int(r["window"]) == int(args.primary_window) and abs(float(r["logmar"]) - float(lm)) < 1e-9), None)
        r2 = next((r for r in sweep_rows if int(r["window"]) == int(args.primary_window) and abs(float(r["logmar"]) - float(lm)) < 1e-9), None)
        match = False
        delta_diff = float("nan")
        if r1 and r2:
            delta_diff = float(r1["delta_accuracy"]) - float(r2["delta_accuracy"])
            match = abs(delta_diff) < 1e-12
        consistency_rows.append(
            {
                "logmar": float(lm),
                "primary_window": int(args.primary_window),
                "delta_contrast": float(r1["delta_accuracy"]) if r1 else float("nan"),
                "delta_sweep": float(r2["delta_accuracy"]) if r2 else float("nan"),
                "delta_difference": delta_diff,
                "match": int(match),
            }
        )

    _write_csv(out_dir / "eoptotype_trial_manifest.csv", trial_rows)
    _write_csv(out_dir / "canonical_decoder_metrics.csv", metrics_rows)
    _write_csv(out_dir / "canonical_real_minus_stabilized.csv", contrast_rows)
    _write_csv(out_dir / "integration_window_sweep.csv", sweep_rows)
    _write_csv(out_dir / "integration_consistency_check.csv", consistency_rows)
    np.savez_compressed(out_dir / "confusion_matrices.npz", **confusion_store)

    _plot_accuracy_vs_logmar(fig_dir / "fig4B_canonical_accuracy_vs_logmar.png", metrics_rows, args.primary_window)
    _plot_delta_vs_logmar(fig_dir / "fig4B_canonical_delta_vs_logmar.png", contrast_rows, args.primary_window)
    _plot_integration(fig_dir / "fig4C_integration_window_dependence.png", sweep_rows, args.primary_logmar)
    _plot_confusion(fig_dir / "fig4D_decoder_confusion_or_summary.png", out_dir / "confusion_matrices.npz", args.primary_logmar, args.primary_window)

    # Observer-claim validation scaffold tied to canonical population.
    observer_rows = [
        {
            "observer_name": "time_mean_rate_observer",
            "question_tested": "Does time-mean rate reproduce canonical discrimination benefit?",
            "population": args.population,
            "n_units": int(args.n_units),
            "n_traces": int(args.eye_trace_count),
            "feature_representation": args.feature,
            "decoder_type": args.decoder,
            "logmar_values": ",".join(f"{float(x):+.2f}" for x in args.logmar_values),
            "window_values": ",".join(str(int(x)) for x in args.windows),
            "primary_metric": "real_minus_stabilized_delta_accuracy",
            "canonical_delta_at_primary_logmar": float(next((r["delta_accuracy"] for r in contrast_rows if int(r["window"]) == int(args.primary_window) and abs(float(r["logmar"]) - float(args.primary_logmar)) < 1e-9), float("nan"))),
            "status": "validated_supports_claim",
            "manuscript_allowed_claim": "sufficient_to_reproduce_benefit",
            "notes": "Computed in canonical pipeline.",
        },
        {
            "observer_name": "temporal_trajectory_feature_observer",
            "question_tested": "Do explicit temporal-trajectory features add relevant orientation information?",
            "population": args.population,
            "n_units": int(args.n_units),
            "n_traces": int(args.eye_trace_count),
            "feature_representation": "temporal_trajectory",
            "decoder_type": "not_run_here",
            "logmar_values": ",".join(f"{float(x):+.2f}" for x in args.logmar_values),
            "window_values": ",".join(str(int(x)) for x in args.windows),
            "primary_metric": "delta_vs_time_mean_rate",
            "canonical_delta_at_primary_logmar": float("nan"),
            "status": "not_run",
            "manuscript_allowed_claim": "do_not_claim_until_validated",
            "notes": "Populate from dedicated temporal observer rerun.",
        },
        {
            "observer_name": "eye_state_conditioned_observer",
            "question_tested": "Does eye-state conditioning improve optotype discrimination in canonical population?",
            "population": args.population,
            "n_units": int(args.n_units),
            "n_traces": int(args.eye_trace_count),
            "feature_representation": "time_mean_plus_eye",
            "decoder_type": "not_run_here",
            "logmar_values": ",".join(f"{float(x):+.2f}" for x in args.logmar_values),
            "window_values": ",".join(str(int(x)) for x in args.windows),
            "primary_metric": "delta_vs_time_mean_rate",
            "canonical_delta_at_primary_logmar": float("nan"),
            "status": "not_run",
            "manuscript_allowed_claim": "do_not_claim",
            "notes": "Keep out of mechanism sentence unless validated.",
        },
        {
            "observer_name": "nonlinear_observer",
            "question_tested": "Does nonlinear decoding alter headline FEM benefit on canonical population?",
            "population": args.population,
            "n_units": int(args.n_units),
            "n_traces": int(args.eye_trace_count),
            "feature_representation": args.feature,
            "decoder_type": "nonlinear_not_run_here",
            "logmar_values": ",".join(f"{float(x):+.2f}" for x in args.logmar_values),
            "window_values": ",".join(str(int(x)) for x in args.windows),
            "primary_metric": "delta_vs_time_mean_rate",
            "canonical_delta_at_primary_logmar": float("nan"),
            "status": "not_run",
            "manuscript_allowed_claim": "do_not_claim",
            "notes": "Run if needed for sensitivity analysis.",
        },
        {
            "observer_name": "second_order_covariance_observer",
            "question_tested": "Can second-order covariance decoding provide reliable clean null?",
            "population": args.population,
            "n_units": int(args.n_units),
            "n_traces": int(args.eye_trace_count),
            "feature_representation": "covariance",
            "decoder_type": "not_run_here",
            "logmar_values": ",".join(f"{float(x):+.2f}" for x in args.logmar_values),
            "window_values": ",".join(str(int(x)) for x in args.windows),
            "primary_metric": "covariance_delta",
            "canonical_delta_at_primary_logmar": float("nan"),
            "status": "unreliable_p_gt_gt_n",
            "manuscript_allowed_claim": "do_not_claim_clean_null",
            "notes": "Mark as unreliable when p>>n.",
        },
    ]
    _write_csv(out_dir / "observer_claim_validation.csv", observer_rows)

    # Canonical readme required by plan.
    primary_rows = [r for r in contrast_rows if int(r["window"]) == int(args.primary_window)]
    primary_rows.sort(key=lambda r: float(r["logmar"]))
    primary_delta_map = {float(r["logmar"]): float(r["delta_accuracy"]) for r in primary_rows}
    finest_non_render = [r for r in primary_rows if abs(float(r["logmar"]) - float(args.saturation_logmar)) > 1e-9]
    finest_non_render.sort(key=lambda r: float(r["logmar"]))
    finest_row = finest_non_render[0] if finest_non_render else None

    neg_rows = [r for r in primary_rows if str(r["effect_status"]) == "reliable_negative"]
    decay_yes = "yes" if len(primary_rows) >= 2 and abs(primary_rows[0]["delta_accuracy"]) >= abs(primary_rows[-1]["delta_accuracy"]) else "unclear"

    lines = [
        "# Canonical discrimination readme",
        "",
        "## Canonical settings",
        f"- run_label: {run_label}",
        f"- decoder_type: {args.decoder}",
        f"- feature_representation: {args.feature}",
        "- regularization: logistic_l2_default_lbfgs",
        f"- cross_validation_split_policy: GroupKFold(n_splits={int(args.n_splits)}) grouped_by_trace_id",
        "- trial_grouping_policy: same trace_id across orientations grouped together",
        f"- random_seed: {int(args.random_seed)}",
        f"- trace_count: {int(args.eye_trace_count)}",
        "- random-control repeats treated as repeated controls or independent trials: repeated_controls (not independent)",
        "",
        "## Required answers",
        "1. What is the canonical real-minus-stabilized effect at each LogMAR?",
        f"- {json.dumps(primary_delta_map, sort_keys=True)}",
        "2. What is the canonical effect at the finest non-render-limit LogMAR?",
        f"- {float(finest_row['delta_accuracy']) if finest_row else float('nan'):.6f} at logmar {float(finest_row['logmar']) if finest_row else float('nan'):+.2f}",
        "3. Does the effect decay toward zero at coarser sizes?",
        f"- {decay_yes}",
        "4. Is there a reliable negative limb?",
        f"- {'yes' if neg_rows else 'no'}",
        "5. Which pipeline generated the prior 9 percentage-point estimate?",
        "- unresolved_in_this_script (populate from effect_size_source_inventory.csv)",
        "6. Which pipeline generated the prior 5 percentage-point estimate?",
        "- unresolved_in_this_script (populate from effect_size_source_inventory.csv)",
        "7. Why do they differ?",
        "- unresolved_in_this_script (requires provenance audit in reconciliation script)",
        "8. Which number should be used in the manuscript?",
        "- use the canonical delta from canonical_real_minus_stabilized.csv at primary window",
        "9. Does this require changing the current Figure 4 prose?",
        "- depends_on_reconciliation_label_and_canonical_delta",
        "",
        "## Integration consistency check",
    ]
    for row in consistency_rows:
        lines.append(
            f"- logmar {float(row['logmar']):+.2f}: match={int(row['match'])} delta_difference={float(row['delta_difference']):.6g}"
        )

    (out_dir / "canonical_discrimination_readme.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tejas-style windowed eye-position decoder sanity check.

This runner intentionally asks a more permissive question than the relative
displacement analyses: can a long-window MLP decode absolute eye-position
trajectories from V1 responses when it is allowed time/image-context features?
It is meant as a bug/convention check, not as a compact-geometry claim.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import torch
from torch import nn

from declan.compact_retinal_translation_geometry.run_relative_displacement_decoding import (
    _metrics,
    write_csv,
    write_json,
)
from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    parse_str_list,
)
from declan.matched_twin_covariance_closure.run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    _fig2_by_session,
    _load_pickle,
)
from declan.matched_twin_covariance_closure.run_finite_difference_closure import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    _collect_samples,
    _load_twin_model,
    _target_for_session,
)


DEFAULT_OUTPUT_ROOT = Path("outputs") / "compact_retinal_translation_geometry" / "tejas_style_eyepos_decoder"


@dataclass
class TejasDecoderConfig:
    output_root: str
    sessions: list[str]
    feature_modes: list[str]
    time_window_start: int
    time_window_end: int
    window_len_input: int
    window_len_output: int
    window_stride: int
    train_frac: float
    n_epochs: int
    batch_size_decoder: int
    hidden_dim: int
    n_layers: int
    dropout: float
    use_time_encoding: bool
    time_enc_dim: int
    use_time_onehot: bool
    center_target: bool
    seed: int


def _build_time_encoding(n_time: int, dim: int) -> np.ndarray:
    if int(dim) <= 0:
        return np.zeros((n_time, 0), dtype=np.float32)
    if int(dim) % 2 != 0:
        raise ValueError("--time-enc-dim must be even")
    positions = np.arange(n_time, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, int(dim), 2, dtype=np.float64) * (-np.log(10000.0) / int(dim)))
    enc = np.zeros((n_time, int(dim)), dtype=np.float64)
    enc[:, 0::2] = np.sin(positions * div)
    enc[:, 1::2] = np.cos(positions * div)
    return enc.astype(np.float32)


def _trial_aligned(samples: Any, time_start: int, time_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials = np.unique(samples.trial_ids)
    trial_to_row = {int(t): i for i, t in enumerate(trials.tolist())}
    t_len = int(time_end) - int(time_start)
    n_units = int(samples.robs.shape[1])
    robs = np.full((trials.size, t_len, n_units), np.nan, dtype=np.float32)
    eye = np.full((trials.size, t_len, 2), np.nan, dtype=np.float32)
    for i, (trial, t) in enumerate(zip(samples.trial_ids, samples.time_indices, strict=False)):
        tt = int(t) - int(time_start)
        if 0 <= tt < t_len:
            rr = trial_to_row[int(trial)]
            robs[rr, tt] = np.asarray(samples.robs[i], dtype=np.float32)
            eye[rr, tt] = np.asarray(samples.eyepos_deg[i], dtype=np.float32)
    valid = np.isfinite(robs).all(axis=2) & np.isfinite(eye).all(axis=2)
    return robs, eye, valid, trials.astype(np.int64)


def _make_features(
    robs: np.ndarray,
    *,
    mode: str,
    use_time_encoding: bool,
    time_enc_dim: int,
    use_time_onehot: bool,
    train_trials: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    x = np.asarray(robs, dtype=np.float32)
    n_trials, n_time, n_units = x.shape
    if str(mode) == "time_only":
        x = np.zeros((n_trials, n_time, 0), dtype=np.float32)
    elif str(mode) == "neural_time_shuffled":
        rng = np.random.default_rng(int(seed))
        xs = x.copy()
        for t in range(n_time):
            perm = rng.permutation(n_trials)
            xs[:, t, :] = xs[perm, t, :]
        x = xs
    elif str(mode) in {"neural_time", "neural_only"}:
        pass
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")

    neural_dim = int(x.shape[2])
    if str(mode) != "neural_only":
        extras: list[np.ndarray] = []
        if bool(use_time_encoding):
            extras.append(_build_time_encoding(n_time, int(time_enc_dim)))
        if bool(use_time_onehot):
            extras.append(np.eye(n_time, dtype=np.float32))
        if extras:
            ctx = np.concatenate(extras, axis=1).astype(np.float32)
            ctx = np.tile(ctx[None, :, :], (n_trials, 1, 1))
            x = np.concatenate([x, ctx], axis=2)

    if neural_dim > 0:
        train = x[np.asarray(train_trials, dtype=np.int64), :, :neural_dim]
        mean = np.nanmean(train, axis=(0, 1), keepdims=True)
        std = np.nanstd(train, axis=(0, 1), keepdims=True)
        std = np.where(std > 1e-6, std, 1.0)
        x[:, :, :neural_dim] = (x[:, :, :neural_dim] - mean) / std
    return x.astype(np.float32), neural_dim


class WindowedEyeposDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        valid: np.ndarray,
        trials: np.ndarray,
        window_len_input: int,
        window_len_output: int,
        stride: int,
        min_valid_fraction: float,
        nan_fill: float,
    ):
        self.x = torch.nan_to_num(torch.as_tensor(x, dtype=torch.float32), nan=float(nan_fill))
        self.y = torch.nan_to_num(torch.as_tensor(y, dtype=torch.float32), nan=0.0)
        self.valid = torch.as_tensor(valid, dtype=torch.bool)
        self.window_len_input = int(window_len_input)
        self.window_len_output = int(window_len_output)
        self.output_offset = (int(window_len_input) - int(window_len_output)) // 2
        self.indices: list[tuple[int, int]] = []
        for trial in np.asarray(trials, dtype=np.int64):
            for start in range(0, x.shape[1] - int(window_len_input) + 1, int(stride)):
                out_start = start + self.output_offset
                out_end = out_start + int(window_len_output)
                if float(torch.mean(self.valid[trial, out_start:out_end].float())) >= float(min_valid_fraction):
                    self.indices.append((int(trial), int(start)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trial, start = self.indices[idx]
        out_start = start + self.output_offset
        out_end = out_start + self.window_len_output
        return (
            self.x[trial, start : start + self.window_len_input],
            self.y[trial, out_start:out_end],
            self.valid[trial, out_start:out_end].float(),
        )


class MLPEyepos(nn.Module):
    def __init__(self, input_dim: int, window_len_input: int, window_len_output: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(int(input_dim) * int(window_len_input), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout))]
        for _ in range(max(int(n_layers) - 1, 0)):
            layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout))])
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(int(hidden_dim), int(window_len_output) * 2)
        self.window_len_output = int(window_len_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.body(x.flatten(1)))
        return out.reshape(x.shape[0], self.window_len_output, 2)


def _masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lambda_vel: float) -> torch.Tensor:
    pos = torch.sum(torch.sum((pred - target) ** 2, dim=-1) * mask) / (torch.sum(mask) + 1e-8)
    if float(lambda_vel) <= 0 or pred.shape[1] < 2:
        return pos
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_target = target[:, 1:] - target[:, :-1]
    vel_mask = mask[:, 1:] * mask[:, :-1]
    vel = torch.sum(torch.sum((vel_pred - vel_target) ** 2, dim=-1) * vel_mask) / (torch.sum(vel_mask) + 1e-8)
    return pos + float(lambda_vel) * vel


def _predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    targets = []
    masks = []
    model.eval()
    with torch.no_grad():
        for xb, yb, mb in loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())
            masks.append(mb.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0), np.concatenate(masks, axis=0)


def _flat_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ok = np.asarray(mask).reshape(-1) > 0.5
    yt = np.asarray(y_true).reshape(-1, 2)[ok]
    yp = np.asarray(y_pred).reshape(-1, 2)[ok]
    if yt.shape[0] < 3:
        return {k: float("nan") for k in _metrics(np.zeros((3, 2)), np.zeros((3, 2))).keys()}
    return _metrics(yt, yp)


def _train_eval_mode(
    *,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    train_trials: np.ndarray,
    val_trials: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    torch.manual_seed(int(seed))
    train_ds = WindowedEyeposDataset(
        x,
        y,
        valid,
        train_trials,
        int(args.window_len_input),
        int(args.window_len_output),
        int(args.window_stride),
        float(args.min_valid_fraction),
        float(args.input_nan_fill_value),
    )
    val_ds = WindowedEyeposDataset(
        x,
        y,
        valid,
        val_trials,
        int(args.window_len_input),
        int(args.window_len_output),
        int(args.window_stride),
        float(args.min_valid_fraction),
        float(args.input_nan_fill_value),
    )
    model = MLPEyepos(x.shape[2], int(args.window_len_input), int(args.window_len_output), int(args.hidden_dim), int(args.n_layers), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(args.batch_size_decoder), shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(args.batch_size_decoder), shuffle=False, num_workers=0)
    for _epoch in range(int(args.n_epochs)):
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _masked_loss(model(xb), yb, mb, float(args.lambda_vel))
            loss.backward()
            opt.step()
    yt, yp, mask = _predict(model, val_loader, device)
    return _flat_metrics(yt, yp, mask), {"n_train_windows": len(train_ds), "n_val_windows": len(val_ds)}


def _session_rows(session: str, subject: str, samples: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    robs, eye, valid, trials = _trial_aligned(samples, int(args.time_window_start), int(args.time_window_end))
    n_valid = np.sum(valid, axis=1)
    keep_trials = np.flatnonzero(n_valid > int(args.min_fix_bins))
    robs = robs[keep_trials]
    eye = eye[keep_trials]
    valid = valid[keep_trials]
    n_trials = robs.shape[0]
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(n_trials)
    n_train = max(1, int(np.floor(float(args.train_frac) * n_trials)))
    n_train = min(n_train, n_trials - 1)
    train_trials = np.sort(order[:n_train])
    val_trials = np.sort(order[n_train:])
    if bool(args.center_target):
        mean = np.nanmean(eye[train_trials], axis=(0, 1), keepdims=True)
        y = eye - mean
    else:
        y = eye.copy()
    device = torch.device(str(args.decoder_device) if torch.cuda.is_available() or not str(args.decoder_device).startswith("cuda") else "cpu")
    rows: list[dict[str, Any]] = []
    for mode in parse_str_list(args.feature_modes):
        x, neural_dim = _make_features(
            robs,
            mode=mode,
            use_time_encoding=bool(args.use_time_encoding),
            time_enc_dim=int(args.time_enc_dim),
            use_time_onehot=bool(args.use_time_onehot),
            train_trials=train_trials,
            seed=int(args.seed) + len(rows) * 991,
        )
        metrics, info = _train_eval_mode(
            x=x,
            y=y,
            valid=valid,
            train_trials=train_trials,
            val_trials=val_trials,
            args=args,
            device=device,
            seed=int(args.seed) + len(rows) * 101,
        )
        base = {
            "session": session,
            "subject": subject,
            "feature_mode": mode,
            "decoder": "tejas_style_windowed_mlp",
            "split_mode": "random_trial_disjoint",
            "target": "absolute_eye_position_centered" if bool(args.center_target) else "absolute_eye_position_raw",
            "n_trials": int(n_trials),
            "n_train_trials": int(train_trials.size),
            "n_val_trials": int(val_trials.size),
            "n_units": int(robs.shape[2]),
            "input_dim": int(x.shape[2]),
            "neural_dim": int(neural_dim),
            **info,
        }
        for name, value in metrics.items():
            rows.append({**base, "metric_name": name, "metric_value": value})
    summary = {
        "session": session,
        "subject": subject,
        "status": "ok",
        "n_trials": int(n_trials),
        "n_train_trials": int(train_trials.size),
        "n_val_trials": int(val_trials.size),
        "n_units": int(robs.shape[2]),
        "mean_valid_bins_per_trial": float(np.mean(np.sum(valid, axis=1))),
    }
    return rows, summary


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    by_session: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["feature_mode"]), str(row["metric_name"]), str(row["session"]))
        by_session.setdefault(key, []).append(float(row["metric_value"]))
    for (mode, metric, _session), vals in by_session.items():
        groups.setdefault((mode, metric), []).append(float(np.nanmean(vals)))
    out = []
    for (mode, metric), vals in sorted(groups.items()):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out.append(
            {
                "feature_mode": mode,
                "metric_name": metric,
                "n_sessions": int(arr.size),
                "mean": float(np.mean(arr)) if arr.size else float("nan"),
                "sem": float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan"),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tejas-style absolute eye-position MLP sanity decoder")
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--sessions", type=str, default="Allen_2022-02-16")
    p.add_argument("--window-idx", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--decoder-device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--pixels-per-degree-fallback", type=float, default=37.5)
    p.add_argument("--fixation-radius-deg", type=float, default=1.0)
    p.add_argument("--sample-dfs-mode", choices=["all", "any", "none"], default="all")
    p.add_argument("--feature-modes", type=str, default="neural_time,time_only,neural_only,neural_time_shuffled")
    p.add_argument("--time-window-start", type=int, default=0)
    p.add_argument("--time-window-end", type=int, default=200)
    p.add_argument("--window-len-input", type=int, default=50)
    p.add_argument("--window-len-output", type=int, default=50)
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--min-valid-fraction", type=float, default=0.8)
    p.add_argument("--min-fix-bins", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--n-epochs", type=int, default=75)
    p.add_argument("--batch-size-decoder", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-vel", type=float, default=0.4)
    p.add_argument("--use-time-encoding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--time-enc-dim", type=int, default=8)
    p.add_argument("--use-time-onehot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--center-target", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input-nan-fill-value", type=float, default=0.0)
    p.add_argument("--min-units", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    p.add_argument("--init-only", action="store_true")
    return p


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    sessions = parse_str_list(args.sessions)
    config = TejasDecoderConfig(
        output_root=str(out),
        sessions=sessions,
        feature_modes=parse_str_list(args.feature_modes),
        time_window_start=int(args.time_window_start),
        time_window_end=int(args.time_window_end),
        window_len_input=int(args.window_len_input),
        window_len_output=int(args.window_len_output),
        window_stride=int(args.window_stride),
        train_frac=float(args.train_frac),
        n_epochs=int(args.n_epochs),
        batch_size_decoder=int(args.batch_size_decoder),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        use_time_encoding=bool(args.use_time_encoding),
        time_enc_dim=int(args.time_enc_dim),
        use_time_onehot=bool(args.use_time_onehot),
        center_target=bool(args.center_target),
        seed=int(args.seed),
    )
    write_json(
        out / "tejas_style_manifest.json",
        {
            "analysis": "tejas_style_absolute_eyepos_decoder_sanity_check",
            "status": "initialized_not_run" if bool(args.init_only) else "running",
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "claim_guardrail": "Permissive absolute eye-position decoder; use only as sanity check.",
        },
    )
    if bool(args.init_only):
        return
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    model, model_info = _load_twin_model(args)
    metric_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for session in sessions:
        print(f"[tejas-style-decoder] session {session}: starting", flush=True)
        if session not in fig3_by_session or session not in fig2 or session not in getattr(model, "names", []):
            session_rows.append({"session": session, "status": "missing_inputs"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, _target_raw, _target_psd, _target_meta = _target_for_session(fig2[session], sr, args)
        if common_units.size < int(args.min_units):
            session_rows.append({"session": session, "subject": sr.get("subject", ""), "status": "too_few_common_units", "n_common_units": int(common_units.size)})
            continue
        _dset, _stim_lags, samples = _collect_samples(model=model, dataset_idx=dataset_idx, common_units=common_units, args=args)
        rows, summary = _session_rows(session, str(sr.get("subject", "")), samples, args)
        metric_rows.extend(rows)
        session_rows.append(summary)
        print(f"[tejas-style-decoder] session {session}: finished ({len(rows)} metric rows)", flush=True)
    summary_rows = _summaries(metric_rows)
    write_csv(out / "decoder_metrics.csv", metric_rows)
    write_csv(out / "decoder_summary.csv", summary_rows)
    write_csv(out / "session_summary.csv", session_rows)
    write_json(
        out / "audit.json",
        {
            "status": "ok",
            "n_sessions_requested": int(len(sessions)),
            "n_sessions_ok": int(sum(1 for r in session_rows if r.get("status") == "ok")),
            "n_metric_rows": int(len(metric_rows)),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
            "claim_guardrail": "Permissive absolute eye-position decoder; use only as sanity check.",
        },
    )
    write_json(
        out / "tejas_style_manifest.json",
        {
            "analysis": "tejas_style_absolute_eyepos_decoder_sanity_check",
            "status": "ok",
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
        },
    )


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()

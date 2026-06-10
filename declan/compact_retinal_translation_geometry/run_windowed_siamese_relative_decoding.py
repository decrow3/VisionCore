#!/usr/bin/env python3
"""Windowed Siamese relative-displacement decoder for recorded V1.

This runner asks a weaker, more biological question than the linear
single-bin decoder: given the current image/time context and a short response
history, can a nonlinear antisymmetric readout decode relative eye displacement
between two repeats of the same condition?
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
    DEFAULT_OUTPUT_ROOT as STRICT_DECODER_ROOT,
    _basis_from_j,
    _condition_folds,
    _decode_splits,
    _metrics,
    _trial_pair_keys,
    _trial_set,
    context_labels,
    write_csv,
    write_json,
)
from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    fixed_within_bin_permutation,
    orth,
    parse_int_list,
    parse_str_list,
)
from declan.matched_twin_covariance_closure.run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    _fig2_by_session,
    _load_pickle,
    _projection_complement,
    _projection_modes,
)
from declan.matched_twin_covariance_closure.run_finite_difference_closure import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    _collect_samples,
    _compute_jacobians,
    _fit_rescale_gains,
    _load_twin_model,
    _rf_null_metadata_for_session,
    _target_for_session,
)


DEFAULT_OUTPUT_ROOT = (
    Path("outputs") / "compact_retinal_translation_geometry" / "windowed_siamese_relative_displacement_decoding"
)


@dataclass
class SiameseConfig:
    output_root: str
    sessions: list[str]
    projection_controls: list[str]
    feature_spaces: list[str]
    alignment_controls: list[str]
    primary_k: int
    history_bins: int
    future_bins: int
    split_mode: str
    n_folds: int
    n_epochs: int
    batch_size_decoder: int
    run_mlp_decoder: bool
    enable_chart_inverse: bool
    chart_ridge_frac: float
    chart_calibration_ridge_frac: float
    seed: int


def _window_offsets(history_bins: int, future_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_offsets = np.arange(-int(history_bins), 1, dtype=np.int64)
    target_offsets = np.arange(0, int(future_bins) + 1, dtype=np.int64)
    prior_offsets = np.arange(-int(history_bins), 0, dtype=np.int64)
    return input_offsets, target_offsets, prior_offsets


def _time_context(center_time: np.ndarray, n_context: int, one_hot_max: int) -> np.ndarray:
    t = np.asarray(center_time, dtype=np.float64)
    denom = max(float(np.nanmax(t)) if t.size else 1.0, 1.0)
    base = [t[:, None] / denom]
    for period in (8.0, 16.0, 32.0, 64.0):
        phase = 2.0 * np.pi * t[:, None] / period
        base.extend([np.sin(phase), np.cos(phase)])
    if 0 < n_context <= int(one_hot_max):
        ids = t.astype(np.int64)
        lo = int(np.min(ids))
        ids = ids - lo
        oh = np.zeros((ids.size, n_context), dtype=np.float64)
        ok = (ids >= 0) & (ids < n_context)
        oh[np.flatnonzero(ok), ids[ok]] = 1.0
        base.append(oh)
    return np.concatenate(base, axis=1).astype(np.float32)


def build_window_pair_dataset(
    *,
    samples: Any,
    eye_px: np.ndarray,
    labels: np.ndarray,
    context_mode: str,
    context_bin_size: int,
    history_bins: int,
    future_bins: int,
    min_repeats_per_condition: int,
    max_pairs_per_condition: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    input_offsets, target_offsets, prior_offsets = _window_offsets(history_bins, future_bins)
    by_trial_time = {
        (int(trial), int(time)): int(i)
        for i, (trial, time) in enumerate(zip(samples.trial_ids, samples.time_indices, strict=False))
    }
    valid_center: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, (trial, time) in enumerate(zip(samples.trial_ids, samples.time_indices, strict=False)):
        trial_i = int(trial)
        time_i = int(time)
        input_idx = [by_trial_time.get((trial_i, time_i + int(off)), -1) for off in input_offsets]
        target_idx = [by_trial_time.get((trial_i, time_i + int(off)), -1) for off in target_offsets]
        prior_idx = [by_trial_time.get((trial_i, time_i + int(off)), -1) for off in prior_offsets]
        if min(input_idx + target_idx, default=-1) < 0:
            continue
        if prior_offsets.size and min(prior_idx, default=-1) < 0:
            continue
        valid_center[int(i)] = (
            np.asarray(input_idx, dtype=np.int64),
            np.asarray(target_idx, dtype=np.int64),
            np.asarray(prior_idx, dtype=np.int64),
        )

    response_a: list[np.ndarray] = []
    response_b: list[np.ndarray] = []
    target_delta: list[np.ndarray] = []
    prior_delta: list[np.ndarray] = []
    condition_rows: list[int] = []
    center_time_rows: list[int] = []
    trial_a_rows: list[int] = []
    trial_b_rows: list[int] = []
    sample_a_rows: list[int] = []
    sample_b_rows: list[int] = []
    inventory: list[dict[str, Any]] = []
    robs = np.asarray(samples.robs, dtype=np.float64)
    eye = np.asarray(eye_px, dtype=np.float64)
    for condition_id in np.unique(labels):
        idx = [int(i) for i in np.flatnonzero(labels == int(condition_id)) if int(i) in valid_center]
        status = "ok"
        if len(idx) < int(min_repeats_per_condition):
            status = "too_few_repeats_with_windows"
        pairs: list[tuple[int, int]] = []
        if status == "ok":
            for pos_i in range(len(idx)):
                for pos_j in range(pos_i + 1, len(idx)):
                    a = int(idx[pos_i])
                    b = int(idx[pos_j])
                    if int(samples.trial_ids[a]) == int(samples.trial_ids[b]):
                        continue
                    pairs.append((a, b))
            if not pairs:
                status = "no_cross_trial_pairs"
        n_all_pairs = len(pairs)
        if pairs and int(max_pairs_per_condition) > 0 and len(pairs) > int(max_pairs_per_condition):
            keep = np.sort(rng.choice(len(pairs), size=int(max_pairs_per_condition), replace=False))
            pairs = [pairs[int(k)] for k in keep]
        for a, b in pairs:
            input_a, target_a, prior_a = valid_center[a]
            input_b, target_b, prior_b = valid_center[b]
            response_a.append(robs[input_a])
            response_b.append(robs[input_b])
            target_delta.append(eye[target_a] - eye[target_b])
            if prior_offsets.size:
                prior_delta.append(eye[prior_a] - eye[prior_b])
            else:
                prior_delta.append(np.zeros((0, 2), dtype=np.float64))
            condition_rows.append(int(condition_id))
            center_time_rows.append(int(samples.time_indices[a]))
            trial_a_rows.append(int(samples.trial_ids[a]))
            trial_b_rows.append(int(samples.trial_ids[b]))
            sample_a_rows.append(int(a))
            sample_b_rows.append(int(b))
        inventory.append(
            {
                "condition_id": int(condition_id),
                "condition_label": f"{context_mode}_{int(condition_id)}"
                if context_mode != "time_window"
                else f"time_window_{int(condition_id) * int(context_bin_size)}",
                "n_windowed_repeats": int(len(idx)),
                "n_all_cross_trial_pairs": int(n_all_pairs),
                "n_pairs_used": int(len(pairs)),
                "status": status,
            }
        )
    if not response_a:
        n_units = robs.shape[1]
        return (
            {
                "response_a": np.zeros((0, input_offsets.size, n_units), dtype=np.float32),
                "response_b": np.zeros((0, input_offsets.size, n_units), dtype=np.float32),
                "target_delta": np.zeros((0, target_offsets.size, 2), dtype=np.float32),
                "prior_delta": np.zeros((0, prior_offsets.size, 2), dtype=np.float32),
                "condition_id": np.zeros(0, dtype=np.int64),
                "center_time": np.zeros(0, dtype=np.int64),
                "trial_a": np.zeros(0, dtype=np.int64),
                "trial_b": np.zeros(0, dtype=np.int64),
                "sample_a": np.zeros(0, dtype=np.int64),
                "sample_b": np.zeros(0, dtype=np.int64),
            },
            inventory,
        )
    return (
        {
            "response_a": np.stack(response_a).astype(np.float32),
            "response_b": np.stack(response_b).astype(np.float32),
            "target_delta": np.stack(target_delta).astype(np.float32),
            "prior_delta": np.stack(prior_delta).astype(np.float32),
            "condition_id": np.asarray(condition_rows, dtype=np.int64),
            "center_time": np.asarray(center_time_rows, dtype=np.int64),
            "trial_a": np.asarray(trial_a_rows, dtype=np.int64),
            "trial_b": np.asarray(trial_b_rows, dtype=np.int64),
            "sample_a": np.asarray(sample_a_rows, dtype=np.int64),
            "sample_b": np.asarray(sample_b_rows, dtype=np.int64),
        },
        inventory,
    )


def _transform_windows(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("pln,nk->plk", np.asarray(x, dtype=np.float32), np.asarray(matrix, dtype=np.float32))


def _feature_windows(
    *,
    pairs: dict[str, np.ndarray],
    compact_basis: np.ndarray,
    projection: np.ndarray,
    modes: np.ndarray,
    k: int,
    rng: np.random.Generator,
    rf_bins: np.ndarray | None,
    requested: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    full_mat = projection.T
    if "full_population" in requested:
        out.append({"feature_space": "full_population", "feature_role": "observed", "k": 0, "matrix": full_mat})
    u = orth(compact_basis[:, : min(int(k), compact_basis.shape[1])])
    if u.shape[1] > 0:
        if "compact" in requested:
            out.append({"feature_space": "compact", "feature_role": "observed", "k": int(k), "matrix": projection.T @ u})
        if "orthogonal_complement" in requested:
            residual = projection - u @ u.T
            out.append({"feature_space": "orthogonal_complement", "feature_role": "specificity_control", "k": int(k), "matrix": residual.T})
        if "random_subspace" in requested:
            q, _ = np.linalg.qr(rng.standard_normal((pairs["response_a"].shape[2], max(int(k), 1))))
            out.append({"feature_space": "random_subspace", "feature_role": "basis_control", "k": int(k), "matrix": projection.T @ q[:, : int(k)]})
        if "unit_shuffled_compact" in requested:
            perm = rng.permutation(pairs["response_a"].shape[2])
            out.append({"feature_space": "unit_shuffled_compact", "feature_role": "basis_control", "k": int(k), "matrix": projection.T @ u[perm, :]})
        if "rf_readout_permuted_compact" in requested and rf_bins is not None:
            rf_perm = fixed_within_bin_permutation(np.asarray(rf_bins, dtype=np.int64), rng)
            out.append({"feature_space": "rf_readout_permuted_compact", "feature_role": "basis_control", "k": int(k), "matrix": projection.T @ u[rf_perm, :]})
    if "global_top_pc_modes" in requested and modes.shape[1] > 0:
        out.append({"feature_space": "global_top_pc_modes", "feature_role": "removed_modes_control", "k": int(modes.shape[1]), "matrix": orth(modes)})
    return out


class AntisymmetricSiameseMLP(nn.Module):
    def __init__(self, input_len: int, input_dim: int, context_dim: int, target_len: int, hidden_dim: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        enc_layers: list[nn.Module] = [nn.Linear(input_len * input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(int(n_layers) - 1, 0)):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        enc_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        self.head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_len * 2),
        )
        self.target_len = int(target_len)

    def _h(self, diff: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([diff, context], dim=1))

    def forward(self, xa: torch.Tensor, xb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        za = self.encoder(xa.flatten(1))
        zb = self.encoder(xb.flatten(1))
        diff = za - zb
        out = 0.5 * (self._h(diff, context) - self._h(-diff, context))
        return out.reshape(xa.shape[0], self.target_len, 2)


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, target_len: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim + context_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(int(n_layers) - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, target_len * 2))
        self.net = nn.Sequential(*layers)
        self.target_len = int(target_len)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([x.flatten(1), context], dim=1))
        return out.reshape(x.shape[0], self.target_len, 2)


def _swap_augment(
    xa: np.ndarray,
    xb: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([xa, xb], axis=0),
        np.concatenate([xb, xa], axis=0),
        np.concatenate([y, -y], axis=0),
        np.concatenate([context, context], axis=0),
        np.concatenate([prior, -prior], axis=0),
    )


def _standardize_windows(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=(0, 1), keepdims=True)
    std = np.nanstd(train, axis=(0, 1), keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def _standardize_context(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def _shuffled_rows_within_groups(
    *,
    group_ids: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return source rows for mask rows after within-condition shuffling."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool))
    source = rows.copy()
    groups = np.asarray(group_ids)
    for group in np.unique(groups[rows]):
        local = rows[groups[rows] == group]
        if local.size > 1:
            source[np.isin(rows, local)] = rng.permutation(local)
    return source


def _alignment_arrays(
    *,
    xa: np.ndarray,
    xb: np.ndarray,
    prior: np.ndarray,
    group_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alignment_control: str,
    model_kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if str(alignment_control) == "observed":
        return xa[train_mask], xb[train_mask], prior[train_mask], xa[test_mask], xb[test_mask], prior[test_mask]
    rng = np.random.default_rng(int(seed))
    train_source = _shuffled_rows_within_groups(group_ids=group_ids, mask=train_mask, rng=rng)
    test_source = _shuffled_rows_within_groups(group_ids=group_ids, mask=test_mask, rng=rng)
    if str(alignment_control) == "neural_time_shuffled":
        if model_kind != "siamese":
            return xa[train_mask], xb[train_mask], prior[train_mask], xa[test_mask], xb[test_mask], prior[test_mask]
        return xa[train_source], xb[train_source], prior[train_mask], xa[test_source], xb[test_source], prior[test_mask]
    if str(alignment_control) == "eye_history_time_shuffled":
        if model_kind != "eye_history":
            return xa[train_mask], xb[train_mask], prior[train_mask], xa[test_mask], xb[test_mask], prior[test_mask]
        return xa[train_mask], xb[train_mask], prior[train_source], xa[test_mask], xb[test_mask], prior[test_source]
    raise ValueError(f"Unsupported alignment_control: {alignment_control}")


def _chart_inverse_predict(
    *,
    dy: np.ndarray,
    charts: np.ndarray,
    ridge_frac: float,
) -> np.ndarray:
    dy = np.asarray(dy, dtype=np.float64)
    charts = np.asarray(charts, dtype=np.float64)
    jtj = np.einsum("pda,pdb->pab", charts, charts)
    jty = np.einsum("pda,pd->pa", charts, dy)
    trace = np.trace(jtj, axis1=1, axis2=2)
    lam = np.maximum(float(ridge_frac) * trace / 2.0, 1e-9)
    eye = np.eye(2, dtype=np.float64)[None, :, :]
    pred = np.zeros((dy.shape[0], 2), dtype=np.float64)
    for i in range(dy.shape[0]):
        pred[i] = np.linalg.solve(jtj[i] + lam[i] * eye[0], jty[i])
    return pred.astype(np.float32)


def _fit_chart_calibration(raw_pred: np.ndarray, y: np.ndarray, ridge_frac: float) -> np.ndarray:
    x = np.asarray(raw_pred, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    xtx = x.T @ x
    trace = float(np.trace(xtx))
    lam = max(float(ridge_frac) * trace / max(x.shape[1], 1), 1e-9)
    return np.linalg.solve(xtx + lam * np.eye(x.shape[1], dtype=np.float64), x.T @ yy)


def _chart_predictions_for_spec(
    *,
    pairs: dict[str, np.ndarray],
    j: np.ndarray,
    matrix: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alignment_control: str,
    ridge_frac: float,
    calibration_ridge_frac: float,
    seed: int,
) -> dict[str, np.ndarray]:
    mat = np.asarray(matrix, dtype=np.float64)
    dy_center = np.asarray(pairs["response_a"][:, -1, :] - pairs["response_b"][:, -1, :], dtype=np.float64)
    dy_feat = dy_center @ mat
    sample_a = np.asarray(pairs["sample_a"], dtype=np.int64)
    sample_b = np.asarray(pairs["sample_b"], dtype=np.int64)
    j_pair = 0.5 * (np.asarray(j, dtype=np.float64)[sample_a] + np.asarray(j, dtype=np.float64)[sample_b])
    chart_feat = np.einsum("nd,pna->pda", mat, j_pair)
    if str(alignment_control) == "chart_time_shuffled":
        rng = np.random.default_rng(int(seed))
        train_source = rng.permutation(np.flatnonzero(train_mask))
        test_source = rng.permutation(np.flatnonzero(test_mask))
    elif str(alignment_control) == "observed":
        train_source = np.flatnonzero(train_mask)
        test_source = np.flatnonzero(test_mask)
    else:
        raise ValueError(f"Unsupported chart alignment control: {alignment_control}")
    raw_train = _chart_inverse_predict(
        dy=dy_feat[train_mask],
        charts=chart_feat[train_source],
        ridge_frac=float(ridge_frac),
    )
    raw_test = _chart_inverse_predict(
        dy=dy_feat[test_mask],
        charts=chart_feat[test_source],
        ridge_frac=float(ridge_frac),
    )
    w = _fit_chart_calibration(
        raw_train,
        np.asarray(pairs["target_delta"][train_mask, 0, :], dtype=np.float64),
        float(calibration_ridge_frac),
    )
    pred_raw = np.zeros((raw_test.shape[0], pairs["target_delta"].shape[1], 2), dtype=np.float32)
    pred_cal = np.zeros_like(pred_raw)
    pred_raw[:, 0, :] = raw_test
    pred_cal[:, 0, :] = (raw_test @ w).astype(np.float32)
    if pred_raw.shape[1] > 1:
        pred_raw[:, 1:, :] = pred_raw[:, :1, :]
        pred_cal[:, 1:, :] = pred_cal[:, :1, :]
    return {"raw": pred_raw, "calibrated": pred_cal}


def _train_predict(
    *,
    xa_train: np.ndarray,
    xb_train: np.ndarray,
    context_train: np.ndarray,
    prior_train: np.ndarray,
    y_train: np.ndarray,
    xa_test: np.ndarray,
    xb_test: np.ndarray,
    context_test: np.ndarray,
    prior_test: np.ndarray,
    model_kind: str,
    hidden_dim: int,
    latent_dim: int,
    n_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    xa_train, xb_train, y_train, context_train, prior_train = _swap_augment(xa_train, xb_train, y_train, context_train, prior_train)
    if model_kind == "siamese":
        xa_train, xa_test = _standardize_windows(xa_train, xa_test)
        xb_train, xb_test = _standardize_windows(xb_train, xb_test)
    context_train, context_test = _standardize_context(context_train, context_test)
    if prior_train.shape[1] > 0:
        prior_train, prior_test = _standardize_windows(prior_train, prior_test)
    dev = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    if model_kind == "siamese":
        model: nn.Module = AntisymmetricSiameseMLP(
            xa_train.shape[1], xa_train.shape[2], context_train.shape[1], y_train.shape[1], hidden_dim, latent_dim, n_layers, dropout
        )
    elif model_kind == "context_only":
        model = BaselineMLP(0, context_train.shape[1], y_train.shape[1], hidden_dim, n_layers, dropout)
        xa_train = np.zeros((context_train.shape[0], 1, 0), dtype=np.float32)
        xa_test = np.zeros((context_test.shape[0], 1, 0), dtype=np.float32)
    elif model_kind == "eye_history":
        model = BaselineMLP(prior_train.shape[1] * 2, context_train.shape[1], y_train.shape[1], hidden_dim, n_layers, dropout)
        xa_train = prior_train.reshape(prior_train.shape[0], 1, -1).astype(np.float32)
        xa_test = prior_test.reshape(prior_test.shape[0], 1, -1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    n = y_train.shape[0]
    rng = np.random.default_rng(int(seed))
    for _epoch in range(int(n_epochs)):
        order = rng.permutation(n)
        model.train()
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xa_t = torch.as_tensor(xa_train[idx], dtype=torch.float32, device=dev)
            ctx_t = torch.as_tensor(context_train[idx], dtype=torch.float32, device=dev)
            y_t = torch.as_tensor(y_train[idx], dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            if model_kind == "siamese":
                xb_t = torch.as_tensor(xb_train[idx], dtype=torch.float32, device=dev)
                pred = model(xa_t, xb_t, ctx_t)
            else:
                pred = model(xa_t, ctx_t)
            loss = torch.mean((pred - y_t) ** 2)
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, xa_test.shape[0], int(batch_size)):
            sl = slice(start, start + int(batch_size))
            xa_t = torch.as_tensor(xa_test[sl], dtype=torch.float32, device=dev)
            ctx_t = torch.as_tensor(context_test[sl], dtype=torch.float32, device=dev)
            if model_kind == "siamese":
                xb_t = torch.as_tensor(xb_test[sl], dtype=torch.float32, device=dev)
                pred = model(xa_t, xb_t, ctx_t)
            else:
                pred = model(xa_t, ctx_t)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def _trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rows = _metrics(y_true.reshape(-1, 2), y_pred.reshape(-1, 2))
    rows = {f"trajectory_{k}": v for k, v in rows.items()}
    rows.update({f"center_{k}": v for k, v in _metrics(y_true[:, 0, :], y_pred[:, 0, :]).items()})
    return rows


def _session_rows(
    *,
    session: str,
    subject: str,
    pairs: dict[str, np.ndarray],
    j: np.ndarray,
    labels: np.ndarray,
    samples: Any,
    target_cov: np.ndarray,
    projection_controls: list[str],
    feature_spaces: set[str],
    alignment_controls: set[str],
    primary_k: int,
    rf_bins: np.ndarray | None,
    split_mode: str,
    n_folds: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    context = _time_context(pairs["center_time"], int(np.unique(pairs["condition_id"]).size), int(args.context_one_hot_max))
    folds = _decode_splits(pairs, int(n_folds), int(args.seed), str(split_mode))
    for fold_idx, (held_ids, train_mask, test_mask) in enumerate(folds):
        train_conditions = set(int(v) for v in np.unique(pairs["condition_id"][train_mask]))
        test_conditions = set(int(v) for v in np.unique(pairs["condition_id"][test_mask]))
        shared_conditions = sorted(train_conditions.intersection(test_conditions))
        train_trials = _trial_set(pairs["trial_a"], pairs["trial_b"], train_mask)
        test_trials = _trial_set(pairs["trial_a"], pairs["trial_b"], test_mask)
        shared_trials = sorted(train_trials.intersection(test_trials))
        train_trial_pairs = _trial_pair_keys(pairs["trial_a"], pairs["trial_b"], train_mask)
        test_trial_pairs = _trial_pair_keys(pairs["trial_a"], pairs["trial_b"], test_mask)
        shared_trial_pairs = sorted(train_trial_pairs.intersection(test_trial_pairs))
        leakage_rows.append(
            {
                "session": session,
                "fold_id": int(fold_idx),
                "split_mode": str(split_mode),
                "n_train_conditions": int(len(train_conditions)),
                "n_test_conditions": int(len(test_conditions)),
                "n_shared_conditions": int(len(shared_conditions)),
                "n_train_trials": int(len(train_trials)),
                "n_test_trials": int(len(test_trials)),
                "n_shared_trials": int(len(shared_trials)),
                "n_shared_trial_pairs": int(len(shared_trial_pairs)),
                "n_train_pairs": int(np.sum(train_mask)),
                "n_test_pairs": int(np.sum(test_mask)),
                "condition_overlap_status": "pass" if not shared_conditions else ("warn" if split_mode == "trial_disjoint" else "fail"),
                "trial_overlap_status": "pass" if not shared_trials else ("fail" if split_mode == "trial_disjoint" else "warn"),
                "status": "fail" if (split_mode == "condition_disjoint" and shared_conditions) or (split_mode == "trial_disjoint" and shared_trials) else "pass",
            }
        )
        if not np.any(train_mask) or not np.any(test_mask):
            continue
        train_sample_mask = ~np.isin(samples.trial_ids, held_ids) if split_mode == "trial_disjoint" else np.isin(labels, list(train_conditions))
        for projection_control in projection_controls:
            modes = _projection_modes(str(projection_control), target_cov)
            projection = _projection_complement(pairs["response_a"].shape[2], modes)
            compact_basis, basis_rank = _basis_from_j(j, train_sample_mask, projection, int(primary_k))
            rng = np.random.default_rng(int(args.seed) + int(fold_idx) * 1009 + len(metric_rows))
            specs = _feature_windows(
                pairs=pairs,
                compact_basis=compact_basis,
                projection=projection,
                modes=modes,
                k=int(primary_k),
                rng=rng,
                rf_bins=rf_bins,
                requested=feature_spaces,
            )
            if "context_only" in feature_spaces:
                specs.append({"feature_space": "context_only", "feature_role": "context_control", "k": 0, "matrix": None})
            if "eye_history" in feature_spaces and pairs["prior_delta"].shape[1] > 0:
                specs.append({"feature_space": "eye_history", "feature_role": "trajectory_prior_control", "k": 0, "matrix": None})
            for spec in specs:
                feature_space = str(spec["feature_space"])
                if feature_space == "context_only":
                    xa = np.zeros((pairs["response_a"].shape[0], 1, 0), dtype=np.float32)
                    xb = xa
                    kind = "context_only"
                elif feature_space == "eye_history":
                    xa = np.zeros((pairs["response_a"].shape[0], 1, 0), dtype=np.float32)
                    xb = xa
                    kind = "eye_history"
                else:
                    mat = np.asarray(spec["matrix"], dtype=np.float64)
                    xa = _transform_windows(pairs["response_a"], mat)
                    xb = _transform_windows(pairs["response_b"], mat)
                    kind = "siamese"
                if kind == "siamese" and bool(args.enable_chart_inverse):
                    for chart_alignment in ("observed", "chart_time_shuffled"):
                        chart_preds = _chart_predictions_for_spec(
                            pairs=pairs,
                            j=j,
                            matrix=mat,
                            train_mask=train_mask,
                            test_mask=test_mask,
                            alignment_control=chart_alignment,
                            ridge_frac=float(args.chart_ridge_frac),
                            calibration_ridge_frac=float(args.chart_calibration_ridge_frac),
                            seed=int(args.seed) + fold_idx * 1429 + len(metric_rows),
                        )
                        for chart_kind, chart_pred in chart_preds.items():
                            metrics = _trajectory_metrics(pairs["target_delta"][test_mask], chart_pred)
                            base = {
                                "session": session,
                                "subject": subject,
                                "fold_id": int(fold_idx),
                                "feature_space": f"{feature_space}_chart_inverse"
                                if chart_kind == "raw"
                                else f"{feature_space}_chart_inverse_calibrated",
                                "feature_role": "local_chart_observer"
                                if chart_kind == "raw"
                                else "local_chart_observer_train_calibrated",
                                "k": int(spec["k"]),
                                "basis_rank": int(basis_rank),
                                "projection_control": projection_control,
                                "alignment_control": chart_alignment,
                                "decoder": "local_chart_ridge_inverse"
                                if chart_kind == "raw"
                                else "local_chart_ridge_inverse_train_calibrated",
                                "split_mode": str(split_mode),
                                "n_train_pairs": int(np.sum(train_mask)),
                                "n_test_pairs": int(np.sum(test_mask)),
                                "target_len": int(pairs["target_delta"].shape[1]),
                                "input_len": int(pairs["response_a"].shape[1]),
                            }
                            for name, value in metrics.items():
                                metric_rows.append({**base, "metric_name": name, "metric_value": value})
                if bool(args.run_mlp_decoder):
                    controls = ["observed"]
                    if kind == "siamese" and "neural_time_shuffled" in alignment_controls:
                        controls.append("neural_time_shuffled")
                    if kind == "eye_history" and "eye_history_time_shuffled" in alignment_controls:
                        controls.append("eye_history_time_shuffled")
                    for alignment_control in controls:
                        xa_train, xb_train, prior_train, xa_test, xb_test, prior_test = _alignment_arrays(
                            xa=xa,
                            xb=xb,
                            prior=pairs["prior_delta"],
                            group_ids=pairs["condition_id"],
                            train_mask=train_mask,
                            test_mask=test_mask,
                            alignment_control=alignment_control,
                            model_kind=kind,
                            seed=int(args.seed) + fold_idx * 1291 + len(metric_rows),
                        )
                        pred = _train_predict(
                            xa_train=xa_train,
                            xb_train=xb_train,
                            context_train=context[train_mask],
                            prior_train=prior_train,
                            y_train=pairs["target_delta"][train_mask],
                            xa_test=xa_test,
                            xb_test=xb_test,
                            context_test=context[test_mask],
                            prior_test=prior_test,
                            model_kind=kind,
                            hidden_dim=int(args.hidden_dim),
                            latent_dim=int(args.latent_dim),
                            n_layers=int(args.n_layers),
                            dropout=float(args.dropout),
                            lr=float(args.learning_rate),
                            weight_decay=float(args.weight_decay),
                            n_epochs=int(args.n_epochs),
                            batch_size=int(args.batch_size_decoder),
                            device=str(args.decoder_device),
                            seed=int(args.seed) + fold_idx * 997 + len(metric_rows),
                        )
                        metrics = _trajectory_metrics(pairs["target_delta"][test_mask], pred)
                        base = {
                            "session": session,
                            "subject": subject,
                            "fold_id": int(fold_idx),
                            "feature_space": feature_space,
                            "feature_role": str(spec["feature_role"]),
                            "k": int(spec["k"]),
                            "basis_rank": int(basis_rank),
                            "projection_control": projection_control,
                            "alignment_control": alignment_control,
                            "decoder": "antisymmetric_siamese_mlp" if kind == "siamese" else f"{kind}_mlp",
                            "split_mode": str(split_mode),
                            "n_train_pairs": int(np.sum(train_mask)),
                            "n_test_pairs": int(np.sum(test_mask)),
                            "target_len": int(pairs["target_delta"].shape[1]),
                            "input_len": int(pairs["response_a"].shape[1]),
                        }
                        for name, value in metrics.items():
                            metric_rows.append({**base, "metric_name": name, "metric_value": value})
    return metric_rows, leakage_rows


def _summaries(metric_rows: list[dict[str, Any]], primary_projection: str, primary_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str, str, str], list[float]] = {}
    by_session: dict[tuple[str, int, str, str, str, str], list[float]] = {}
    for row in metric_rows:
        key = (
            str(row["feature_space"]),
            int(row["k"]),
            str(row["projection_control"]),
            str(row.get("alignment_control", "observed")),
            str(row["metric_name"]),
        )
        by_session.setdefault((*key, str(row["session"])), []).append(float(row["metric_value"]))
    for key_sess, vals in by_session.items():
        groups.setdefault(key_sess[:5], []).append(float(np.nanmean(vals)))
    summary = []
    for (feature_space, k, projection_control, alignment_control, metric_name), vals in sorted(groups.items()):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        summary.append(
            {
                "feature_space": feature_space,
                "k": int(k),
                "projection_control": projection_control,
                "alignment_control": alignment_control,
                "metric_name": metric_name,
                "n_sessions": int(arr.size),
                "observed_mean": float(np.mean(arr)) if arr.size else float("nan"),
                "observed_sem": float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan"),
            }
        )
    primary = [
        r
        for r in summary
        if str(r["projection_control"]) == str(primary_projection)
        and str(r["alignment_control"]) == "observed"
        and str(r["metric_name"]) == "center_R2_mean"
        and int(r["k"]) in {0, int(primary_k), 2}
    ]
    vals = {(str(r["feature_space"]), int(r["k"])): float(r["observed_mean"]) for r in primary}
    comparison = [
        {
            "projection_control": primary_projection,
            "primary_k": int(primary_k),
            "full_population_center_R2_mean": vals.get(("full_population", 0), float("nan")),
            "compact_center_R2_mean": vals.get(("compact", int(primary_k)), float("nan")),
            "orthogonal_complement_center_R2_mean": vals.get(("orthogonal_complement", int(primary_k)), float("nan")),
            "random_subspace_center_R2_mean": vals.get(("random_subspace", int(primary_k)), float("nan")),
            "rf_readout_permuted_compact_center_R2_mean": vals.get(("rf_readout_permuted_compact", int(primary_k)), float("nan")),
            "global_top_pc_modes_center_R2_mean": vals.get(("global_top_pc_modes", 2), float("nan")),
            "context_only_center_R2_mean": vals.get(("context_only", 0), float("nan")),
            "eye_history_center_R2_mean": vals.get(("eye_history", 0), float("nan")),
        }
    ]
    c = comparison[0]
    compact = float(c["compact_center_R2_mean"])
    for key in ("full_population", "orthogonal_complement", "random_subspace", "rf_readout_permuted_compact", "context_only", "eye_history"):
        val = float(c.get(f"{key}_center_R2_mean", float("nan")))
        c[f"compact_minus_{key}_center_R2"] = compact - val if np.isfinite(compact) and np.isfinite(val) else float("nan")
    shuffled_vals = {
        (str(r["feature_space"]), int(r["k"]), str(r["alignment_control"])): float(r["observed_mean"])
        for r in summary
        if str(r["projection_control"]) == str(primary_projection)
        and str(r["metric_name"]) == "center_R2_mean"
        and int(r["k"]) in {0, int(primary_k), 2}
    }
    for feature_space, k, shuffled_name in (
        ("full_population", 0, "neural_time_shuffled"),
        ("compact", int(primary_k), "neural_time_shuffled"),
        ("orthogonal_complement", int(primary_k), "neural_time_shuffled"),
        ("random_subspace", int(primary_k), "neural_time_shuffled"),
        ("rf_readout_permuted_compact", int(primary_k), "neural_time_shuffled"),
        ("global_top_pc_modes", 2, "neural_time_shuffled"),
        ("eye_history", 0, "eye_history_time_shuffled"),
    ):
        observed = shuffled_vals.get((feature_space, k, "observed"), float("nan"))
        shuffled = shuffled_vals.get((feature_space, k, shuffled_name), float("nan"))
        c[f"{feature_space}_alignment_null_center_R2_mean"] = shuffled
        c[f"{feature_space}_alignment_gain_center_R2"] = observed - shuffled if np.isfinite(observed) and np.isfinite(shuffled) else float("nan")
    return summary, comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Windowed Siamese relative displacement decoding")
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--step-px", type=float, default=0.25)
    p.add_argument("--pixels-per-degree-fallback", type=float, default=37.5)
    p.add_argument("--fixation-radius-deg", type=float, default=1.0)
    p.add_argument("--sample-dfs-mode", choices=["all", "any", "none"], default="all")
    p.add_argument("--rescale-mode", choices=["none", "globalgain", "gain", "globalaffine", "affine"], default="affine")
    p.add_argument("--projection-controls", type=str, default="global_rate+target_pc1")
    p.add_argument("--primary-projection-control", type=str, default="global_rate+target_pc1")
    p.add_argument("--target-variant", choices=["raw", "psd"], default="psd")
    p.add_argument("--feature-spaces", type=str, default="full_population,compact,orthogonal_complement,random_subspace,rf_readout_permuted_compact,global_top_pc_modes,context_only,eye_history")
    p.add_argument("--alignment-controls", type=str, default="observed,neural_time_shuffled,eye_history_time_shuffled")
    p.add_argument("--primary-k", type=int, default=10)
    p.add_argument("--history-bins", type=int, default=10)
    p.add_argument("--future-bins", type=int, default=0)
    p.add_argument("--context-mode", choices=["time_bin", "time_window"], default="time_bin")
    p.add_argument("--context-bin-size", type=int, default=10)
    p.add_argument("--context-one-hot-max", type=int, default=128)
    p.add_argument("--min-repeats-per-condition", type=int, default=3)
    p.add_argument("--max-pairs-per-condition", type=int, default=100)
    p.add_argument("--split-mode", choices=["condition_disjoint", "trial_disjoint"], default="trial_disjoint")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--batch-size-decoder", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--run-mlp-decoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-chart-inverse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--chart-ridge-frac", type=float, default=1e-3)
    p.add_argument("--chart-calibration-ridge-frac", type=float, default=1e-3)
    p.add_argument("--min-units", type=int, default=50)
    p.add_argument("--enable-rf-readout-null", action="store_true", default=True)
    p.add_argument("--rf-null-min-bin-units", type=int, default=6)
    p.add_argument("--rf-null-bin-features", type=str, default="rf_xy,tangent_norm,mean_rate,ccnorm")
    p.add_argument("--rf-null-session-yaml-dir", type=Path, default=Path("experiments") / "dataset_configs" / "sessions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    p.add_argument("--init-only", action="store_true")
    return p


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    projection_controls = parse_str_list(args.projection_controls)
    feature_spaces = set(parse_str_list(args.feature_spaces))
    alignment_controls = set(parse_str_list(args.alignment_controls))
    valid_alignment_controls = {"observed", "neural_time_shuffled", "eye_history_time_shuffled"}
    unknown_alignment_controls = sorted(alignment_controls.difference(valid_alignment_controls))
    if unknown_alignment_controls:
        raise ValueError(f"Unknown alignment controls: {unknown_alignment_controls}")
    alignment_controls.add("observed")
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    sessions = parse_str_list(args.sessions)
    config = SiameseConfig(
        output_root=str(out),
        sessions=sessions,
        projection_controls=projection_controls,
        feature_spaces=sorted(feature_spaces),
        alignment_controls=sorted(alignment_controls),
        primary_k=int(args.primary_k),
        history_bins=int(args.history_bins),
        future_bins=int(args.future_bins),
        split_mode=str(args.split_mode),
        n_folds=int(args.n_folds),
        n_epochs=int(args.n_epochs),
        batch_size_decoder=int(args.batch_size_decoder),
        run_mlp_decoder=bool(args.run_mlp_decoder),
        enable_chart_inverse=bool(args.enable_chart_inverse),
        chart_ridge_frac=float(args.chart_ridge_frac),
        chart_calibration_ridge_frac=float(args.chart_calibration_ridge_frac),
        seed=int(args.seed),
    )
    write_json(
        out / "windowed_siamese_manifest.json",
        {
            "analysis": "windowed_siamese_relative_displacement_decoding",
            "status": "initialized_not_run" if bool(args.init_only) else "running",
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "strict_decoder_reference": str(STRICT_DECODER_ROOT),
            "claim_guardrail": "Context-aware relative decoder; do not describe as absolute eye-position decoding.",
        },
    )
    if bool(args.init_only):
        return
    model, model_info = _load_twin_model(args)
    session_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []
    for session in sessions:
        print(f"[siamese-decoding] session {session}: starting", flush=True)
        if session not in fig3_by_session or session not in fig2 or session not in getattr(model, "names", []):
            session_rows.append({"session": session, "status": "missing_inputs"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, target_raw, target_psd, target_meta = _target_for_session(fig2[session], sr, args)
        if common_units.size < int(args.min_units):
            session_rows.append({"session": session, "status": "too_few_common_units", "n_common_units": int(common_units.size)})
            continue
        dset, stim_lags, samples = _collect_samples(model=model, dataset_idx=dataset_idx, common_units=common_units, args=args)
        labels = context_labels(samples, str(args.context_mode), int(args.context_bin_size))
        eye_px = samples.eyepos_deg * float(samples.pixels_per_degree)
        pairs, inventory = build_window_pair_dataset(
            samples=samples,
            eye_px=eye_px,
            labels=labels,
            context_mode=str(args.context_mode),
            context_bin_size=int(args.context_bin_size),
            history_bins=int(args.history_bins),
            future_bins=int(args.future_bins),
            min_repeats_per_condition=int(args.min_repeats_per_condition),
            max_pairs_per_condition=int(args.max_pairs_per_condition),
            seed=int(args.seed) + dataset_idx * 101,
        )
        for row in inventory:
            row.update({"session": session, "subject": sr.get("subject", "")})
        inventory_rows.extend(inventory)
        if pairs["response_a"].shape[0] < max(20, int(args.n_folds) * 3):
            session_rows.append({"session": session, "subject": sr.get("subject", ""), "status": "too_few_window_pairs", "n_pairs": int(pairs["response_a"].shape[0])})
            continue
        print(f"[siamese-decoding] session {session}: {pairs['response_a'].shape[0]} window pairs", flush=True)
        gains, rescale_status = _fit_rescale_gains(model=model, dset=dset, stim_lags=stim_lags, samples=samples, common_units=common_units, dataset_idx=dataset_idx, args=args)
        j = _compute_jacobians(model=model, dset=dset, stim_lags=stim_lags, samples=samples, common_units=common_units, gains=gains, dataset_idx=dataset_idx, args=args)
        rf_meta = _rf_null_metadata_for_session(session=session, subject=str(sr.get("subject", "")), common_units=common_units, sr=sr, samples=samples, j=j, gains=gains, args=args)
        rf_rows.extend(rf_meta.unit_rows)
        target = target_psd if str(args.target_variant) == "psd" else target_raw
        m_rows, l_rows = _session_rows(
            session=session,
            subject=str(sr.get("subject", "")),
            pairs=pairs,
            j=j,
            labels=labels,
            samples=samples,
            target_cov=target,
            projection_controls=projection_controls,
            feature_spaces=feature_spaces,
            alignment_controls=alignment_controls,
            primary_k=int(args.primary_k),
            rf_bins=rf_meta.bins,
            split_mode=str(args.split_mode),
            n_folds=int(args.n_folds),
            args=args,
        )
        metric_rows.extend(m_rows)
        leakage_rows.extend(l_rows)
        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": dataset_idx,
                "n_common_units": int(common_units.size),
                "n_samples_used": int(samples.source_indices.size),
                "n_window_pairs": int(pairs["response_a"].shape[0]),
                "n_pair_conditions": int(np.unique(pairs["condition_id"]).size),
                "input_len": int(pairs["response_a"].shape[1]),
                "target_len": int(pairs["target_delta"].shape[1]),
                "rescale_status": rescale_status,
                "rf_null_status": rf_meta.status,
                **target_meta,
            }
        )
        print(f"[siamese-decoding] session {session}: finished ({len(m_rows)} metric rows)", flush=True)
    summary_rows, comparison_rows = _summaries(metric_rows, str(args.primary_projection_control), int(args.primary_k))
    write_csv(out / "session_summary.csv", session_rows)
    write_csv(out / "pair_inventory.csv", inventory_rows)
    write_csv(out / "decoder_metrics.csv", metric_rows)
    write_csv(out / "decoder_summary.csv", summary_rows)
    write_csv(out / "feature_space_comparison.csv", comparison_rows)
    write_csv(out / "split_leakage_audit.csv", leakage_rows)
    write_csv(out / "rf_readout_unit_bins.csv", rf_rows)
    comp = comparison_rows[0] if comparison_rows else {}
    compact = float(comp.get("compact_center_R2_mean", float("nan"))) if comp else float("nan")
    compact_minus_orth = float(comp.get("compact_minus_orthogonal_complement_center_R2", float("nan"))) if comp else float("nan")
    compact_minus_random = float(comp.get("compact_minus_random_subspace_center_R2", float("nan"))) if comp else float("nan")
    compact_minus_rf = float(comp.get("compact_minus_rf_readout_permuted_compact_center_R2", float("nan"))) if comp else float("nan")
    leakage_failures = int(sum(1 for r in leakage_rows if r.get("status") == "fail"))
    decision = "diagnostic"
    if np.isfinite(compact) and compact > 0 and np.isfinite(compact_minus_orth) and compact_minus_orth > 0 and np.isfinite(compact_minus_random) and compact_minus_random > 0 and (not np.isfinite(compact_minus_rf) or compact_minus_rf > 0) and leakage_failures == 0:
        decision = "candidate_positive"
    write_json(
        out / "audit.json",
        {
            "status": "ok",
            "decision": decision,
            "n_sessions_requested": int(len(sessions)),
            "n_sessions_ok": int(sum(1 for r in session_rows if r.get("status") == "ok")),
            "n_metric_rows": int(len(metric_rows)),
            "n_leakage_failures": leakage_failures,
            "primary_feature_comparison": comp,
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
            "claim_guardrail": "Context-aware relative decoder; do not describe as absolute eye-position decoding.",
        },
    )
    write_json(
        out / "windowed_siamese_manifest.json",
        {
            "analysis": "windowed_siamese_relative_displacement_decoding",
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

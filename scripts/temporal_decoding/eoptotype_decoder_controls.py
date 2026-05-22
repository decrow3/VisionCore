"""E-optotype decoder controls: FEM vs stabilized (D1–D3).

Goal
----
Test how information under real FEM is represented and read out, not just whether a
single decoder performs better.

This script is intentionally *cache-only*: it consumes the `.npz` rate caches written
by `scripts/temporal_decoding/rate_computation.save_rates()` and the cached eye traces
(`scripts/temporal_decoding/data/eye_traces.npz`). It avoids importing the full
simulation stack.

Decoders
--------
D1  Window-averaged rates (time-mean over last W frames).
D2a Eye-aware linear decoder: concat([rate_mean, eye_mean]).
D2b (Optional) Eye-conditioned response correction: subtract position-bin mean.
D3  (Optional) Supervised mean-trajectory subspace (PCA on class means only).

Outputs
-------
- Accuracy-vs-window plots (stabilized vs real; plus D2/D3 overlays)
- A small summary table at W ∈ {1, 24, 60}
- Cached `.npz` results under `scripts/temporal_decoding/data/results/`

Usage
-----
  /home/declan/VisionCore/.venv/bin/python scripts/temporal_decoding/eoptotype_decoder_controls.py \
    --logmar -0.20 --n_splits 5 --windows 1,6,12,24,48,60

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_WINDOWS = (1, 6, 12, 24, 48, 60)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_CONDITIONS = ("real", "stabilized")


def _format_logmar_for_filename(logmar: float) -> str:
    # Filenames look like: rates_lm0.40_... or rates_hires_lm-0.20_...
    # i.e. no leading '+' for positive values.
    return f"{logmar:.2f}"


def _load_rates_npz(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load rates saved by `rate_computation.save_rates()`.

    Returns:
      rates_list: list length M, each (T_i, N)
      lengths: (M,) int
    """
    d = np.load(path, allow_pickle=True)
    rates_padded = d["rates"]  # (M, T_max, N) float32, NaN padded
    lengths = d["lengths"].astype(int)  # (M,)
    rates_list = [rates_padded[i, : lengths[i]] for i in range(rates_padded.shape[0])]
    return rates_list, lengths


def _window_last_frames(x: np.ndarray, window: int) -> np.ndarray:
    """Take the last `window` frames, padding by repeating the first frame."""
    if window <= 0:
        raise ValueError("window must be positive")
    T = int(x.shape[0])
    if T >= window:
        return x[-window:]
    pad = np.repeat(x[[0]], window - T, axis=0)
    return np.concatenate([pad, x], axis=0)


def build_rates_by_condition(
    rates_dir: Path,
    logmar: float,
    hires_threshold: float = 0.35,
    orientations: Iterable[int] = DEFAULT_ORIENTATIONS,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
) -> tuple[dict[str, dict[str, list[np.ndarray]]], dict[str, dict[str, np.ndarray]]]:
    """Load cached rate lists into {cond: {ori_key: [rates...]}}.

    Also returns lengths in the same dict shape.
    """
    rates_by_condition: dict[str, dict[str, list[np.ndarray]]] = {}
    lengths_by_condition: dict[str, dict[str, np.ndarray]] = {}

    use_hires = float(logmar) < float(hires_threshold)
    prefix = "rates_hires_lm" if use_hires else "rates_lm"
    lm_str = _format_logmar_for_filename(float(logmar))

    for cond in conditions:
        by_stim: dict[str, list[np.ndarray]] = {}
        by_len: dict[str, np.ndarray] = {}
        for ori in orientations:
            fname = f"{prefix}{lm_str}_ori{int(ori)}_{cond}.npz"
            path = rates_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing cached rates: {path}")
            rates_list, lengths = _load_rates_npz(path)
            by_stim[f"ori{int(ori)}"] = rates_list
            by_len[f"ori{int(ori)}"] = lengths
        rates_by_condition[cond] = by_stim
        lengths_by_condition[cond] = by_len

    return rates_by_condition, lengths_by_condition


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    traces = d["traces"].astype(np.float32)  # (M, T_max, 2)
    durations = d["durations"].astype(int)  # (M,)
    return traces, durations


def _eye_padded_for_rates(
    eye_traces: np.ndarray,
    eye_durations: np.ndarray,
    trace_index: int,
    target_T: int,
) -> np.ndarray:
    """Return eye trace sequence length `target_T` aligned to cached rates.

    Empirically, the cached rates have length `durations + 1` (one extra initial frame).
    We align by prepending one sample (repeat the first valid sample) when needed.

    Falls back to left-padding by repeating the first sample if target_T > duration.
    """
    dur = int(eye_durations[trace_index])
    if dur <= 0:
        raise ValueError(f"Invalid duration for trace {trace_index}: {dur}")

    eye = np.asarray(eye_traces[trace_index, :dur], dtype=np.float32)

    if target_T == dur:
        return eye
    if target_T > dur:
        pad = np.repeat(eye[[0]], target_T - dur, axis=0)
        return np.concatenate([pad, eye], axis=0)

    # target_T < dur: truncate (should be rare)
    return eye[-target_T:]


def _build_time_mean_dataset(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, groups) for the D1-style time-mean decoder.

    This matches the existing temporal-decoding convention:
      - Equalize trial counts across classes to M_min
      - groups = trace index (0..M_min-1), repeated across classes
    """
    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)
    if K < 2:
        raise ValueError("Need at least 2 classes")

    M_min = min(len(rates_by_stim[sid]) for sid in stim_ids)
    X_list, y_list = [], []
    for label, sid in enumerate(stim_ids):
        feats = []
        for r in rates_by_stim[sid][:M_min]:
            r_win = _window_last_frames(np.asarray(r), window)
            feats.append(r_win.mean(axis=0))
        X_list.append(np.stack(feats, axis=0))
        y_list.append(np.full(M_min, label, dtype=int))

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups_all = np.tile(np.arange(M_min, dtype=int), K)
    return X_all, y_all, groups_all


def _decode_groupkfold_logreg(
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    n_splits: int = 5,
    C_logistic: float = 1.0,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

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


def decode_d1_time_mean(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
    n_splits: int = 5,
    C_logistic: float = 1.0,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Grouped-CV decoding using only the time-mean over the last W frames."""
    X_all, y_all, groups_all = _build_time_mean_dataset(rates_by_stim, window)
    return _decode_groupkfold_logreg(
        X_all,
        y_all,
        groups_all,
        n_splits=n_splits,
        C_logistic=C_logistic,
        rng_seed=rng_seed,
    )


@dataclass(frozen=True)
class MLPConfig:
    hidden1: int = 128
    hidden2: int = 64
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    val_frac: float = 0.2
    batch_size: int = 128
    device: str = "cpu"


def _decode_groupkfold_mlp(
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    cfg: MLPConfig,
    n_splits: int = 5,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Grouped CV for an MLP classifier with early stopping inside each fold.

    Outer split: GroupKFold by trace.
    Inner early-stop: split the training groups into train/val by groups.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    import torch
    import torch.nn as nn

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []

    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.int64)
    groups_all = np.asarray(groups_all, dtype=np.int64)

    device = torch.device(cfg.device)

    class _MLP(nn.Module):
        def __init__(self, in_dim: int, h1: int, h2: int, p: float, out_dim: int = 4) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(h2, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        pred = torch.argmax(logits, dim=1)
        return float((pred == y).float().mean().item())

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups_all)):
        # Deterministic split of training groups into train/val.
        unique_groups = np.unique(groups_all[train_idx])
        rng = np.random.default_rng(rng_seed + 1000 * fold_idx)
        rng.shuffle(unique_groups)

        n_val_groups = int(np.ceil(float(cfg.val_frac) * float(unique_groups.size)))
        n_val_groups = max(1, min(n_val_groups, unique_groups.size - 1))
        val_groups = set(unique_groups[:n_val_groups].tolist())

        train_sub_idx = np.array([i for i in train_idx if int(groups_all[i]) not in val_groups], dtype=int)
        val_sub_idx = np.array([i for i in train_idx if int(groups_all[i]) in val_groups], dtype=int)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_sub_idx]).astype(np.float32)
        y_tr = y_all[train_sub_idx]
        X_va = scaler.transform(X_all[val_sub_idx]).astype(np.float32)
        y_va = y_all[val_sub_idx]
        X_te = scaler.transform(X_all[test_idx]).astype(np.float32)
        y_te = y_all[test_idx]

        # Torch tensors
        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        X_va_t = torch.from_numpy(X_va).to(device)
        y_va_t = torch.from_numpy(y_va).to(device)
        X_te_t = torch.from_numpy(X_te).to(device)
        y_te_t = torch.from_numpy(y_te).to(device)

        # Model
        torch.manual_seed(rng_seed + fold_idx)
        model = _MLP(in_dim=int(X_all.shape[1]), h1=int(cfg.hidden1), h2=int(cfg.hidden2), p=float(cfg.dropout)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        loss_fn = nn.CrossEntropyLoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        bad_epochs = 0

        # Mini-batch training
        n = int(X_tr_t.shape[0])
        bs = int(cfg.batch_size)
        idx_all = torch.arange(n, device=device)

        for _epoch in range(int(cfg.max_epochs)):
            model.train()
            # Shuffle indices each epoch
            perm = idx_all[torch.randperm(n)]
            for start in range(0, n, bs):
                batch = perm[start : start + bs]
                xb = X_tr_t[batch]
                yb = y_tr_t[batch]
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_va_t)
                val_loss = float(loss_fn(val_logits, y_va_t).item())

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= int(cfg.patience):
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        with torch.no_grad():
            test_logits = model(X_te_t)
            fold_accs.append(_accuracy(test_logits, y_te_t))

    fold_accs_arr = np.asarray(fold_accs, dtype=float)
    return float(fold_accs_arr.mean()), float(fold_accs_arr.std()), fold_accs_arr


def decode_d1_time_mean_mlp(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
    cfg: MLPConfig,
    n_splits: int = 5,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Nonlinear version of D1: MLP on the same time-mean feature."""
    X_all, y_all, groups_all = _build_time_mean_dataset(rates_by_stim, window)
    return _decode_groupkfold_mlp(
        X_all,
        y_all,
        groups_all,
        cfg=cfg,
        n_splits=n_splits,
        rng_seed=rng_seed,
    )


def decode_d2a_time_mean_plus_eye(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
    eye_traces: np.ndarray,
    eye_durations: np.ndarray,
    use_stabilized_eye: bool,
    n_splits: int = 5,
    C_logistic: float = 1.0,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Eye-aware linear decoder: concat([rate_mean, eye_mean]).

    Notes on stabilized eye features
    -------------------------------
    The stabilized condition has no motion, but is typically rendered at a trace-specific
    mean position. When `use_stabilized_eye=True`, we provide the constant mean position
    per trace as the eye feature (repeated over time). When False, we use the real trace
    segment aligned to the trial (useful mainly for debugging).
    """
    X_rate, y_all, groups_all = _build_time_mean_dataset(rates_by_stim, window)

    # Compute per-trace eye mean once, then repeat across classes.
    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)
    M_min = min(len(rates_by_stim[sid]) for sid in stim_ids)

    eye_means = np.zeros((M_min, 2), dtype=np.float32)
    for trace_idx in range(M_min):
        # Use the first class to get the rates length (rate caches are aligned across oris)
        r0 = np.asarray(rates_by_stim[stim_ids[0]][trace_idx])
        eye_seq = _eye_padded_for_rates(
            eye_traces=eye_traces,
            eye_durations=eye_durations,
            trace_index=trace_idx,
            target_T=int(r0.shape[0]),
        )

        if use_stabilized_eye:
            eye_mean_full = np.nanmean(eye_seq, axis=0)
            eye_means[trace_idx] = eye_mean_full
        else:
            # For the requested W<=60 regime, eye_seq is always long enough.
            eye_means[trace_idx] = np.nanmean(eye_seq[-window:], axis=0)

    eye_all = np.tile(eye_means, (K, 1))
    X_all = np.concatenate([X_rate, eye_all], axis=1)

    return _decode_groupkfold_logreg(
        X_all,
        y_all,
        groups_all,
        n_splits=n_splits,
        C_logistic=C_logistic,
        rng_seed=rng_seed,
    )


def decode_d2b_eye_conditioned_correction(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
    eye_traces: np.ndarray,
    eye_durations: np.ndarray,
    n_bins: int = 9,
    min_bin_count: int = 200,
    n_splits: int = 5,
    C_logistic: float = 1.0,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Eye-conditioned response correction before D1 decoding.

    For each CV fold:
      1) Build eye-position bins from training data.
      2) Estimate per-bin mean response μ(bin) pooling over *all* classes.
      3) Subtract μ(bin(t)) from each timepoint, then compute mean over last W.

    This is intentionally simple and conservative (no temporal model of μ).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)
    M_min = min(len(rates_by_stim[sid]) for sid in stim_ids)

    # Pre-build per-class lists of aligned (rates, eye) for the first M_min traces.
    per_class_rates: list[list[np.ndarray]] = []
    per_class_eye: list[list[np.ndarray]] = []
    for sid in stim_ids:
        r_list = []
        e_list = []
        for trace_idx, r in enumerate(rates_by_stim[sid][:M_min]):
            r = np.asarray(r, dtype=np.float32)
            r_list.append(r)
            e_list.append(
                _eye_padded_for_rates(eye_traces, eye_durations, trace_idx, target_T=int(r.shape[0]))
            )
        per_class_rates.append(r_list)
        per_class_eye.append(e_list)

    # Build (X_raw, y, groups) where X_raw is a list of per-sample trajectories.
    traj_list: list[np.ndarray] = []
    eye_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[int] = []
    for label, sid in enumerate(stim_ids):
        for trace_idx in range(M_min):
            traj_list.append(per_class_rates[label][trace_idx])
            eye_list.append(per_class_eye[label][trace_idx])
            y_list.append(label)
            groups_list.append(trace_idx)

    y_all = np.asarray(y_list, dtype=int)
    groups_all = np.asarray(groups_list, dtype=int)

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []

    for train_idx, test_idx in gkf.split(np.zeros((len(y_all), 1)), y_all, groups=groups_all):
        # Collect training eye positions and responses across all timepoints.
        train_eye = []
        train_r = []
        for i in train_idx:
            r = traj_list[i]
            e = eye_list[i]
            # Use all timepoints (not just last W) to estimate μ(bin)
            T = min(r.shape[0], e.shape[0])
            train_eye.append(e[:T])
            train_r.append(r[:T])

        train_eye = np.concatenate(train_eye, axis=0)  # (T_total, 2)
        train_r = np.concatenate(train_r, axis=0)      # (T_total, N)

        # Define bins based on training eye position range.
        # Add a tiny epsilon so max values fall inside.
        eps = 1e-6
        x_min, y_min = np.nanmin(train_eye, axis=0) - eps
        x_max, y_max = np.nanmax(train_eye, axis=0) + eps
        x_edges = np.linspace(x_min, x_max, n_bins + 1)
        y_edges = np.linspace(y_min, y_max, n_bins + 1)

        def _bin_index(exy: np.ndarray) -> np.ndarray:
            xi = np.clip(np.digitize(exy[:, 0], x_edges) - 1, 0, n_bins - 1)
            yi = np.clip(np.digitize(exy[:, 1], y_edges) - 1, 0, n_bins - 1)
            return xi + n_bins * yi

        train_bin = _bin_index(train_eye)
        n_total_bins = n_bins * n_bins

        N = int(train_r.shape[1])
        mu = np.zeros((n_total_bins, N), dtype=np.float32)
        counts = np.zeros((n_total_bins,), dtype=np.int64)

        # Accumulate means efficiently
        for b in range(n_total_bins):
            idx = np.where(train_bin == b)[0]
            counts[b] = int(idx.size)
            if idx.size > 0:
                mu[b] = train_r[idx].mean(axis=0)

        global_mu = train_r.mean(axis=0)

        # Build fold features: corrected mean rate over last W
        def _featurize(sample_indices: np.ndarray) -> np.ndarray:
            feats = []
            for i in sample_indices:
                r = traj_list[i]
                e = eye_list[i]
                T = min(r.shape[0], e.shape[0])
                r = r[:T]
                e = e[:T]
                b = _bin_index(e)

                # Subtract per-bin mean; back off to global mean if bin sparse
                r_corr = r.copy()
                for t in range(T):
                    bt = int(b[t])
                    if counts[bt] >= min_bin_count:
                        r_corr[t] = r_corr[t] - mu[bt]
                    else:
                        r_corr[t] = r_corr[t] - global_mu

                r_win = _window_last_frames(r_corr, window)
                feats.append(r_win.mean(axis=0))
            return np.stack(feats, axis=0)

        X_tr = _featurize(train_idx)
        X_te = _featurize(test_idx)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

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


def decode_d3_supervised_mean_trajectory(
    rates_by_stim: dict[str, list[np.ndarray]],
    window: int,
    n_splits: int = 5,
    C_logistic: float = 1.0,
    rng_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Supervised mean-trajectory decoder (PCA on class means only).

    Per fold:
      - Compute class-mean trajectories over the last W frames (padding if needed)
      - Fit PCA on the K class mean vectors only (K=4), so ≤ 3 dims
      - Project each single-trial flattened trajectory into this subspace
      - Decode with logistic regression
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    stim_ids = sorted(rates_by_stim.keys())
    K = len(stim_ids)
    if K < 2:
        raise ValueError("Need at least 2 classes")

    M_min = min(len(rates_by_stim[sid]) for sid in stim_ids)

    # Build list of flattened trajectories
    X_flat_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[int] = []

    for label, sid in enumerate(stim_ids):
        for trace_idx, r in enumerate(rates_by_stim[sid][:M_min]):
            r_win = _window_last_frames(np.asarray(r), window)
            X_flat_list.append(r_win.reshape(-1))
            y_list.append(label)
            groups_list.append(trace_idx)

    X_flat = np.stack(X_flat_list, axis=0)  # (K*M_min, W*N)
    y_all = np.asarray(y_list, dtype=int)
    groups_all = np.asarray(groups_list, dtype=int)

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []

    for train_idx, test_idx in gkf.split(X_flat, y_all, groups=groups_all):
        # Compute class means using training samples only
        class_means = []
        for k in range(K):
            idx_k = train_idx[y_all[train_idx] == k]
            class_means.append(X_flat[idx_k].mean(axis=0))
        means_matrix = np.stack(class_means, axis=0)  # (K, W*N)

        pca = PCA(n_components=min(K - 1, means_matrix.shape[1]))
        pca.fit(means_matrix)

        X_tr = pca.transform(X_flat[train_idx])
        X_te = pca.transform(X_flat[test_idx])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

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
class Curve:
    windows: list[int]
    mean: np.ndarray
    std: np.ndarray


def _compute_curve(
    fn,
    windows: list[int],
) -> Curve:
    m, s = [], []
    for W in windows:
        print(f"  W={W:>3d} ...", flush=True)
        mean_acc, std_acc, _fold = fn(W)
        m.append(mean_acc)
        s.append(std_acc)
    return Curve(windows=windows, mean=np.asarray(m, dtype=float), std=np.asarray(s, dtype=float))


def _save_plot(
    out_path: Path,
    title: str,
    curves: dict[str, Curve],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=160)
    for label, c in curves.items():
        ax.errorbar(c.windows, c.mean, yerr=c.std, marker="o", linewidth=2, capsize=3, label=label)

    ax.set_xlabel("Integration window W (frames)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.2, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _print_summary_table(
    windows: list[int],
    results: dict[str, dict[int, tuple[float, float]]],
    key_windows: Iterable[int] = (1, 24, 60),
) -> None:
    key_windows = [int(w) for w in key_windows]
    print("\n=== Summary table (mean ± std across folds) ===")
    header = "decoder".ljust(14) + "  " + "  ".join([f"W={w}".rjust(14) for w in key_windows])
    print(header)
    print("-" * len(header))

    for dec_name in sorted(results.keys()):
        row = dec_name.ljust(14)
        for w in key_windows:
            if w not in results[dec_name]:
                row += "  " + "(n/a)".rjust(14)
                continue
            m, s = results[dec_name][w]
            row += "  " + f"{m:.3f}±{s:.3f}".rjust(14)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logmar", type=float, required=True)
    parser.add_argument("--rates_dir", type=str, default="scripts/temporal_decoding/data/rates")
    parser.add_argument("--eye_traces_path", type=str, default="scripts/temporal_decoding/data/eye_traces.npz")
    parser.add_argument("--hires_threshold", type=float, default=0.35)
    parser.add_argument("--windows", type=str, default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--C_logistic", type=float, default=1.0)
    parser.add_argument(
        "--use_stabilized_eye",
        action="store_true",
        help="For D2a, use per-trace constant mean eye position (recommended).",
    )

    # Nonlinear D1 (MLP) controls
    parser.add_argument("--run_mlp", action="store_true", help="Run nonlinear D1 (MLP) on the same time-mean feature")
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        help="Only run D1 linear + D1 MLP (skip D2a/D2b/D3) for faster nonlinear-rescue tests",
    )
    parser.add_argument("--mlp_hidden1", type=int, default=128)
    parser.add_argument("--mlp_hidden2", type=int, default=64)
    parser.add_argument("--mlp_dropout", type=float, default=0.15)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_max_epochs", type=int, default=200)
    parser.add_argument("--mlp_patience", type=int, default=20)
    parser.add_argument("--mlp_val_frac", type=float, default=0.2)
    parser.add_argument("--mlp_batch_size", type=int, default=128)

    # Optional decoders
    parser.add_argument("--auto", action="store_true", help="Auto-run D2b/D3 based on D1/D2a outcomes")
    parser.add_argument("--run_d2b", action="store_true", help="Force-run D2b")
    parser.add_argument("--run_d3", action="store_true", help="Force-run D3")
    parser.add_argument("--d2b_n_bins", type=int, default=9)
    parser.add_argument("--d2b_min_bin_count", type=int, default=200)
    parser.add_argument("--auto_eps", type=float, default=0.01, help="Threshold for considering a decoder improvement")

    args = parser.parse_args()

    rates_dir = Path(args.rates_dir)
    eye_path = Path(args.eye_traces_path)

    windows = [int(x) for x in str(args.windows).split(",") if str(x).strip()]

    print("\n" + "=" * 72)
    print("E-optotype decoder controls")
    print("=" * 72)
    print(f"logmar={args.logmar:+.2f} | windows={windows} | n_splits={args.n_splits} | C={args.C_logistic}")

    rates_by_cond, _lengths_by_cond = build_rates_by_condition(
        rates_dir=rates_dir,
        logmar=float(args.logmar),
        hires_threshold=float(args.hires_threshold),
    )

    eye_traces, eye_durations = _load_eye_traces(eye_path)

    mlp_cfg: MLPConfig | None = None
    if bool(args.run_mlp):
        # Prefer CUDA if available, but default to CPU if not.
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        mlp_cfg = MLPConfig(
            hidden1=int(args.mlp_hidden1),
            hidden2=int(args.mlp_hidden2),
            dropout=float(args.mlp_dropout),
            lr=float(args.mlp_lr),
            weight_decay=float(args.mlp_weight_decay),
            max_epochs=int(args.mlp_max_epochs),
            patience=int(args.mlp_patience),
            val_frac=float(args.mlp_val_frac),
            batch_size=int(args.mlp_batch_size),
            device=str(device),
        )

    # D1 curves for both conditions
    d1_curves: dict[str, Curve] = {}
    d1_at_w: dict[str, dict[int, tuple[float, float]]] = {"D1_real": {}, "D1_stabilized": {}}

    for cond in DEFAULT_CONDITIONS:
        rates_by_stim = rates_by_cond[cond]

        def _fn(W: int):
            return decode_d1_time_mean(
                rates_by_stim=rates_by_stim,
                window=W,
                n_splits=int(args.n_splits),
                C_logistic=float(args.C_logistic),
            )

        curve = _compute_curve(_fn, windows)
        d1_curves[f"D1 {cond}"] = curve
        for w, m, s in zip(curve.windows, curve.mean, curve.std, strict=True):
            d1_at_w[f"D1_{cond}"][int(w)] = (float(m), float(s))

    # Nonlinear D1 (MLP) curves for both conditions
    d1_mlp_curves: dict[str, Curve] = {}
    d1_mlp_at_w: dict[str, dict[int, tuple[float, float]]] = {"D1mlp_real": {}, "D1mlp_stabilized": {}}
    if mlp_cfg is not None:
        for cond in DEFAULT_CONDITIONS:
            rates_by_stim = rates_by_cond[cond]

            def _fn(W: int):
                return decode_d1_time_mean_mlp(
                    rates_by_stim=rates_by_stim,
                    window=W,
                    cfg=mlp_cfg,
                    n_splits=int(args.n_splits),
                )

            curve = _compute_curve(_fn, windows)
            d1_mlp_curves[f"D1mlp {cond}"] = curve
            for w, m, s in zip(curve.windows, curve.mean, curve.std, strict=True):
                d1_mlp_at_w[f"D1mlp_{cond}"][int(w)] = (float(m), float(s))

    # D2a (eye-aware) for real and stabilized (for completeness)
    d2_curves: dict[str, Curve] = {"D1 stabilized": d1_curves["D1 stabilized"], "D1 real": d1_curves["D1 real"]}
    d2_at_w: dict[str, dict[int, tuple[float, float]]] = {
        "D1_stabilized": d1_at_w["D1_stabilized"],
        "D1_real": d1_at_w["D1_real"],
        "D2a_real": {},
        "D2a_stabilized": {},
    }

    if not bool(args.mlp_only):
        for cond in DEFAULT_CONDITIONS:
            rates_by_stim = rates_by_cond[cond]
            use_stab_eye = bool(args.use_stabilized_eye and cond == "stabilized")

            def _fn(W: int):
                return decode_d2a_time_mean_plus_eye(
                    rates_by_stim=rates_by_stim,
                    window=W,
                    eye_traces=eye_traces,
                    eye_durations=eye_durations,
                    use_stabilized_eye=use_stab_eye,
                    n_splits=int(args.n_splits),
                    C_logistic=float(args.C_logistic),
                )

            curve = _compute_curve(_fn, windows)
            d2_curves[f"D2a {cond}"] = curve
            for w, m, s in zip(curve.windows, curve.mean, curve.std, strict=True):
                d2_at_w[f"D2a_{cond}"][int(w)] = (float(m), float(s))

    # Decide whether to run D2b and/or D3
    do_d2b = bool(args.run_d2b)
    do_d3 = bool(args.run_d3)

    if (not bool(args.mlp_only)) and args.auto and not (do_d2b or do_d3):
        # D2b only makes sense if D2a improves over D1 for the real condition.
        d2a_real = d2_curves["D2a real"].mean
        d1_real = d1_curves["D1 real"].mean
        d2a_improve = float(np.max(d2a_real - d1_real))
        if d2a_improve >= float(args.auto_eps):
            do_d2b = True

        # D3 only if neither D1 nor D2a (nor D2b) closes the gap to stabilized.
        # We check after D2b, but if D2b isn't planned, decide here.
        d1_stab = d1_curves["D1 stabilized"].mean
        max_real = float(np.max(np.maximum(d1_real, d2a_real)))
        max_stab = float(np.max(d1_stab))
        if (max_stab - max_real) >= float(args.auto_eps):
            do_d3 = True

        print("\nAuto-decoder selection:")
        print(f"  max(D2a_real - D1_real) = {d2a_improve:+.3f} -> run D2b: {do_d2b}")
        print(f"  max(stabilized D1) - max(real best) = {max_stab-max_real:+.3f} -> run D3: {do_d3}")

    extra_curves: dict[str, Curve] = {}
    extra_at_w: dict[str, dict[int, tuple[float, float]]] = {}

    if do_d2b:
        rates_by_stim_real = rates_by_cond["real"]

        def _fn(W: int):
            return decode_d2b_eye_conditioned_correction(
                rates_by_stim=rates_by_stim_real,
                window=W,
                eye_traces=eye_traces,
                eye_durations=eye_durations,
                n_bins=int(args.d2b_n_bins),
                min_bin_count=int(args.d2b_min_bin_count),
                n_splits=int(args.n_splits),
                C_logistic=float(args.C_logistic),
            )

        curve = _compute_curve(_fn, windows)
        extra_curves["D2b real"] = curve
        extra_at_w["D2b_real"] = {int(w): (float(m), float(s)) for w, m, s in zip(curve.windows, curve.mean, curve.std, strict=True)}

    if do_d3:
        rates_by_stim_real = rates_by_cond["real"]

        def _fn(W: int):
            return decode_d3_supervised_mean_trajectory(
                rates_by_stim=rates_by_stim_real,
                window=W,
                n_splits=int(args.n_splits),
                C_logistic=float(args.C_logistic),
            )

        curve = _compute_curve(_fn, windows)
        extra_curves["D3 real"] = curve
        extra_at_w["D3_real"] = {int(w): (float(m), float(s)) for w, m, s in zip(curve.windows, curve.mean, curve.std, strict=True)}

    # Save plots
    figures_dir = Path("scripts/temporal_decoding/figures")
    results_dir = Path("scripts/temporal_decoding/data/results")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    lm_tag = f"lm{float(args.logmar):+.2f}".replace("+", "")

    out_suffix = ""
    if bool(args.run_mlp):
        out_suffix += "_mlp"
    if bool(args.mlp_only):
        out_suffix += "_only"

    d1_fig = figures_dir / f"fig_decoder_controls_D1{out_suffix}_{lm_tag}.png"
    d2_fig = figures_dir / f"fig_decoder_controls_D2{out_suffix}_{lm_tag}.png"

    _save_plot(
        d1_fig,
        title=f"D1 time-mean decoder (logmar={float(args.logmar):+.2f})",
        curves=(
            {
                "stabilized (linear)": d1_curves["D1 stabilized"],
                "real FEM (linear)": d1_curves["D1 real"],
            }
            | (
                {
                    "stabilized (MLP)": d1_mlp_curves["D1mlp stabilized"],
                    "real FEM (MLP)": d1_mlp_curves["D1mlp real"],
                }
                if mlp_cfg is not None
                else {}
            )
        ),
    )

    if not bool(args.mlp_only):
        d2_plot_curves = {
            "stabilized D1": d1_curves["D1 stabilized"],
            "real D1": d1_curves["D1 real"],
            "real D2a (rate+eye)": d2_curves["D2a real"],
        }
        if do_d2b:
            d2_plot_curves["real D2b (eye-corrected)"] = extra_curves["D2b real"]
        if do_d3:
            d2_plot_curves["real D3 (traj subspace)"] = extra_curves["D3 real"]

        _save_plot(
            d2_fig,
            title=f"Decoder controls (logmar={float(args.logmar):+.2f})",
            curves=d2_plot_curves,
        )

    # Save numeric results
    out_npz = results_dir / f"decoder_controls{out_suffix}_{lm_tag}.npz"

    def _pack(curve: Curve) -> dict[str, np.ndarray]:
        return {
            "windows": np.asarray(curve.windows, dtype=int),
            "mean": np.asarray(curve.mean, dtype=float),
            "std": np.asarray(curve.std, dtype=float),
        }

    payload = {
        "logmar": np.asarray([float(args.logmar)], dtype=float),
        "windows": np.asarray(windows, dtype=int),
        "n_splits": np.asarray([int(args.n_splits)], dtype=int),
        "C_logistic": np.asarray([float(args.C_logistic)], dtype=float),
        "D1_real_mean": d1_curves["D1 real"].mean,
        "D1_real_std": d1_curves["D1 real"].std,
        "D1_stabilized_mean": d1_curves["D1 stabilized"].mean,
        "D1_stabilized_std": d1_curves["D1 stabilized"].std,
        "mlp_only": np.asarray([bool(args.mlp_only)], dtype=bool),
    }
    if not bool(args.mlp_only):
        payload["D2a_real_mean"] = d2_curves["D2a real"].mean
        payload["D2a_real_std"] = d2_curves["D2a real"].std
        payload["D2a_stabilized_mean"] = d2_curves["D2a stabilized"].mean
        payload["D2a_stabilized_std"] = d2_curves["D2a stabilized"].std
    if mlp_cfg is not None:
        payload["D1mlp_real_mean"] = d1_mlp_curves["D1mlp real"].mean
        payload["D1mlp_real_std"] = d1_mlp_curves["D1mlp real"].std
        payload["D1mlp_stabilized_mean"] = d1_mlp_curves["D1mlp stabilized"].mean
        payload["D1mlp_stabilized_std"] = d1_mlp_curves["D1mlp stabilized"].std
        payload["mlp_hidden1"] = np.asarray([int(mlp_cfg.hidden1)], dtype=int)
        payload["mlp_hidden2"] = np.asarray([int(mlp_cfg.hidden2)], dtype=int)
        payload["mlp_dropout"] = np.asarray([float(mlp_cfg.dropout)], dtype=float)
        payload["mlp_lr"] = np.asarray([float(mlp_cfg.lr)], dtype=float)
        payload["mlp_weight_decay"] = np.asarray([float(mlp_cfg.weight_decay)], dtype=float)
        payload["mlp_max_epochs"] = np.asarray([int(mlp_cfg.max_epochs)], dtype=int)
        payload["mlp_patience"] = np.asarray([int(mlp_cfg.patience)], dtype=int)
        payload["mlp_val_frac"] = np.asarray([float(mlp_cfg.val_frac)], dtype=float)
        payload["mlp_batch_size"] = np.asarray([int(mlp_cfg.batch_size)], dtype=int)
        payload["mlp_device"] = np.asarray([str(mlp_cfg.device)], dtype=object)
    if do_d2b:
        payload["D2b_real_mean"] = extra_curves["D2b real"].mean
        payload["D2b_real_std"] = extra_curves["D2b real"].std
    if do_d3:
        payload["D3_real_mean"] = extra_curves["D3 real"].mean
        payload["D3_real_std"] = extra_curves["D3 real"].std

    np.savez(out_npz, **payload)
    print(f"\nSaved results: {out_npz}")
    print(f"Saved figures: {d1_fig}")
    if not bool(args.mlp_only):
        print(f"Saved figures: {d2_fig}")

    # Print summary table at key windows
    summary: dict[str, dict[int, tuple[float, float]]] = {
        "D1_stabilized": d1_at_w["D1_stabilized"],
        "D1_real": d1_at_w["D1_real"],
    }
    if not bool(args.mlp_only):
        summary["D2a_real"] = d2_at_w["D2a_real"]
    if mlp_cfg is not None:
        summary["D1mlp_stabilized"] = d1_mlp_at_w["D1mlp_stabilized"]
        summary["D1mlp_real"] = d1_mlp_at_w["D1mlp_real"]
    if do_d2b:
        summary["D2b_real"] = extra_at_w["D2b_real"]
    if do_d3:
        summary["D3_real"] = extra_at_w["D3_real"]

    _print_summary_table(windows, summary, key_windows=(1, 24, 60))

    # Brief interpretation hints (based on success criteria)
    d1_real = d1_curves["D1 real"].mean
    d1_stab = d1_curves["D1 stabilized"].mean
    if not bool(args.mlp_only):
        d2a_real = d2_curves["D2a real"].mean
        best_real = np.maximum(d1_real, d2a_real)
        best_real_name = "D2a" if float(np.max(d2a_real - d1_real)) > 1e-9 else "D1"
    else:
        best_real = d1_real
        best_real_name = "D1"

    print("\n=== Quick interpretation (heuristic) ===")
    print(f"Best real decoder among D1/D2a: {best_real_name}")
    if not bool(args.mlp_only):
        print(f"max(real D1) = {float(np.max(d1_real)):.3f} | max(real D2a) = {float(np.max(d2a_real)):.3f} | max(stabilized D1) = {float(np.max(d1_stab)):.3f}")
    else:
        print(f"max(real D1) = {float(np.max(d1_real)):.3f} | max(stabilized D1) = {float(np.max(d1_stab)):.3f}")

    if float(np.max(d1_real)) >= float(np.max(d1_stab)) + float(args.auto_eps):
        print("Case: D1 crosses stabilized -> strong evidence multi-position sampling adds orientation info")
    elif float(np.max(best_real)) >= float(np.max(d1_stab)) + float(args.auto_eps):
        print("Case: D2 exceeds stabilized -> eye-state dependent / transformation-contingent info")
    elif float(np.max(d1_real)) >= float(np.max(d1_stab)) - float(args.auto_eps):
        print("Case: D1 approaches stabilized -> consistent with smooth manifold / averaging recovers")
    else:
        print("Case: none exceed stabilized -> SSI gain likely reflects position info, not orientation")

    if mlp_cfg is not None:
        d1_mlp_real = d1_mlp_curves["D1mlp real"].mean
        d1_mlp_stab = d1_mlp_curves["D1mlp stabilized"].mean
        rescue_real = d1_mlp_real - d1_real
        rescue_stab = d1_mlp_stab - d1_stab
        rescue_diff = rescue_real - rescue_stab
        print("\n=== Nonlinear rescue metric (MLP vs linear D1) ===")
        print(f"max(rescue_real) = {float(np.max(rescue_real)):+.3f} | max(rescue_stabilized) = {float(np.max(rescue_stab)):+.3f}")
        print(f"max( rescue_real - rescue_stabilized ) = {float(np.max(rescue_diff)):+.3f}")


if __name__ == "__main__":
    main()

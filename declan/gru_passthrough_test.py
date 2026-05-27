"""
GRU Passthrough Test
====================
Tests whether the ConvGRU is doing meaningful temporal integration within its
32-frame window, or is effectively a passthrough of the most recent frame.

Two complementary tests:

Test 1 – Temporal integration (primary)
    For each windowed sample, compare predictions using:
    (a) the real 32-frame history (dynamic)
    (b) all history frames replaced by the current frame (static)
    If R² ≈ 1.0 → GRU ignores its history; the 32-frame context buys nothing.

Test 2 – Representational similarity analysis (secondary)
    Compare the structure of convnet features (pre-GRU) vs GRU features
    (post-GRU) using representational dissimilarity matrices.
    If RSA ≈ 1.0 → GRU is a feature-level passthrough.

Usage
-----
python declan/gru_passthrough_test.py
python declan/gru_passthrough_test.py --n-frames 500
python declan/gru_passthrough_test.py --image BrightTrees.JPG
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, '/home/declan/DataYatesV1')

import DataYatesV1  # noqa: F401
from spatial_info import make_counterfactual_stim
from utils import get_model_and_dataset_configs

FIGURES_DIR = os.path.join(ROOT, 'declan', 'gru_passthrough_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

N_LAGS   = 32
OUT_SIZE = (151, 151)
PPD      = 37.5
BATCH_PRED = 64
BATCH_FEAT = 8
DATASET_IDX = 0   # Yates/McFarland dataset


# ── Model loading ─────────────────────────────────────────────────────────────

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    model, _ = get_model_and_dataset_configs()
    device = get_device()
    model = model.to(device).eval()
    return model


# ── Stimulus generation ────────────────────────────────────────────────────────

def load_image(name: str) -> np.ndarray:
    bg_dir = '/home/declan/DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds'
    from PIL import Image
    path = os.path.join(bg_dir, name)
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.clip(arr, 0, 255)


def make_fem_trace(T: int, sigma_deg: float = 0.03, seed: int = 0) -> np.ndarray:
    """Brownian random walk eye trace with FEM-like amplitude."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, sigma_deg, size=(T, 2)).astype(np.float32)
    return np.cumsum(steps, axis=0)


def build_stim(image_gray: np.ndarray, eyepos: np.ndarray) -> torch.Tensor:
    """Returns (T_out, 1, N_LAGS, H, W) normalized stimulus on CPU.

    Keeping the full stimulus stack on CPU avoids pinning large tensors in GPU
    memory across multiple tests. Batches are moved to GPU during inference.
    """
    full_stack = np.repeat(image_gray[np.newaxis], eyepos.shape[0] + N_LAGS, axis=0)
    eyepos_t   = torch.from_numpy(eyepos)
    stim       = make_counterfactual_stim(
        full_stack, eyepos_t, out_size=OUT_SIZE, n_lags=N_LAGS, ppd=PPD
    )
    stim_norm  = (stim.to(torch.float32) - 127.0) / 255.0
    return stim_norm


def make_static_stim(stim_dynamic: torch.Tensor) -> torch.Tensor:
    """Replace all history frames with the current frame (lag 0 = index 0 in dim 2).

    Returns a view (no allocation) when possible.
    """
    cur = stim_dynamic[:, :, 0:1]
    return cur.expand_as(stim_dynamic)


# ── Inference helpers ──────────────────────────────────────────────────────────

def diagnose_shapes(model, device: str) -> int:
    """
    Run one synthetic forward pass and print T at each model stage.
    Returns the GRU output T (number of timesteps the GRU actually processes).
    This tells us whether the GRU is a temporal integrator or a spatial nonlinearity.
    """
    x = torch.zeros(1, 1, N_LAGS, 64, 64, device=device)
    print('Shape diagnostics (T dimension at each stage):')
    print(f'  Input          : {tuple(x.shape)}  → T={x.shape[2]}')
    with torch.no_grad():
        x_fe  = model.model.frontend(x)
        print(f'  After frontend : {tuple(x_fe.shape)}  → T={x_fe.shape[2]}')
        x_cn  = model.model.convnet(x_fe)
        print(f'  After convnet  : {tuple(x_cn.shape)}  → T={x_cn.shape[2]}')
        x_gru = model.model.recurrent(x_cn)
        print(f'  After GRU      : {tuple(x_gru.shape)}  → T={x_gru.shape[2]}')
        # confirm readout accepts 5D and takes last timestep
        out   = model.model.readouts[DATASET_IDX](x_gru)
        print(f'  After readout  : {tuple(out.shape)}')

    gru_t = int(x_gru.shape[2])
    if gru_t == 1:
        print('\n  *** T=1 entering the GRU.  The GRU computes exactly one step —')
        print('      it is a spatial nonlinearity, not a temporal integrator.')
        print('      All temporal processing lives in the frontend + convnet.')
        print('      The make_static_stim test probes FRONTEND temporal sensitivity,')
        print('      not GRU recurrence.\n')
    else:
        print(f'\n  GRU processes {gru_t} temporal steps — it has genuine recurrence.')
        print(f'  make_static_stim tests whether those {gru_t} steps are used.\n')
    return gru_t


def predict_batched(
    model,
    stim: torch.Tensor,
    device: str,
    static_history: bool = False,
    batch_size: int = BATCH_PRED,
) -> np.ndarray:
    """
    Run model predictions in batches. Returns (T_out, N_neurons) float32 array.

    `stim` is expected on CPU; batches are moved to `device` for inference.
    """
    preds = []
    use_amp = (device == 'cuda')

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(stim), batch_size):
            batch = stim[i:i + batch_size].to(device, non_blocking=True)

            if static_history:
                batch = batch.clone()
                batch[:, :, 1:] = batch[:, :, 0:1]

            core = model.model.core_forward(batch, behavior=None)  # (B, C, T, H, W)
            out  = model.model.readouts[DATASET_IDX](core)         # (B, N)
            out  = model.model.activation(out)                     # softplus
            preds.append(out.float().cpu().numpy())

    return np.concatenate(preds, axis=0)


def extract_features_batched(
    model,
    stim: torch.Tensor,
    device: str,
    batch_size: int = BATCH_FEAT,
):
    """
    Extract convnet (pre-GRU) and GRU (post-GRU) features at the last timestep.
    Returns two arrays of shape (T_out, D).

    Uses a smaller batch size than prediction by default to avoid CUDA OOM.
    """
    conv_feats = []
    gru_feats  = []
    use_amp = (device == 'cuda')

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(stim), batch_size):
            batch = stim[i:i + batch_size].to(device, non_blocking=True)

            # manual forward through components (no adapter, matching core_forward)
            x     = model.model.frontend(batch)
            fconv = model.model.convnet(x)        # (B, C_conv, T, H, W)
            fgru  = model.model.recurrent(fconv)  # (B, C_gru,  T, H, W)

            # take last timestep, flatten spatial dims
            B = fconv.shape[0]
            conv_feats.append(fconv[:, :, -1].float().reshape(B, -1).cpu().numpy())
            gru_feats.append(fgru[:,  :, -1].float().reshape(B, -1).cpu().numpy())

            del batch, x, fconv, fgru

    return (np.concatenate(conv_feats, axis=0),
            np.concatenate(gru_feats,  axis=0))


# ── Analysis ───────────────────────────────────────────────────────────────────

def pearson_r_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pairwise Pearson r between rows of X (N, D).
    Returns (N, N) correlation matrix.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn = Xc / norms
    return Xn @ Xn.T


def rsa_correlation(feat_a: np.ndarray, feat_b: np.ndarray,
                    n_samples: int = 500) -> float:
    """
    Representational similarity: Pearson r between upper triangles of the
    pairwise correlation matrices of feat_a and feat_b.
    Subsample rows if N > n_samples to keep it fast.
    """
    N = feat_a.shape[0]
    if N > n_samples:
        idx = np.random.default_rng(42).choice(N, n_samples, replace=False)
        feat_a = feat_a[idx]
        feat_b = feat_b[idx]

    rsa_a = pearson_r_matrix(feat_a)
    rsa_b = pearson_r_matrix(feat_b)

    triu = np.triu_indices(rsa_a.shape[0], k=1)
    ra, rb = rsa_a[triu], rsa_b[triu]
    r, _ = pearsonr(ra, rb)
    return float(r)


def per_neuron_r2(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """R² between pred_a and pred_b for each neuron (column). Returns (N_neurons,)."""
    r2 = np.zeros(pred_a.shape[1])
    for n in range(pred_a.shape[1]):
        a, b = pred_a[:, n], pred_b[:, n]
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((b - b.mean()) ** 2)
        r2[n] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_temporal_integration(pred_dynamic, pred_static, r2_per_neuron, label):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'Temporal integration test — {label}', fontsize=12)

    # 1. Scatter: one neuron (median R²)
    ax = axes[0]
    med_idx = int(np.argsort(r2_per_neuron)[len(r2_per_neuron) // 2])
    ax.scatter(pred_static[:, med_idx], pred_dynamic[:, med_idx],
               s=4, alpha=0.3, color='steelblue')
    lo = min(pred_static[:, med_idx].min(), pred_dynamic[:, med_idx].min())
    hi = max(pred_static[:, med_idx].max(), pred_dynamic[:, med_idx].max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    ax.set_xlabel('pred (static window)')
    ax.set_ylabel('pred (dynamic window)')
    ax.set_title(f'Median-R² neuron (R²={r2_per_neuron[med_idx]:.3f})')

    # 2. Histogram of per-neuron R²
    ax = axes[1]
    ax.hist(r2_per_neuron, bins=30, color='steelblue', edgecolor='white')
    ax.axvline(np.median(r2_per_neuron), color='red', linestyle='--',
               label=f'median={np.median(r2_per_neuron):.3f}')
    ax.set_xlabel('R² (dynamic vs. static window)')
    ax.set_ylabel('# neurons')
    ax.set_title('Per-neuron R²')
    ax.legend()

    # 3. Mean prediction traces (first 200 time points)
    ax = axes[2]
    mean_dyn = pred_dynamic[:200].mean(axis=1)
    mean_sta = pred_static[:200].mean(axis=1)
    ax.plot(mean_dyn, label='dynamic', linewidth=0.8)
    ax.plot(mean_sta, label='static',  linewidth=0.8)
    ax.set_xlabel('time (frames)')
    ax.set_ylabel('mean rate (arb.)')
    ax.set_title('Population mean rate (first 200 frames)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f'temporal_integration_{label}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {out_path}')


def plot_rsa(conv_feats, gru_feats, rsa_r, label, n_show=200):
    idx = np.random.default_rng(0).choice(len(conv_feats),
                                          min(n_show, len(conv_feats)),
                                          replace=False)
    rsa_c = pearson_r_matrix(conv_feats[idx])
    rsa_g = pearson_r_matrix(gru_feats[idx])
    triu  = np.triu_indices(len(idx), k=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'Feature RSA — {label}  (r={rsa_r:.3f})', fontsize=12)

    im0 = axes[0].imshow(rsa_c, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('ConvNet RDM')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(rsa_g, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('GRU RDM')
    plt.colorbar(im1, ax=axes[1])

    axes[2].scatter(rsa_c[triu], rsa_g[triu], s=1, alpha=0.2, color='steelblue')
    lo, hi = -1, 1
    axes[2].plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    axes[2].set_xlabel('ConvNet pairwise r')
    axes[2].set_ylabel('GRU pairwise r')
    axes[2].set_title(f'RSA scatter (r={rsa_r:.3f})')

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f'feature_rsa_{label}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def run(image_name: str, n_frames: int, label: str):
    device = get_device()
    print(f'\n=== {label} | {image_name} | T={n_frames} ===')

    # Load image
    image_gray = load_image(image_name)

    # Build stimulus on CPU (move to GPU per-batch during inference)
    print('Building stimuli ...')
    eyepos_fem = make_fem_trace(n_frames, sigma_deg=0.03, seed=1)
    stim_dynamic = build_stim(image_gray, eyepos_fem)   # (T_out, 1, 32, H, W) on CPU

    T_out = stim_dynamic.shape[0]
    print(f'  Windows: {T_out}')

    # ── Shape diagnostics ─────────────────────────────────────────────────────
    model = load_model()
    gru_t = diagnose_shapes(model, device)

    # ── Test 1: Temporal integration ──────────────────────────────────────────
    print('Test 1: Temporal integration ...')
    pred_dyn = predict_batched(model, stim_dynamic, device=device, static_history=False)  # (T_out, N)
    pred_sta = predict_batched(model, stim_dynamic, device=device, static_history=True)   # (T_out, N)

    r2_neurons = per_neuron_r2(pred_dyn, pred_sta)
    med_r2 = float(np.median(r2_neurons))
    mean_r2 = float(np.mean(r2_neurons))

    print(f'  Median per-neuron R²(dynamic vs static) = {med_r2:.4f}')
    print(f'  Mean   per-neuron R²(dynamic vs static) = {mean_r2:.4f}')
    print(f'  Min / Max R² = {r2_neurons.min():.4f} / {r2_neurons.max():.4f}')

    if gru_t == 1:
        component = 'frontend+convnet filterbank'
        if med_r2 > 0.98:
            verdict = f'({component}) temporal history adds <2% variance — model is effectively single-frame'
        elif med_r2 > 0.90:
            verdict = f'({component}) temporal history adds modest variance — some temporal filtering'
        else:
            verdict = f'({component}) temporal history substantially changes predictions — active temporal filtering'
    else:
        component = f'GRU ({gru_t}-step recurrence)'
        if med_r2 > 0.98:
            verdict = f'({component}) IS a passthrough — temporal history adds <2% explained variance'
        elif med_r2 > 0.90:
            verdict = f'({component}) has MINOR temporal integration — history adds modest variance'
        else:
            verdict = f'({component}) has MEANINGFUL temporal integration — history substantially changes predictions'
    print(f'\n  VERDICT: {verdict}\n')

    plot_temporal_integration(pred_dyn, pred_sta, r2_neurons, label)

    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── Test 2: Feature RSA ───────────────────────────────────────────────────
    print('Test 2: Feature RSA ...')
    conv_feats, gru_feats = extract_features_batched(model, stim_dynamic, device=device)
    print(f'  ConvNet feature shape : {conv_feats.shape}')
    print(f'  GRU    feature shape  : {gru_feats.shape}')

    rsa_r = rsa_correlation(conv_feats, gru_feats, n_samples=500)
    print(f'  RSA (ConvNet vs GRU, pairwise correlation) = {rsa_r:.4f}')

    if gru_t == 1:
        rsa_note = '(T=1: GRU does one step, so RSA measures a single nonlinear transform)'
    else:
        rsa_note = f'(T={gru_t}: RSA measures whether GRU reshapes the temporal-mean representation)'
    if rsa_r > 0.98:
        rsa_verdict = f'features are nearly identical — GRU barely transforms the representation  {rsa_note}'
    elif rsa_r > 0.90:
        rsa_verdict = f'features are moderately similar — GRU applies a partial transformation  {rsa_note}'
    else:
        rsa_verdict = f'features are structurally different — GRU substantially reshapes the representation  {rsa_note}'
    print(f'  VERDICT: {rsa_verdict}\n')

    plot_rsa(conv_feats, gru_feats, rsa_r, label)

    return {
        'label':       label,
        'n_windows':   T_out,
        'med_r2':      med_r2,
        'mean_r2':     mean_r2,
        'rsa_r':       rsa_r,
        'r2_neurons':  r2_neurons,
        'verdict_ti':  verdict,
        'verdict_rsa': rsa_verdict,
    }


def build_eoptotype_stim(logmar: float, orientation: int, n_frames: int) -> torch.Tensor:
    """
    Build a GRU-passthrough stimulus from the high-res E-optotype pipeline.

    Uses a synthetic Brownian FEM trace (same as make_fem_trace) so that the
    temporal structure is comparable to the natural-image test. The output format
    is identical to build_stim: (T_out, 1, N_LAGS, H, W) float32 on CPU.
    """
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'temporal_decoding'))
    from stimulus_hires import hires_counterfactual_stim

    eyepos = make_fem_trace(n_frames, sigma_deg=0.03, seed=1)
    stim = hires_counterfactual_stim(
        orientation_deg=float(orientation),
        logmar=float(logmar),
        eyepos=eyepos,
        condition='real',
        device='cpu',
    )  # (T_valid, 1, n_lags, H, W) float32
    return stim


def run_eoptotype(logmar: float, orientation: int, n_frames: int) -> dict:
    """E-optotype variant of run() — replaces natural image with E at given LogMAR."""
    label = f'eoptotype_lm{logmar:+.2f}_ori{orientation}'
    device = get_device()
    print(f'\n=== {label} | T={n_frames} ===')

    print('Building E-optotype stimuli ...')
    stim_dynamic = build_eoptotype_stim(logmar, orientation, n_frames)
    T_out = stim_dynamic.shape[0]
    print(f'  Windows: {T_out}')

    model = load_model()
    gru_t = diagnose_shapes(model, device)

    print('Test 1: Temporal integration ...')
    pred_dyn = predict_batched(model, stim_dynamic, device=device, static_history=False)
    pred_sta = predict_batched(model, stim_dynamic, device=device, static_history=True)

    r2_neurons = per_neuron_r2(pred_dyn, pred_sta)
    med_r2 = float(np.median(r2_neurons))
    mean_r2 = float(np.mean(r2_neurons))

    print(f'  Median per-neuron R²(dynamic vs static) = {med_r2:.4f}')
    print(f'  Mean   per-neuron R²(dynamic vs static) = {mean_r2:.4f}')

    # Qualitative verdict (same thresholds as run())
    if med_r2 > 0.98:
        verdict = 'temporal history adds <2% variance — effectively single-frame'
    elif med_r2 > 0.90:
        verdict = 'temporal history adds modest variance — some temporal filtering'
    else:
        verdict = 'temporal history substantially changes predictions — active temporal integration'
    print(f'\n  VERDICT: {verdict}\n')

    plot_temporal_integration(pred_dyn, pred_sta, r2_neurons, label)

    if device == 'cuda':
        torch.cuda.empty_cache()

    print('Test 2: Feature RSA ...')
    conv_feats, gru_feats = extract_features_batched(model, stim_dynamic, device=device)
    rsa_r = rsa_correlation(conv_feats, gru_feats, n_samples=500)
    print(f'  RSA (ConvNet vs GRU) = {rsa_r:.4f}')
    plot_rsa(conv_feats, gru_feats, rsa_r, label)

    return {
        'label': label,
        'n_windows': T_out,
        'med_r2': med_r2,
        'mean_r2': mean_r2,
        'rsa_r': rsa_r,
        'r2_neurons': r2_neurons,
        'verdict_ti': verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stimulus', default='natural', choices=['natural', 'eoptotype'],
                    help='Stimulus type: natural image or E-optotype (default: natural)')
    ap.add_argument('--image', default='BrightTrees.JPG',
                    help='Background image filename (natural stimulus only)')
    ap.add_argument('--logmar', type=float, default=-0.40,
                    help='LogMAR letter size for eoptotype stimulus (default: -0.40)')
    ap.add_argument('--orientation', type=int, default=0,
                    help='E orientation in degrees for eoptotype stimulus (default: 0)')
    ap.add_argument('--n-frames', type=int, default=500,
                    help='Number of movie frames to generate (default: 500)')
    args = ap.parse_args()

    if args.stimulus == 'eoptotype':
        results = run_eoptotype(args.logmar, args.orientation, args.n_frames)
    else:
        results = run(args.image, args.n_frames, label=args.image.split('.')[0])

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f"  Stimulus           : {results['label']}")
    print(f"  Windows analysed   : {results['n_windows']}")
    print(f"  Temporal R² (med)  : {results['med_r2']:.4f}  "
          f"← does history matter for predictions?")
    if 'rsa_r' in results:
        print(f"  Feature RSA        : {results['rsa_r']:.4f}  "
              f"← are conv and GRU features similar?")
    print()
    print(f"  Temporal verdict   : {results['verdict_ti']}")
    print(f"  RSA verdict        : {results['verdict_rsa']}")
    print()
    print(f'Figures saved to: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()

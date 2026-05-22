#%%
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path('../declan/overnight_backimage_long_sweeps_20s_re').expanduser().resolve()
# files produced by the updated sweep script are named like "sweep20s_<image>.pkl"
GLOB = 'sweep20s_*.pkl'

# Plot defaults
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 160,
})

RESULTS_DIR.resolve()
#%%
def scan_results(results_dir: Path = RESULTS_DIR):
    pkls = sorted(results_dir.glob(GLOB))
    errs = sorted(results_dir.glob('*.ERROR.txt'))
    return pkls, errs

pkls, errs = scan_results()
print(f'Found {len(pkls)} completed result files')
print(f'Found {len(errs)} error markers')
if errs:
    print('Errors:')
    for p in errs[:20]:
        print(' -', p.name)

#%%
def load_one(path: Path) -> dict:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f'Expected dict in {path}, got {type(obj)}')
    # Accept both legacy and newer field names. Map common alternatives to canonical names
    aliases = {
        'saccade_rates': ('saccade_rates', 'saccade_rate_list', 'saccade_rates_hz'),
        'eye_scale_list': ('eye_scale_list', 'eye_scales', 'eye_scale'),
    }
    for canon, candidates in aliases.items():
        if canon not in obj:
            for c in candidates:
                if c in obj:
                    obj[canon] = obj[c]
                    break

    required = ['image_file', 'saccade_rates', 'eye_scale_list', 'i_spikes', 'i_rates', 'I_t']
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f'Missing keys in {path}: {missing}')

    # Normalize to numpy arrays for downstream stacking
    for k in ['saccade_rates', 'eye_scale_list', 'i_spikes', 'i_rates', 'I_t']:
        try:
            obj[k] = np.asarray(obj[k])
        except Exception:
            raise TypeError(f'Unable to convert key {k} to numpy array in {path}')
    return obj

def load_all(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        try:
            out.append(load_one(p))
        except Exception as e:
            print(f'Failed to load {p.name}: {e}')
    return out

results = load_all(pkls)
print('Loaded:', len(results))
if results:
    r0 = results[0]
    print('Example image:', r0['image_file'])
    print('i_spikes shape (sacc_rates x eye_scales):', r0['i_spikes'].shape)

#%%
def check_grid_compatibility(results: list[dict]):
    if not results:
        return True
    s0 = results[0]['saccade_rates']
    e0 = results[0]['eye_scale_list']
    ok = True
    for r in results[1:]:
        if r['saccade_rates'].shape != s0.shape or np.any(r['saccade_rates'] != s0):
            print('Mismatch saccade_rates for', r['image_file'])
            ok = False
        if r['eye_scale_list'].shape != e0.shape or np.any(r['eye_scale_list'] != e0):
            print('Mismatch eye_scale_list for', r['image_file'])
            ok = False
    return ok

grid_ok = check_grid_compatibility(results)
print('Common grid:', grid_ok)
#%%
def stack_metric(results: list[dict], key: str) -> np.ndarray:
    # returns array of shape (n_images, n_sacc, n_eye)
    arrs_raw = [np.asarray(r[key], dtype=np.float32) for r in results]
    normed = []
    shapes = []
    for a in arrs_raw:
        # Accept either (n_sacc, n_eye) or (n_sacc, n_trials, n_eye).
        if a.ndim == 3:
            # average across trials axis -> (n_sacc, n_eye)
            a2 = np.nanmean(a, axis=1)
        elif a.ndim == 2:
            a2 = a
        else:
            raise ValueError(f"Unexpected array ndim {a.ndim} for key {key}; expected 2 or 3")
        normed.append(a2.astype(np.float32))
        shapes.append(a2.shape)

    # Ensure all shapes match
    uniq = set(shapes)
    if len(uniq) != 1:
        # helpful error with per-file shapes
        for r, s in zip(results, shapes):
            print(f"MISMATCH {r.get('image_file','?')}: {key} -> {s}")
        raise ValueError(f"all input arrays must have the same shape after trial-averaging; seen shapes: {uniq}")

    return np.stack(normed, axis=0)

if results and grid_ok:
    saccade_rates = results[0]['saccade_rates']
    eye_scales = results[0]['eye_scale_list']

    I_sp = stack_metric(results, 'i_spikes')
    I_rt = stack_metric(results, 'i_rates')
    I_t = stack_metric(results, 'I_t')

    print('Stacked shapes:', I_sp.shape, I_rt.shape, I_t.shape)
#%% Aggregate heatmaps
def mean_sem(x: np.ndarray, axis=0):
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=axis)
    sem = np.nanstd(x, axis=axis) / np.sqrt(np.maximum(1, np.sum(~np.isnan(x), axis=axis)))
    return mean, sem

def plot_heatmap(Z: np.ndarray, *, title: str, saccade_rates: np.ndarray, eye_scales: np.ndarray, cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # Z is typically shaped (n_saccade_rates, n_eye_scales). We plot saccade rate on x and eye scale on y.
    im = ax.imshow(Z.T, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Saccade rate (Hz)')
    ax.set_ylabel('Eye scale')

    # x ticks (saccade rates)
    xt = np.arange(len(saccade_rates))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(v) for v in saccade_rates])

    # y ticks (eye scales)
    yt = np.arange(len(eye_scales))
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{v:.2g}' for v in eye_scales])

    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    return fig

if results and grid_ok:
    mean_sp, sem_sp = mean_sem(I_sp, axis=0)
    mean_rt, sem_rt = mean_sem(I_rt, axis=0)
    mean_It, sem_It = mean_sem(I_t, axis=0)

    plot_heatmap(mean_sp, title=f'Mean bits/spike across {len(results)} images', saccade_rates=saccade_rates, eye_scales=eye_scales)
    plot_heatmap(mean_rt, title=f'Mean bits/sec across {len(results)} images', saccade_rates=saccade_rates, eye_scales=eye_scales)
    plot_heatmap(mean_It, title=f'Mean I_t across {len(results)} images', saccade_rates=saccade_rates, eye_scales=eye_scales)
    plt.show()

#%% Per image check
def plot_image(idx: int, metric: str = 'i_spikes'):
    r = results[idx]
    Z = np.asarray(r[metric], dtype=np.float32)
    # If a trials axis is present (n_sacc, n_trials, n_eye), average over trials
    if Z.ndim == 3:
        Z = np.nanmean(Z, axis=1)
    return plot_heatmap(
        Z,
        title=f"{metric} for {r['image_file']}",
        saccade_rates=r['saccade_rates'],
        eye_scales=r['eye_scale_list'],
    )

if results:
    idx = 1
    plot_image(idx, 'i_spikes')
    plt.show()


#%% Summary table
def best_point_per_image(metric: str = 'i_spikes'):
    rows = []
    for r in results:
        Z = np.asarray(r[metric], dtype=np.float32)
        # If a trials axis exists (n_sacc, n_trials, n_eye), average across trials
        if Z.ndim == 3:
            Z2 = np.nanmean(Z, axis=1)
        else:
            Z2 = Z

        if not np.isfinite(Z2).any():
            continue
        flat = np.nanargmax(Z2)
        i_sacc, i_eye = np.unravel_index(flat, Z2.shape)
        rows.append({
            'image_file': r['image_file'],
            'best_value': float(Z2[i_sacc, i_eye]),
            'best_saccade_rate': float(r['saccade_rates'][i_sacc]),
            'best_eye_scale': float(r['eye_scale_list'][i_eye]),
            'n_trials': int(r.get('n_trials', -1)),
        })
    rows = sorted(rows, key=lambda d: -d['best_value'])
    return rows

rows = best_point_per_image('i_spikes')
for r in rows[:20]:
    print(
        f"{r['best_value']:.4f} bits/spike | sacc={r['best_saccade_rate']:.2g}Hz | eye={r['best_eye_scale']:.3g} | {r['image_file']}"
    )
    
#%% Live refresh
pkls, errs = scan_results()
results = load_all(pkls)
print(f'Now loaded {len(results)} result files; {len(errs)} errors')
if results:
    grid_ok = check_grid_compatibility(results)
    print('Common grid:', grid_ok)
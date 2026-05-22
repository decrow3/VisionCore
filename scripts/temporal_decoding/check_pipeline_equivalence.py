"""
Pipeline equivalence check.

Compares the lo-res (e_optotype_stack + make_counterfactual_stim) and hi-res
(stimulus_hires.py) pipelines at a comfortably-resolved LogMAR value (default 0.4).

If the two pipelines produce systematically different inputs or model responses,
cross-regime conclusions between the resolved and hyperacuity sweeps are unsafe.

Checks:
  1. Stimulus statistics: mean, std, histogram of input frames
  2. Per-neuron mean response: Pearson correlation between pipelines
  3. Model A decoding accuracy for each pipeline

Usage:
    python check_pipeline_equivalence.py [--logmar 0.4] [--n_traces 20]
                                          [--n_splits 3]

Output:
    Printed comparison table + figures/diag_pipeline_equivalence_lm{logmar}.png
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
EYE_TRACES_PATH = os.path.join(DATA_DIR, 'eye_traces.npz')
PKL_PATH = os.path.join(SCRIPT_DIR, '..', 'mcfarland_outputs_mono.pkl')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

ORIENTATIONS = [0, 90, 180, 270]


def load_model_readout(device):
    import dill
    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout
    model, _ = get_model_and_dataset_configs(mode='standard')
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    model = model.to(device)
    with open(PKL_PATH, 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)
    return model, readout


def get_lores_rates(model, readout, orientation, logmar, eye_traces,
                    durations, n_traces, device):
    """Compute rates via lo-res pipeline for n_traces stabilized trials."""
    from stimulus import e_optotype_stack
    from rate_computation import compute_population_rates

    stim_stack = e_optotype_stack(orientation, logmar)
    result = compute_population_rates(
        model, readout, stim_stack, eye_traces[:n_traces], durations[:n_traces],
        condition='stabilized',
        verbose=False,
    )
    return result['rates']   # list of (T_m, N)


def get_hires_rates(model, readout, orientation, logmar, eye_traces,
                    durations, n_traces, device):
    """Compute rates via hi-res pipeline for n_traces stabilized trials."""
    from rate_computation import compute_population_rates_hires

    result = compute_population_rates_hires(
        model, readout, orientation, logmar,
        eye_traces[:n_traces], durations[:n_traces],
        condition='stabilized',
        verbose=False,
    )
    return result['rates']   # list of (T_m, N)


def stim_statistics(rates_list, label):
    """Summarize statistics of a list of (T, N) rate arrays."""
    all_rates = np.concatenate([r.flatten() for r in rates_list])
    stats = {
        'label': label,
        'mean': float(all_rates.mean()),
        'std': float(all_rates.std()),
        'min': float(all_rates.min()),
        'max': float(all_rates.max()),
        'p5': float(np.percentile(all_rates, 5)),
        'p95': float(np.percentile(all_rates, 95)),
        'n_frames': sum(r.shape[0] for r in rates_list),
        'N': rates_list[0].shape[1],
    }
    return stats, all_rates


def per_neuron_mean_response(rates_list):
    """Return (N,) mean response per neuron, averaged across frames and trials."""
    all_rates = np.concatenate(rates_list, axis=0)  # (T_total, N)
    return all_rates.mean(axis=0)  # (N,)


def run_model_A_accuracy(rates_list, n_splits, label):
    """
    Run a minimal Model-A decoding on a list of per-orientation rate lists.
    rates_by_stim: dict ori → list of (T, N) arrays
    """
    from decoding import run_decoding_ladder
    rates_by_stim = {f'ori{ori}': rates_list[i] for i, ori in enumerate(ORIENTATIONS)}
    ladder = run_decoding_ladder(
        rates_by_stim,
        models=['A'],
        n_splits=n_splits,
        verbose=False,
    )
    return ladder['A']['mean_acc'], ladder['A']['std_acc']


def main():
    parser = argparse.ArgumentParser(description='Pipeline equivalence check')
    parser.add_argument('--logmar', type=float, default=0.4)
    parser.add_argument('--n_traces', type=int, default=20)
    parser.add_argument('--n_splits', type=int, default=3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"LogMAR={args.logmar}, n_traces={args.n_traces}, n_splits={args.n_splits}")

    if args.logmar > 0.3:
        pass  # both pipelines applicable
    else:
        print(f"WARNING: LogMAR={args.logmar} is below the hi-res threshold. "
              f"Lo-res pipeline may lose orientation information at this scale.")

    # Load eye traces
    from extract_eye_traces import load_eye_traces, print_summary
    td = load_eye_traces(EYE_TRACES_PATH)
    print_summary(td)
    eye_traces = td['traces']
    durations = td['durations']

    # Load model
    print("Loading model...")
    model, readout = load_model_readout(device)

    # Compute rates for all orientations via both pipelines
    print(f"\nComputing lo-res rates (LogMAR={args.logmar})...")
    lores_by_ori = []
    for ori in ORIENTATIONS:
        print(f"  Orientation {ori}°...", flush=True)
        rates = get_lores_rates(model, readout, ori, args.logmar,
                                eye_traces, durations, args.n_traces, device)
        lores_by_ori.append(rates)

    print(f"\nComputing hi-res rates (LogMAR={args.logmar})...")
    hires_by_ori = []
    for ori in ORIENTATIONS:
        print(f"  Orientation {ori}°...", flush=True)
        rates = get_hires_rates(model, readout, ori, args.logmar,
                                eye_traces, durations, args.n_traces, device)
        hires_by_ori.append(rates)

    # --- Check 1: response statistics ---
    all_lores = [r for rates in lores_by_ori for r in rates]
    all_hires = [r for rates in hires_by_ori for r in rates]
    stats_lo, flat_lo = stim_statistics(all_lores, 'lo-res')
    stats_hi, flat_hi = stim_statistics(all_hires, 'hi-res')

    print(f"\n{'='*60}")
    print("1. Response statistics comparison:")
    print(f"{'Metric':>15}  {'lo-res':>10}  {'hi-res':>10}  {'ratio':>8}  {'flag':>5}")
    print('-' * 55)
    for metric in ['mean', 'std', 'p5', 'p95']:
        lo_val = stats_lo[metric]
        hi_val = stats_hi[metric]
        ratio = hi_val / (lo_val + 1e-12) if abs(lo_val) > 1e-8 else float('nan')
        flag = '⚠️ ' if abs(ratio - 1.0) > 0.2 else 'ok'
        print(f"{metric:>15}  {lo_val:>10.4f}  {hi_val:>10.4f}  {ratio:>8.3f}  {flag:>5}")

    # --- Check 2: per-neuron mean response correlation ---
    mn_lo = per_neuron_mean_response(all_lores)
    mn_hi = per_neuron_mean_response(all_hires)
    r_corr, p_val = pearsonr(mn_lo, mn_hi)
    flag_corr = '⚠️ ' if r_corr < 0.9 else 'ok'
    print(f"\n2. Per-neuron mean response correlation:")
    print(f"   Pearson r = {r_corr:.4f} (p={p_val:.2e})  [{flag_corr}]")
    print(f"   Threshold: r ≥ 0.90")

    # --- Check 3: Model A decoding accuracy ---
    print(f"\n3. Model A decoding accuracy:")
    acc_lo, std_lo = run_model_A_accuracy(lores_by_ori, args.n_splits, 'lo-res')
    acc_hi, std_hi = run_model_A_accuracy(hires_by_ori, args.n_splits, 'hi-res')
    delta_acc = abs(acc_hi - acc_lo)
    flag_acc = '⚠️ ' if delta_acc > 0.05 else 'ok'
    print(f"   lo-res:  {acc_lo:.4f} ± {std_lo:.4f}")
    print(f"   hi-res:  {acc_hi:.4f} ± {std_hi:.4f}")
    print(f"   |Δ| = {delta_acc:.4f}  [{flag_acc}]")
    print(f"   Threshold: |Δ| < 0.05")

    # --- Check 4: Normalization convention ---
    print(f"\n4. Normalization check:")
    print(f"   Hi-res divides by 127.0 (in stimulus_hires.py line ~381)")
    mean_lo = stats_lo['mean']
    mean_hi = stats_hi['mean']
    print(f"   Mean response lo-res: {mean_lo:.4f}")
    print(f"   Mean response hi-res: {mean_hi:.4f}")
    if abs(mean_lo - mean_hi) > 0.1 * max(abs(mean_lo), abs(mean_hi), 0.1):
        print(f"   ⚠️  Mean responses differ by >10% — check normalization in lo-res path")
        print(f"      (confirm make_counterfactual_stim also divides by 127)")
    else:
        print(f"   ok — Mean responses agree within 10%")

    # --- Overall verdict ---
    passed = (abs(stats_lo['mean'] / (stats_hi['mean'] + 1e-12) - 1.0) < 0.2 and
              r_corr >= 0.9 and delta_acc <= 0.05)
    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS ✓' if passed else 'FAIL ⚠️  — cross-regime comparisons may not be safe'}")
    if not passed:
        print("  Recommended action: fix normalization / resampling mismatch before")
        print("  interpreting any resolved vs hyperacuity accuracy comparisons.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: response distribution
    ax = axes[0]
    bins = np.linspace(min(flat_lo.min(), flat_hi.min()),
                       max(flat_lo.max(), flat_hi.max()), 60)
    ax.hist(flat_lo, bins=bins, alpha=0.5, label='lo-res', color='steelblue', density=True)
    ax.hist(flat_hi, bins=bins, alpha=0.5, label='hi-res', color='tomato', density=True)
    ax.set_xlabel('Neural response (a.u.)')
    ax.set_ylabel('Density')
    ax.set_title(f'Response distribution\n(all neurons, all trials, all oris)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: scatter of per-neuron mean response
    ax = axes[1]
    ax.scatter(mn_lo, mn_hi, alpha=0.3, s=8, color='gray')
    lim = [min(mn_lo.min(), mn_hi.min()), max(mn_lo.max(), mn_hi.max())]
    ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Lo-res mean response')
    ax.set_ylabel('Hi-res mean response')
    ax.set_title(f'Per-neuron mean response\nr = {r_corr:.3f}')
    ax.grid(True, alpha=0.3)

    # Panel 3: accuracy bar
    ax = axes[2]
    bars = ax.bar(['lo-res', 'hi-res'], [acc_lo, acc_hi],
                   yerr=[std_lo, std_hi], capsize=6,
                   color=['steelblue', 'tomato'], alpha=0.8)
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.3, label='Chance')
    ax.axhline(0.625, color='k', linestyle='--', alpha=0.5, label='62.5% threshold')
    ax.set_ylim([0.0, 1.1])
    ax.set_ylabel('Model A accuracy')
    ax.set_title(f'Model A accuracy comparison\nLogMAR={args.logmar:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, acc in zip(bars, [acc_lo, acc_hi]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=9)

    lm_str = f'{args.logmar:.2f}'.replace('-', 'neg')
    fig.suptitle(f'Pipeline equivalence: lo-res vs hi-res  |  LogMAR={args.logmar:.2f}',
                 fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f'diag_pipeline_equivalence_lm{lm_str}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

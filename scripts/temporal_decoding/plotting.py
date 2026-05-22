"""
Shared plotting utilities for the temporal decoding analysis.

Provides consistent color schemes, figure saving, and layout helpers
across all analysis modules.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ─── Style ────────────────────────────────────────────────────────────────────

# Color palette
COLORS = {
    'real': 'royalblue',
    'stabilized': 'tomato',
    'scaled_0.5': 'cornflowerblue',
    'scaled_2.0': 'navy',
    'matched_null': 'gray',
    'shuffled': 'dimgray',
}

MODEL_COLORS = {
    'A': 'tomato',
    'B': 'darkorange',
    'C': 'royalblue',
    'D': 'purple',
    'C_mlp': 'mediumblue',
}

MODEL_LABELS = {
    'A': 'Model A (rate only)',
    'B': 'Model B (mean trajectory)',
    'C': 'Model C (full trajectory)',
    'D': 'Model D (+ residual covariance)',
    'C_mlp': 'Model C (MLP)',
}

CONDITION_LABELS = {
    'real': 'Real FEM',
    'stabilized': 'Stabilized',
    'scaled_0.5': 'Half-amplitude FEM',
    'scaled_2.0': 'Double-amplitude FEM',
    'matched_null': 'Matched-budget null',
    'shuffled': 'Shuffled traces',
}


def set_publication_style():
    """Apply publication-quality matplotlib style."""
    mpl.rcParams.update({
        'figure.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.constrained_layout.use': True,
    })


def save_figure(fig: plt.Figure, path: str, dpi: int = 150, **kwargs) -> None:
    """Save figure, creating parent directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"Saved: {path}")


# ─── Summary figures ──────────────────────────────────────────────────────────

def plot_decoding_ladder_bar(
    ladder_results_by_condition: dict,
    models: list = ('A', 'B', 'C'),
    chance: float = 0.25,
    figsize=(8, 5),
) -> plt.Figure:
    """
    Bar chart of decoding accuracy for each model under each condition.
    """
    conditions = list(ladder_results_by_condition.keys())
    n_cond = len(conditions)
    n_models = len(models)

    x = np.arange(n_cond)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(models):
        accs = []
        stds = []
        for cond in conditions:
            r = ladder_results_by_condition[cond].get(model_name, {})
            accs.append(r.get('mean_acc', 0))
            stds.append(r.get('std_acc', 0))

        ax.bar(
            x + offsets[i], accs, width,
            yerr=stds,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name, 'gray'),
            alpha=0.85, capsize=3,
        )

    ax.axhline(y=chance, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=15)
    ax.set_ylabel('Decoding accuracy')
    ax.set_title('Ablation Ladder: E Orientation Discrimination')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.05])

    return fig


def plot_comparison_scatter(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    label_x: str,
    label_y: str,
    figsize=(5, 5),
) -> plt.Figure:
    """
    Scatter plot comparing two decoding conditions (one point per LogMAR).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(acc_x, acc_y, alpha=0.7, s=60)
    lim = [min(acc_x.min(), acc_y.min()) - 0.05,
           max(acc_x.max(), acc_y.max()) + 0.05]
    ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    return fig


def plot_covariance_summary(
    covariance_results: dict,
    alpha_values: dict,
    figsize=(10, 4),
) -> plt.Figure:
    """
    Summary figure for covariance analysis: alignment fractions and SNR.
    """
    conditions = list(covariance_results.keys())
    x = np.arange(len(conditions))

    alphas = [alpha_values[c][0] for c in conditions if c in alpha_values]
    alphas_chance = [alpha_values[c][1] for c in conditions if c in alpha_values]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    bars = ax.bar(x, alphas, color=[COLORS.get(c, 'gray') for c in conditions], alpha=0.8)
    ax.plot(x, alphas_chance, 'k--', label='chance level')
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=15)
    ax.set_ylabel('Alignment fraction α')
    ax.set_title('FEM–Signal Alignment')
    ax.legend()

    ax = axes[1]
    snrs = [covariance_results[c].get('snr', 0) for c in conditions
            if c in covariance_results]
    ax.bar(x[:len(snrs)], snrs,
           color=[COLORS.get(c, 'gray') for c in conditions[:len(snrs)]],
           alpha=0.8)
    ax.set_xticks(x[:len(snrs)])
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions[:len(snrs)]],
                       rotation=15)
    ax.set_ylabel('Signal subspace SNR')
    ax.set_title('Signal-to-FEM-Noise Ratio')

    plt.suptitle('Population Geometry', fontsize=13)
    return fig


def make_summary_figure(
    neurometric_results: dict,
    ladder_results: dict,
    integration_results: dict,
    out_dir: str,
) -> None:
    """
    Generate and save the full set of summary figures.

    Args:
        neurometric_results: output of neurometric.compute_neurometric_curve()
        ladder_results: dict condition → output of decoding.run_decoding_ladder()
        integration_results: output of integration_time.integration_time_curve()
        out_dir: directory to save figures
    """
    from neurometric import plot_neurometric_curves
    from integration_time import plot_integration_time_curves

    set_publication_style()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Figure 1: Neurometric curves
    fig = plot_neurometric_curves(neurometric_results)
    save_figure(fig, os.path.join(out_dir, 'fig_neurometric.png'))
    plt.close(fig)

    # Figure 2: Ablation ladder bar chart
    if ladder_results:
        fig = plot_decoding_ladder_bar(ladder_results)
        save_figure(fig, os.path.join(out_dir, 'fig_ablation_ladder.png'))
        plt.close(fig)

    # Figure 3: Integration time curves
    if integration_results:
        fig = plot_integration_time_curves(integration_results)
        save_figure(fig, os.path.join(out_dir, 'fig_integration_time.png'))
        plt.close(fig)

    print(f"All figures saved to {out_dir}")

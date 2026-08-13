#!/usr/bin/env python3
"""Build the static HTML source and synthesis figure for the FEM power report.

The PDF itself is printed from the generated HTML with headless Chrome.  This
script deliberately reads the saved checkpoint tables and figures rather than
recomputing model responses.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/fem_dynamic_power_results_report_v1"
ASSETS = OUT / "assets"

CP01 = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
CP11 = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
CP12A = ROOT / "outputs/fig4_active_sensing/rr100_retinal_spectral_rank_checkpoint_12a_v1"
CP12B = ROOT / "outputs/fig4_active_sensing/rr100_neural_effect_rank_checkpoint_12b_v1"
CP15C = ROOT / "outputs/fig4_active_sensing/rr100_fit_quality_population_checkpoint_15c_v1"
CP15D = ROOT / "outputs/fig4_active_sensing/rr100_empirical_surface_predictor_checkpoint_15d_v1"
CP16 = ROOT / "outputs/fig4_active_sensing/rr100_corrected_figure4_cache_checkpoint_16_v1"


def copy_asset(source: Path, target_name: str) -> str:
    target = ASSETS / target_name
    shutil.copy2(source, target)
    return f"assets/{target_name}"


def build_predictor_comparison() -> dict[str, float]:
    variants = pd.read_csv(CP11 / "predictor_variant_per_unit_explainability.csv")
    composition = pd.read_csv(
        CP11 / "total_power_plus_spectral_composition_per_unit_explainability.csv"
    )

    variants = variants.loc[variants["quality_cohort"].astype(bool)].copy()
    keep = variants.pivot(
        index="rr100_index",
        columns="predictor_variant",
        values="cv_r2_vs_train_mean_baseline",
    )
    comp = composition.loc[composition["quality_cohort"].astype(bool)].set_index("rr100_index")
    keep["total_plus_composition"] = comp["cv_r2_vs_train_mean_baseline"]

    unit = keep["primary_unit_specific_amplitude"]
    total = keep["total_power_amplitude_no_unit_tuning"]
    total_comp = keep["total_plus_composition"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.35), constrained_layout=True)

    ax = axes[0]
    lim_lo = float(min(total.min(), unit.min(), -0.4))
    lim_hi = float(max(total.max(), unit.max(), 0.9))
    ax.scatter(total, unit, s=35, color="#356fa3", alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#777777", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="#c6cbd1", linewidth=0.8)
    ax.axvline(0, color="#c6cbd1", linewidth=0.8)
    ax.set(xlim=(lim_lo, lim_hi), ylim=(lim_lo, lim_hi))
    ax.set_xlabel("Total dynamic power CV $R^2$")
    ax.set_ylabel("Unit-specific SF×TF overlap CV $R^2$")
    ax.set_title("A  Unit-specific routing rarely improves prediction", loc="left", weight="bold")
    ax.text(
        0.04,
        0.96,
        f"{100 * np.mean(unit > total):.0f}% above diagonal",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
    )

    ax = axes[1]
    labels = ["Total\npower", "Unit SF×TF\noverlap", "Total +\ncomposition"]
    cols = [total.to_numpy(), unit.to_numpy(), total_comp.to_numpy()]
    colors = ["#d65238", "#356fa3", "#8e6bb3"]
    parts = ax.violinplot(cols, positions=np.arange(3), widths=0.78, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
    rng = np.random.default_rng(7)
    for x, values, color in zip(range(3), cols, colors):
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=12, color=color, alpha=0.58)
        med = float(np.nanmedian(values))
        ax.plot([x - 0.24, x + 0.24], [med, med], color="#15191d", linewidth=2.2)
        ax.text(x, min(1.04, med + 0.08), f"median {med:.2f}", ha="center", fontsize=9, weight="bold")
    ax.axhline(0, color="#9ca3aa", linestyle="--", linewidth=1)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(-0.85, 1.12)
    ax.set_ylabel("Held-out pair CV $R^2$")
    ax.set_title("B  The simpler predictor has the best median", loc="left", weight="bold")

    ax = axes[2]
    lim_lo2 = float(min(total.min(), total_comp.min(), -0.4))
    lim_hi2 = float(max(total.max(), total_comp.max(), 0.9))
    ax.scatter(total, total_comp, s=35, color="#8e6bb3", alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.plot([lim_lo2, lim_hi2], [lim_lo2, lim_hi2], color="#777777", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="#c6cbd1", linewidth=0.8)
    ax.axvline(0, color="#c6cbd1", linewidth=0.8)
    ax.set(xlim=(lim_lo2, lim_hi2), ylim=(lim_lo2, lim_hi2))
    ax.set_xlabel("Total dynamic power CV $R^2$")
    ax.set_ylabel("Total + spectral composition CV $R^2$")
    ax.set_title("C  Added spectral composition usually hurts", loc="left", weight="bold")
    ax.text(
        0.04,
        0.96,
        f"{100 * np.mean(total_comp > total):.0f}% above diagonal",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
    )

    for ax in axes:
        ax.grid(color="#e9ecef", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Held-out prediction across 66 RR100 units and 16 complete image–trajectory pairs",
        fontsize=15,
        weight="bold",
    )
    out = ASSETS / "predictor_comparison_synthesis.png"
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)

    return {
        "n_quality_units": int(len(keep)),
        "median_total_r2": float(total.median()),
        "median_unit_r2": float(unit.median()),
        "median_total_composition_r2": float(total_comp.median()),
        "fraction_unit_beats_total": float(np.mean(unit > total)),
        "fraction_composition_beats_total": float(np.mean(total_comp > total)),
        "fraction_unit_positive": float(np.mean(unit > 0)),
        "fraction_unit_ge_025": float(np.mean(unit >= 0.25)),
    }


def page(n: int, title: str, kicker: str, body: str, cls: str = "") -> str:
    return f"""
    <section class="page {cls}">
      <div class="page-head"><span>{kicker}</span><span>FEMs and dynamic power</span></div>
      <h1>{title}</h1>
      {body}
      <div class="page-num">{n}</div>
    </section>
    """


def build_html(metrics: dict[str, float], images: dict[str, str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FEMs and dynamic power: RR100 results and discussion</title>
<style>
  @page {{ size: Letter landscape; margin: 0; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #e8ecef; color: #172029; font-family: Arial, Helvetica, sans-serif; }}
  .page {{ position: relative; width: 11in; height: 8.5in; padding: .48in .58in .42in; background: white; page-break-after: always; overflow: hidden; }}
  .page:last-child {{ page-break-after: auto; }}
  .page-head {{ display:flex; justify-content:space-between; border-bottom:1px solid #cfd7de; padding-bottom:5px; color:#5d6a75; font-size:9px; text-transform:uppercase; letter-spacing:.09em; }}
  .page-num {{ position:absolute; right:.58in; bottom:.22in; color:#77838d; font-size:9px; }}
  h1 {{ font-size:27px; line-height:1.08; margin:14px 0 10px; letter-spacing:-.025em; }}
  h2 {{ font-size:17px; line-height:1.15; margin:0 0 7px; color:#1d3446; }}
  h3 {{ font-size:13px; margin:0 0 5px; }}
  p, li {{ font-size:11.4px; line-height:1.42; }}
  p {{ margin:5px 0 8px; }}
  ul {{ margin:5px 0 0; padding-left:20px; }}
  li {{ margin:3px 0; }}
  .lede {{ font-size:18px; line-height:1.34; max-width:9.3in; margin:12px 0 18px; color:#22323f; }}
  .accent {{ color:#b6402a; }}
  .blue {{ color:#2f6f9f; }}
  .muted {{ color:#5f6c76; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:15px; }}
  .card {{ border:1px solid #d9e0e5; border-radius:8px; padding:13px 15px; background:#fbfcfd; }}
  .card.result {{ border-top:5px solid #d65238; }}
  .card.neutral {{ border-top:5px solid #356fa3; }}
  .card.limit {{ border-top:5px solid #8e6bb3; }}
  .metric {{ font-size:29px; font-weight:700; line-height:1; margin:4px 0 5px; color:#1e3547; }}
  .small {{ font-size:9.6px; line-height:1.35; }}
  .figure {{ width:100%; object-fit:contain; display:block; }}
  .fig-large {{ max-height:5.55in; }}
  .fig-medium {{ max-height:4.8in; }}
  .caption {{ font-size:9.4px; line-height:1.35; color:#58656f; margin-top:6px; }}
  .callout {{ background:#f3f7f9; border-left:5px solid #356fa3; padding:11px 14px; margin:10px 0; font-size:12px; line-height:1.4; }}
  .warning {{ background:#fff6e9; border-left-color:#d99025; }}
  .equation {{ font-family: Georgia, 'Times New Roman', serif; text-align:center; font-size:16px; padding:8px; margin:5px 0; background:#f7f9fa; border-radius:6px; }}
  .flow {{ display:grid; grid-template-columns:1.2fr .18fr 1.2fr .18fr 1.2fr .18fr 1.2fr; align-items:center; gap:8px; margin:15px 0 17px; }}
  .flow .box {{ min-height:80px; border:1.5px solid #bcc9d2; border-radius:9px; padding:12px; background:#f8fafb; text-align:center; }}
  .flow .arrow {{ text-align:center; font-size:24px; color:#72808a; }}
  table {{ width:100%; border-collapse:collapse; font-size:10.4px; }}
  th {{ text-align:left; background:#eef3f6; color:#253744; }}
  th, td {{ padding:7px 8px; border-bottom:1px solid #dfe5e9; vertical-align:top; }}
  .cover {{ background:linear-gradient(135deg,#f6fafc 0%,#ffffff 60%,#fdf3ee 100%); }}
  .cover h1 {{ font-size:41px; max-width:8.8in; margin-top:.72in; }}
  .cover .tag {{ display:inline-block; background:#17384d; color:white; padding:6px 10px; border-radius:20px; font-size:10px; letter-spacing:.08em; text-transform:uppercase; }}
  .summary-line {{ font-size:15px; line-height:1.42; max-width:9.1in; border-left:6px solid #d65238; padding-left:18px; margin:24px 0; }}
  .footnote {{ position:absolute; left:.58in; bottom:.2in; font-size:8.5px; color:#75808a; max-width:8.9in; }}
</style>
</head>
<body>

{page(1, "FEM-created dynamic power predicts RR100 responses, with outcome-dependent tuning specificity", "Results + discussion", f'''
  <div class="tag">Frozen RR100 · natural-image retinal movies · fixed-eye F0 tuning</div>
  <p class="lede">Fixational eye movements (FEMs) convert static spatial structure into temporal retinal modulation. In the original 16 paired movies, total supported power best predicts temporal-response modulation. In the corrected Figure-4 cache, unit-specific SF×TF weighting modestly improves prediction of <strong>mean-rate change</strong>, but none of the power predictors explain spatial-SSI change.</p>
  <div class="grid3">
    <div class="card result"><div class="metric">0.28</div><h3>Temporal modulation</h3><p>Median held-out R² from total power across the original 16 paired 128-frame movies.</p></div>
    <div class="card neutral"><div class="metric">58.0%</div><h3>Corrected-cache rate PC1</h3><p>Shared variance across 512 crossed image–trace pairs and 91 responsive units, using full 32-frame mean rates at 120 Hz.</p></div>
    <div class="card limit"><div class="metric">0.052</div><h3>Corrected-cache mean rate</h3><p>Median crossed held-out R² for parametric SF×TF weighting versus 0.004 for total power.</p></div>
  </div>
  <div class="summary-line"><strong>Interpretation:</strong> FEM produces a strong shared mean-rate component, organized mainly by eye trace, and a more image-specific spatial-SSI component. SF×TF routing is detectable for mean rate in the corrected crossed cache, but its predictive effect is small and does not generalize to SSI.</div>
  <p class="muted small">Technical report · generated 12 August 2026 · internal model analysis. All TF-resolved results use canonical retinal reconstruction at 120 Hz. The corrected-cache analysis contains 16 images × 32 eye traces × 32 explicitly timed conditions.</p>
''', 'cover')}

{page(2, "The direct test separates retinal power, unit sensitivity, and neural response", "Analysis construction", '''
  <div class="flow">
    <div class="box"><h3>Static image + eye trace</h3><p>Construct exact 51×51 retinal frames under true zero gaze and measured FEM.</p></div><div class="arrow">→</div>
    <div class="box"><h3>Retinal spectrum</h3><p><em>P</em><sub>c</sub>(SF,TF): positive-TF power generated for condition <em>c</em>.</p></div><div class="arrow">→</div>
    <div class="box"><h3>Fixed-eye unit tuning</h3><p><em>W</em><sub>u</sub>(SF,TF): separable F0 grating fit for unit <em>u</em>.</p></div><div class="arrow">→</div>
    <div class="box"><h3>Frozen response effect</h3><p><em>Y</em><sub>uc</sub>: temporal SD of FEM minus zero response.</p></div>
  </div>
  <div class="grid2">
    <div>
      <h2>Two competing predictors</h2>
      <div class="equation">Total power: &nbsp; A<sub>c</sub> = √[Σ P<sub>c</sub>(SF,TF)]</div>
      <p>The same scalar is supplied to every unit. A separate nonnegative scale and intercept are learned for each unit.</p>
      <div class="equation">Unit overlap: &nbsp; X<sub>uc</sub> = g<sub>u</sub>√[Σ P<sub>c</sub>(SF,TF) W<sub>u</sub>(SF,TF)<sup>2</sup>]</div>
      <p>This asks whether power falling inside each unit's independently measured fixed-eye SF×TF passband better predicts its FEM modulation.</p>
    </div>
    <div>
      <h2>Cohorts and validation</h2>
      <table>
        <tr><th>Component</th><th>Definition</th></tr>
        <tr><td>Exact condition set</td><td>16 natural-image / measured-eye-trajectory pairs; true zero gaze versus FEM; 97 valid response frames.</td></tr>
        <tr><td>Tuning cohort</td><td>66/100 units passing SF R² ≥ 0.70, TF R² ≥ 0.70, and joint surface R² ≥ 0.50.</td></tr>
        <tr><td>Neural rank cohort</td><td>91/100 units with FEM-effect temporal SD &gt; 10<sup>−4</sup> Hz.</td></tr>
        <tr><td>Prediction test</td><td>Leave one complete image–trajectory pair out; R² relative to the corresponding training-pair mean baseline.</td></tr>
        <tr><td>Corrected-cache extension</td><td>16 images × 32 eye traces; original timing versus static; crossed four-fold validation holds out both the test images and test traces.</td></tr>
      </table>
      <div class="callout"><strong>Why this is a stringent test:</strong> the SF×TF surface is fitted independently from controlled fixed-eye gratings; it is not adjusted to natural-image FEM responses.</div>
    </div>
  </div>
''')}

{page(3, "True zero gaze is temporally static; FEM creates broadband temporal retinal power", "Result 1 · retinal mechanism", f'''
  <img class="figure fig-large" src="{images['input_power']}" alt="Exact retinal reconstruction and SF-TF power redistribution">
  <p class="caption"><strong>Evidence.</strong> One audibly selected natural-image example reconstructed with the canonical 51×51 renderer at 120 Hz. Zero gaze repeats an identical frame (maximum frame difference 0; positive-TF power 0). The measured FEM trajectory translates the retinal image and creates positive-TF power. This panel contains no unit tuning and no neural response weighting.</p>
  <div class="callout warning"><strong>Spectral coverage matters:</strong> for this example, only 30.0% of positive-TF dynamic power falls inside the jointly fitted SF (1–11.31 cpd) and TF (0.5–32 Hz) support; 64.5% lies above the fitted TF maximum. The tuning-weighted model therefore tests the measured support, not all FEM-created power.</div>
''')}

{page(4, "Across the 16 exact conditions, amplitude dominates—but spectral shape also changes", "Result 2 · retinal variation", f'''
  <div class="grid3" style="margin-bottom:10px">
    <div class="card result"><div class="metric">4,960×</div><h3>Total-power range</h3><p>Supported dynamic power differs enormously across the 16 conditions.</p></div>
    <div class="card neutral"><div class="metric">97.5%</div><h3>Raw rank-1 energy</h3><p>High-power conditions dominate an unnormalized SVD.</p></div>
    <div class="card limit"><div class="metric">60.9%</div><h3>Equal-total rank-1 energy</h3><p>After giving each spectral map equal total power, one template is incomplete.</p></div>
  </div>
  <img class="figure fig-medium" src="{images['spectral_rank']}" alt="Retinal spectral amplitude and shape diagnostics">
  <p class="caption"><strong>Interpretation.</strong> The retinal input contains a very large scalar amplitude component, consistent with a shared population effect, but spectra are not exact scalar multiples of one common SF×TF template (median pairwise normalized-shape cosine 0.57; worst 0.09). Raw spectral rank alone cannot establish a one-factor mechanism.</p>
''')}

{page(5, "Most responsive RR100 units share the ordering of the 16 FEM conditions", "Result 3 · neural structure", f'''
  <img class="figure fig-large" src="{images['neural_rank']}" alt="Shared neural FEM-effect rank and reliability">
  <p class="caption"><strong>Evidence.</strong> For 91 responsive units, centered PC1 after standardizing each unit across conditions explains 65.2% of variance, versus a 13.5% image-shuffle 99th percentile (p=0.0002). Random unit halves recover nearly identical condition scores (median Pearson r=0.99). Condition 6 supplies 46% of squared PC1 score energy, but removing it leaves 60.0% PC1 variance and median pairwise unit-profile Spearman 0.64.</p>
  <div class="callout"><strong>Bounded claim:</strong> this is a common ordering within one frozen RR100 population and these 16 image–trajectory pairs. PC1 is an algebraic description, not evidence for a single biological latent factor, generalization to experimental neurons, or additive versus multiplicative gain.</div>
''')}

{page(6, "For temporal modulation in the original 16 pairs, total power predicts best", "Result 4 · direct mechanistic test", f'''
  <img class="figure fig-medium" src="{images['predictor']}" alt="Held-out predictor comparison across RR100 units">
  <div class="grid3" style="margin-top:8px">
    <div class="card result"><h3>Total supported power</h3><p>Median held-out R² = <strong>{metrics['median_total_r2']:.2f}</strong>. It knows the condition's total dynamic power, but nothing about unit tuning.</p></div>
    <div class="card neutral"><h3>Unit SF×TF overlap</h3><p>Median R² = <strong>{metrics['median_unit_r2']:.2f}</strong>; {100*metrics['fraction_unit_positive']:.0f}% positive and {100*metrics['fraction_unit_ge_025']:.0f}% at least 0.25. It beats total power for only <strong>{100*metrics['fraction_unit_beats_total']:.0f}%</strong> of units.</p></div>
    <div class="card limit"><h3>Total + spectral composition</h3><p>Median R² = <strong>{metrics['median_total_composition_r2']:.2f}</strong> and beats total power for only <strong>{100*metrics['fraction_composition_beats_total']:.0f}%</strong> of units.</p></div>
  </div>
  <p class="caption"><strong>Meaning of the negative result.</strong> It does not show that SF/TF tuning is absent. It shows that this particular orientation-collapsed, scalar power-overlap construction does not explain additional held-out variation beyond the broad amount of dynamic power in this dataset.</p>
''')}

{page(7, "Better grating fits do not produce a larger tuning advantage", "Diagnostic 1 · fit quality", f'''
  <img class="figure fig-large" src="{images['fit_quality']}" alt="Population fit-quality thresholds and covariate tests">
  <p class="caption"><strong>Evidence.</strong> Across the same 66-unit cohort, joint-fit quality is not positively associated with the advantage of unit-specific SF×TF prediction (Spearman ρ=−0.23, p=0.060). Raising the joint-fit threshold from 0.50 to 0.85 changes the median paired ΔR² from −0.04 to −0.17; the best-fit quartile has median ΔR²=−0.12. RMSE, MAE, and condition-ordering differences give the same qualitative result.</p>
  <div class="callout"><strong>Covariate check:</strong> after adjusting for response-effect magnitude, preferred SF/TF, and SF/TF bandwidth, the standardized joint-fit coefficient for ΔR² remains negative (−0.34; unit-bootstrap 95% interval −0.63 to −0.13). Fit quality is therefore not merely being masked by these measured unit properties.</div>
''')}

{page(8, "Measured grating surfaces still do not outperform total dynamic power", "Diagnostic 2 · parametric misspecification", f'''
  <img class="figure fig-large" src="{images['empirical_surface']}" alt="Parametric and empirical grating-surface predictor comparison">
  <p class="caption"><strong>Direct misspecification test.</strong> The exact supported retinal-power maps were reweighted by four alternatives under the same leave-one-pair-out calibration: total power, the log-Gaussian separable fit, the measured empirical rank-1 surface, and the full measured 2-D positive-F0 grid. Median held-out R² values are 0.283, 0.223, 0.246, and 0.275, respectively.</p>
  <div class="grid2">
    <div class="callout"><strong>Relative to total power:</strong> the full measured 2-D surface has median ΔR²=−0.027 and wins for 44% of units. The empirical rank-1 surface has median ΔR²=−0.039 and wins for 39%.</div>
    <div class="callout warning"><strong>What this rules out:</strong> parametric smoothing and separability are not the primary population-level bottleneck. It does not rule out incomplete stimulus support or a mismatch between grating power sensitivity and natural-image computation.</div>
  </div>
''')}

{page(9, "The corrected Figure-4 cache separates shared rate drive from image-specific SSI", "Corrected native-time cache", f'''
  <img class="figure fig-large" src="{images['corrected_cache']}" alt="Corrected native-time Figure-4 cache population analysis">
  <p class="caption"><strong>Corrected-cache analysis.</strong> The cache crosses 16 images with 32 independently selected eye traces and 32 explicitly timed conditions; all movies contain 32 frames at 120 Hz. Across 91 responsive units, original timing minus stabilized static gives a signed mean-rate PC1 explaining 58.0% of unit-standardized variance and median pairwise unit-profile Spearman ρ=0.56. Signed SSI PC1 explains only 27.0%, with median ρ=0.15.</p>
  <div class="grid2">
    <div class="callout"><strong>Different organizing variables:</strong> 75.2% of population-mean rate-change variance is attributable to eye-trace identity, versus 19.5% to image identity. For SSI change, 82.1% is attributable to image identity and only 9.4% to trace identity.</div>
    <div class="callout warning"><strong>Prediction:</strong> among 66 quality-fit units, crossed held-out median R² for mean-rate change is 0.052 for parametric SF×TF overlap, 0.042 for the measured 2-D surface, and 0.004 for total power. For SSI change all medians are approximately zero or negative.</div>
  </div>
''')}

{page(10, "FEM effects share a retinal origin but separate by neural outcome", "Discussion + next tests", '''
  <div class="grid2">
    <div>
      <h2>What the evidence supports</h2>
      <ul>
        <li>FEMs deterministically convert static spatial structure into nonzero temporal retinal power.</li>
        <li>The amount of created power varies greatly across conditions, while spectral shape also varies.</li>
        <li>The frozen RR100 population has a strong shared mean-rate component across crossed image–trace combinations; this component is primarily trace-organized.</li>
        <li>Spatial-SSI effects are less rank-1 and are primarily image-organized, so “the FEM effect” cannot be treated as one scalar response phenotype.</li>
        <li>Total power best predicts temporal-response modulation across the original 16 paired movies.</li>
        <li>In the corrected crossed cache, SF×TF weighting modestly improves mean-rate prediction over total power, but explains little absolute variance and does not predict SSI.</li>
      </ul>
      <h2 style="margin-top:14px">What remains unestablished</h2>
      <ul>
        <li>Whether the effect is additive or multiplicative relative to each unit's zero-gaze image response.</li>
        <li>Whether a shared biological gain mechanism exists.</li>
        <li>Why grating tuning helps slightly for mean rate but not spatial SSI—possible factors include orientation, phase, contrast, context, nonlinear integration, and missing frequency support.</li>
        <li>Generalization to another twin seed, new native-time movies, or recorded neurons.</li>
      </ul>
    </div>
    <div>
      <h2>Most informative next analyses</h2>
      <ol style="margin-top:5px; padding-left:20px">
        <li><strong>Fit additive and multiplicative models directly:</strong> compare FEM−zero change with zero-gaze response level using held-out conditions and nested model comparisons.</li>
        <li><strong>Fit trace and image components explicitly:</strong> predict the dominant trace-led rate axis and image-led SSI axis separately, with image- and trace-disjoint validation.</li>
        <li><strong>Move beyond radial power overlap:</strong> retain 2-D spatial frequency/orientation, response phase or energy where appropriate, nonlinear contrast normalization, and power outside the current grating-fit support.</li>
        <li><strong>Seek external replication:</strong> rerun on another frozen twin/model seed and, where stimulus reconstruction permits, compare with recorded-unit FEM effects.</li>
      </ol>
      <div class="callout warning"><strong>Working conclusion:</strong> FEM-created dynamic power is a credible source of shared response modulation. The corrected cache provides modest evidence for unit-specific SF×TF routing of mean-rate changes, alongside a distinct image-specific SSI effect that the current power model does not explain.</div>
    </div>
  </div>
  <h2 style="margin-top:14px">Audit trail</h2>
  <table>
    <tr><th>Checkpoint</th><th>Role</th><th>Primary saved evidence</th></tr>
    <tr><td>01</td><td>Exact retinal mechanism</td><td>Retinal frames, SF×TF power maps, support audit</td></tr>
    <tr><td>11</td><td>Held-out predictor comparison</td><td>Per-unit predictions, R² tables, control models</td></tr>
    <tr><td>12A / 12B</td><td>Retinal amplitude/shape and neural rank</td><td>SVD/PCA arrays, nulls, removals, split halves</td></tr>
    <tr><td>15A–15D</td><td>Grating-fit limitation diagnostics</td><td>Raw units, selected maps, thresholds/covariates, empirical surfaces</td></tr>
    <tr><td>16A / 16B</td><td>Corrected Figure-4 cache</td><td>Native-time response maps, rank/split halves, crossed image-and-trace prediction</td></tr>
  </table>
  <p class="small muted">Conceptual precedent: Kuang et al. (2012), “Temporal encoding of spatial information during active visual fixation,” Current Biology 22:510–514. All numerical claims in this report are drawn from the saved VisionCore checkpoint artifacts listed in the accompanying manifest.</p>
''')}

</body>
</html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    images = {
        "input_power": copy_asset(
            CP01 / "checkpoint_01_kuang_input_power_redistribution.png",
            "checkpoint_01_input_power.png",
        ),
        "spectral_rank": copy_asset(
            CP12A / "checkpoint_12a_rank_and_shape_diagnostics.png",
            "checkpoint_12a_rank_and_shape.png",
        ),
        "neural_rank": copy_asset(
            CP12B / "checkpoint_12b_shared_factor_rank_and_reliability.png",
            "checkpoint_12b_neural_rank.png",
        ),
        "corrected_cache": copy_asset(
            CP16 / "checkpoint_16b_corrected_cache_population_analysis.png",
            "checkpoint_16b_corrected_cache_population.png",
        ),
        "fit_quality": copy_asset(
            CP15C / "checkpoint_15c_fit_quality_population_tests.png",
            "checkpoint_15c_fit_quality_population.png",
        ),
        "empirical_surface": copy_asset(
            CP15D / "checkpoint_15d_empirical_surface_predictor_comparison.png",
            "checkpoint_15d_empirical_surface_predictors.png",
        ),
        "predictor": "assets/predictor_comparison_synthesis.png",
    }
    metrics = build_predictor_comparison()
    html = build_html(metrics, images)
    (OUT / "fem_dynamic_power_results_and_discussion.html").write_text(html, encoding="utf-8")

    source_files = [
        CP01 / "manifest.json",
        CP11 / "per_unit_leave_one_image_out_explainability.csv",
        CP11 / "predictor_variant_per_unit_explainability.csv",
        CP11 / "total_power_plus_spectral_composition_per_unit_explainability.csv",
        CP12A / "README.md",
        CP12B / "README.md",
        CP15C / "fit_quality_threshold_summary.csv",
        CP15C / "fit_quality_metric_associations.csv",
        CP15C / "fit_quality_covariate_models.csv",
        CP15D / "empirical_surface_population_comparison_summary.csv",
        CP15D / "per_unit_empirical_surface_predictor_metrics.csv",
        CP16 / "manifest.json",
        CP16 / "corrected_cache_rank_summary.csv",
        CP16 / "corrected_cache_crossed_cv_metrics.csv",
    ]
    manifest = {
        "report": "FEMs and dynamic power: RR100 results and discussion",
        "created_date": "2026-08-12",
        "static_source": "fem_dynamic_power_results_and_discussion.html",
        "pdf_output": "fem_dynamic_power_results_and_discussion.pdf",
        "synthesis_metrics": metrics,
        "source_files": [str(path.relative_to(ROOT)) for path in source_files],
        "scope_note": "No model responses were recomputed; the corrected-cache extension uses saved full response/SSI summaries and reconstructs retinal spectra from its exact saved 32-frame traces.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# FEMs and dynamic power: results and discussion\n\n"
        "This directory contains a ten-page technical report that synthesizes the saved "
        "RR100 Figure 4 checkpoints. It does not rerun the frozen model.\n\n"
        "- `fem_dynamic_power_results_and_discussion.html`: static source of truth\n"
        "- `fem_dynamic_power_results_and_discussion.pdf`: requested report\n"
        "- `assets/`: copied checkpoint figures plus the report-only predictor synthesis\n"
        "- `manifest.json`: numerical synthesis values and source tables\n"
        "- `extracted_text.txt` and `page_previews/`: PDF verification artifacts\n\n"
        "The central conclusion is outcome-specific. Total supported power best predicts temporal "
        "modulation in the original 16 paired movies. In the corrected 16-image by 32-trace cache, "
        "SF×TF weighting modestly improves prediction of mean-rate changes, whereas no power "
        "predictor generalizes to spatial-SSI changes. The report does not claim additive or "
        "multiplicative gain, privileged image–trace matching, or external biological generalization.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

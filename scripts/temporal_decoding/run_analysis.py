"""
Task 5.1: Main Pipeline Script — Full Temporal Decoding Analysis

Orchestrates the complete analysis pipeline end-to-end, with caching at each phase.

Usage:
    # Full pipeline
    python run_analysis.py --phase all

    # Individual phases
    python run_analysis.py --phase 1            # Generate data
    python run_analysis.py --phase 2            # Primary decoding
    python run_analysis.py --phase 3            # Mechanistic analyses
    python run_analysis.py --phase 4            # Generate figures

    # Quick test run
    python run_analysis.py --phase all --n_traces 20 --logmar_subset --reduced

    # Specify LogMAR for threshold decoding
    python run_analysis.py --phase 2 --threshold_logmar 0.4
"""
import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory layout
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RATES_DIR = os.path.join(DATA_DIR, 'rates')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

EYE_TRACES_PATH = os.path.join(DATA_DIR, 'eye_traces.npz')
PKL_PATH = os.path.join(SCRIPT_DIR, '..', 'mcfarland_outputs_mono.pkl')

for d in [DATA_DIR, RATES_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


def _load_rates_list_npz(path: str) -> list[np.ndarray]:
    """Lightweight loader for cached rates.

    Loads the format written by `rate_computation.save_rates()`:
      - rates: (M, T_max, N) float32
      - lengths: (M,) int

    Returns a list of length M with arrays shaped (T_i, N).
    This avoids importing `rate_computation` (which pulls in external deps).
    """
    d = np.load(path, allow_pickle=True)
    rates_padded = d['rates']
    lengths = d['lengths'].astype(int)
    return [rates_padded[i, :lengths[i]] for i in range(rates_padded.shape[0])]


def _rate_cache_path_dual_regime(logmar: float, ori: int, cond: str,
                                 hires_threshold: float = 0.35,
                                 rate_file_tag: str = "") -> str:
    """Match `run_dual_regime.py` caching convention."""
    use_hires = (float(logmar) < float(hires_threshold))
    prefix = 'rates_hires' if use_hires else 'rates'
    tag = str(rate_file_tag or "").strip()
    if tag != "" and not tag.startswith("_"):
        tag = "_" + tag
    return os.path.join(RATES_DIR, f'{prefix}_lm{logmar:.2f}_ori{ori}_{cond}{tag}.npz')


def run_phase_2_cached(args):
    """Phase 2, but decode-only from cached rate matrices.

    This mode intentionally does not import the simulation/model stack.
    It requires that cached `.npz` files already exist under `RATES_DIR`.
    """
    from decoding import run_decoding_ladder
    from integration_time import integration_time_curve, plot_integration_time_curves
    from plotting import save_figure
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("PHASE 2 (CACHED): Decode-only from cached rates")
    print("=" * 60)

    # --- 2.2 Integration time sweep (cached) ---
    print(f"\n--- 2.2 Integration Time Sweep (cached; method={args.integration_method}) ---")
    logmar_threshold = float(args.threshold_logmar)
    it_tag = str(getattr(args, 'rate_file_tag', '') or '').strip()
    if it_tag != '' and not it_tag.startswith('_'):
        it_tag = '_' + it_tag
    int_time_cache = os.path.join(
        RESULTS_DIR,
        f'integration_time_{args.integration_method}_lm{logmar_threshold:+.2f}{it_tag}.pkl',
    )

    if os.path.exists(int_time_cache) and not args.force:
        print(f"  Loading cached: {int_time_cache}")
        with open(int_time_cache, 'rb') as f:
            int_results = pickle.load(f)
    else:
        orientations = [0, 90, 180, 270]
        rates_by_condition = {}
        for cond in ['real', 'stabilized']:
            rates_by_stim = {}
            for ori in orientations:
                path = _rate_cache_path_dual_regime(
                    logmar_threshold, ori, cond,
                    hires_threshold=float(args.hires_threshold),
                    rate_file_tag=str(getattr(args, 'rate_file_tag', '') or ''),
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing cached rates for integration-time sweep: {path}\n"
                        "Either run the full pipeline that computes rates, or choose a LogMAR where caches exist."
                    )
                rates_by_stim[f'ori{ori}'] = _load_rates_list_npz(path)
            rates_by_condition[cond] = rates_by_stim

        windows = [1, 3, 6, 12, 24, 36, 48, 60] if not args.reduced else [1, 6, 24, 60]
        int_results = integration_time_curve(
            rates_by_condition,
            windows=windows,
            n_splits=args.n_splits,
            C_logistic=1.0,
            verbose=True,
            method=args.integration_method,
        )
        with open(int_time_cache, 'wb') as f:
            pickle.dump(int_results, f)
        print(f"  Saved: {int_time_cache}")

    if not args.no_figures:
        fig = plot_integration_time_curves(int_results)
        out_fig = os.path.join(FIGURES_DIR, f'fig_integration_time_{args.integration_method}.png')
        save_figure(fig, out_fig)
        plt.close(fig)
        print(f"  Saved: {out_fig}")

    # --- 2.3 Neurometric curves (cached, dual-regime) ---
    if args.skip_neurometric:
        print("\n  Skipping neurometric curves (--skip_neurometric)")
        return

    print(f"\n--- 2.3 Neurometric Curves (cached; dual-regime, hires_threshold={args.hires_threshold}) ---")

    def _resolve_output_path(base_dir: str, name_or_path: str) -> str:
        """Resolve a user-provided output name/path.

        If `name_or_path` contains a path separator, treat it as a path.
        Otherwise, place it under `base_dir`.
        """
        s = str(name_or_path or "").strip()
        if s == "":
            raise ValueError("Output name/path must be non-empty")
        return s if (os.sep in s) else os.path.join(base_dir, s)

    def _parse_float_csv(s: str) -> list[float]:
        parts = [p.strip() for p in str(s).split(',') if p.strip()]
        return [float(p) for p in parts]

    # Match the dual-regime grid used elsewhere in the repo.
    if args.neurometric_logmars is not None:
        logmar_values = _parse_float_csv(args.neurometric_logmars)
    else:
        logmar_values = [
            1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.15, -0.2, -0.25, -0.3,
        ]
        if args.neurometric_extend_negative:
            logmar_values = logmar_values + [-0.35, -0.40, -0.45, -0.50]

    conditions = ['real', 'stabilized']
    if not args.reduced and not args.no_matched_null:
        conditions.append('matched_null')

    from neurometric import (
        fit_neurometric_threshold,
        load_neurometric_results,
        plot_neurometric_curves,
        save_neurometric_results,
    )

    neuro_cache = _resolve_output_path(
        RESULTS_DIR,
        getattr(args, 'neurometric_cache_name', 'neurometric_cached_dual_regime.npz'),
    )

    def _lm_key(x: float) -> float:
        # LogMAR values are only ever used/serialized at 2 decimal places in cache file names.
        return float(np.round(float(x), 2))

    # Optional reuse of existing neurometric results.
    # This makes extensions (e.g., adding -0.35..-0.50) fast by computing only missing points.
    reuse = {
        'logmar_keys': set(),
        'accuracy': {},
        'accuracy_std': {},
    }
    if os.path.exists(neuro_cache) and not args.force:
        try:
            prev = load_neurometric_results(neuro_cache)
            prev_lm_keys = [_lm_key(v) for v in prev.get('logmar_values', [])]

            # Only reuse if previous has matching model set (A/C) and overlaps conditions.
            prev_models = set(prev.get('models', []))
            if prev_models.issuperset({'A', 'C'}):
                reuse['logmar_keys'] = set(prev_lm_keys)
                for cond in conditions:
                    for mod in ['A', 'C']:
                        key = (cond, mod)
                        if key not in prev.get('accuracy', {}):
                            continue
                        acc_arr = np.asarray(prev['accuracy'][key], dtype=float)
                        std_arr = np.asarray(prev['accuracy_std'][key], dtype=float)
                        if acc_arr.shape[0] != len(prev_lm_keys):
                            continue
                        reuse['accuracy'][key] = {k: float(acc_arr[i]) for i, k in enumerate(prev_lm_keys)}
                        reuse['accuracy_std'][key] = {k: float(std_arr[i]) for i, k in enumerate(prev_lm_keys)}

                if reuse['logmar_keys']:
                    n_reuse = sum(1 for v in logmar_values if _lm_key(v) in reuse['logmar_keys'])
                    print(f"  Reusing cached neurometric points from: {neuro_cache} ({n_reuse}/{len(logmar_values)} LogMAR values)")
        except Exception as e:
            print(f"  Warning: failed to reuse {neuro_cache} ({type(e).__name__}: {e}); recomputing sweep")

    results = {
        'logmar_values': logmar_values,
        'conditions': conditions,
        'models': ['A', 'C'],
        'accuracy': {},
        'accuracy_std': {},
        'threshold': {},
    }
    for cond in conditions:
        for mod in ['A', 'C']:
            results['accuracy'][(cond, mod)] = []
            results['accuracy_std'][(cond, mod)] = []

    orientations = [0, 90, 180, 270]
    for logmar in logmar_values:
        lm_k = _lm_key(logmar)
        print(f"\n=== LogMAR = {logmar:+.2f} ===")
        for cond in conditions:
            # Reuse if available for both A and C for this condition/logmar.
            can_reuse = (
                lm_k in reuse['logmar_keys']
                and (cond, 'A') in reuse['accuracy'] and (cond, 'C') in reuse['accuracy']
                and lm_k in reuse['accuracy'][(cond, 'A')] and lm_k in reuse['accuracy'][(cond, 'C')]
                and lm_k in reuse['accuracy_std'].get((cond, 'A'), {}) and lm_k in reuse['accuracy_std'].get((cond, 'C'), {})
            )
            if can_reuse:
                a = reuse['accuracy'][(cond, 'A')][lm_k]
                c = reuse['accuracy'][(cond, 'C')][lm_k]
                a_std = reuse['accuracy_std'][(cond, 'A')][lm_k]
                c_std = reuse['accuracy_std'][(cond, 'C')][lm_k]
                results['accuracy'][(cond, 'A')].append(a)
                results['accuracy'][(cond, 'C')].append(c)
                results['accuracy_std'][(cond, 'A')].append(a_std)
                results['accuracy_std'][(cond, 'C')].append(c_std)
                print(
                    f"  {cond:11s}  A={a:.3f}  C={c:.3f}  (C-A={c-a:+.3f})  [reused]"
                )
                continue

            rates_by_stim = {}
            for ori in orientations:
                path = _rate_cache_path_dual_regime(
                    float(logmar), ori, cond,
                    hires_threshold=float(args.hires_threshold),
                    rate_file_tag=str(getattr(args, 'rate_file_tag', '') or ''),
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing cached rates: {path}\n"
                        "This cached-mode sweep requires all LogMAR/ori/cond .npz files to exist."
                    )
                rates_by_stim[f'ori{ori}'] = _load_rates_list_npz(path)

            ladder = run_decoding_ladder(
                rates_by_stim,
                models=['A', 'C'],
                n_splits=args.n_splits,
                n_components_C=args.n_pca_C,
                verbose=False,
            )
            for mod in ['A', 'C']:
                results['accuracy'][(cond, mod)].append(ladder[mod]['mean_acc'])
                results['accuracy_std'][(cond, mod)].append(ladder[mod]['std_acc'])
            print(
                f"  {cond:11s}  A={ladder['A']['mean_acc']:.3f}  C={ladder['C']['mean_acc']:.3f}  "
                f"(C-A={ladder['C']['mean_acc']-ladder['A']['mean_acc']:+.3f})"
            )

    lm_arr = np.array(logmar_values)
    for cond in conditions:
        for mod in ['A', 'C']:
            acc = np.array(results['accuracy'][(cond, mod)], dtype=float)
            std = np.array(results['accuracy_std'][(cond, mod)], dtype=float)
            results['accuracy'][(cond, mod)] = acc
            results['accuracy_std'][(cond, mod)] = std
            results['threshold'][(cond, mod)] = fit_neurometric_threshold(lm_arr, acc)

    t_stab_A = results['threshold'].get(('stabilized', 'A'))
    t_real_C = results['threshold'].get(('real', 'C'))
    if t_stab_A is not None and t_real_C is not None:
        results['delta_logmar'] = t_stab_A - t_real_C
        print(f"\n=== ΔLogMAR = {results['delta_logmar']:.3f} ===")
    else:
        results['delta_logmar'] = None
        print(f"\nThresholds missing: stabilized/A={t_stab_A}, real/C={t_real_C}")

    save_neurometric_results(results, neuro_cache)
    print(f"  Saved: {neuro_cache}")

    if not args.no_figures:
        fig = plot_neurometric_curves(results)
        out_fig = _resolve_output_path(
            FIGURES_DIR,
            getattr(args, 'neurometric_fig_name', 'fig_neurometric_cached_dual_regime.png'),
        )
        save_figure(fig, out_fig)
        plt.close(fig)
        print(f"  Saved: {out_fig}")


def load_model_and_readout(mode: str = 'standard', device: str = None):
    """Load the VisionCore model and population readout."""
    import dill
    import torch

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model (mode={mode}, device={device})...")
    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout

    model, _ = get_model_and_dataset_configs(mode=mode)
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    model = model.to(device)

    print(f"Loading readout from {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)

    return model, readout, outputs


# ─── Phase 1: Data Generation ────────────────────────────────────────────────

def run_phase_1(args):
    """Phase 1: Extract eye traces and verify stimulus generation."""
    print("\n" + "=" * 60)
    print("PHASE 1: Data Generation")
    print("=" * 60)

    # 1.1 Verify E optotype stimulus generation
    print("\n--- 1.1 E Optotype Stimulus Generator ---")
    from stimulus import e_optotype_stack, visualize_e_optotypes, letter_size_pixels

    for logmar in [0.0, 0.5, 1.0]:
        stack = e_optotype_stack(0, logmar)
        size = letter_size_pixels(logmar)
        print(f"  LogMAR={logmar:.1f}: letter={size:.2f}px, stack={stack.shape}, dtype={stack.dtype}")

    if not args.no_figures:
        from plotting import save_figure
        fig = visualize_e_optotypes(
            logmar_values=[-0.2, 0.0, 0.3, 0.6, 1.0],
            orientations=[0, 90, 180, 270],
        )
        save_figure(fig, os.path.join(FIGURES_DIR, 'phase1_e_optotypes.png'))
        import matplotlib.pyplot as plt
        plt.close(fig)

    # 1.2 Extract eye traces
    print("\n--- 1.2 Eye Trace Extraction ---")
    if os.path.exists(EYE_TRACES_PATH) and not args.force:
        print(f"  Loading cached: {EYE_TRACES_PATH}")
        from extract_eye_traces import load_eye_traces, print_summary
        traces_data = load_eye_traces(EYE_TRACES_PATH)
    else:
        model, readout, outputs = load_model_and_readout(mode=args.mode)
        from extract_eye_traces import extract_eye_traces, save_eye_traces, print_summary
        traces_data = extract_eye_traces(model, outputs)
        save_eye_traces(traces_data, EYE_TRACES_PATH)

    from extract_eye_traces import print_summary
    print_summary(traces_data)

    # Apply n_traces limit
    traces = traces_data['traces']
    durations = traces_data['durations']
    rms = traces_data['rms']

    if args.n_traces is not None:
        idx = np.random.choice(len(traces), min(args.n_traces, len(traces)), replace=False)
        traces = traces[idx]
        durations = durations[idx]
        rms = rms[idx]
        print(f"  Using {len(traces)} traces (subset of {len(traces_data['traces'])})")

    # 1.4 Test null trace generation
    print("\n--- 1.4 Null Trace Generation ---")
    from null_traces import generate_phase_randomized_traces, verify_null_properties
    null_sample = generate_phase_randomized_traces(traces[:10], n_nulls=3, seed=42)
    ok = verify_null_properties(traces[:10], null_sample)
    print(f"  Null trace generation: {'OK' if ok else 'WARNING - RMS not preserved'}")

    print("\nPhase 1 complete.")
    return traces, durations, rms, traces_data


# ─── Phase 2: Primary Decoding ────────────────────────────────────────────────

def run_phase_2(args, model, readout, traces, durations):
    """Phase 2: Run decoding ladder, integration time curve, neurometric curves."""
    print("\n" + "=" * 60)
    print("PHASE 2: Primary Decoding")
    print("=" * 60)

    from stimulus import e_optotype_stack
    from rate_computation import compute_population_rates, compute_population_rates_hires
    from decoding import run_decoding_ladder, print_ladder_results
    from null_traces import generate_phase_randomized_traces

    logmar_threshold = args.threshold_logmar
    conditions = ['real', 'stabilized']
    if not args.reduced:
        conditions += ['matched_null']

    orientations = [0, 90, 180, 270]

    def _use_hires_for_logmar(logmar: float) -> bool:
        return float(logmar) < float(args.hires_threshold)

    # 2.0 Generate null traces (needed for matched_null condition)
    null_traces_arr = None
    if 'matched_null' in conditions:
        print(f"\n--- Generating null traces ---")
        null_traces_arr = generate_phase_randomized_traces(traces, n_nulls=5, seed=42)

    # 2.1 Ablation Ladder at threshold LogMAR
    print(f"\n--- 2.1 Ablation Ladder (LogMAR={logmar_threshold:.2f}) ---")
    ladder_cache = os.path.join(RESULTS_DIR, f'ladder_lm{logmar_threshold:.2f}.pkl')

    if os.path.exists(ladder_cache) and not args.force:
        print(f"  Loading cached: {ladder_cache}")
        with open(ladder_cache, 'rb') as f:
            ladder_results_by_condition = pickle.load(f)
    else:
        ladder_results_by_condition = {}

        for cond in conditions:
            print(f"\n  Condition: {cond}")
            rates_by_stim = {}

            for ori in orientations:
                rate_cache = _rate_cache_path_dual_regime(
                    float(logmar_threshold), ori, cond,
                    hires_threshold=float(args.hires_threshold),
                    rate_file_tag='',
                )
                if os.path.exists(rate_cache) and not args.force:
                    from rate_computation import load_rates
                    rates_by_stim[f'ori{ori}'] = load_rates(rate_cache)['rates']
                else:
                    use_hires = _use_hires_for_logmar(logmar_threshold)
                    if use_hires:
                        result = compute_population_rates_hires(
                            model, readout, ori, float(logmar_threshold), traces, durations,
                            condition=cond,
                            null_traces=null_traces_arr if cond == 'matched_null' else None,
                            stim_params={'logmar': float(logmar_threshold), 'orientation': ori},
                            verbose=True,
                        )
                    else:
                        stim_stack = e_optotype_stack(ori, float(logmar_threshold))
                        result = compute_population_rates(
                            model, readout, stim_stack, traces, durations,
                            condition=cond,
                            null_traces=null_traces_arr if cond == 'matched_null' else None,
                            stim_params={'logmar': float(logmar_threshold), 'orientation': ori},
                            verbose=True,
                        )
                    rates_by_stim[f'ori{ori}'] = result['rates']
                    from rate_computation import save_rates
                    save_rates(result, rate_cache)

            models_to_run = ['A', 'B', 'C']
            if not args.reduced:
                models_to_run.append('D')

            results = run_decoding_ladder(
                rates_by_stim,
                models=models_to_run,
                n_splits=args.n_splits,
                run_mlp_control=not args.reduced,
                verbose=True,
            )
            ladder_results_by_condition[cond] = results
            print_ladder_results(results)

        with open(ladder_cache, 'wb') as f:
            pickle.dump(ladder_results_by_condition, f)
        print(f"  Saved ladder results to {ladder_cache}")

    # 2.2 Integration time sweep
    print(f"\n--- 2.2 Integration Time Sweep (method={args.integration_method}) ---")
    int_time_cache = os.path.join(RESULTS_DIR, f'integration_time_{args.integration_method}.pkl')

    if os.path.exists(int_time_cache) and not args.force:
        print(f"  Loading cached: {int_time_cache}")
        with open(int_time_cache, 'rb') as f:
            int_results = pickle.load(f)
    else:
        from integration_time import integration_time_curve, DEFAULT_WINDOWS

        # Load or compute rates at threshold LogMAR for real and stabilized
        rates_by_condition = {}
        for cond in ['real', 'stabilized']:
            rates_by_stim = {}
            for ori in orientations:
                rate_cache = _rate_cache_path_dual_regime(
                    float(logmar_threshold), ori, cond,
                    hires_threshold=float(args.hires_threshold),
                    rate_file_tag='',
                )
                from rate_computation import load_rates
                if os.path.exists(rate_cache):
                    rates_by_stim[f'ori{ori}'] = load_rates(rate_cache)['rates']
                else:
                    use_hires = _use_hires_for_logmar(logmar_threshold)
                    if use_hires:
                        result = compute_population_rates_hires(
                            model, readout, ori, float(logmar_threshold), traces, durations,
                            condition=cond, verbose=False,
                        )
                    else:
                        stim_stack = e_optotype_stack(ori, float(logmar_threshold))
                        result = compute_population_rates(
                            model, readout, stim_stack, traces, durations,
                            condition=cond, verbose=False,
                        )
                    rates_by_stim[f'ori{ori}'] = result['rates']
            rates_by_condition[cond] = rates_by_stim

        windows = [1, 3, 6, 12, 24, 36, 48, 60] if not args.reduced else [1, 6, 24, 60]
        int_results = integration_time_curve(
            rates_by_condition,
            windows=windows,
            n_splits=args.n_splits,
            verbose=True,
            method=args.integration_method,
        )

        with open(int_time_cache, 'wb') as f:
            pickle.dump(int_results, f)

    if not args.no_figures:
        from integration_time import plot_integration_time_curves
        from plotting import save_figure
        import matplotlib.pyplot as plt
        fig = plot_integration_time_curves(int_results)
        save_figure(fig, os.path.join(FIGURES_DIR, f'fig_integration_time_{args.integration_method}.png'))
        plt.close(fig)

    # 2.3 Neurometric curves (most expensive — only run if requested or not reduced)
    if not args.skip_neurometric:
        print(f"\n--- 2.3 Neurometric Curves ---")
        from neurometric import (compute_neurometric_curve, save_neurometric_results,
                                  load_neurometric_results, LOGMAR_REDUCED, LOGMAR_FULL,
                                  CONDITIONS_REDUCED, CONDITIONS_FULL)

        neuro_cache = os.path.join(RESULTS_DIR, 'neurometric.npz')
        if os.path.exists(neuro_cache) and not args.force:
            print(f"  Loading cached: {neuro_cache}")
            neuro_results = load_neurometric_results(neuro_cache)
        else:
            logmar_vals = LOGMAR_REDUCED if args.reduced else LOGMAR_FULL
            conds = CONDITIONS_REDUCED if args.reduced else CONDITIONS_FULL
            neuro_results = compute_neurometric_curve(
                model, readout, traces, durations,
                logmar_values=logmar_vals,
                conditions=conds,
                null_traces=null_traces_arr if 'matched_null' in conds else None,
                cache_dir=RATES_DIR,
                n_splits=args.n_splits,
                verbose=True,
            )
            save_neurometric_results(neuro_results, neuro_cache)

        print(f"  ΔLogMAR = {neuro_results.get('delta_logmar')}")

        if not args.no_figures:
            from neurometric import plot_neurometric_curves
            from plotting import save_figure
            import matplotlib.pyplot as plt
            fig = plot_neurometric_curves(neuro_results)
            save_figure(fig, os.path.join(FIGURES_DIR, 'fig_neurometric.png'))
            plt.close(fig)
    else:
        neuro_results = None
        print("  Skipping neurometric curves (--skip_neurometric)")

    # 2.4 Sequential entropy reduction
    if not args.reduced:
        print(f"\n--- 2.4 Sequential Entropy Reduction ---")
        entropy_cache = os.path.join(RESULTS_DIR, f'entropy_lm{float(logmar_threshold):+.2f}.pkl')

        if os.path.exists(entropy_cache) and not args.force:
            print(f"  Loading cached: {entropy_cache}")
            with open(entropy_cache, 'rb') as f:
                entropy_results = pickle.load(f)
        else:
            from entropy import compute_sequential_entropy

            entropy_results = {}
            for cond in ['real', 'stabilized']:
                rates_by_stim = {}
                for ori in orientations:
                    rate_cache = _rate_cache_path_dual_regime(
                        float(logmar_threshold), ori, cond,
                        hires_threshold=float(args.hires_threshold),
                        rate_file_tag='',
                    )
                    from rate_computation import load_rates
                    if os.path.exists(rate_cache):
                        r_list = load_rates(rate_cache)['rates']
                        min_T = min(r.shape[0] for r in r_list)
                        rates_by_stim[f'ori{ori}'] = np.stack([r[:min_T] for r in r_list])
                    # Skip if rates not computed

                if rates_by_stim:
                    entropy_results[cond] = compute_sequential_entropy(
                        rates_by_stim, n_splits=args.n_splits, verbose=True,
                    )

            with open(entropy_cache, 'wb') as f:
                pickle.dump(entropy_results, f)

        if entropy_results and not args.no_figures:
            from entropy import plot_sequential_entropy
            from plotting import save_figure
            import matplotlib.pyplot as plt
            fig = plot_sequential_entropy(entropy_results)
            save_figure(fig, os.path.join(FIGURES_DIR, 'fig_entropy.png'))
            plt.close(fig)

    print("\nPhase 2 complete.")
    return ladder_results_by_condition, int_results


# ─── Phase 3: Mechanistic Analyses ───────────────────────────────────────────

def run_phase_3(args, model, readout, traces, durations, rms):
    """Phase 3: Covariances, alignment, intervention, budget stratification."""
    print("\n" + "=" * 60)
    print("PHASE 3: Mechanistic Analyses")
    print("=" * 60)

    from rate_computation import load_rates
    from stimulus import e_optotype_stack
    from rate_computation import compute_population_rates, compute_population_rates_hires

    logmar_threshold = float(args.threshold_logmar)
    orientations = [0, 90, 180, 270]

    use_hires = float(logmar_threshold) < float(args.hires_threshold)

    # Load or compute rates for real and stabilized conditions
    rates_by_condition = {}
    for cond in ['real', 'stabilized']:
        rates_by_stim = {}
        for ori in orientations:
            rate_cache = _rate_cache_path_dual_regime(
                float(logmar_threshold), ori, cond,
                hires_threshold=float(args.hires_threshold),
                rate_file_tag='',
            )
            if os.path.exists(rate_cache):
                r_list = load_rates(rate_cache)['rates']
                min_T = min(r.shape[0] for r in r_list)
                rates_by_stim[f'ori{ori}'] = np.stack([r[:min_T] for r in r_list])
            else:
                print(f"  Computing rates: {cond}, ori={ori}...")
                if use_hires:
                    result = compute_population_rates_hires(
                        model, readout, ori, float(logmar_threshold), traces, durations,
                        condition=cond, verbose=False,
                    )
                else:
                    stim_stack = e_optotype_stack(ori, float(logmar_threshold))
                    result = compute_population_rates(
                        model, readout, stim_stack, traces, durations,
                        condition=cond, verbose=False,
                    )
                from rate_computation import save_rates
                save_rates(result, rate_cache)
                min_T = min(r.shape[0] for r in result['rates'])
                rates_by_stim[f'ori{ori}'] = np.stack(
                    [r[:min_T] for r in result['rates']]
                )
        rates_by_condition[cond] = rates_by_stim

    # 3.1 Signal and noise covariances
    print(f"\n--- 3.1 Covariance Analysis ---")
    from geometry import compute_covariances, alignment_fraction, compute_subspace_snr

    cov_cache = os.path.join(RESULTS_DIR, f'covariances_lm{logmar_threshold:+.2f}.pkl')
    if os.path.exists(cov_cache) and not args.force:
        print(f"  Loading cached: {cov_cache}")
        with open(cov_cache, 'rb') as f:
            cov_results = pickle.load(f)
    else:
        cov_results = {}
        for cond in ['real', 'stabilized']:
            print(f"  Computing covariances for {cond}...")
            cov_results[cond] = compute_covariances(
                rates_by_condition[cond], mode='instantaneous'
            )
            fem_trace = float(np.trace(cov_results[cond]['C_FEM']))
            signal_trace = float(np.trace(cov_results[cond]['C_signal']))
            fem_is_negligible = fem_trace < 1e-6 * signal_trace

            alpha, alpha_chance = alignment_fraction(
                cov_results[cond]['C_signal'],
                cov_results[cond]['C_FEM'],
                d=5,
            )
            snr = compute_subspace_snr(
                cov_results[cond]['C_signal'],
                cov_results[cond]['C_FEM'],
                d=5,
            )
            cov_results[cond]['alpha'] = alpha
            cov_results[cond]['alpha_chance'] = alpha_chance
            cov_results[cond]['snr'] = snr
            cov_results[cond]['fem_is_negligible'] = fem_is_negligible
            if fem_is_negligible:
                print(f"    α=N/A (C_FEM≈0, trace={fem_trace:.2e} — stabilized, α undefined)")
            else:
                print(f"    α={alpha:.4f} (chance={alpha_chance:.4f}), SNR={snr:.3f}")

        with open(cov_cache, 'wb') as f:
            pickle.dump(cov_results, f)

    if not args.no_figures:
        from geometry import plot_eigenspectra
        from plotting import save_figure
        import matplotlib.pyplot as plt
        fig = plot_eigenspectra(cov_results, n_show=15)
        save_figure(fig, os.path.join(FIGURES_DIR, 'fig_eigenspectra.png'))
        plt.close(fig)

    # 3.2 Representational intervention
    print(f"\n--- 3.2 Representational Intervention ---")
    from geometry import representational_intervention
    from decoding import run_decoding_ladder

    intervention_cache = os.path.join(RESULTS_DIR, f'intervention_lm{logmar_threshold:+.2f}.pkl')
    if os.path.exists(intervention_cache) and not args.force:
        with open(intervention_cache, 'rb') as f:
            intervention_results = pickle.load(f)
    else:
        intervention_results = {}
        for cond in ['real']:
            print(f"  Running intervention for {cond}...")
            rates_cleaned, _ = representational_intervention(
                rates_by_condition[cond],
                cov_results[cond],
                d_remove=5,
            )

            # Decode on cleaned rates
            results_cleaned = run_decoding_ladder(
                {sid: [rates_cleaned[sid][i] for i in range(rates_cleaned[sid].shape[0])]
                 for sid in rates_cleaned},
                models=['A', 'C'], n_splits=args.n_splits, verbose=False,
            )
            # Decode on original rates
            results_orig = run_decoding_ladder(
                {sid: [rates_by_condition[cond][sid][i]
                       for i in range(rates_by_condition[cond][sid].shape[0])]
                 for sid in rates_by_condition[cond]},
                models=['A', 'C'], n_splits=args.n_splits, verbose=False,
            )
            intervention_results[cond] = {
                'original': results_orig,
                'cleaned': results_cleaned,
                'change_C': (results_cleaned['C']['mean_acc']
                             - results_orig['C']['mean_acc']),
            }
            print(f"  Change in Model C accuracy: "
                  f"{intervention_results[cond]['change_C']:+.4f}")

        with open(intervention_cache, 'wb') as f:
            pickle.dump(intervention_results, f)

    # 3.3 Budget stratification
    print(f"\n--- 3.3 Budget Stratification ---")
    budget_cache = os.path.join(RESULTS_DIR, f'budget_lm{logmar_threshold:+.2f}.pkl')

    if os.path.exists(budget_cache) and not args.force:
        with open(budget_cache, 'rb') as f:
            budget_results = pickle.load(f)
    else:
        from budget_analysis import run_budget_stratification

        budget_results = run_budget_stratification(
            model, readout, logmar_threshold, traces, durations, rms,
            n_bins=3, n_splits=args.n_splits, verbose=True,
        )

        with open(budget_cache, 'wb') as f:
            pickle.dump(budget_results, f)

    if not args.no_figures:
        from budget_analysis import plot_budget_gain
        from plotting import save_figure
        import matplotlib.pyplot as plt
        fig = plot_budget_gain(budget_results)
        save_figure(fig, os.path.join(FIGURES_DIR, 'fig_budget_stratification.png'))
        plt.close(fig)

    print(f"\nTrend: {budget_results.get('trend', 'unknown')}")
    print("\nPhase 3 complete.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Temporal decoding analysis pipeline'
    )
    parser.add_argument('--phase', default='all',
                        choices=['1', '2', '3', '4', 'all'],
                        help='Which phase(s) to run')
    parser.add_argument('--n_traces', type=int, default=None,
                        help='Number of eye traces to use (None = all)')
    parser.add_argument('--threshold_logmar', type=float, default=0.0,
                        help='LogMAR value for ablation ladder and mechanistic analyses. '
                             'Default 0.0 = Snellen 20/20 (1 arcmin gap = 0.625px at 37.5ppd). '
                             'Use -0.1 or -0.2 if Model A is still at ceiling here.')
    parser.add_argument('--logmar_subset', action='store_true',
                        help='Use reduced LogMAR grid for neurometric curves')
    parser.add_argument('--reduced', action='store_true',
                        help='Use reduced settings for faster testing')
    parser.add_argument('--skip_neurometric', action='store_true',
                        help='Skip the expensive neurometric curve computation')
    parser.add_argument('--decode_only', action='store_true',
                        help='Decode-only from cached rate .npz files (no model/sim imports). '
                             'Useful if external data packages are unavailable.')
    parser.add_argument('--rate_file_tag', type=str, default='',
                        help="Optional tag appended to cached rate filenames in decode_only mode (e.g. 'full' -> *_full.npz).")
    parser.add_argument('--integration_method', default='flat_pca',
                        choices=['flat_pca', 'time_mean'],
                        help='Integration-time decoder: flat_pca (original) or time_mean (accumulation control).')
    parser.add_argument('--hires_threshold', type=float, default=0.35,
                        help='LogMAR below which cached mode prefers rates_hires*.npz (dual-regime convention).')
    parser.add_argument('--no_matched_null', action='store_true',
                        help='In cached neurometric sweep, skip matched_null even if caches exist.')
    parser.add_argument('--neurometric_extend_negative', action='store_true',
                        help='In cached neurometric sweep, extend the default LogMAR grid down to -0.50 (adds -0.35,-0.40,-0.45,-0.50).')
    parser.add_argument('--neurometric_logmars', default=None,
                        help='Override cached neurometric LogMAR grid with a comma-separated list of floats (quote if values are negative). '
                             "Example: --neurometric_logmars='1.0,0.8,0.6,0.4,0.2,0.0,-0.2,-0.3,-0.4'")
    parser.add_argument('--neurometric_cache_name', type=str, default='neurometric_cached_dual_regime.npz',
                        help='In decode_only cached mode, filename (or path) for neurometric cache output. '
                             'Defaults to neurometric_cached_dual_regime.npz under scripts/temporal_decoding/data/results/.')
    parser.add_argument('--neurometric_fig_name', type=str, default='fig_neurometric_cached_dual_regime.png',
                        help='In decode_only cached mode, filename (or path) for neurometric figure output. '
                             'Defaults to fig_neurometric_cached_dual_regime.png under scripts/temporal_decoding/figures/.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if cached results exist')
    parser.add_argument('--no_figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n_pca_C', type=int, default=50,
                        help='PCA components for Model C temporal residual (neurometric decoding)')
    parser.add_argument('--mode', default='standard',
                        help='Model loading mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.reduced:
        print("Running in REDUCED mode (fast testing)")
        if args.n_traces is None:
            args.n_traces = 20
        args.skip_neurometric = True

    phases_to_run = set()
    if args.phase == 'all':
        phases_to_run = {'1', '2', '3'}
    else:
        phases_to_run = {args.phase}

    model = None  # lazy-load

    if args.decode_only:
        # Decode-only mode: relies entirely on cached `RATES_DIR` files.
        if '2' in phases_to_run:
            run_phase_2_cached(args)
        else:
            print("decode_only currently supports phase 2 only")
    else:
        # Phase 1: Data generation
        traces, durations, rms, traces_data = run_phase_1(args)

        if '2' in phases_to_run or '3' in phases_to_run:
            if model is None:
                model, readout, outputs = load_model_and_readout(mode=args.mode)

        # Phase 2: Primary decoding
        if '2' in phases_to_run:
            ladder_results, int_results = run_phase_2(args, model, readout, traces, durations)

        # Phase 3: Mechanistic
        if '3' in phases_to_run:
            run_phase_3(args, model, readout, traces, durations, rms)

    # Phase 4: Summary figures
    if '4' in phases_to_run or args.phase == 'all':
        print("\n" + "=" * 60)
        print("PHASE 4: Summary Figures")
        print("=" * 60)

        results_to_plot = {}

        neuro_cache = os.path.join(RESULTS_DIR, 'neurometric.npz')
        if os.path.exists(neuro_cache):
            from neurometric import load_neurometric_results, plot_neurometric_curves
            from plotting import save_figure
            import matplotlib.pyplot as plt
            neuro = load_neurometric_results(neuro_cache)
            fig = plot_neurometric_curves(neuro)
            save_figure(fig, os.path.join(FIGURES_DIR, 'fig_neurometric.png'))
            plt.close(fig)
            print(f"  ΔLogMAR = {neuro.get('delta_logmar')}")

        print(f"\nAll output figures: {FIGURES_DIR}")

    print("\n=== Pipeline complete ===")

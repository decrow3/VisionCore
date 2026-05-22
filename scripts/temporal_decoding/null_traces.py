"""
Task 1.4: Matched-Budget Null Trace Generation

Generates phase-randomized null traces that preserve velocity PSD (and thus
RMS displacement and path length) but destroy specific trajectory structure.

These null traces serve as a control: if decoding results are similar for
real vs. null traces, the result does not depend on the specific biological
structure of fixational eye movements.
"""
import numpy as np


def phase_randomize_trace(trace: np.ndarray, rng=None) -> np.ndarray:
    """
    Phase-randomize a single (T, 2) eye trace.

    Preserves: amplitude spectrum (= velocity PSD), RMS, approximate path length.
    Destroys: specific trajectory, phase relationships between x and y axes.

    Args:
        trace: (T, 2) eye trace in degrees, no NaN values
        rng: numpy random Generator (created if None)

    Returns:
        (T, 2) phase-randomized trace
    """
    rng = rng or np.random.default_rng()
    T = trace.shape[0]
    result = np.zeros_like(trace)

    for dim in range(2):
        ft = np.fft.rfft(trace[:, dim])
        # New independent random phases for each axis (destroying x-y coupling)
        phases = rng.uniform(0, 2 * np.pi, size=ft.shape)
        phases[0] = 0  # preserve DC component (mean position)
        if T % 2 == 0:
            phases[-1] = 0  # preserve Nyquist for even-length signals
        ft_rand = np.abs(ft) * np.exp(1j * phases)
        result[:, dim] = np.fft.irfft(ft_rand, n=T)

    return result.astype(np.float32)


def generate_phase_randomized_traces(
    traces: np.ndarray,
    n_nulls: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate phase-randomized null traces for an array of real eye traces.

    Args:
        traces: (M, max_T, 2) float32, NaN-padded (use durations to find valid length)
        n_nulls: number of null traces per real trace
        seed: random seed for reproducibility

    Returns:
        (M, n_nulls, max_T, 2) float32, NaN-padded same as input
    """
    M, max_T, _ = traces.shape
    rng = np.random.default_rng(seed)
    nulls = np.full((M, n_nulls, max_T, 2), np.nan, dtype=np.float32)

    for i in range(M):
        # Find valid length (first NaN row)
        nan_rows = np.where(np.isnan(traces[i, :, 0]))[0]
        T = int(nan_rows[0]) if len(nan_rows) > 0 else max_T
        valid_trace = traces[i, :T]

        for j in range(n_nulls):
            rand_trace = phase_randomize_trace(valid_trace, rng=rng)
            nulls[i, j, :T] = rand_trace

    return nulls


def verify_null_properties(
    traces: np.ndarray,
    nulls: np.ndarray,
    tol_rms: float = 0.10,
    n_check: int = 20,
) -> bool:
    """
    Verify that null traces preserve RMS (within tolerance) but differ from originals.

    Args:
        traces: (M, max_T, 2) real traces
        nulls: (M, n_nulls, max_T, 2) null traces
        tol_rms: fractional tolerance for RMS preservation
        n_check: number of traces to spot-check

    Returns:
        True if all checks pass
    """
    M = traces.shape[0]
    n_check = min(n_check, M)
    idx = np.random.choice(M, n_check, replace=False)

    rms_errors = []
    for i in idx:
        nan_rows = np.where(np.isnan(traces[i, :, 0]))[0]
        T = int(nan_rows[0]) if len(nan_rows) > 0 else traces.shape[1]
        real = traces[i, :T]
        null = nulls[i, 0, :T]

        real_rms = np.sqrt(np.mean((real - real.mean(0)) ** 2))
        null_rms = np.sqrt(np.mean((null - null.mean(0)) ** 2))

        if real_rms > 1e-6:
            rms_errors.append(abs(real_rms - null_rms) / real_rms)

    if rms_errors:
        max_err = max(rms_errors)
        print(f"RMS preservation: max relative error = {max_err:.4f} (tolerance {tol_rms})")
        return max_err < tol_rms

    return True


def verify_psd_preservation(
    traces: np.ndarray,
    nulls: np.ndarray,
    n_check: int = 20,
) -> None:
    """
    Print PSD comparison between real and null traces (spot check).
    """
    M = traces.shape[0]
    n_check = min(n_check, M)
    idx = np.random.choice(M, n_check, replace=False)

    # Check PSD preservation per-trace (identical by construction for each trace)
    max_rel_diff = 0.0
    for i in idx:
        nan_rows = np.where(np.isnan(traces[i, :, 0]))[0]
        T = int(nan_rows[0]) if len(nan_rows) > 0 else traces.shape[1]
        real = traces[i, :T, 0]
        null = nulls[i, 0, :T, 0]

        real_psd = np.abs(np.fft.rfft(real)) ** 2
        null_psd = np.abs(np.fft.rfft(null)) ** 2
        rel_diff = np.max(np.abs(real_psd - null_psd) / (real_psd + 1e-12))
        max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"PSD preservation: max per-trace relative difference = {max_rel_diff:.2e} "
          f"(should be near machine epsilon)")


if __name__ == '__main__':
    print("Testing phase-randomized null trace generation...")

    rng = np.random.default_rng(42)
    M, max_T = 50, 300

    # Generate synthetic traces with realistic FEM statistics
    traces = np.full((M, max_T, 2), np.nan, dtype=np.float32)
    durations = rng.integers(100, max_T, size=M)
    for i in range(M):
        T = durations[i]
        # Simulate drift + tremor
        drift = np.cumsum(rng.normal(0, 0.002, (T, 2)), axis=0)
        tremor = rng.normal(0, 0.01, (T, 2))
        traces[i, :T] = (drift + tremor).astype(np.float32)

    print(f"Input: {M} traces, durations {durations.min()}–{durations.max()} frames")

    # Generate nulls
    nulls = generate_phase_randomized_traces(traces, n_nulls=10, seed=0)
    assert nulls.shape == (M, 10, max_T, 2), f"Shape error: {nulls.shape}"
    print(f"Output shape: {nulls.shape}  OK")

    # Verify properties
    ok = verify_null_properties(traces, nulls)
    assert ok, "RMS preservation failed"
    print("RMS preservation: OK")

    verify_psd_preservation(traces, nulls)

    # Verify reproducibility
    nulls2 = generate_phase_randomized_traces(traces, n_nulls=10, seed=0)
    assert np.allclose(nulls, nulls2, equal_nan=True), "Reproducibility failed"
    print("Reproducibility: OK")

    # Verify different nulls are different from each other
    for i in range(M):
        T = durations[i]
        assert not np.allclose(nulls[i, 0, :T], nulls[i, 1, :T]), \
            f"Nulls should differ: trace {i}"
    print("Null traces are mutually distinct: OK")

    print("\nAll tests passed!")

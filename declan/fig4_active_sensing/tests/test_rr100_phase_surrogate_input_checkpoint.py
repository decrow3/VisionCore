import numpy as np

from declan.fig4_active_sensing.make_rr100_phase_surrogate_input_checkpoint import (
    apply_coherent_allpass,
    contrast_match_intact,
    full_st_amplitude_relative_error,
    localized_allpass_transfer,
    spatial_contrast_rms,
)


def test_localized_allpass_preserves_full_spatiotemporal_amplitude() -> None:
    rng = np.random.default_rng(3)
    movie = rng.normal(size=(11, 17, 17)).astype(np.float32)
    transfer, _impulse, audit = localized_allpass_transfer(
        (17, 17), sigma_px=2.0, seed=9
    )
    control = apply_coherent_allpass(movie, transfer)

    assert audit["transfer_magnitude_relative_error"] < 1e-12
    assert full_st_amplitude_relative_error(movie, control) < 1e-6
    assert np.allclose(movie.mean(axis=(1, 2)), control.mean(axis=(1, 2)), atol=1e-6)


def test_contrast_match_preserves_frame_means_and_matches_target_rms() -> None:
    rng = np.random.default_rng(11)
    intact = rng.normal(loc=100.0, scale=20.0, size=(8, 9, 9)).astype(np.float32)
    target = rng.normal(loc=85.0, scale=7.0, size=(8, 9, 9)).astype(np.float32)
    matched, scale = contrast_match_intact(intact, target)

    assert 0.0 < scale < 1.0
    assert np.allclose(
        matched.mean(axis=(1, 2)), intact.mean(axis=(1, 2)), atol=1e-5
    )
    assert np.isclose(
        spatial_contrast_rms(matched), spatial_contrast_rms(target), rtol=1e-6
    )

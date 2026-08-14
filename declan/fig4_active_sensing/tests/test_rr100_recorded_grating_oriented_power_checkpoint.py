import numpy as np

from declan.fig4_active_sensing.make_rr100_recorded_grating_oriented_power_checkpoint import (
    GRATING_ORIENTATIONS,
    display_channel_fractions,
    localized_oriented_spectrum,
    synthetic_drifting_grating,
)
from declan.fig4_active_sensing.make_rr100_orientation_routing_input_checkpoint import (
    four_grating_channels,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import ORIENTATION_EDGES_DEG


def test_oriented_power_sums_to_radial_power() -> None:
    rng = np.random.default_rng(18)
    movie = rng.normal(size=(40, 51, 51))
    aperture = np.outer(np.hanning(51), np.hanning(51))
    radial, oriented = localized_oriented_spectrum(movie, ppd=20.0, spatial_aperture=aperture)
    np.testing.assert_allclose(oriented.sum(axis=-1), radial, rtol=1e-10, atol=1e-8)


def test_synthetic_grating_maps_to_expected_bar_orientation_channel() -> None:
    aperture = np.outer(np.hanning(51), np.hanning(51))
    for orientation in GRATING_ORIENTATIONS:
        movie = synthetic_drifting_grating(ppd=20.0, grating_orientation_deg=float(orientation))
        _, oriented = localized_oriented_spectrum(movie, ppd=20.0, spatial_aperture=aperture)
        channels = four_grating_channels(oriented, ORIENTATION_EDGES_DEG).sum(axis=(0, 1))
        assert int(np.argmax(channels)) == int(np.flatnonzero(GRATING_ORIENTATIONS == orientation)[0])


def test_offset_display_orientations_are_power_preservingly_assigned() -> None:
    displayed = np.asarray([11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75])
    fractions = display_channel_fractions(displayed)
    np.testing.assert_allclose(fractions.sum(), 1.0)
    np.testing.assert_allclose(fractions, np.full(4, 0.25))

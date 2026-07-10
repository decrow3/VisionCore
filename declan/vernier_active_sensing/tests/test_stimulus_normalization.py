import numpy as np
import torch

from declan.vernier_active_sensing.forward import renderer_raw_to_model_pixelnorm, build_vernier_movie
from declan.vernier_active_sensing.stimulus import RenderGeometry, VernierSpec


def test_renderer_raw_to_model_pixelnorm_maps_vernier_local_range_to_pixelnorm() -> None:
    geometry = RenderGeometry()
    raw = torch.tensor([0.0, geometry.max_raw / 2.0, geometry.max_raw], dtype=torch.float32)

    actual = renderer_raw_to_model_pixelnorm(raw, max_raw=float(geometry.max_raw))
    expected = torch.tensor([-127.0 / 255.0, 0.5 / 255.0, 128.0 / 255.0], dtype=torch.float32)

    torch.testing.assert_close(actual, expected)


def test_build_vernier_movie_returns_near_zero_neutral_background() -> None:
    geometry = RenderGeometry(world_size=(32, 32), retina_size=(8, 8))
    spec = VernierSpec(contrast=0.0)
    trace = np.zeros((4, 2), dtype=np.float32)

    stim = build_vernier_movie(spec, trace, geometry=geometry, n_lags=3, device="cpu")

    expected_neutral = ((geometry.background_raw * (255.0 / geometry.max_raw)) - 127.0) / 255.0
    assert stim.shape == (4, 1, 3, 8, 8)
    assert torch.isfinite(stim).all()
    torch.testing.assert_close(stim, torch.full_like(stim, expected_neutral))

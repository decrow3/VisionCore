from __future__ import annotations

import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.screen_backimage_local_feature_spaces import (
    _image_metadata_targets,
    _pyramid_feature_targets,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    _cross_validated_decode,
    _mean_r2,
)


def test_pyramid_feature_targets_split_signed_and_energy_channels() -> None:
    n = 2
    n_scales = 2
    n_orientations = 2
    n_channels = 3
    n_grid = 4
    tensor = np.zeros((n, n_scales, n_orientations, n_channels, n_grid), dtype=np.float32)
    tensor[:, :, :, 0, :] = np.arange(n * n_scales * n_orientations * n_grid).reshape(
        n, n_scales, n_orientations, n_grid
    )
    tensor[:, :, :, 1, :] = tensor[:, :, :, 0, :] + 100.0
    tensor[:, :, :, 2, :] = tensor[:, :, :, 0, :] + 200.0
    latents = {"pyramid_local_field": tensor.reshape(n, -1)}

    targets, manifest = _pyramid_feature_targets(
        latents,
        pyramid_latent_name="pyramid_local_field",
        n_scales=n_scales,
        n_orientations=n_orientations,
        n_channels=n_channels,
    )

    assert targets["pyramid_signed_grid"].shape == (n, n_scales * n_orientations * 2 * n_grid)
    assert targets["pyramid_energy_grid"].shape == (n, n_scales * n_orientations * n_grid)
    expected_band_energy = tensor[:, :, :, 2, :].mean(axis=-1).reshape(n, -1)
    assert np.allclose(targets["pyramid_band_energy"], expected_band_energy)
    assert {row["latent"] for row in manifest if row["included"]} >= {
        "pyramid_local_field",
        "pyramid_signed_grid",
        "pyramid_energy_grid",
        "pyramid_band_energy",
    }


def test_image_contour_axis_code_uses_180_degree_axis_encoding() -> None:
    images = pd.DataFrame(
        {
            "image_edge_axis_deg": [0.0, 90.0],
            "image_orientation_coherence": [0.5, 1.0],
        }
    )

    targets, _manifest = _image_metadata_targets(images, include_motion_diagnostic_targets=False)

    assert "image_contour_axis_code" in targets
    contour = targets["image_contour_axis_code"]
    assert contour.shape == (2, 2)
    assert np.allclose(contour[0], [0.5, 0.5], atol=1e-6)
    assert np.allclose(contour[1], [1.0, -1.0], atol=1e-6)


def test_mean_r2_accepts_single_pc_vector_prediction() -> None:
    y_true = np.arange(6, dtype=np.float64)[:, None]
    y_pred = np.arange(6, dtype=np.float64)

    assert np.isclose(_mean_r2(y_true, y_pred), 1.0)


def test_cross_validated_decode_accepts_scalar_target() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 5))
    Z = (X[:, [0]] + 0.1 * rng.normal(size=(24, 1))).astype(np.float64)
    groups = np.repeat(np.arange(12), 2)

    result = _cross_validated_decode(
        X,
        Z,
        groups,
        k=4,
        alphas=[1.0],
        alpha_mode="fixed",
        fixed_alpha=1.0,
        outer_folds=3,
        inner_folds=2,
        seed=0,
    )

    assert result["target_dim"] == 1
    assert result["per_window_score"].shape == (24,)
    assert np.isfinite(result["mean_neg_mse"])

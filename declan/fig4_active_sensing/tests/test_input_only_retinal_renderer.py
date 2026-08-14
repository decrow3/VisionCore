import numpy as np
import torch

from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _trace_xy_to_twin_helper_order,
)


def test_lag_zero_renderer_matches_validated_helper() -> None:
    common = _load_twin_common()
    rng = np.random.default_rng(17)
    patch = rng.uniform(0, 255, size=(540, 540)).astype(np.float32)
    trace = rng.normal(0, 0.03, size=(72, 2)).astype(np.float32)
    direct = render_retinal_frames_lag_zero(common, patch, trace, ppd=37.50476617)
    stack = np.broadcast_to(patch[None], (105, *patch.shape)).copy()
    reference = common.make_counterfactual_stim(
        stack, torch.from_numpy(_trace_xy_to_twin_helper_order(trace)),
        ppd=37.50476617, n_lags=32, out_size=(51, 51),
    )[1:73, 0, 0]
    assert torch.equal(direct, reference)


def test_lag_zero_renderer_accepts_nonproduction_length() -> None:
    common = _load_twin_common()
    rng = np.random.default_rng(23)
    patch = rng.uniform(0, 255, size=(540, 540)).astype(np.float32)
    trace = rng.normal(0, 0.02, size=(32, 2)).astype(np.float32)
    direct = render_retinal_frames_lag_zero(common, patch, trace, ppd=37.50476617)
    stack = np.broadcast_to(patch[None], (65, *patch.shape)).copy()
    reference = common.make_counterfactual_stim(
        stack,
        torch.from_numpy(_trace_xy_to_twin_helper_order(trace)),
        ppd=37.50476617,
        n_lags=32,
        out_size=(51, 51),
    )[1:33, 0, 0]
    assert torch.equal(direct, reference)

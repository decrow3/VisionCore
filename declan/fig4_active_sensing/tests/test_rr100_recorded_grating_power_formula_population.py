import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_recorded_grating_power_formula_population import (
    session_balanced_bootstrap,
)


def test_session_balanced_bootstrap_point_weights_sessions_equally() -> None:
    frame = pd.DataFrame(
        {
            "session": ["large"] * 4 + ["small"],
            "value": [0.0, 0.0, 0.0, 0.0, 2.0],
        }
    )
    point, low, high = session_balanced_bootstrap(
        frame, "value", n_bootstraps=100, rng=np.random.default_rng(4)
    )
    assert point == 1.0
    assert low <= point <= high

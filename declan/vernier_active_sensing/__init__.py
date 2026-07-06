"""Vernier active-sensing analysis helpers."""

from .stimulus import RenderGeometry, VernierSpec
from .synthetic_trajectory_priors import (
    SyntheticTrajectoryPriorConfig,
    SyntheticTrajectoryPriorResult,
    generate_synthetic_trajectory_prior,
    recommended_empirical_confined_config,
)

__all__ = [
    "RenderGeometry",
    "SyntheticTrajectoryPriorConfig",
    "SyntheticTrajectoryPriorResult",
    "VernierSpec",
    "generate_synthetic_trajectory_prior",
    "recommended_empirical_confined_config",
]

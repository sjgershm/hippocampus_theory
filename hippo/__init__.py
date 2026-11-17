"""Hippocampal generator/resolvent network model.

Implements the model from Gershman, "Prediction, imagination, memory:
variations on a hippocampal theme."
"""

from .model import (
    HippocampalModel,
    ContextHippocampalModel,
    AntitheticStream,
    project_to_generator,
    is_generator,
)
from . import environments
from .ensembles import context_gates, bump_inputs, flat_corr
from .pacemaker import pacemaker_phases, branch_stream, SIGMA_ETA

# --- Physical scales, applied consistently across all simulations ------------ #
# Spatial: cm per discretization bin (so 1D tracks / 2D arenas are physical,
# e.g. a 40-bin track = 160 cm, a 52x44 arena = 208x176 cm ~ Pfeiffer-Foster).
BIN_CM = 4.0
# Temporal: one theta cycle (125 ms) as a finite-depth horizon, in model time
# units. All SR readouts use the truncated resolvent (Eq. 14) with this horizon,
# i.e. one theta sweep of look-ahead rather than the infinite-horizon resolvent.
THETA_PERIOD = 6.0

# Mean of the exponential distribution of leak rates, p(lambda) ~ e^{-lambda/mu}
# (a default; the multi-scale/discounting sim now takes leak rates from CA1 data).
LEAK_RATE_MEAN = 1.0

__all__ = [
    "HippocampalModel",
    "ContextHippocampalModel",
    "AntitheticStream",
    "project_to_generator",
    "is_generator",
    "environments",
    "context_gates",
    "bump_inputs",
    "flat_corr",
    "pacemaker_phases",
    "branch_stream",
    "SIGMA_ETA",
    "BIN_CM",
    "THETA_PERIOD",
    "LEAK_RATE_MEAN",
]

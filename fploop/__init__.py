"""
Forecast–Price Feedback Loop
File: fploop/__init__.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Top-level package for the forecast–price feedback-loop simulator. Re-exports the
core public API — the world/observation types, the demand-world, forecaster and
policy base classes, and the simulation and full-information-oracle entry points.
"""

from __future__ import annotations

from fploop.forecasters.base import Forecaster
from fploop.generators.base import DemandWorld
from fploop.loop import run_simulation
from fploop.oracle import full_information_oracle
from fploop.policies.base import Policy
from fploop.types import (
    History,
    MarketState,
    Observation,
    RunResult,
    WorldConfig,
)

__version__ = "0.1.0"

__all__ = [
    "WorldConfig",
    "MarketState",
    "Observation",
    "History",
    "RunResult",
    "DemandWorld",
    "Forecaster",
    "Policy",
    "run_simulation",
    "full_information_oracle",
    "__version__",
]

"""
Forecast–Price Feedback Loop
File: fploop/generators/__init__.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Demand-world generators: the ground-truth environments that produce demand from
prices. Exports the :class:`DemandWorld` abstract base and the concrete
:class:`LinearLogitWorld` constant-elasticity world.
"""

from __future__ import annotations

from fploop.generators.base import DemandWorld
from fploop.generators.linear_logit import LinearLogitWorld

__all__ = ["DemandWorld", "LinearLogitWorld"]

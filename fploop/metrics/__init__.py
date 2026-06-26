"""
Forecast–Price Feedback Loop
File: fploop/metrics/__init__.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Metrics subpackage: groups the evaluation metrics for the feedback loop —
cumulative regret, residual elasticity bias, performative stability, and the
risk-aware (CVaR / worst-case) regret summaries — and re-exports them.
"""

from __future__ import annotations

from fploop.metrics.bias import residual_elasticity_bias
from fploop.metrics.regret import cumulative_regret
from fploop.metrics.risk import cvar_regret, worst_case_over_regimes
from fploop.metrics.stability import performative_gap

__all__ = [
    "cumulative_regret",
    "residual_elasticity_bias",
    "performative_gap",
    "cvar_regret",
    "worst_case_over_regimes",
]

"""
Forecast–Price Feedback Loop
File: fploop/policies/__init__.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Policies subpackage: groups the pricing intervention arms — the greedy baseline,
the exploration family, the causal (IV / DML / censoring-aware) family, and the
decision-focused family — and re-exports them along with the abstract base.
"""

from __future__ import annotations

from fploop.policies.base import Policy
from fploop.policies.baseline import GreedyBaseline
from fploop.policies.causal import (
    CensoringAwarePolicy,
    ControlFunctionPolicy,
    DMLPolicy,
)
from fploop.policies.decision_focused import DecisionFocusedPolicy, DRORobustPolicy
from fploop.policies.exploration import (
    ControlledVariancePricing,
    EpsilonGreedyPricing,
    SPSAPricing,
)

__all__ = [
    "Policy",
    "GreedyBaseline",
    "ControlledVariancePricing",
    "EpsilonGreedyPricing",
    "SPSAPricing",
    "ControlFunctionPolicy",
    "DMLPolicy",
    "CensoringAwarePolicy",
    "DecisionFocusedPolicy",
    "DRORobustPolicy",
]

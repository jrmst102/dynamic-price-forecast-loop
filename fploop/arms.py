"""
Forecast–Price Feedback Loop
File: fploop/arms.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

The single source of truth for the intervention arms: ``name -> (class, family)``.
The offline sweep engine (:mod:`fploop.sweep`) and the figure/export layer
(:mod:`fploop.crossover`) both import this registry, so the arm vocabulary cannot
drift between them. It is deliberately torch-free (the sweep runs headless), so
neither side can own it. Display concerns (labels, colours) live with each
presentation layer, not here.
"""

from __future__ import annotations

from fploop.policies import (
    CensoringAwarePolicy,
    ControlFunctionPolicy,
    ControlledVariancePricing,
    DecisionFocusedPolicy,
    DMLPolicy,
    DRORobustPolicy,
    EpsilonGreedyPricing,
    GreedyBaseline,
    SPSAPricing,
)

# name -> (policy class, family). Family groups arms for the map's colouring and
# the taxonomy; order here is the canonical legend order.
ARM_REGISTRY: dict[str, tuple[type, str]] = {
    "greedy": (GreedyBaseline, "baseline"),
    "controlled_variance": (ControlledVariancePricing, "exploration"),
    "epsilon_greedy": (EpsilonGreedyPricing, "exploration"),
    "spsa": (SPSAPricing, "exploration"),
    "twosls": (ControlFunctionPolicy, "causal"),
    "dml": (DMLPolicy, "causal"),
    "censoring_aware": (CensoringAwarePolicy, "causal"),
    "decision_focused": (DecisionFocusedPolicy, "decision_focused"),
    "dro": (DRORobustPolicy, "decision_focused"),
}

# One principled arm per family — the cross-scenario default for the offline sweep.
DEFAULT_ARMS = ["greedy", "controlled_variance", "spsa", "twosls", "dml", "dro"]

# Arms that only earn their keep under one scenario: :func:`fploop.sweep.run_sweep`
# auto-adds them there and excludes them everywhere else (e.g. censoring-aware EM
# only matters where capacity binds; elsewhere it would just tie greedy).
SCENARIO_ARMS: dict[str, list[str]] = {"censoring": ["censoring_aware"]}

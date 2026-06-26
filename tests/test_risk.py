"""
Forecast–Price Feedback Loop
File: tests/test_risk.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4c §C1.3 — risk-aware regret metrics.

The mean hides tails; CVaR and the minimax worst-case make them first-class so the
crossover map can be re-coloured by a risk-aware criterion.
"""

from __future__ import annotations

import numpy as np

from fploop.metrics import cvar_regret, worst_case_over_regimes


def test_cvar_equals_hand_computed_tail_mean():
    # Worst 10% of ten seeds is the single largest value; worst 20% is the top two.
    regret = np.arange(1.0, 11.0)  # 1..10
    assert cvar_regret(regret, alpha=0.9) == 10.0
    assert cvar_regret(regret, alpha=0.8) == 9.5  # mean of [9, 10]


def test_cvar_single_seed_is_that_seed():
    assert cvar_regret([3.5], alpha=0.9) == 3.5


def test_cvar_at_least_mean():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=rng.integers(1, 40))
        assert cvar_regret(x, alpha=0.9) >= float(np.mean(x)) - 1e-9


def test_worst_case_selects_minimax_arm():
    # B never exceeds 3; A spikes to 5 on one cell -> minimax prefers B.
    worst = worst_case_over_regimes({"A": [1.0, 5.0, 2.0], "B": [3.0, 3.0, 3.0]})
    assert worst == {"A": 5.0, "B": 3.0}
    assert min(worst, key=worst.get) == "B"

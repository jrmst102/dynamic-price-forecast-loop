"""
Forecast–Price Feedback Loop
File: tests/test_aggregation.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4c §C1.4 — winning-arm aggregation with the load-bearing tie rule.

Cells where arms are statistically indistinguishable must be marked ``tie``, not
painted a winner, or the map invents sharp crossovers that aren't real.
"""

from __future__ import annotations

import pandas as pd

from fploop.sweep import winning_arm


def _cell(arm: str, regrets: list[float]) -> list[dict]:
    """Rows for one arm in a single (base/gbt/0.0/0.1) cell, one row per seed."""
    return [
        {
            "scenario": "base",
            "forecaster": "gbt",
            "lambda": 0.0,
            "cost_std": 0.1,
            "arm": arm,
            "seed": i,
            "final_regret": r,
        }
        for i, r in enumerate(regrets)
    ]


def test_picks_lower_regret_arm():
    table = pd.DataFrame(_cell("greedy", [10.0, 10.0]) + _cell("twosls", [4.0, 4.0]))
    win = winning_arm(table, criterion="mean")
    assert len(win) == 1
    assert win.iloc[0]["arm"] == "twosls"


def test_near_equal_arms_tie():
    # 4.0 vs 4.1 -> 2.5% apart, under the 5% default tie threshold.
    table = pd.DataFrame(_cell("greedy", [4.1, 4.1]) + _cell("twosls", [4.0, 4.0]))
    assert winning_arm(table, criterion="mean", tie_rel=0.05).iloc[0]["arm"] == "tie"
    # A tighter threshold separates them again.
    assert winning_arm(table, criterion="mean", tie_rel=0.01).iloc[0]["arm"] == "twosls"


def test_cvar_can_pick_different_winner_than_mean():
    # A: low mean (5.0) but a fat tail (one 20.0 seed). B: higher mean (7.0), no tail.
    # mean prefers A; CVaR (worst seed) prefers B -> the risk-aware ranking diverges.
    table = pd.DataFrame(_cell("A", [0.0, 0.0, 0.0, 20.0]) + _cell("B", [7.0, 7.0, 7.0, 7.0]))
    assert winning_arm(table, criterion="mean").iloc[0]["arm"] == "A"
    assert winning_arm(table, criterion="cvar").iloc[0]["arm"] == "B"


def _bias_cell(arm: str, regrets: list[float], biases: list[float]) -> list[dict]:
    """Rows carrying both final_regret and mean_abs_bias_2h for the bias criterion."""
    return [
        {
            "scenario": "base",
            "forecaster": "gbt",
            "lambda": 0.0,
            "cost_std": 0.1,
            "arm": arm,
            "seed": i,
            "final_regret": r,
            "mean_abs_bias_2h": b,
        }
        for i, (r, b) in enumerate(zip(regrets, biases, strict=True))
    ]


def test_bias_criterion_diverges_from_regret_and_drops_non_estimating_arms():
    # The load-bearing reference case (4b): 2SLS is regret-optimal (orthogonal
    # instrument) but DML recovers the true elasticity better. SPSA never reports
    # an elasticity -> all-nan bias -> dropped, never crowned the bias winner.
    nan = float("nan")
    table = pd.DataFrame(
        _bias_cell("twosls", [4.0, 4.0], [0.5, 0.5])
        + _bias_cell("dml", [4.0, 4.0], [0.1, 0.1])
        + _bias_cell("spsa", [9.0, 9.0], [nan, nan])
    )
    # On regret, 2SLS and DML tie (identical) and SPSA is worst -> tie crowned.
    assert winning_arm(table, criterion="mean").iloc[0]["arm"] == "tie"
    # On bias, DML wins outright and the non-estimating SPSA is excluded.
    assert winning_arm(table, criterion="bias").iloc[0]["arm"] == "dml"


def test_bias_criterion_requires_its_column():
    # A regret-only table cannot be ranked by bias -> a clear error, not a KeyError.
    table = pd.DataFrame(_cell("greedy", [1.0]) + _cell("twosls", [2.0]))
    try:
        winning_arm(table, criterion="bias")
    except ValueError as exc:
        assert "mean_abs_bias_2h" in str(exc)
    else:
        raise AssertionError("expected ValueError for a bias rank on a regret-only table")

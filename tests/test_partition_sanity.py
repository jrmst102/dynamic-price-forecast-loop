"""
Forecast–Price Feedback Loop
File: tests/test_partition_sanity.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Closeout §1.2 — partition-sanity checks on the cached base/reference GBT sweeps.

Validates the rendered crossover partition against the corrected expectation, and
pins the one substantive divergence from the original hypothesis. The corrected
expectation holds: where the instrument is dead an exploration arm wins, and the
``reference`` regret map is owned by 2SLS (never DML dominating). The divergence:
in the ``reference`` world DML collapses onto 2SLS — its EMA-of-price controls go
nearly collinear with log-price, the residualised instrument weakens, and it holds
its prior just like 2SLS — so the **bias** map is won by the price-varying arms
(controlled-variance / DRO) that fix identification at the source, not by DML. DML
instead earns its keep on risk-aware (CVaR) regret and, per §2.1, on drift regret.

The cached sweeps are gitignored build artifacts; these tests skip when absent
(regenerate with ``python -m fploop.sweep --scenario {base,reference} ...``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fploop.sweep import SweepResult, winning_arm

SWEEPS = Path(__file__).resolve().parents[1] / "sweeps"
EXPLORATION = {"greedy", "controlled_variance", "spsa"}


def _load(scenario: str) -> SweepResult:
    """Load a cached GBT sweep, or skip if its parquet is absent."""
    d = SWEEPS / f"{scenario}_gbt"
    if not (d / "results.parquet").exists():
        pytest.skip(f"cached sweep {d} absent; run `python -m fploop.sweep --scenario {scenario}`")
    return SweepResult.load(d)


def _cols(table) -> tuple[list[float], list[float]]:
    """Sorted (lambda grid, cost_std grid) for the sweep."""
    return sorted(table["lambda"].unique()), sorted(table["cost_std"].unique())


def _winner(table, *, criterion: str, lam: float, cost: float) -> str:
    """Winning arm in one cell under ``criterion`` (the map's own logic)."""
    w = winning_arm(table, criterion=criterion)
    row = w[np.isclose(w["lambda"], lam) & np.isclose(w["cost_std"], cost)]
    assert len(row) == 1, f"expected one cell at lambda={lam}, cost_std={cost}"
    return str(row.iloc[0]["arm"])


# --- base -------------------------------------------------------------------


def test_base_dead_instrument_regret_is_exploration():
    t = _load("base").table
    lams, costs = _cols(t)  # cost_std=0 -> dead instrument; strong-endogeneity cell
    assert _winner(t, criterion="mean", lam=lams[-1], cost=costs[0]) in EXPLORATION


def test_base_dead_instrument_bias_is_exploration():
    t = _load("base").table
    lams, costs = _cols(t)
    assert _winner(t, criterion="bias", lam=lams[-1], cost=costs[0]) in EXPLORATION


def test_base_live_instrument_bias_is_twosls():
    t = _load("base").table
    lams, costs = _cols(t)  # cost_std max -> live instrument
    assert _winner(t, criterion="bias", lam=lams[-1], cost=costs[-1]) == "twosls"


# --- reference --------------------------------------------------------------


def test_reference_live_instrument_regret_is_twosls_not_dml():
    """Spec §1.2: the reference regret map is owned by 2SLS (or tie), never DML."""
    t = _load("reference").table
    lams, costs = _cols(t)
    arm = _winner(t, criterion="mean", lam=lams[-1], cost=costs[-1])
    assert arm in {"twosls", "tie"}
    assert arm != "dml"


def test_reference_dead_instrument_bias_is_exploration():
    t = _load("reference").table
    lams, costs = _cols(t)
    assert _winner(t, criterion="bias", lam=lams[-1], cost=costs[0]) in EXPLORATION


def test_reference_bias_winner_is_not_dml():
    """Accepted divergence from the original hypothesis: reference bias is not DML."""
    t = _load("reference").table
    lams, costs = _cols(t)
    assert _winner(t, criterion="bias", lam=lams[-1], cost=costs[-1]) != "dml"


def test_reference_dml_collapses_onto_twosls_on_bias():
    """DML's EMA controls go collinear with price, so it holds its prior like 2SLS:
    both sit near the bias floor while a price-varying arm wins by a wide margin."""
    t = _load("reference").table
    lams, costs = _cols(t)
    cell = t[np.isclose(t["lambda"], lams[-1]) & np.isclose(t["cost_std"], costs[-1])]
    per_arm = cell.groupby("arm")["mean_abs_bias_2h"].mean()
    # DML and 2SLS within a tight band of each other (the collapse)...
    assert abs(per_arm["dml"] - per_arm["twosls"]) < 0.05
    # ...and the price-varying winner beats both causal arms by a wide margin.
    causal_or_nonest = ["dml", "twosls", "spsa"]
    winner_bias = per_arm.drop(labels=[a for a in causal_or_nonest if a in per_arm.index]).min()
    assert winner_bias < 0.5 * per_arm["dml"]


# --- competition / drift / censoring (closeout §2.1) ------------------------


def test_competition_live_instrument_bias_is_dml():
    """Competition is where DML earns its keep: it owns the bias map (cf. reference)."""
    t = _load("competition").table
    lams, costs = _cols(t)
    assert _winner(t, criterion="bias", lam=lams[-1], cost=costs[-1]) == "dml"


def test_competition_live_instrument_regret_is_causal_or_tie():
    """The regret-optimal arm under competition is causal (2SLS/DML), tied at the min."""
    t = _load("competition").table
    lams, costs = _cols(t)
    assert _winner(t, criterion="mean", lam=lams[-1], cost=costs[-1]) in {"tie", "twosls", "dml"}


def test_drift_has_dml_regret_wins():
    """Drift is the one scenario where DML wins mean-regret cells outright (its regret moment)."""
    t = _load("drift").table
    w = winning_arm(t, criterion="mean")
    assert (w["arm"] == "dml").sum() > 0


def test_censoring_aware_wins_under_binding_capacity():
    """With the censoring-aware arm in the sweep it wins regret AND bias where capacity binds."""
    t = _load("censoring").table
    lams, costs = _cols(t)
    assert "censoring_aware" in set(t["arm"]), "censoring sweep must include censoring-aware"
    assert _winner(t, criterion="mean", lam=lams[-1], cost=costs[-1]) == "censoring_aware"
    assert _winner(t, criterion="bias", lam=lams[-1], cost=costs[-1]) == "censoring_aware"

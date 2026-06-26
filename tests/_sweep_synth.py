"""
Forecast–Price Feedback Loop
File: tests/_sweep_synth.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Shared helper: synthesise a tiny in-memory SweepResult for the 4c.2 tests.

Building the table directly (instead of running a real sweep) keeps the
presentation tests fast and lets them force exact ties and a mean/CVaR
divergence that a stochastic sweep could not reliably produce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from fploop.sweep import TABLE_COLUMNS, SweepResult


def _r2(cost_std: float) -> float:
    """A monotone instrument-strength curve so the R² contours are well defined."""
    return min(0.95, 4.0 * cost_std)


def synthetic_sweep(
    cells: Mapping[tuple[float, float, str], Sequence[float]],
    *,
    scenario: str = "base",
    forecaster: str = "gbt",
) -> SweepResult:
    """Build a SweepResult from ``{(lambda, cost_std, arm): [per-seed regret]}``."""
    rows = []
    for (lam, cost_std, arm), regrets in cells.items():
        for seed, regret in enumerate(regrets):
            rows.append(
                {
                    "scenario": scenario,
                    "lambda": float(lam),
                    "cost_std": float(cost_std),
                    "arm": arm,
                    "forecaster": forecaster,
                    "seed": seed,
                    "final_regret": float(regret),
                    "mean_abs_bias_2h": 0.0,
                    "final_perf_gap": 0.0,
                    "first_stage_r2": _r2(cost_std),
                }
            )
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    manifest = {"scenario": scenario, "forecaster": forecaster, "git_commit": "deadbeef"}
    return SweepResult(table=table, manifest=manifest)

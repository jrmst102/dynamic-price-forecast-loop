"""
Forecast–Price Feedback Loop
File: tests/test_overlay_render.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Closeout §1.3 — overlay integration with the real calibrated markets.

The committed ``data/calibrated_markets.csv`` (full 26-category Dominick's
calibration) must drop one labelled dot per category on the crossover map without
error, stacked near lambda~=0 (the realized-historical reading) and spread up the
instrument-strength axis by each market's first-stage R^2.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from fploop.crossover import crossover_map_figure, load_calibration
from tests._sweep_synth import synthetic_sweep

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "calibrated_markets.csv"


def _grid_sweep():
    # R^2 curve (synthetic_sweep uses min(0.95, 4*cost)) spans the categories' range.
    cells = {}
    for lam in (0.0, 0.3, 0.6, 0.9):
        for cost in (0.05, 0.10, 0.15):
            cells[(lam, cost, "greedy")] = [1.0, 1.1, 0.9]
            cells[(lam, cost, "twosls")] = [2.0, 2.1, 1.9]
    return synthetic_sweep(cells)


def test_real_calibration_overlay_renders():
    calibration = load_calibration(REAL_CSV)
    assert calibration is not None and len(calibration) == 26  # full Dominick's calibration

    fig = crossover_map_figure(_grid_sweep(), calibration=calibration)
    assert isinstance(fig, go.Figure)
    scatter = next(t for t in fig.data if t.type == "scatter")
    # One dot per calibrated market, labelled by category.
    assert len(scatter.x) == len(calibration)
    assert set(scatter.text) == set(calibration["category"])
    # Stacked near lambda~=0 (realized-historical regimes), spread up the R^2 axis.
    # The full calibration's implied lambda tops out near 0.06 (FRD, SDR).
    assert all(x < 0.1 for x in scatter.x)
    assert len({round(float(y), 4) for y in scatter.y}) >= 3

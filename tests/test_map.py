"""
Forecast–Price Feedback Loop
File: tests/test_map.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4c §C2.1 — the crossover-map figure.

The map-building function must return a Plotly figure from a cached sweep, paint
the neutral ``tie`` colour where arms are statistically indistinguishable, and
re-colour when the criterion switches from mean to CVaR.
"""

from __future__ import annotations

import plotly.graph_objects as go

from fploop.crossover import TIE_COLOR, crossover_map_figure
from tests._sweep_synth import synthetic_sweep


def _heatmap(fig: go.Figure) -> go.Heatmap:
    return next(t for t in fig.data if t.type == "heatmap")


def _colours(fig: go.Figure) -> set[str]:
    return {colour for _pos, colour in _heatmap(fig).colorscale}


def test_returns_plotly_figure():
    sr = synthetic_sweep(
        {
            (0.0, 0.05, "greedy"): [1.0, 1.1, 0.9],
            (0.0, 0.05, "twosls"): [2.0, 2.1, 1.9],
            (0.9, 0.20, "greedy"): [3.0, 3.1, 2.9],
            (0.9, 0.20, "twosls"): [1.0, 1.1, 0.9],
        }
    )
    fig = crossover_map_figure(sr)
    assert isinstance(fig, go.Figure)
    # greedy wins the low-λ cell, 2SLS wins the high-λ cell: both colours appear.
    assert _colours(fig) >= {"#808080", "#7030a0"}


def test_tie_colour_appears_when_arms_equal():
    # Two arms with identical regret in every cell -> every cell ties.
    sr = synthetic_sweep(
        {
            (0.0, 0.05, "greedy"): [2.0, 2.0, 2.0],
            (0.0, 0.05, "twosls"): [2.0, 2.0, 2.0],
            (0.9, 0.20, "greedy"): [2.0, 2.0, 2.0],
            (0.9, 0.20, "twosls"): [2.0, 2.0, 2.0],
        }
    )
    fig = crossover_map_figure(sr)
    assert TIE_COLOR in _colours(fig)


def test_criterion_switch_changes_colouring():
    # In every cell arm A has the lower MEAN but a fat tail (higher CVaR); arm B is
    # tight. So 'mean' colours the map by A and 'cvar' colours it by B.
    cells = {}
    for lam in (0.0, 0.9):
        for cost in (0.05, 0.20):
            cells[(lam, cost, "greedy")] = [1.0, 1.0, 1.0, 1.0, 10.0]  # mean 2.8, CVaR 10
            cells[(lam, cost, "twosls")] = [3.0, 3.0, 3.0, 3.0, 3.0]  # mean 3.0, CVaR 3
    sr = synthetic_sweep(cells)
    mean_colours = _colours(crossover_map_figure(sr, criterion="mean"))
    cvar_colours = _colours(crossover_map_figure(sr, criterion="cvar"))
    # mean paints the map greedy-grey; CVaR repaints it 2SLS-purple.
    assert "#808080" in mean_colours
    assert "#7030a0" in cvar_colours
    assert mean_colours != cvar_colours

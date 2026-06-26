"""
Forecast–Price Feedback Loop
File: tests/test_overlay.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4c §C2.2 — the calibration overlay ("you are here").

With a ``calibrated_markets.csv`` present, each market is dropped on the map as a
labelled dot; with the file absent the map must still build (graceful, never a
hard fail).
"""

from __future__ import annotations

import plotly.graph_objects as go

from fploop.crossover import crossover_map_figure, load_calibration
from tests._sweep_synth import synthetic_sweep

CALIBRATION_CSV = (
    "category,lambda_implied,first_stage_r2,beta_iv,beta_ols\n"
    "coffee,0.30,0.40,-1.8,-1.2\n"
    "cereal,0.70,0.65,-2.1,-1.0\n"
)


def _grid_sweep():
    cells = {}
    for lam in (0.0, 0.3, 0.6, 0.9):
        for cost in (0.05, 0.10, 0.20):
            cells[(lam, cost, "greedy")] = [1.0, 1.1, 0.9]
            cells[(lam, cost, "twosls")] = [2.0, 2.1, 1.9]
    return synthetic_sweep(cells)


def test_overlay_places_dots(tmp_path):
    path = tmp_path / "calibrated_markets.csv"
    path.write_text(CALIBRATION_CSV)
    calibration = load_calibration(path)
    assert calibration is not None and len(calibration) == 2

    fig = crossover_map_figure(_grid_sweep(), calibration=calibration)
    scatter = next(t for t in fig.data if t.type == "scatter")
    assert list(scatter.text) == ["coffee", "cereal"]
    assert list(scatter.x) == [0.30, 0.70]
    # y is read off the R² axis and must land inside the swept cost_std range.
    assert all(0.05 <= y <= 0.20 for y in scatter.y)


def test_missing_file_is_graceful(tmp_path):
    # load_calibration returns None for an absent file, and the map still builds.
    assert load_calibration(tmp_path / "does_not_exist.csv") is None
    fig = crossover_map_figure(_grid_sweep(), calibration=None)
    assert isinstance(fig, go.Figure)
    assert not any(t.type == "scatter" for t in fig.data)

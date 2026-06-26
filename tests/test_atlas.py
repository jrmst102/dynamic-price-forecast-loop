"""
Forecast–Price Feedback Loop
File: tests/test_atlas.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Closeout §2.3 — the vulnerability atlas (forward-looking-lambda view).

For each calibrated category, a horizontal slice through the map at that category's
first-stage R^2 shows the winning arm as lambda rises. Strong-instrument categories
stay causal-rescuable far up the axis; weak-instrument ones flip to exploration at
low lambda, marked by a crossover lambda.
"""

from __future__ import annotations

import plotly.graph_objects as go

from fploop.crossover import load_calibration, vulnerability_atlas_figure
from tests._sweep_synth import synthetic_sweep

# soup -> R^2 0.80 (maps to cost_std 0.20); beer -> R^2 0.20 (maps to cost_std 0.05).
CALIBRATION_CSV = (
    "category,lambda_implied,first_stage_r2,beta_iv,beta_ols\n"
    "soup,0.0,0.80,-1.5,-1.5\n"
    "beer,0.0,0.20,-2.5,-3.5\n"
)


def _atlas_sweep():
    """Strong instrument (cost 0.20): 2SLS wins all lambda. Weak (cost 0.05): greedy
    at low lambda, exploration (spsa) at high lambda -> a crossover at lambda=0.6."""
    cells = {}
    for lam in (0.0, 0.3, 0.6, 0.9):
        cells[(lam, 0.20, "twosls")] = [1.0, 1.0, 1.0]
        cells[(lam, 0.20, "greedy")] = [2.0, 2.0, 2.0]
        cells[(lam, 0.20, "spsa")] = [2.0, 2.0, 2.0]
        weak_winner = "greedy" if lam < 0.6 else "spsa"
        for arm in ("greedy", "spsa", "twosls"):
            cells[(lam, 0.05, arm)] = [1.0, 1.0, 1.0] if arm == weak_winner else [2.0, 2.0, 2.0]
    return synthetic_sweep(cells)


def test_atlas_one_slice_per_category_with_crossover(tmp_path):
    path = tmp_path / "calibrated_markets.csv"
    path.write_text(CALIBRATION_CSV)
    calibration = load_calibration(path)

    fig = vulnerability_atlas_figure(_atlas_sweep(), calibration, criterion="mean")
    heat = next(t for t in fig.data if t.type == "heatmap")
    # Rows are ordered by instrument strength (ascending R^2): beer (weak) then soup.
    assert list(heat.y) == ["beer", "soup"]
    rows = {cat: list(txt) for cat, txt in zip(heat.y, heat.text, strict=True)}
    # Weak-instrument beer flips greedy -> exploration as lambda rises.
    assert rows["beer"] == ["greedy", "greedy", "SPSA", "SPSA"]
    # Strong-instrument soup stays 2SLS-rescuable all the way up the axis.
    assert rows["soup"] == ["2SLS control-fn"] * 4

    # The crossover marker is placed for beer (at lambda=0.6) and not for soup.
    scatter = next(t for t in fig.data if t.type == "scatter")
    assert list(scatter.y) == ["beer"]
    assert list(scatter.x) == [0.6]


def test_atlas_graceful_without_calibration():
    # No CSV -> a placeholder figure, never a crash.
    fig = vulnerability_atlas_figure(_atlas_sweep(), None)
    assert isinstance(fig, go.Figure)
    assert not any(t.type == "heatmap" for t in fig.data)

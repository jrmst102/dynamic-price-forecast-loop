"""
Forecast–Price Feedback Loop
File: fploop/calibration/__init__.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Calibration: real-market coordinates for the crossover map.

Estimate, per Dominick's category, the regime coordinates the Phase-4c crossover
map overlays as "you are here" dots — the IV elasticity ``beta_iv``, the realized
first-stage instrument R² ``first_stage_r2``, and an implied endogeneity strength
``lambda_implied`` on the same scale as the map's x-axis. The pipeline is three
stages:

- :mod:`fploop.calibration.ingest` — read + clean a category's movement/UPC files
  into a tidy ``(store, upc, week)`` panel with ``lq``, ``lp``, ``lc``.
- :mod:`fploop.calibration.estimate` — OLS/IV elasticity, first-stage R², shock
  parameters, and the implied λ (matched against the project's own world).
- :mod:`fploop.calibration.run` — per-category orchestration -> ``calibrated_markets.csv``.

This workstream only *produces* the CSV the existing overlay consumes; it changes
no simulation, sweep, or map code. ``import fploop`` stays torch-free and
calib-free — the SAS/IV dependencies live behind the optional ``calib`` extra and
are imported lazily inside the submodules.
"""

from __future__ import annotations

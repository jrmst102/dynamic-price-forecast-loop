"""
Forecast–Price Feedback Loop
File: tests/test_imports.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Smoke test of the ``fploop`` package surface: the ``__version__`` is a string and
the documented public symbols (WorldConfig, DemandWorld, Forecaster, Policy,
run_simulation, full_information_oracle) are importable from the top level.
"""

from __future__ import annotations

import fploop


def test_version_is_string():
    assert isinstance(fploop.__version__, str)


def test_public_api_present():
    for name in [
        "WorldConfig",
        "DemandWorld",
        "Forecaster",
        "Policy",
        "run_simulation",
        "full_information_oracle",
    ]:
        assert hasattr(fploop, name), f"missing public symbol: {name}"

"""
Forecast–Price Feedback Loop
File: tests/test_interfaces.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Core interface contracts: the abstract DemandWorld cannot be instantiated, the
deferred strategic-consumer toggle still raises NotImplementedError, the
Observation/History dataclasses construct and log to a frame, and policy classes
carry their family tag.
"""

from __future__ import annotations

import pytest

from fploop import DemandWorld, History, Observation, WorldConfig
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.policies import GreedyBaseline


def test_abc_not_instantiable():
    with pytest.raises(TypeError):
        DemandWorld(WorldConfig())  # type: ignore[abstract]


def test_unimplemented_toggle_raises_not_implemented():
    # Phase 4a implements drift/censoring/competition; the strategic-consumer
    # extension stays deferred (§A.8) and still raises.
    world = LinearLogitWorld(WorldConfig(strategic=True))
    with pytest.raises(NotImplementedError):
        world.reset()


def test_dataclasses_construct_and_log():
    import numpy as np

    cfg = WorldConfig(n_products=1)
    assert cfg.n_products == 1
    hist = History()
    hist.add(
        Observation(
            period=0,
            prices=np.array([10.0]),
            observed_demand=np.array([90.0]),
            revenue=900.0,
        )
    )
    assert len(hist) == 1
    assert hist.as_frame().shape[0] == 1


def test_policy_family_tags():
    assert GreedyBaseline.family == "baseline"

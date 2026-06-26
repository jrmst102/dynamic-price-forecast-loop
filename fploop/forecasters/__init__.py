"""
Forecast–Price Feedback Loop
File: fploop/forecasters/__init__.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Forecaster implementations and the lazy registry. Exports the :class:`Forecaster`
and :class:`DesignMatrixForecaster` bases and the always-available
:class:`GBTForecaster`; the torch-backed feedforward and RNN forecasters are
detected without importing torch and imported lazily on first access so plain
``import fploop`` stays torch-free.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

from fploop.forecasters.base import DesignMatrixForecaster, Forecaster
from fploop.forecasters.gbt import GBTForecaster

# Detect torch WITHOUT importing it, so ``import fploop`` (which triggers this
# package __init__ via ``fploop.forecasters.base``) stays torch-free even when
# the optional ``nn`` extra is installed. The NN forecasters are imported lazily
# on first access (see ``__getattr__``).
HAS_TORCH = importlib.util.find_spec("torch") is not None

__all__ = ["Forecaster", "DesignMatrixForecaster", "GBTForecaster"]
if HAS_TORCH:
    __all__ += ["FeedforwardForecaster", "RNNForecaster"]

if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from fploop.forecasters.feedforward import FeedforwardForecaster
    from fploop.forecasters.rnn import RNNForecaster

_LAZY = {
    "FeedforwardForecaster": "fploop.forecasters.feedforward",
    "RNNForecaster": "fploop.forecasters.rnn",
}


def __getattr__(name: str) -> object:
    """Lazily import the torch-backed forecasters only when actually accessed."""
    module_path = _LAZY.get(name)
    if module_path is not None:
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

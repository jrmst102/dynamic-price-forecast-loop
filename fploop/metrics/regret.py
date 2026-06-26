"""
Forecast–Price Feedback Loop
File: fploop/metrics/regret.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Regret metric: the cumulative profit shortfall of a policy relative to the
full-information oracle over the horizon.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cumulative_regret(realized_profit: NDArray, oracle_profit: NDArray) -> NDArray:
    """Cumulative profit regret vs the full-information oracle.

    Returns ``np.cumsum(oracle_profit - realized_profit)``.
    """
    return np.cumsum(np.asarray(oracle_profit) - np.asarray(realized_profit))

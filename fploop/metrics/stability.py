"""
Forecast–Price Feedback Loop
File: fploop/metrics/stability.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Stability metric: the rolling standard deviation of log price (the performative
gap), used to detect whether a price path has converged or is oscillating.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def performative_gap(prices: NDArray, window: int = 10) -> NDArray:
    """Rolling std of log price over ``window``.

    Low values mean the price path has converged; rising values flag oscillation
    or spiral. Returns an array aligned to ``prices`` where the first
    ``window - 1`` entries are NaN.
    """
    logp = np.log(np.asarray(prices).ravel())
    out = np.full(logp.shape, np.nan)
    for i in range(window - 1, len(logp)):
        out[i] = np.std(logp[i - window + 1 : i + 1])
    return out

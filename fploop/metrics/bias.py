"""
Forecast–Price Feedback Loop
File: fploop/metrics/bias.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Elasticity-bias metric: the signed per-period gap between a policy's estimated
elasticity and the true elasticity, the diagnostic for endogeneity bias.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def residual_elasticity_bias(estimated: NDArray, truth: NDArray) -> NDArray:
    """Signed per-period elasticity bias ``estimated - truth``.

    Warm-up periods (before the first fit) carry no estimate and arrive as NaN.
    Coercing both operands to float keeps those as NaN (so they propagate as gaps)
    and tolerates ``None`` from a JSON round-trip, instead of raising on
    ``None - float`` when the inputs land as an object-dtype array.
    """
    return np.asarray(estimated, dtype=float) - np.asarray(truth, dtype=float)

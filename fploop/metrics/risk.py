"""
Forecast–Price Feedback Loop
File: fploop/metrics/risk.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Risk-aware regret metrics (Phase 4c §C1.3).

The wins matrix ranked arms by the *mean* final regret. The mean hides tails: a
policy that wins on average can still blow up on a handful of seeds (the 3.5
outlier seed that the mean concealed). These metrics make the tail and the
worst case first-class, so the crossover map can be re-coloured by a risk-aware
criterion and the user can watch the ranking diverge from the mean.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


def cvar_regret(regret_by_seed: NDArray | Iterable[float], alpha: float = 0.9) -> float:
    """Tail risk: the mean of the worst ``(1 - alpha)`` fraction of seeds.

    Operates on the *upper* tail of regret (large regret is bad), so
    ``alpha=0.9`` returns the mean of the worst 10% of seeds — the metric that
    would have flagged the 3.5 outlier seed the plain mean hid. At least one seed
    always enters the tail (``ceil`` of the fraction, floored at 1), so for a
    single seed CVaR equals that seed's regret. Because it averages the largest
    values it is always ``>= mean`` (see :func:`cvar_regret` test).

    Parameters
    ----------
    regret_by_seed : array-like
        Final cumulative regret, one value per seed.
    alpha : float, optional
        Tail level in ``[0, 1)``; the worst ``1 - alpha`` fraction is averaged.

    Returns
    -------
    float
        The conditional value-at-risk of regret at level ``alpha``.
    """
    r = np.sort(np.asarray(list(regret_by_seed), dtype=float))  # ascending
    n = r.size
    if n == 0:
        raise ValueError("cvar_regret needs at least one seed")
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}")
    k = max(1, int(np.ceil((1.0 - alpha) * n)))  # number of worst seeds in the tail
    return float(np.mean(r[-k:]))  # mean of the k largest (worst) regrets


def worst_case_over_regimes(
    arm_regret_by_cell: Mapping[str, Iterable[float]],
) -> dict[str, float]:
    """Per-arm worst-case regret across the swept cells — the minimax criterion.

    Takes a mapping ``arm -> iterable of already-aggregated per-cell regrets``
    (each cell summarised by its mean or CVaR across seeds) and returns, per arm,
    the maximum over cells. The **minimax arm** is the one that minimises this:
    ``min(result, key=result.get)``. It is the arm you would pick if you had to
    commit to one intervention before knowing which regime you land in.

    Parameters
    ----------
    arm_regret_by_cell : Mapping[str, Iterable[float]]
        Arm name -> per-cell aggregated regret values.

    Returns
    -------
    dict[str, float]
        Arm name -> worst (largest) per-cell regret.
    """
    out: dict[str, float] = {}
    for arm, cells in arm_regret_by_cell.items():
        vals = np.asarray(list(cells), dtype=float)
        if vals.size == 0:
            raise ValueError(f"arm {arm!r} has no cells")
        out[arm] = float(np.max(vals))
    return out

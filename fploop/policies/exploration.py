"""
Forecast–Price Feedback Loop
File: fploop/policies/exploration.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Exploration pricing arms (Family A): break the feedback loop with deliberate price
variation. Provides controlled-variance jitter, epsilon-greedy forced exploration,
and a model-free SPSA profit optimiser on log price.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fploop.generators.linear_logit import optimal_price
from fploop.policies.base import Policy
from fploop.types import History, MarketState, Observation


class ControlledVariancePricing(Policy):
    """Family A: break the loop with deliberate price variation.

    Identical to :class:`GreedyBaseline` but always multiplies the proposed price
    by exploration noise, guaranteeing dispersion beyond what cost variation
    provides. This relieves incomplete learning (poorly identified slope); it
    does **not** remove the endogeneity bias.
    """

    family = "exploration"

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        beta = self.forecaster.estimated_elasticity()[0]
        if len(history) < self.warmup or not np.isfinite(beta):
            return self._warmup_price(state)
        self._beta_hat = float(beta)
        price = optimal_price(state.cost, self._beta_hat)
        # Controlled variance: dispersion anneals as identification improves, and each
        # draw is truncated at +/-2 sigma so a single shock cannot send the price to
        # extreme multiples of the optimum. Without this the multiplicative noise
        # compounds an already-biased, over-inflated greedy price into ruinous regret
        # at high endogeneity (the price ran to ~95x oracle in early runs).
        t = len(history)
        eff_std = self.explore_std * float(np.sqrt(self.warmup / max(t, self.warmup)))
        z = np.clip(self.rng.normal(0.0, 1.0, size=price.shape), -2.0, 2.0)
        return price * np.exp(eff_std * z)

    def update(self, observation: Observation, history: History) -> None:
        self.forecaster.fit(history)


class EpsilonGreedyPricing(Policy):
    """Family A: greedy with probability ``1-epsilon``, random exploration otherwise.

    Unlike :class:`ControlledVariancePricing` (which always jitters the greedy
    price), this occasionally jumps to a price drawn uniformly across the observed
    log-price range — coarser but standard bandit-style forced exploration. With
    ``epsilon=0`` it reduces exactly to :class:`GreedyBaseline`.
    """

    family = "exploration"

    def __init__(self, *args: object, epsilon: float = 0.1, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        beta = self.forecaster.estimated_elasticity()[0]
        if len(history) < self.warmup or not np.isfinite(beta):
            return self._warmup_price(state)
        self._beta_hat = float(beta)
        if self.rng.random() < self.epsilon:
            logp = np.log(np.array([p[0] for p in history.prices]))
            return np.array([np.exp(self.rng.uniform(logp.min(), logp.max()))])
        return optimal_price(state.cost, self._beta_hat)

    def update(self, observation: Observation, history: History) -> None:
        self.forecaster.fit(history)


class SPSAPricing(Policy):
    """Family A: model-free simultaneous-perturbation profit optimiser on log price.

    Estimates no demand model and no elasticity — an honest contrast to the
    model-based arms. Each gradient step spans two periods: a ``+`` period at
    ``exp(theta + c_k*Delta)`` and a ``-`` period at ``exp(theta - c_k*Delta)``
    with ``Delta`` a fresh +/-1 Rademacher; after the pair the profit difference
    forms the SPSA gradient estimate and ``theta`` ascends. Standard decay gains
    ``a_k = a/(k+1+A)**0.602`` and ``c_k = c/(k+1)**0.101``.

    A forecaster is accepted (for a uniform constructor) but never used;
    ``current_elasticity`` returns ``nan``. Retrains every period by default so the
    +/- pairing is not throttled by the loop's ``retrain_every`` gate.
    """

    family = "exploration"

    def __init__(
        self,
        *args: object,
        a: float = 0.1,
        c: float = 0.1,
        A: float = 10.0,
        retrain_every: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, retrain_every=retrain_every, **kwargs)
        self.a, self.c, self.A = a, c, A
        self._theta: float | None = None  # current log-price centre
        self._k: int = 0  # gradient-step counter
        self._half: int = 0  # 0 -> "+" period, 1 -> "-" period
        self._delta: float = 1.0
        self._profit_plus: float = 0.0

    def _c_k(self) -> float:
        return self.c / (self._k + 1) ** 0.101

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        if len(history) < self.warmup:
            return self._warmup_price(state)
        if self._theta is None:  # seed the centre at the mean warm-up log price
            self._theta = float(np.mean(np.log([p[0] for p in history.prices])))
        c_k = self._c_k()
        if self._half == 0:
            self._delta = 1.0 if self.rng.random() < 0.5 else -1.0
            return np.array([np.exp(self._theta + c_k * self._delta)])
        return np.array([np.exp(self._theta - c_k * self._delta)])

    def update(self, observation: Observation, history: History) -> None:
        profit = float(((observation.prices - observation.cost) * observation.observed_demand)[0])
        if self._half == 0:
            self._profit_plus = profit
            self._half = 1
            return
        # "-" period complete: form the SPSA gradient and ascend.
        c_k = self._c_k()
        a_k = self.a / (self._k + 1 + self.A) ** 0.602
        ghat = (self._profit_plus - profit) / (2.0 * c_k * self._delta)
        self._theta = float(self._theta + a_k * ghat)
        self._k += 1
        self._half = 0

    def current_elasticity(self) -> NDArray:
        return np.array([np.nan])

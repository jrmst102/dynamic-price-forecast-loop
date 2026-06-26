"""
Forecast–Price Feedback Loop
File: fploop/generators/linear_logit.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Single-product constant-elasticity (log-log) demand world. Implements
:class:`LinearLogitWorld` with optional endogeneity, reference-price effects,
intercept drift, competition, and capacity censoring, plus a log-normal cost
shifter that serves as the excluded instrument. Also provides the full-information
:func:`optimal_price` markup rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fploop.generators.base import DemandWorld
from fploop.types import MarketState, Observation, WorldConfig


def optimal_price(cost: NDArray, beta: float, *, floor: float = -1.05) -> NDArray:
    """Full-information optimal price: a constant markup over marginal cost.

    For constant-elasticity demand the profit-maximising price is
    ``cost * beta / (beta + 1)``. The ``floor`` keeps the markup finite and the
    firm in the elastic region when pricing off a *noisy estimated* ``beta`` that
    might wander toward the ``beta = -1`` singularity.

    Parameters
    ----------
    cost : NDArray
        Marginal cost(s), shape ``(n_products,)``.
    beta : float
        Own-price elasticity (true or estimated), expected ``< -1``.
    floor : float, optional
        Upper bound applied to ``beta`` (i.e. ``beta`` is clamped no closer to
        zero than ``floor``) to avoid the singularity, by default ``-1.05``.

    Returns
    -------
    NDArray
        Optimal price(s), same shape as ``cost``.
    """
    b = min(beta, floor)  # keep markup finite and the firm in the elastic region
    return cost * (b / (b + 1.0))


_DRIFT_KINDS = ("none", "gradual", "abrupt")


class LinearLogitWorld(DemandWorld):
    """Single-product constant-elasticity (log-log) demand world.

    Demand composes four optional Phase-4a features on top of the base log-log
    curve ``log q = alpha + beta * log p + xi``:

    - **endogeneity** — the effective transacted price responds to the
      contemporaneous shock, ``log p = log(p_base) + lambda * xi`` (biases a naive
      regression);
    - **reference effect** — a gain term ``theta_ref * (log r_t - log p_eff)``
      where the reference price ``r_t`` is an EMA of the firm's own past effective
      prices (an instrument-independent lock-in channel);
    - **drift** — a deterministic gradual/abrupt schedule on the intercept
      ``alpha_t`` (nonstationarity);
    - **competition** — an observed log-normal rival price ``pc_t`` entering via
      ``beta_cross * log pc_t``;
    - **censoring** — observed demand truncated at ``capacity`` (stockouts).

    With every toggle off (``reference_effect=0``, ``drift_kind="none"``,
    ``competition=False``, ``censoring=False``) the equation reduces exactly to the
    Phase-3.5 form ``log q = alpha + beta * log p_eff + xi``.

    A log-normal marginal-cost shifter provides both the profit objective and the
    excluded instrument used by the control-function policy. ``reset(seed)``
    pre-generates the full seeded shock, cost, drift, and competitor paths so a
    policy run and the oracle run on the same seed see identical exogenous draws;
    ``step`` consumes them by index. The reference price is path-dependent on the
    firm's own prices and so is carried in the state, not pre-generated.
    """

    def __init__(self, config: WorldConfig) -> None:
        super().__init__(config)
        self._state: MarketState | None = None
        self._xi: NDArray | None = None
        self._cost: NDArray | None = None
        self._alpha: NDArray | None = None
        self._pc: NDArray | None = None
        self._t: int = 0

    def _alpha_path(self, alpha0: float, T: int) -> NDArray:
        """Deterministic drifting-intercept schedule (§A.3)."""
        config = self.config
        if config.drift_kind == "none":
            return np.full(T, alpha0)
        if config.drift_kind == "gradual":
            period = config.drift_period or (T // 2)
            return alpha0 + config.drift_magnitude * np.sin(2.0 * np.pi * np.arange(T) / period)
        # "abrupt": level shift at the midpoint
        alpha = np.full(T, alpha0)
        alpha[T // 2 :] = alpha0 + config.drift_magnitude
        return alpha

    def reset(self, seed: int | None = None) -> MarketState:
        config = self.config
        if config.strategic or config.n_products != 1:
            raise NotImplementedError("Phase 4a optional extension — see §A.8")
        if config.drift_kind not in _DRIFT_KINDS:
            raise ValueError(f"drift_kind must be one of {_DRIFT_KINDS}; got {config.drift_kind!r}")

        beta = config.elasticity
        if beta >= -1:
            raise ValueError(f"elasticity must be < -1 (elastic region); got {beta}")
        theta_ref = config.reference_effect
        if theta_ref != 0.0 and not (beta - theta_ref < -1.0):
            raise ValueError(
                "reference_effect leaves the effective contemporaneous elasticity "
                f"inelastic: beta - reference_effect = {beta - theta_ref} must be < -1"
            )
        alpha0 = np.log(config.base_demand)

        rng = np.random.default_rng(seed if seed is not None else config.seed)
        T = config.horizon

        # Demand shock: AR(1) (rho = 0 -> i.i.d.)
        nu = rng.normal(0.0, config.shock_std, size=T)
        xi = np.empty(T)
        xi[0] = nu[0]
        for t in range(1, T):
            xi[t] = config.shock_ar1 * xi[t - 1] + nu[t]

        # Cost shifter (the instrument): log-normal around c_bar
        z = rng.normal(0.0, 1.0, size=T)
        cost = config.marginal_cost * np.exp(config.cost_shifter_std * z)  # shape (T,)

        # Drifting intercept (deterministic) and competitor price path. The
        # competitor draw is taken last and only when competition is on, so with
        # it off the RNG stream matches Phase 3.5 exactly (bit-identical back-compat).
        alpha_path = self._alpha_path(alpha0, T)
        if config.competition:
            zc = rng.normal(0.0, 1.0, size=T)
            pc = config.competitor_price_mean * np.exp(config.competitor_price_std * zc)
        else:
            pc = np.ones(T)  # log 1 = 0, so the competition term vanishes

        self._xi, self._cost, self._alpha, self._pc, self._t = xi, cost, alpha_path, pc, 0
        self._state = MarketState(
            period=0,
            reference_price=np.array([config.marginal_cost]),
            latent_params={"alpha": float(alpha_path[0]), "beta": beta},
            cost=np.array([cost[0]]),
            competitor_price=np.array([pc[0]]),
        )
        return self._state

    def step(self, prices: NDArray) -> Observation:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        config = self.config
        alpha_t = self._state.latent_params["alpha"]
        beta = self._state.latent_params["beta"]
        theta_ref = config.reference_effect
        beta_cross = config.cross_elasticity
        t = self._t
        xi_t, c_t, pc_t = self._xi[t], self._cost[t], self._pc[t]
        r_t = self._state.reference_price  # shape (1,), path-dependent on own prices

        # Endogeneity: the effective transacted price responds to the shock.
        log_p_eff = np.log(np.atleast_1d(prices)) + config.endogeneity_strength * xi_t
        p_eff = np.exp(log_p_eff)
        log_q = (
            alpha_t
            + beta * log_p_eff
            + theta_ref * (np.log(r_t) - log_p_eff)  # reference: gain below reference
            + beta_cross * np.log(pc_t)  # competition (0 when off, pc_t == 1)
            + xi_t
        )
        q_true = np.exp(log_q)
        if config.censoring and config.capacity is not None:
            q_obs = np.minimum(q_true, config.capacity)
            censored = q_true > config.capacity
        else:
            q_obs = q_true
            censored = np.zeros_like(q_true, dtype=bool)
        revenue = float((p_eff * q_obs)[0])  # sell only what's available

        obs = Observation(
            period=t,
            prices=p_eff,
            observed_demand=q_obs,
            revenue=revenue,
            cost=np.array([c_t]),
            competitor_prices=np.array([pc_t]),
            censored=censored,
            reference_price=np.asarray(r_t).copy(),  # r_t entering this period's demand
        )

        # advance reference with the firm's effective price, then t
        gamma = config.reference_memory
        r_next = gamma * r_t + (1.0 - gamma) * p_eff
        self._t += 1
        if self._t < config.horizon:
            self._state = MarketState(
                period=self._t,
                reference_price=r_next,
                latent_params={"alpha": float(self._alpha[self._t]), "beta": beta},
                cost=np.array([self._cost[self._t]]),
                competitor_price=np.array([self._pc[self._t]]),
            )
        return obs

    def true_elasticity(self) -> NDArray:
        # Short-run own-price elasticity: the one-period response holding r_t and
        # pc_t fixed (exactly what the grid-slope readout estimates). The long-run
        # elasticity, after the reference fully adjusts, is beta.
        return np.array([self.config.elasticity - self.config.reference_effect])

    def optimal_prices(self) -> NDArray:
        # Myopic full-information per-period optimum. Contemporaneous demand is
        # constant-elasticity with effective elasticity e = beta - theta_ref.
        config = self.config
        e = config.elasticity - config.reference_effect
        state = self.current_state
        p_star = optimal_price(state.cost, e)
        if config.censoring and config.capacity is not None:
            # log_q(p) = K_t + e*log p; raise the price until demand == capacity.
            k_t = (
                state.latent_params["alpha"]
                + config.reference_effect * np.log(state.reference_price)
                + config.cross_elasticity * np.log(state.competitor_price)
            )
            if float(np.exp(k_t + e * np.log(p_star))[0]) > config.capacity:
                p_cap = np.exp((np.log(config.capacity) - k_t) / e)
                p_star = np.maximum(p_star, p_cap)
        return p_star

    def shock_path(self) -> NDArray:
        if self._xi is None:
            raise RuntimeError("call reset() before reading shock_path()")
        return self._xi

    def cost_path(self) -> NDArray:
        if self._cost is None:
            raise RuntimeError("call reset() before reading cost_path()")
        return self._cost

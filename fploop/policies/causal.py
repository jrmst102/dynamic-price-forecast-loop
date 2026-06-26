"""
Forecast–Price Feedback Loop
File: fploop/policies/causal.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Causal pricing arms (Family B): correct the simultaneity bias rather than just
explore around it. Provides the 2SLS control-function arm, the cross-fitted
double/debiased-ML arm, and the EM/Tobit censoring-aware capacity-pricing arm.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from fploop.features import build_design
from fploop.forecasters.base import DesignMatrixForecaster
from fploop.generators.linear_logit import optimal_price
from fploop.policies.base import Policy
from fploop.types import History, MarketState, Observation


def _twosls_slope(log_p: NDArray, log_c: NDArray, log_q: NDArray) -> tuple[float, float]:
    """2SLS slope of ``log_q`` on ``log_p``, instrumented by ``log_c``.

    First stage regresses ``log_p`` on ``[1, log_c]``; the second stage regresses
    ``log_q`` on ``[1, p_hat]`` where ``p_hat`` is the first-stage fit. The slope
    on ``p_hat`` is the instrumented (structural) elasticity.

    Parameters
    ----------
    log_p, log_c, log_q : NDArray
        Log effective price, log cost (the instrument), and log demand.

    Returns
    -------
    tuple[float, float]
        ``(beta_hat, first_stage_r2)`` — the slope and the first-stage R²
        (instrument strength).
    """
    Z = np.column_stack([np.ones_like(log_c), log_c])
    coef1, *_ = np.linalg.lstsq(Z, log_p, rcond=None)
    p_hat = Z @ coef1
    ss_res = np.sum((log_p - p_hat) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    X2 = np.column_stack([np.ones_like(p_hat), p_hat])
    coef2, *_ = np.linalg.lstsq(X2, log_q, rcond=None)
    return float(coef2[1]), float(r2)


class ControlFunctionPolicy(Policy):
    """Family B: correct the simultaneity bias with an instrument.

    Implemented via its linear special case, **two-stage least squares**, which is
    exact and low-variance in this log-log world. (A flexible control function
    with a tree second stage is ill-posed — the "∂/∂p at the median residual"
    object is not the structural slope when the model can interact price and
    residual — so the causal arm is deliberately forecaster-independent here. The
    orthogonalised / cross-fitted DML generalisation is Phase 4.)

    The excluded cost shifter ``log_c`` is the instrument. Relies on cost
    variation (``cost_shifter_std > 0``) for instrument strength; when the
    instrument is too weak (the collapse regime) or the sample is too small, the
    arm holds its prior rather than chasing a near-unidentified slope.
    """

    family = "causal"

    def __init__(
        self,
        *args: object,
        min_first_stage_r2: float = 0.05,
        min_samples: int = 40,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_first_stage_r2 = min_first_stage_r2
        # In the weak-instrument demo regime (cost_shifter_std ~ 0.05) the first
        # stage can spike spuriously on a handful of early points; accepting that
        # estimate locks the arm onto a bad slope. Require a minimum sample before
        # trusting any 2SLS update so the arm degrades gracefully on every seed.
        self.min_samples = min_samples
        self._beta_hat = -2.0  # prior until the first valid 2SLS update

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        if len(history) < self.warmup:
            return self._warmup_price(state)
        return optimal_price(state.cost, self._beta_hat)

    def update(self, observation: Observation, history: History) -> None:
        log_p = np.log(np.array([p[0] for p in history.prices]))
        log_c = np.log(np.array([c[0] for c in history.costs]))
        log_q = np.log(np.clip(np.array([d[0] for d in history.observed_demand]), 1e-9, None))
        beta_hat, r2 = _twosls_slope(log_p, log_c, log_q)
        # Instrument guard: a weak first stage (the dead-instrument collapse
        # regime) or too few samples means the slope is near-unidentified, so keep
        # the prior and degrade gracefully rather than blowing up.
        if len(history) >= self.min_samples and r2 >= self.min_first_stage_r2:
            self._beta_hat = beta_hat

    def current_elasticity(self) -> NDArray:
        return np.array([self._beta_hat])


def _dml_pliv(
    Y: NDArray,
    D: NDArray,
    Z: NDArray,
    X: NDArray,
    fit_nuisance: Callable[[NDArray, NDArray], Callable[[NDArray], NDArray]],
    K: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Cross-fitted partially-linear IV estimate of ``beta`` in ``Y=beta*D+g(X)+e``.

    ``fit_nuisance(Xtr, ytr) -> predictor(Xte) -> yhat`` fits a flexible regression
    of a target on the controls ``X``. For each of ``K`` folds the three nuisances
    ``E[Y|X]``, ``E[D|X]``, ``E[Z|X]`` are fit on the *other* folds and used to
    residualise the held-out fold; the pooled residuals give the IV moment
    ``beta = Σ(Z̃·Ỹ)/Σ(Z̃·D̃)``. With no controls (``X`` has zero columns) the
    nuisance collapses to the train-fold mean, so residualising is just demeaning
    and ``beta`` equals the ordinary just-identified 2SLS slope — the bare-world
    sanity case.

    Returns ``(beta_hat, strength)`` where ``strength = |Σ(Z̃·D̃)|/n`` is the
    residualised first-stage strength, a weak-instrument guard signal.
    """
    n = Y.shape[0]
    rng = rng or np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), K)
    Yt, Dt, Zt = np.empty(n), np.empty(n), np.empty(n)
    has_controls = X is not None and X.shape[1] > 0
    for fold in folds:
        te = np.zeros(n, dtype=bool)
        te[fold] = True
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            Yt[te], Dt[te], Zt[te] = Y[te], D[te], Z[te]
            continue
        if has_controls:
            ell = fit_nuisance(X[tr], Y[tr])
            rhat = fit_nuisance(X[tr], D[tr])
            mhat = fit_nuisance(X[tr], Z[tr])
            Yt[te] = Y[te] - ell(X[te])
            Dt[te] = D[te] - rhat(X[te])
            Zt[te] = Z[te] - mhat(X[te])
        else:  # demean on the train fold -> pooled IV == 2SLS
            Yt[te] = Y[te] - Y[tr].mean()
            Dt[te] = D[te] - D[tr].mean()
            Zt[te] = Z[te] - Z[tr].mean()
    denom = float(np.sum(Zt * Dt))
    strength = abs(denom) / n
    beta = float(np.sum(Zt * Yt) / denom) if denom != 0.0 else np.nan
    return beta, strength


class DMLPolicy(Policy):
    """Family B: cross-fitted double/debiased ML — the forecaster-dependent causal arm.

    Targets the partially-linear IV model ``Y = beta*D + g(X) + e`` with
    ``E[e|X,Z]=0``, where ``Y=log q``, ``D=log p`` (endogenous), ``Z=log c``
    (instrument), and the controls ``X`` are the nonlinear 4a confounders
    ``{reference, competitor, time}``. The demand forecaster's own class is the
    nuisance learner (so GBT/FF/RNN genuinely differ here): for each fold it fits
    ``E[Y|X]``, ``E[D|X]``, ``E[Z|X]`` on the complement, residualises the held-out
    fold, and pools the residualised IV moment (see :func:`_dml_pliv`).

    Its edge over 2SLS appears **only** against nonlinear confounders: with no
    controls (the bare world) DML coincides with 2SLS by construction — correct,
    not a bug. It earns its keep in the reference and competition worlds, where
    omitting ``r_t`` / ``pc_t`` biases the price-only and 2SLS slopes alike.

    Nuisance fits across folds are costly, so the arm updates on a coarse schedule
    (``retrain_every=10`` default) and, like the 2SLS arm, holds its prior when the
    residualised instrument is too weak (the collapse regime).
    """

    family = "causal"

    def __init__(
        self,
        *args: object,
        controls: tuple[str, ...] = ("reference", "competitor", "time"),
        reference_memory: float = 0.7,
        K: int = 2,
        min_strength: float = 1e-3,
        min_samples: int = 40,
        retrain_every: int = 10,
        nuisance_forecaster_factory: Callable[[], DesignMatrixForecaster] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, retrain_every=retrain_every, **kwargs)
        self.controls = controls
        self.reference_memory = reference_memory
        self.K = K
        self.min_strength = min_strength
        self.min_samples = min_samples
        self._nuisance_factory = nuisance_forecaster_factory
        self._beta_hat = -2.0  # prior until the first valid DML update

    def _fit_nuisance(self, Xtr: NDArray, ytr: NDArray) -> Callable[[NDArray], NDArray]:
        factory = self._nuisance_factory or type(self.forecaster)
        model = factory()
        model.fit_design(Xtr, ytr)
        return model.predict_logq

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        if len(history) < self.warmup:
            return self._warmup_price(state)
        return optimal_price(state.cost, self._beta_hat)

    def update(self, observation: Observation, history: History) -> None:
        if len(history) < self.min_samples:
            return
        X_full, y = build_design(history, self.controls, self.reference_memory)
        logp = X_full[:, 0]
        controls = X_full[:, 1:]
        # Drop near-constant control columns (e.g. competitor price when
        # competition is off, pc_t == 1): they carry no information and would only
        # add nuisance variance. If all controls vanish, DML reduces to 2SLS.
        keep = [j for j in range(controls.shape[1]) if np.std(controls[:, j]) > 1e-8]
        controls = controls[:, keep]
        logc = np.log(np.array([c[0] for c in history.costs]))
        sl = slice(self.warmup, None)  # post-warmup history only
        beta_hat, strength = _dml_pliv(
            y[sl], logp[sl], logc[sl], controls[sl], self._fit_nuisance, K=self.K, rng=self.rng
        )
        # Weak residualised instrument (the dead-instrument collapse) -> hold prior.
        if strength >= self.min_strength and np.isfinite(beta_hat):
            self._beta_hat = beta_hat

    def current_elasticity(self) -> NDArray:
        return np.array([self._beta_hat])


def _phi(x: NDArray) -> NDArray:
    """Standard-normal pdf."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _mills(a: NDArray) -> NDArray:
    """Inverse Mills ratio ``phi(a)/(1-Phi(a))`` — ``E[N(0,1) | N(0,1) > a]``."""
    upper = np.clip(0.5 * (1.0 - erf(a / np.sqrt(2.0))), 1e-12, None)  # 1 - Phi(a)
    return _phi(a) / upper


class CensoringAwarePolicy(Policy):
    """Family B: EM/Tobit demand unconstraining under capacity censoring.

    A naive forecaster trains on ``q_obs = min(q_true, capacity)``, so when
    capacity binds it underestimates demand and the elasticity readout is biased.
    This arm reconstructs the latent demand first. Because the demand is log-linear
    with a Gaussian shock, the censored conditional mean is closed-form (the
    truncated-normal / inverse-Mills correction), giving an EM loop:

    1. fit the demand regression on the current (initially censored) ``log q``;
    2. for each censored point, impute the latent ``log q`` as the truncated-normal
       conditional mean above ``log(capacity)`` given the current fit and residual
       scale;
    3. refit on the completed series and iterate (``em_iters`` times).

    It then prices off the **unconstrained-fit** elasticity (grid-slope on the
    completed data). Censoring is detected from the data — a spike of observed
    demand at its maximum — so with censoring off (no spike) the arm reduces
    exactly to the greedy price-only fit.
    """

    family = "causal"

    def __init__(
        self,
        *args: object,
        em_iters: int = 5,
        sigma_floor: float = 1e-3,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.em_iters = em_iters
        self.sigma_floor = sigma_floor
        self._capacity_hat: float | None = None  # inferred binding capacity, None when slack
        # Log-linear Tobit fit, set only while censoring binds (else None -> GBT readout).
        self._em_beta: float | None = None
        self._em_intercept: float = 0.0

    @staticmethod
    def _censored_mask(q_obs: NDArray) -> NDArray:
        """Censored points pile up exactly at the observed-demand maximum (the cap).

        Uncensored demand is continuous, so its maximum is unique (one tie) and no
        censoring is inferred; a binding capacity produces a spike of several points
        at the same value.
        """
        cap = float(q_obs.max())
        mask = q_obs >= cap * (1.0 - 1e-9)
        return mask if int(mask.sum()) > 1 else np.zeros_like(mask, dtype=bool)

    def update(self, observation: Observation, history: History) -> None:
        X, y = build_design(history, controls=())  # price-only, like greedy
        q_obs = np.array([d[0] for d in history.observed_demand])
        censored = self._censored_mask(q_obs)
        if not censored.any():  # no binding capacity -> ordinary greedy fit
            self._capacity_hat = None
            self._em_beta = None  # signal propose_price/readout to use the GBT slope
            self.forecaster.fit_design(X, y)
            return
        self._capacity_hat = float(q_obs[censored].max())  # inferred capacity (the spike level)
        threshold = np.log(self._capacity_hat)  # log(capacity)
        logp = X[:, 0]  # column 0 is log price (price-only design)

        # EM/Tobit on a LOG-LINEAR demand model, not the GBT. Demand is log-linear by
        # construction, and a linear fit *extrapolates* the slope above the cap; the
        # piecewise-constant tree cannot, so under heavy censoring (e.g. a fully
        # censored warmup) the tree's imputation never escalates and the elasticity
        # collapses to ~0 (markup-floor overpricing). The linear Tobit recovers the
        # slope from the price variation even when the low-price region is censored.
        y_em = y.copy()
        slope, intercept = self._linear_fit(logp, y_em)
        for _ in range(self.em_iters):
            mu = intercept + slope * logp
            resid = (y_em - mu)[~censored]
            if resid.size < 2:  # near-total censoring: fall back to all-point residuals
                resid = y_em - mu
            sigma = float(np.std(resid)) if resid.size else 0.0
            if not np.isfinite(sigma) or sigma < self.sigma_floor:
                sigma = self.sigma_floor
            a = (threshold - mu[censored]) / sigma
            imputed = mu[censored] + sigma * _mills(a)  # truncated-normal mean above the cap
            y_em[censored] = np.where(np.isfinite(imputed), imputed, threshold)
            slope, intercept = self._linear_fit(logp, y_em)
        self._em_beta, self._em_intercept = float(slope), float(intercept)
        self.forecaster.fit_design(X, y_em)  # keep the forecaster fitted on the completed series

    @staticmethod
    def _linear_fit(logp: NDArray, y: NDArray) -> tuple[float, float]:
        """OLS of ``y`` on ``logp`` -> ``(slope, intercept)``; flat if price is frozen."""
        if float(np.ptp(logp)) < 1e-9:  # no price variation -> slope undefined
            return 0.0, float(np.mean(y))
        slope, intercept = np.polyfit(logp, y, 1)
        return float(slope), float(intercept)

    def current_elasticity(self) -> NDArray:
        # While censoring binds, report the Tobit slope; otherwise the GBT grid-slope
        # (so with censoring off the readout matches greedy exactly).
        if self._em_beta is not None and np.isfinite(self._em_beta):
            return np.array([self._em_beta])
        return super().current_elasticity()

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        # Price on the Tobit slope while censoring binds, else the GBT slope (greedy).
        if self._em_beta is not None and np.isfinite(self._em_beta):
            beta = self._em_beta
        else:
            beta = self.forecaster.estimated_elasticity()[0]
        if len(history) < self.warmup or not np.isfinite(beta):
            return self._warmup_price(state)
        self._beta_hat = float(beta)
        p_unc = optimal_price(state.cost, self._beta_hat)
        # Capacity-aware rule (oracle's): if a binding capacity was inferred and the
        # unconstrained optimum would still over-demand it, raise the price until the
        # *estimated* demand chokes down to the inferred capacity. Pricing at the
        # unconstrained optimum under a binding cap leaves money on the table — the
        # firm sells only `capacity` units and could charge more for them. With no
        # censoring (`_capacity_hat is None`) this reduces exactly to the greedy price.
        if self._capacity_hat is not None and self._capacity_hat > 0 and self._em_beta is not None:
            # Estimated unconstrained demand at p_unc from the fitted log-linear model.
            q_hat = float(np.exp(self._em_intercept + self._beta_hat * np.log(float(p_unc[0]))))
            if np.isfinite(q_hat) and q_hat > self._capacity_hat:
                # log p_cap = log p_unc + (log(cap) - log q_hat) / beta_hat
                ratio = np.log(self._capacity_hat) - np.log(q_hat)
                p_cap = p_unc * np.exp(ratio / self._beta_hat)
                # Guard against a degenerate slope blowing the choke price up to inf,
                # which would poison the next fit; fall back to the unconstrained price.
                if np.all(np.isfinite(p_cap)) and np.all(p_cap > 0):
                    return np.maximum(p_unc, p_cap)
        return p_unc

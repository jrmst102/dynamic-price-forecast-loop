"""
Forecast–Price Feedback Loop
File: tests/test_dml.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4b.2 — cross-fitted double/debiased ML (§B2.1, §B2.3).

DML is the forecaster-dependent causal arm. Its win over 2SLS appears only when a
*nonlinear confounder* the 4a worlds introduce (the reference price ``r_t``, the
competitor price ``pc_t``) biases the price-only and 2SLS slopes alike. In the bare
world it coincides with 2SLS by construction — a sanity check, not a defect.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.policies import ControlFunctionPolicy, DMLPolicy, GreedyBaseline
from fploop.policies.causal import _dml_pliv
from fploop.types import WorldConfig

# DML's nuisance learner is the demand forecaster's own class. The default
# (regularised) GBT is deliberately smooth, which is exactly what a faithful
# E[Y|X]/E[D|X] wants when the confounder enters demand smoothly — a less
# regularised nuisance overfits the residuals and erodes the win. The advantage
# over 2SLS is modest, so the comparisons average over several seeds.
SEEDS = range(6)


def _mean_abs_bias(policy_cls, cfg, short_run, **kw):
    biases = []
    for seed in SEEDS:
        world = LinearLogitWorld(cfg)
        policy = policy_cls(GBTForecaster(), rng=np.random.default_rng(1), **kw)
        est = run_simulation(world, policy, seed=seed).estimated_elasticity.ravel()
        half = len(est) // 2
        biases.append(np.nanmean(np.abs(est[half:] - short_run)))
    return float(np.mean(biases))


def test_dml_pliv_reduces_to_2sls_with_no_controls():
    # With zero control columns the cross-fit residualisation is pure demeaning, so
    # the pooled IV moment equals the just-identified 2SLS slope cov(Z,Y)/cov(Z,D).
    rng = np.random.default_rng(0)
    n = 400
    Z = rng.normal(size=n)
    D = 0.8 * Z + rng.normal(size=n)  # endogenous price driven by the instrument
    Y = -2.0 * D + rng.normal(size=n)
    beta, strength = _dml_pliv(Y, D, Z, np.empty((n, 0)), None, K=2, rng=rng)
    iv = np.cov(Z, Y)[0, 1] / np.cov(Z, D)[0, 1]
    assert abs(beta - iv) < 0.05
    assert strength > 0.0


REFERENCE = WorldConfig(
    elasticity=-1.5,
    reference_effect=0.5,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.15,
    horizon=200,
)


def test_dml_beats_2sls_in_reference_world():
    # The +1.18 OVB regime with a live instrument: the price-only greedy slope is
    # badly biased toward zero. The cost instrument is (near-)orthogonal to r_t, so
    # 2SLS largely escapes the bias — but conditioning the nuisances on r_t lets DML
    # trim the residual finite-sample bias below 2SLS as well.
    short_run = REFERENCE.elasticity - REFERENCE.reference_effect  # -2.0
    g_bias = _mean_abs_bias(GreedyBaseline, REFERENCE, short_run)
    s_bias = _mean_abs_bias(ControlFunctionPolicy, REFERENCE, short_run)
    d_bias = _mean_abs_bias(DMLPolicy, REFERENCE, short_run, controls=("reference",))
    assert d_bias < g_bias, f"DML |bias| {d_bias:.3f} not below greedy {g_bias:.3f}"
    assert d_bias < s_bias, f"DML |bias| {d_bias:.3f} not below 2SLS {s_bias:.3f}"


COMPETITION = WorldConfig(
    elasticity=-1.5,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.15,
    horizon=200,
    competition=True,
    cross_elasticity=0.6,
    competitor_price_std=0.25,
)


def test_dml_beats_2sls_in_competition_world():
    # Endogeneity biases the price-only greedy slope; an omitted competitor price
    # adds further noise to it. 2SLS corrects the endogeneity; DML, conditioning the
    # nuisances on pc_t, trims the residual bias below 2SLS. (Per §B2.3, the guard
    # below checks the world actually produces a sizeable price-only bias to fix.)
    short_run = COMPETITION.elasticity  # -1.5
    g_bias = _mean_abs_bias(GreedyBaseline, COMPETITION, short_run)
    s_bias = _mean_abs_bias(ControlFunctionPolicy, COMPETITION, short_run)
    assert g_bias > 0.1, f"competition world produced no price-only bias ({g_bias:.3f})"
    d_bias = _mean_abs_bias(DMLPolicy, COMPETITION, short_run, controls=("competitor",))
    assert d_bias < s_bias, f"DML |bias| {d_bias:.3f} not below 2SLS {s_bias:.3f}"


def test_dml_agrees_with_2sls_in_bare_world():
    # No nonlinear controls: DML must coincide with 2SLS to tolerance (sanity).
    bare = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.6,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=200,
    )

    def last_mean(policy_cls, **kw):
        world = LinearLogitWorld(bare)
        policy = policy_cls(GBTForecaster(), rng=np.random.default_rng(1), **kw)
        est = run_simulation(world, policy, seed=0).estimated_elasticity.ravel()
        return float(np.nanmean(est[-50:]))

    s_est = last_mean(ControlFunctionPolicy)
    d_est = last_mean(DMLPolicy, controls=())
    assert abs(d_est - s_est) < 0.15, f"DML {d_est:.3f} disagrees with 2SLS {s_est:.3f} bare"

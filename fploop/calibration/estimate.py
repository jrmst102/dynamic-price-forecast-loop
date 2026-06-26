"""
Forecast–Price Feedback Loop
File: fploop/calibration/estimate.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Elasticity / IV / implied-λ estimation on a cleaned panel (methodology LOCKED).

Per category, on the tidy panel from :mod:`fploop.calibration.ingest`, absorbing
**store×UPC and week-of-year fixed effects** (within transformation) and clustering
by unit:

- ``beta_ols`` — OLS of ``lq`` on ``lp`` (endogeneity-contaminated).
- ``beta_iv`` — 2SLS of ``lq`` on ``lp`` instrumented by ``lc`` (the cost shifter):
  the consistent structural elasticity.
- ``first_stage_r2`` — partial R² of the instrument ``lc`` in the first stage
  ``lp ~ lc`` (after FE): the realized instrument strength, the map's y-coordinate.
- ``sigma_xi`` / ``rho`` — residual sd and within-unit lag-1 autocorrelation of the
  IV demand residual (sweep-grid realism).
- ``lambda_implied`` — the λ in :class:`~fploop.types.WorldConfig` that reproduces
  the data's OLS–IV gap, matched against the project's **own world** so the implied
  λ is on the same scale as the crossover map's x-axis.

Reading λ (important framing). Calibrating against *historical* data measures the
endogeneity that was **realized** in the retailer's past pricing — not a structural
inevitability. ``lambda_implied`` is therefore "where this market's historical
pricing regime sat on the loop-strength axis", a descriptive coordinate, not a
forecast. A category with a strong instrument but ``lambda_implied ≈ 0`` (the OLS–IV
gap is null or negative) is telling you the feedback loop was **not biting** in that
market's history — an honest landing in the low-λ region where greedy/exploration
suffices and IV is not the fix — even though the same market could move along the
axis if pricing behaviour changed.

``linearmodels`` (imported lazily) supplies the clustered IV standard error; the
point estimates and partial R² are computed directly so the file stays light.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fploop.calibration.ingest import PANEL_COLUMNS
from fploop.generators.linear_logit import LinearLogitWorld, optimal_price
from fploop.types import WorldConfig

_WEAK_IV_F = 10.0  # first-stage F below this flags a weak instrument


# --- Synthetic panel from the project's own world (for the implied-λ match) ---


def simulate_world_panel(
    *,
    beta: float,
    lambda_: float,
    sigma_xi: float,
    sigma_c: float,
    sigma_price_noise: float = 0.05,
    n_units: int = 8,
    n_weeks: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a tidy panel from :class:`LinearLogitWorld` with known parameters.

    Each unit is an independent world run priced at the cost-markup optimum plus
    i.i.d. log-price noise, so the cost shifter is a valid instrument and the
    endogeneity channel ``log p_eff = log p + lambda * xi`` biases OLS exactly as in
    the real loop. Used both to validate the estimator (synthetic recovery) and to
    invert the OLS–IV gap into :func:`implied_lambda`.

    Parameters
    ----------
    beta : float
        True own-price elasticity (``< -1``).
    lambda_ : float
        Endogeneity strength (the quantity being recovered).
    sigma_xi : float
        Demand-shock sd.
    sigma_c : float
        Log cost-shifter sd (instrument strength).
    sigma_price_noise : float
        Sd of exogenous log-price noise on top of the cost markup.
    n_units, n_weeks : int
        Panel dimensions (units × weeks).
    seed : int
        Base seed; each unit and its price noise are independently seeded from it.

    Returns
    -------
    pd.DataFrame
        Panel with columns :data:`~fploop.calibration.ingest.PANEL_COLUMNS`.
    """
    rows: list[dict] = []
    for u in range(n_units):
        cfg = WorldConfig(
            elasticity=float(beta),
            endogeneity_strength=float(lambda_),
            shock_std=float(sigma_xi),
            cost_shifter_std=float(sigma_c),
            horizon=int(n_weeks),
        )
        world = LinearLogitWorld(cfg)
        world.reset(seed=seed * 100003 + u)
        costs = world.cost_path()
        rng = np.random.default_rng(seed * 99991 + u + 1)
        noise = rng.normal(0.0, sigma_price_noise, size=n_weeks)
        for t in range(n_weeks):
            price = optimal_price(np.array([costs[t]]), beta) * np.exp(noise[t])
            obs = world.step(price)
            rows.append(
                {
                    "store": u,
                    "upc": 0,
                    "week": t,
                    "woy": (t % 52) + 1,
                    "lq": float(np.log(obs.observed_demand[0])),
                    "lp": float(np.log(obs.prices[0])),
                    "lc": float(np.log(obs.cost[0])),
                    "deal": "",
                }
            )
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


# --- Fixed-effect absorption + point estimates -------------------------------


def _absorb_within(
    values: np.ndarray, group_codes: list[np.ndarray], *, tol: float = 1e-8, max_iter: int = 100
) -> np.ndarray:
    """Within-transform ``values`` by alternating projections over the FE groups.

    Subtracts group means for each FE in turn until a full sweep moves every column
    by less than ``tol`` — the standard way to absorb multi-way fixed effects
    without forming a dummy matrix.
    """
    X = np.atleast_2d(values.astype(float))
    if X.shape[0] == 1:
        X = X.T
    if not group_codes:
        return X - X.mean(axis=0, keepdims=True)
    for _ in range(max_iter):
        max_change = 0.0
        for code in group_codes:
            counts = np.bincount(code)
            for j in range(X.shape[1]):
                means = np.bincount(code, weights=X[:, j]) / counts
                adj = means[code]
                X[:, j] -= adj
                if adj.size:
                    max_change = max(max_change, float(np.max(np.abs(adj))))
        if max_change < tol:
            break
    return X


def _unit_codes(panel: pd.DataFrame) -> np.ndarray:
    """Factorize the (store, upc) pair into a single unit code."""
    pair = list(zip(panel["store"].to_numpy(), panel["upc"].to_numpy(), strict=True))
    return pd.factorize(pd.Series(pair))[0]


def _gap_from_demeaned(lq_d: np.ndarray, lp_d: np.ndarray, lc_d: np.ndarray) -> float:
    """OLS–IV elasticity gap on already-demeaned columns (just-identified IV)."""
    beta_ols = float(lp_d @ lq_d / (lp_d @ lp_d))
    beta_iv = float(lc_d @ lq_d / (lc_d @ lp_d))
    return beta_ols - beta_iv


# --- Implied λ: invert the OLS–IV gap against the project's own world ---------


def implied_lambda(
    *,
    target_gap: float,
    beta_iv: float,
    sigma_xi: float,
    sigma_c: float,
    first_stage_r2: float,
    grid: np.ndarray | None = None,
    seed: int = 7,
) -> float:
    """Find the world's λ that reproduces the data's OLS–IV gap.

    Runs a synthetic mini-sweep over candidate λ with :func:`simulate_world_panel`
    (holding ``beta_iv``, ``sigma_xi``, ``sigma_c`` fixed and the log-price noise set
    so the synthetic first-stage R² matches the data's), measures each synthetic
    OLS–IV gap, and interpolates the λ whose gap equals ``target_gap``. Because the
    synthetic bias is monotone in λ over the relevant range, the match is well
    defined; a ``target_gap`` at or below zero maps to ``0.0`` and one above the
    grid's reach is clamped to the grid's top (an extreme-endogeneity read).

    Returns
    -------
    float
        The implied endogeneity strength, on the same scale as
        ``WorldConfig.endogeneity_strength``.
    """
    if not np.isfinite(target_gap) or target_gap <= 0:
        return 0.0
    if grid is None:
        grid = np.arange(0.0, 0.96, 0.05)
    # Hold total log-price variance at the data's value across the grid. The data's
    # first stage gives sigma_c^2 + sigma_noise^2 + lam^2 sigma_xi^2 = sigma_c^2 / R2;
    # so at each candidate lam the exogenous price-noise variance is the remainder,
    # keeping Var(log p_eff) constant (the OLS-IV gap is then ~linear in lam).
    r2 = float(np.clip(first_stage_r2, 1e-3, 0.999))
    price_var_total = sigma_c**2 / r2

    gaps = []
    for lam in grid:
        noise_var = max(price_var_total - sigma_c**2 - lam**2 * sigma_xi**2, 0.0)
        panel = simulate_world_panel(
            beta=beta_iv,
            lambda_=float(lam),
            sigma_xi=sigma_xi,
            sigma_c=sigma_c,
            sigma_price_noise=float(np.sqrt(noise_var)),
            n_units=6,
            n_weeks=250,
            seed=seed,
        )
        codes = [_unit_codes(panel), pd.factorize(panel["woy"])[0]]
        m = _absorb_within(panel[["lq", "lp", "lc"]].to_numpy(), codes)
        gaps.append(_gap_from_demeaned(m[:, 0], m[:, 1], m[:, 2]))
    gaps_arr = np.asarray(gaps)

    # Restrict to the monotone-increasing prefix so np.interp is well-posed.
    peak = int(np.argmax(gaps_arr))
    xs = gaps_arr[: peak + 1]
    ys = grid[: peak + 1]
    if target_gap >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(target_gap, xs, ys))


# --- Full per-category estimation --------------------------------------------


def estimate_panel(panel: pd.DataFrame, *, cvar_seed: int = 7) -> dict:
    """Estimate all calibration coordinates from a cleaned panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Tidy panel with :data:`~fploop.calibration.ingest.PANEL_COLUMNS`.
    cvar_seed : int
        Seed for the implied-λ synthetic mini-sweep (kept fixed for reproducibility).

    Returns
    -------
    dict
        ``beta_ols, beta_iv, beta_iv_se, first_stage_r2, sigma_xi, rho, n_obs,
        lambda_implied, first_stage_f, weak_iv``.
    """
    if len(panel) < 10:
        raise ValueError(f"panel too small to estimate ({len(panel)} rows)")
    units = _unit_codes(panel)
    codes = [units, pd.factorize(panel["woy"])[0]]
    m = _absorb_within(panel[["lq", "lp", "lc"]].to_numpy(), codes)
    lq_d, lp_d, lc_d = m[:, 0], m[:, 1], m[:, 2]

    beta_ols = float(lp_d @ lq_d / (lp_d @ lp_d))
    beta_iv = float(lc_d @ lq_d / (lc_d @ lp_d))
    # Partial R² of the single instrument after FE = squared correlation of the
    # demeaned price and cost; the first-stage F follows for the weak-IV flag.
    denom = float(np.sqrt((lp_d @ lp_d) * (lc_d @ lc_d)))
    first_stage_r2 = float((lp_d @ lc_d) ** 2 / (denom**2)) if denom > 0 else 0.0
    n_obs = int(len(panel))
    dof = max(n_obs - len(np.unique(units)) - 1, 1)
    first_stage_f = float(first_stage_r2 / max(1.0 - first_stage_r2, 1e-9) * dof)

    resid = lq_d - beta_iv * lp_d
    sigma_xi = float(np.std(resid, ddof=1))
    rho = _within_lag1_autocorr(resid, units, panel["week"].to_numpy())
    beta_iv_se = _clustered_iv_se(lq_d, lp_d, lc_d, units)

    sigma_c = float(np.std(lc_d, ddof=1))
    lam = implied_lambda(
        target_gap=beta_ols - beta_iv,
        beta_iv=beta_iv,
        sigma_xi=sigma_xi,
        sigma_c=sigma_c,
        first_stage_r2=first_stage_r2,
        seed=cvar_seed,
    )

    return {
        "beta_ols": beta_ols,
        "beta_iv": beta_iv,
        "beta_iv_se": beta_iv_se,
        "first_stage_r2": first_stage_r2,
        "first_stage_f": first_stage_f,
        "sigma_xi": sigma_xi,
        "rho": rho,
        "n_obs": n_obs,
        "lambda_implied": lam,
        "weak_iv": bool(first_stage_f < _WEAK_IV_F),
    }


def _within_lag1_autocorr(resid: np.ndarray, units: np.ndarray, weeks: np.ndarray) -> float:
    """Pooled lag-1 autocorrelation of ``resid`` within units, ordered by week."""
    num = den = 0.0
    for u in np.unique(units):
        mask = units == u
        order = np.argsort(weeks[mask])
        r = resid[mask][order]
        if r.size < 3:
            continue
        num += float(np.sum(r[:-1] * r[1:]))
        den += float(np.sum(r[:-1] ** 2))
    return num / den if den > 0 else 0.0


def _clustered_iv_se(
    lq_d: np.ndarray, lp_d: np.ndarray, lc_d: np.ndarray, clusters: np.ndarray
) -> float:
    """Cluster-robust SE of the IV elasticity via linearmodels (lazy import)."""
    from linearmodels.iv import IV2SLS

    data = pd.DataFrame({"lq": lq_d, "lp": lp_d, "lc": lc_d, "const": 1.0})
    res = IV2SLS(data["lq"], data[["const"]], data["lp"], data["lc"]).fit(
        cov_type="clustered", clusters=pd.Series(clusters, index=data.index)
    )
    return float(res.std_errors["lp"])

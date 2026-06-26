"""
Forecast-Price Feedback Loop
File: fploop/calibration/worlds.py

Load calibrated grocery categories into WorldConfig objects, so the sweep can be
seeded from real markets instead of hand-set numbers.

Default mode: one world per category at its calibrated lambda anchor (the "you are
here" points). Pass `lambda_grid` to sweep each category's endogeneity upward from
its real anchor -- the forward-looking trajectory into the algorithmic-pricing regime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fploop.types import WorldConfig

DEFAULT_CONFIG = "data/calibrated_worlds_config.csv"


@dataclass(frozen=True)
class CalibratedWorld:
    """A WorldConfig built from one calibrated category, plus its provenance."""

    category: str
    world: WorldConfig
    lambda_value: float  # endogeneity_strength used for THIS world
    lambda_anchor: float  # the category's calibrated lambda (where it really sits)
    first_stage_r2: float  # instrument strength -> the regime-map's other axis
    direction: str  # "loop" or "promo"
    is_anchor: bool  # True if this world sits at the calibrated anchor


def load_calibrated_worlds(
    config_path: str = DEFAULT_CONFIG,
    *,
    horizon: int = 200,
    seed: int = 0,
    lambda_grid: Iterable[float] | Callable[[float], Iterable[float]] | None = None,
) -> list[CalibratedWorld]:
    """Build WorldConfig objects from the calibrated-worlds config CSV.

    Parameters
    ----------
    config_path : CSV with columns category, elasticity, shock_std, shock_ar1,
        first_stage_r2, cost_shifter_std, lambda_anchor, direction
        (the output of scripts/build_worlds_config.py).
    horizon, seed : passed through to every WorldConfig.
    lambda_grid : controls which endogeneity values each category is instantiated at.
        * None -> one world per category at its calibrated lambda anchor
          (the "you are here" dots).
        * an iterable of floats -> a uniform grid applied to every category
          (each category swept across the same lambda axis, with its real
          elasticity / shocks / instrument held fixed) -- use for a common
          regime-map grid.
        * a callable anchor -> iterable -> a PER-CATEGORY sweep computed from each
          category's own anchor, e.g. `lambda a: np.linspace(a, 0.9, 10)` traces
          each real market from where it actually sits up into the danger zone.
        In every case the calibrated anchor is preserved in `lambda_anchor`, and any
        instantiated world sitting at the anchor is flagged via `is_anchor`.

    Returns a list of CalibratedWorld.
    """
    rows = pd.read_csv(Path(config_path))
    required = {
        "category",
        "elasticity",
        "shock_std",
        "shock_ar1",
        "first_stage_r2",
        "cost_shifter_std",
        "lambda_anchor",
        "direction",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"{config_path} is missing columns: {sorted(missing)}")

    out: list[CalibratedWorld] = []
    for _, r in rows.iterrows():
        anchor = float(r["lambda_anchor"])
        if lambda_grid is None:
            lambdas = [anchor]
        elif callable(lambda_grid):
            lambdas = [float(x) for x in lambda_grid(anchor)]
        else:
            lambdas = [float(x) for x in lambda_grid]
        for lam in lambdas:
            world = WorldConfig(
                elasticity=float(r["elasticity"]),
                shock_std=float(r["shock_std"]),
                shock_ar1=float(r["shock_ar1"]),
                endogeneity_strength=lam,
                cost_shifter_std=float(r["cost_shifter_std"]),
                horizon=horizon,
                seed=seed,
            )
            out.append(
                CalibratedWorld(
                    category=str(r["category"]),
                    world=world,
                    lambda_value=lam,
                    lambda_anchor=anchor,
                    first_stage_r2=float(r["first_stage_r2"]),
                    direction=str(r["direction"]),
                    is_anchor=(lambda_grid is None or abs(lam - anchor) < 1e-9),
                )
            )
    return out

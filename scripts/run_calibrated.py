"""
Forecast-Price Feedback Loop
File: scripts/run_calibrated.py

Run intervention arms across the calibrated grocery worlds, over one or more seeds,
and write a tidy CSV of outcomes -- plus a printed verdict on which family wins.

Examples
--------
# baseline across the anchor worlds
python scripts/run_calibrated.py

# confirmation sweep: 3 arms, lambda grid, 3 seeds, with a winner verdict
python scripts/run_calibrated.py --arms greedy,controlled_variance,twosls \
    --lambda-grid 0,0.2,0.4,0.6,0.8 --seeds 0,1,2 --out results/sweep.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from fploop import run_simulation
from fploop.arms import ARM_REGISTRY
from fploop.calibration.worlds import load_calibrated_worlds
from fploop.forecasters import GBTForecaster
from fploop.generators import LinearLogitWorld


def run_one(cw, arm_name: str, seed: int) -> dict:
    cls, family = ARM_REGISTRY[arm_name]
    result = run_simulation(LinearLogitWorld(cw.world), cls(GBTForecaster()), seed=seed)
    est = np.asarray(result.estimated_elasticity, float).ravel()
    tru = np.asarray(result.true_elasticity, float).ravel()
    valid = ~np.isnan(est)
    final_est = float(est[valid][-1]) if valid.any() else np.nan
    final_bias = abs(final_est - float(tru[-1])) if valid.any() else np.nan
    regret = np.cumsum(
        np.asarray(result.oracle_revenue, float) - np.asarray(result.realized_revenue, float)
    )
    oracle_total = float(np.asarray(result.oracle_revenue, float).sum())
    regret_pct = 100.0 * float(regret[-1]) / oracle_total if oracle_total else np.nan
    return dict(
        category=cw.category,
        family=family,
        arm=arm_name,
        seed=seed,
        lambda_value=round(cw.lambda_value, 4),
        lambda_anchor=round(cw.lambda_anchor, 4),
        first_stage_r2=round(cw.first_stage_r2, 4),
        direction=cw.direction,
        true_elasticity=round(float(tru[-1]), 4),
        final_estimated_elasticity=round(final_est, 4) if valid.any() else np.nan,
        final_abs_bias=round(final_bias, 4) if valid.any() else np.nan,
        final_regret=round(float(regret[-1]), 2),
        regret_pct_of_oracle=round(regret_pct, 2),
    )


def verdict(df: pd.DataFrame) -> None:
    """Aggregate over seeds and report who wins each (category, lambda) cell."""
    fam = dict(zip(df.arm, df.family, strict=True))
    agg = (
        df.groupby(["category", "lambda_value", "arm"])["regret_pct_of_oracle"].mean().reset_index()
    )
    pivot = agg.pivot_table(
        index=["category", "lambda_value"], columns="arm", values="regret_pct_of_oracle"
    )
    winner = pivot.idxmin(axis=1)
    win_fam = winner.map(fam)
    print("\n================ VERDICT (mean regret % over seeds) ================")
    print(pivot.round(1).to_string())
    print("\nwinning family per cell:")
    print(win_fam.value_counts().to_string())
    non_causal = win_fam[win_fam != "causal"]
    if len(non_causal) == 0:
        print("\n=> CAUSAL CORRECTION WINS EVERY CELL (no exploration/baseline win).")
    else:
        print(f"\n=> {len(non_causal)} cell(s) NOT won by causal correction:")
        for (cat, lam), f in non_causal.items():
            print(f"     {cat} lambda={lam}: winner family = {f} ({winner[(cat, lam)]})")


def main():
    ap = argparse.ArgumentParser(description="Run arms across calibrated worlds.")
    ap.add_argument("--config", default="data/calibrated_worlds_config.csv")
    ap.add_argument("--arms", default="greedy")
    ap.add_argument("--out", default="results/calibrated_outcomes.csv")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument(
        "--lambda-grid",
        default="",
        help="e.g. '0,0.2,0.4,0.6,0.8'; blank = each category at its anchor",
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = [float(x) for x in args.lambda_grid.split(",")] if args.lambda_grid.strip() else None
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}. choices: {list(ARM_REGISTRY)}")

    rows = []
    for seed in seeds:
        worlds = load_calibrated_worlds(
            args.config, horizon=args.horizon, seed=seed, lambda_grid=grid
        )
        runnable = [cw for cw in worlds if cw.world.elasticity < -1.0]
        excluded = sorted(
            {
                f"{cw.category}({cw.world.elasticity})"
                for cw in worlds
                if cw.world.elasticity >= -1.0
            }
        )
        if excluded and seed == seeds[0]:
            print(f"skipped inelastic (elasticity >= -1): {', '.join(excluded)}", flush=True)
        for cw in runnable:
            for arm in arms:
                try:
                    rows.append(run_one(cw, arm, seed))
                except Exception as exc:
                    print(f"  ! seed{seed}/{cw.category}/{arm} failed: {exc}", flush=True)
            print(f"  seed {seed}: {cw.category} done (lambda={cw.lambda_value})", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(
        f"\nwrote {args.out}: {len(df)} rows "
        f"({df.category.nunique()} cats x {df.lambda_value.nunique()} lambda "
        f"x {len(arms)} arms x {len(seeds)} seeds)",
        flush=True,
    )
    if len(arms) > 1:
        verdict(df)


if __name__ == "__main__":
    main()

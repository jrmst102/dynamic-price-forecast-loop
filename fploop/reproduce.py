"""
Forecast–Price Feedback Loop
File: fploop/reproduce.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Reproduce-the-figures path (closeout §3.1).

A single entry point that regenerates every paper figure from pinned configs and
seeds: run (or reuse) each scenario's GBT sweep, place the calibrated-markets CSV
where the overlay reads it, then export every crossover map and vulnerability atlas
on all criteria plus the per-cell risk CSVs into ``figures/``. A generated
``figures/README.md`` links each artifact to its sweep config, criterion, and the
git commit it was produced at.

Usage::

    python -m fploop.reproduce                 # full grid, 20 seeds, reuse cached sweeps
    python -m fploop.reproduce --force         # recompute sweeps even if cached
    python -m fploop.reproduce --quick         # tiny grid/seeds for a smoke run

A clean checkout plus ``pip install -e ".[nn,viz,calib]"`` plus this command
regenerates all figures.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fploop.sweep import (
    COST_STD_GRID,
    LAMBDA_GRID,
    _git_commit,
    export_sweep,
    run_sweep,
)

# Pinned reproduction set. base + the four 4a scenarios, GBT, both regret criteria
# and the bias criterion (the bias map is where DML earns its keep in reference).
# The censoring scenario also picks up the censoring-aware EM arm automatically
# (run_sweep adds it via fploop.arms.SCENARIO_ARMS).
SCENARIOS = ["base", "reference", "competition", "drift", "censoring"]
CRITERIA = ["mean", "cvar", "bias"]
FORECASTER = "gbt"
CALIBRATION_SRC = Path("data/calibrated_markets.csv")
CALIBRATION_DST = Path("sweeps/calibrated_markets.csv")


def _ensure_sweep(
    scenario: str, sweeps_root: Path, *, seeds: int, quick: bool, force: bool
) -> Path:
    """Run the scenario's GBT sweep, or reuse a cached one unless ``force``."""
    out = sweeps_root / f"{scenario}_{FORECASTER}"
    if out.joinpath("results.parquet").exists() and not force:
        print(f"  reuse cached sweep {out}")
        return out
    grid = dict(lambda_grid=[0.0, 0.6], cost_std_grid=[0.05, 0.2]) if quick else {}
    horizon = 30 if quick else 200
    print(f"  running sweep scenario={scenario} seeds={seeds} horizon={horizon} ...")
    result = run_sweep(
        scenario=scenario, forecaster=FORECASTER, seeds=range(seeds), horizon=horizon, **grid
    )
    result.save(out)
    return out


def reproduce(
    *,
    scenarios: list[str] | None = None,
    seeds: int = 20,
    out_root: str | Path = "figures",
    sweeps_root: str | Path = "sweeps",
    quick: bool = False,
    force: bool = False,
) -> dict:
    """Regenerate all figures; returns a manifest of what was written."""
    scenarios = scenarios or SCENARIOS
    sweeps_root = Path(sweeps_root)
    out_root = Path(out_root)

    # Place the calibrated markets where the overlay/atlas read them (kept in sync
    # with the canonical data/ copy here, so the two never drift).
    if CALIBRATION_SRC.exists():
        CALIBRATION_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CALIBRATION_SRC, CALIBRATION_DST)

    manifest: dict[str, list[str]] = {}
    for scenario in scenarios:
        cached = _ensure_sweep(scenario, sweeps_root, seeds=seeds, quick=quick, force=force)
        for criterion in CRITERIA:
            written = export_sweep(cached, out_root / criterion, criterion=criterion)
            manifest[f"{scenario}/{criterion}"] = written["figures"] + written["tables"]

    # Slide-ready pre/post intervention figures (same world, same seed).
    from fploop.figures.pre_post import export_pre_post

    horizon = 30 if quick else 200
    pp = export_pre_post(
        "reference",
        ["controlled_variance", "twosls", "dml"],
        seed=2,
        horizon=horizon,
        out_dir=out_root / "pre_post",
    )
    manifest["pre_post"] = pp["written"]

    _write_figures_readme(out_root, scenarios, seeds=seeds, quick=quick)
    return manifest


def _write_figures_readme(out_root: Path, scenarios: list[str], *, seeds: int, quick: bool) -> None:
    """Generate ``figures/README.md`` linking each figure to its config + commit."""
    commit = _git_commit() or "unknown"
    grid = "2x2 (quick)" if quick else f"{len(LAMBDA_GRID)}x{len(COST_STD_GRID)}"
    lines = [
        "# Reproduced figures",
        "",
        f"Produced by `python -m fploop.reproduce` at commit `{commit[:12]}`.",
        "",
        f"- Forecaster: **{FORECASTER}**; grid: **{grid}**; seeds: **{seeds}**.",
        f"- Scenarios: {', '.join(scenarios)}.",
        f"- Criteria (subfolders): {', '.join(CRITERIA)}.",
        "",
        "Each `<criterion>/` folder holds, per scenario:",
        "`crossover_<scenario>_gbt.{png,svg}` (the map), "
        "`atlas_<scenario>_gbt.{png,svg}` (the vulnerability atlas), "
        "`risk_<scenario>_gbt.csv` (per-cell mean/CVaR/worst-case), and `summary.md`.",
        "",
        "`pre_post/` holds the slide-ready before/after intervention figures "
        "(mechanism panels per intervention + the headline before/after bars).",
        "",
        "Overlay coordinates come from `data/calibrated_markets.csv` (5 Dominick's",
        "categories). On the bias map the price-varying arms win in the reference "
        "regime (DML collapses onto 2SLS there); DML's wins are in risk-aware and "
        "drift regret.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(lines) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fploop.reproduce",
        description="Regenerate all crossover maps, atlases, and risk CSVs (closeout §3.1).",
    )
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default="figures", help="output root for figures")
    p.add_argument("--sweeps", default="sweeps", help="cached-sweep root")
    p.add_argument("--scenarios", default=None, help="comma-separated subset (default: all)")
    p.add_argument("--quick", action="store_true", help="tiny grid/seeds for a smoke run")
    p.add_argument("--force", action="store_true", help="recompute sweeps even if cached")
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    scenarios = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
    manifest = reproduce(
        scenarios=scenarios,
        seeds=args.seeds,
        out_root=args.out,
        sweeps_root=args.sweeps,
        quick=args.quick,
        force=args.force,
    )
    n = sum(len(v) for v in manifest.values())
    print(f"Reproduced {n} artifacts across {len(manifest)} combos -> {args.out}/")


if __name__ == "__main__":
    main()

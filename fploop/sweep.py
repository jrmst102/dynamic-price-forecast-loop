"""
Forecast–Price Feedback Loop
File: fploop/sweep.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Offline regime sweep + winning-arm aggregation (Phase 4c §C1).

A real crossover map is thousands of 200-cycle runs and is far too heavy to compute
on demand. This module is the **offline engine**: a CLI precomputes the cross
product of regimes x arms x forecasters x seeds and caches a tidy table to disk; the
figure/export layer (:mod:`fploop.crossover`) only ever *loads* and renders it. Keep
that separation strict — the presentation layer never calls :func:`run_sweep`.

The two swept axes are the ones that govern the answer: endogeneity strength
``lambda`` (``WorldConfig.endogeneity_strength``) and instrument strength (the
cost-shifter level ``cost_std``, whose economically meaningful read is the
realized first-stage R^2 of price-on-cost). A **scenario** fixes the Phase-4a
toggles, so there is one map per scenario.

Compute note: the default ``base`` grid is 7x7 x 6 arms x 20 seeds ~= 5.9k runs
at horizon 200 — bounded for a laptop with joblib. DML (cross-fitted,
``retrain_every=10``) is the costliest arm; the FF/RNN forecasters multiply cost
and belong in separate, smaller sweeps (do not put NN forecasters in the default
grid). Determinism: every ``(lambda, cost_std, arm, seed)`` cell is independently
seeded, so the table is identical regardless of the joblib scheduling order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from fploop.arms import ARM_REGISTRY, DEFAULT_ARMS, SCENARIO_ARMS
from fploop.forecasters import HAS_TORCH
from fploop.forecasters.base import Forecaster
from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.metrics import cumulative_regret, cvar_regret, performative_gap
from fploop.types import WorldConfig

# --- Grid defaults (the two axes that govern the answer) --------------------

LAMBDA_GRID = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]  # endogeneity_strength
COST_STD_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30]  # cost_shifter_std

# The arm registry (name -> class, family), DEFAULT_ARMS, and the scenario-specific
# arm map are the canonical vocabulary, defined once in fploop.arms.
# Re-exported here so existing `from fploop.sweep import ARM_REGISTRY` callers keep
# working.

# --- Scenarios: the Phase-4a toggles each map fixes. cost_shifter_std is a
# swept axis, so it is deliberately absent here (the grid sets it). ----------

SCENARIOS: dict[str, dict] = {
    "base": {},
    "reference": {"reference_effect": 0.5},
    "competition": {"competition": True, "cross_elasticity": 0.5},
    "censoring": {"censoring": True, "capacity": 40.0},
    "drift": {"drift_kind": "gradual", "drift_magnitude": 0.5},
}

TABLE_COLUMNS = [
    "scenario",
    "lambda",
    "cost_std",
    "arm",
    "forecaster",
    "seed",
    "final_regret",
    "mean_abs_bias_2h",
    "final_perf_gap",
    "first_stage_r2",
]


def _forecaster_factory(name: str) -> Callable[[], Forecaster]:
    """Resolve a forecaster name to a fresh-instance factory.

    ``gbt`` is always available; ``ff``/``rnn`` need the optional ``nn`` extra and
    are meant for the separate small NN sweeps, never the default grid.
    """
    name = name.lower()
    if name == "gbt":
        return GBTForecaster
    if name in ("ff", "feedforward", "rnn"):
        if not HAS_TORCH:
            raise RuntimeError(f"forecaster {name!r} needs the nn extra (pip install -e '.[nn]')")
        from fploop.forecasters import FeedforwardForecaster, RNNForecaster

        return FeedforwardForecaster if name in ("ff", "feedforward") else RNNForecaster
    raise ValueError(f"unknown forecaster {name!r}")


def _make_world_config(
    base_config: WorldConfig, scenario: str, lam: float, cost_std: float, horizon: int
) -> WorldConfig:
    """Apply the scenario toggles and the two swept axes onto the base config."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    return replace(
        base_config,
        horizon=horizon,
        endogeneity_strength=float(lam),
        cost_shifter_std=float(cost_std),
        **SCENARIOS[scenario],
    )


def _first_stage_r2(log_p: NDArray, log_c: NDArray) -> float:
    """R^2 of regressing realized log effective price on log cost (instrument strength).

    The economically meaningful read of the cost-shifter axis: how much of the
    realized price variation the instrument explains. Returns ``nan`` when cost is
    constant (``cost_std == 0`` — the dead-instrument axis edge), where the
    regression is undefined.
    """
    if np.std(log_c) < 1e-12:
        return float("nan")
    Z = np.column_stack([np.ones_like(log_c), log_c])
    coef, *_ = np.linalg.lstsq(Z, log_p, rcond=None)
    p_hat = Z @ coef
    ss_res = float(np.sum((log_p - p_hat) ** 2))
    ss_tot = float(np.sum((log_p - log_p.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _run_cell(
    scenario: str,
    lam: float,
    cost_std: float,
    arm: str,
    forecaster: str,
    seed: int,
    base_config: WorldConfig,
    horizon: int,
) -> dict:
    """Run one ``(scenario, lambda, cost_std, arm, forecaster, seed)`` cell -> one row.

    Builds the scenario's world, runs the simulation, and records final cumulative
    regret, second-half mean ``|bias|``, final performative gap, and the realized
    first-stage R^2. Module-level (not a closure) so joblib can pickle it.
    """
    cfg = _make_world_config(base_config, scenario, lam, cost_std, horizon)
    world = LinearLogitWorld(cfg)
    arm_cls = ARM_REGISTRY[arm][0]
    policy = arm_cls(_forecaster_factory(forecaster)(), rng=np.random.default_rng(seed))
    res = run_simulation(world, policy, seed=seed)

    regret = cumulative_regret(np.asarray(res.realized_revenue), np.asarray(res.oracle_revenue))
    bias = np.abs(res.estimated_elasticity.ravel() - res.true_elasticity.ravel())
    half = len(bias) // 2
    # Some arms never report an elasticity (SPSA estimates the *gradient*, not the
    # slope), so the second half can be all-nan -> record nan without a warning.
    second_half = bias[half:]
    mean_abs_bias_2h = float(np.nanmean(second_half)) if np.any(~np.isnan(second_half)) else np.nan
    pgap = performative_gap(np.asarray(res.prices))
    log_p = np.log(res.prices.ravel())
    log_c = np.log(world.cost_path()[: len(log_p)])  # same-seed cost path of this run
    return {
        "scenario": scenario,
        "lambda": float(lam),
        "cost_std": float(cost_std),
        "arm": arm,
        "forecaster": forecaster,
        "seed": int(seed),
        "final_regret": float(regret[-1]),
        "mean_abs_bias_2h": mean_abs_bias_2h,
        "final_perf_gap": float(pgap[-1]),
        "first_stage_r2": _first_stage_r2(log_p, log_c),
    }


def _git_commit() -> str | None:
    """Short HEAD SHA for the manifest, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class SweepResult:
    """Tidy sweep output: one row per cell plus a reproducibility manifest.

    ``table`` has one row per ``(scenario, lambda, cost_std, arm, forecaster,
    seed)`` with columns ``final_regret, mean_abs_bias_2h, final_perf_gap,
    first_stage_r2``. ``manifest`` captures the full config, the grids, the seed
    list, the git commit, and a timestamp so a cached sweep is fully reproducible.
    """

    table: pd.DataFrame
    manifest: dict = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        """Write ``results.parquet`` (the table) and ``manifest.json`` to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(d / "results.parquet", index=False)
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> SweepResult:
        """Round-trip a sweep written by :meth:`save`."""
        d = Path(directory)
        table = pd.read_parquet(d / "results.parquet")
        manifest_path = d / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return cls(table=table, manifest=manifest)


def run_sweep(
    *,
    scenario: str = "base",
    lambda_grid: Sequence[float] = LAMBDA_GRID,
    cost_std_grid: Sequence[float] = COST_STD_GRID,
    arms: Sequence[str] = tuple(DEFAULT_ARMS),
    forecaster: str = "gbt",
    seeds: Sequence[int] = range(20),
    horizon: int = 200,
    base_config: WorldConfig | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> SweepResult:
    """Evaluate the cross product of regimes x arms x seeds for one scenario.

    For each ``(lambda, cost_std, arm, seed)``: build the scenario's
    :class:`WorldConfig`, :func:`run_simulation`, and record final cumulative
    regret, second-half mean ``|bias|``, final ``performative_gap``, and the
    realized first-stage R^2 of price-on-cost. Parallelised with joblib and
    deterministic given the seeds (each cell is independently seeded).

    Parameters
    ----------
    scenario : str
        One of :data:`SCENARIOS` — fixes the Phase-4a toggles for this map.
    lambda_grid, cost_std_grid : Sequence[float]
        The two swept axes (endogeneity strength, cost-shifter sd).
    arms : Sequence[str]
        Arm keys from :data:`ARM_REGISTRY` (default: one per family).
    forecaster : str
        ``gbt`` for the main map; ``ff``/``rnn`` for the separate small NN sweeps.
    seeds : Sequence[int]
        Paired seeds shared by every cell.
    horizon : int
        Pricing cycles per run.
    base_config : WorldConfig, optional
        Calibrated/demo defaults; scenario toggles and the swept axes override it.
    n_jobs : int
        joblib worker count (``-1`` = all cores).
    verbose : int
        joblib verbosity (the CLI sets this to draw a progress bar).

    Returns
    -------
    SweepResult
        Tidy table plus a reproducibility manifest.
    """
    base = base_config or WorldConfig()
    arm_list = list(arms)
    # Auto-include any scenario-specific arm (e.g. censoring-aware EM under
    # 'censoring'), so callers need not remember it and other scenarios stay lean.
    for a in SCENARIO_ARMS.get(scenario, []):
        if a not in arm_list:
            arm_list.append(a)
    seed_list = list(seeds)
    for arm in arm_list:
        if arm not in ARM_REGISTRY:
            raise ValueError(f"unknown arm {arm!r}; choose from {list(ARM_REGISTRY)}")

    # Deterministic cell order; joblib preserves submission order, so the table is
    # identical run-to-run regardless of how the workers are scheduled.
    cells = [
        (lam, cost_std, arm, seed)
        for lam in lambda_grid
        for cost_std in cost_std_grid
        for arm in arm_list
        for seed in seed_list
    ]
    rows = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run_cell)(scenario, lam, cost_std, arm, forecaster, seed, base, horizon)
        for (lam, cost_std, arm, seed) in cells
    )
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)

    manifest = {
        "scenario": scenario,
        "scenario_toggles": SCENARIOS[scenario],
        "lambda_grid": list(map(float, lambda_grid)),
        "cost_std_grid": list(map(float, cost_std_grid)),
        "arms": arm_list,
        "arm_families": {a: ARM_REGISTRY[a][1] for a in arm_list},
        "forecaster": forecaster,
        "seeds": seed_list,
        "horizon": horizon,
        "base_config": asdict(base),
        "n_runs": len(cells),
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return SweepResult(table=table, manifest=manifest)


# --- Aggregation: winning arm per cell (§C1.4) ------------------------------

# Columns identifying a cell (a point on the map). forecaster is included so a
# table mixing forecasters aggregates each one separately.
_CELL_KEYS = ["scenario", "forecaster", "lambda", "cost_std"]


# Which table column each criterion ranks, and what it means. All three are
# "lower is better". mean/cvar rank the final-regret distribution; bias ranks the
# second-half mean |elasticity error|. The bias map is load-bearing: where the
# instrument is orthogonal, 2SLS is already regret-optimal so DML only ties on
# regret — its win shows up in *bias* (it recovers the true elasticity 2SLS does
# not). A regret-only map would read as "DML is a do-nothing wrapper" (4b finding).
_CRITERION_COLUMN = {"mean": "final_regret", "cvar": "final_regret", "bias": "mean_abs_bias_2h"}


def _aggregate(group: pd.Series, criterion: str) -> float:
    vals = group.to_numpy()
    if criterion == "mean":
        return float(np.mean(vals))
    if criterion == "cvar":
        return cvar_regret(vals)
    if criterion == "bias":
        # Arms that never report an elasticity (e.g. SPSA estimates the *gradient*,
        # not the slope) have all-nan bias -> nan, and drop out of the bias ranking.
        return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
    raise ValueError(f"criterion must be one of {list(_CRITERION_COLUMN)}; got {criterion!r}")


def winning_arm(
    table: pd.DataFrame, *, criterion: str = "mean", tie_rel: float = 0.05
) -> pd.DataFrame:
    """Winning arm per ``(scenario, forecaster, lambda, cost_std)`` cell.

    Ranks arms by ``criterion`` and returns the best (lowest) arm. **If the best
    two arms are within ``tie_rel`` relative value** (their CIs effectively
    overlap) the cell is marked ``'tie'`` rather than painted a winner — otherwise
    the map invents sharp crossovers that aren't real (the same honesty discipline
    as the 3.5 fix).

    Parameters
    ----------
    table : pd.DataFrame
        A sweep table (or the ``SweepResult.table``).
    criterion : str
        ``'mean'`` or ``'cvar'`` of ``final_regret``, or ``'bias'`` (second-half
        mean ``|elasticity error|``). The **bias** map is what shows DML earning
        its keep where 2SLS is already regret-optimal (orthogonal instrument); the
        risk-aware (cvar) ranking can also diverge from the mean.
    tie_rel : float
        Relative threshold below which the top two arms tie.

    Returns
    -------
    pd.DataFrame
        One row per cell with columns ``[*cell keys, arm, winner_value,
        criterion]``, where ``arm`` is the winning arm key or ``'tie'``.
    """
    col = _CRITERION_COLUMN.get(criterion)
    if col is None:
        raise ValueError(f"criterion must be one of {list(_CRITERION_COLUMN)}; got {criterion!r}")
    if col not in table.columns:
        raise ValueError(f"criterion {criterion!r} needs column {col!r}, absent from the table")

    keys = [k for k in _CELL_KEYS if k in table.columns]
    out_rows = []
    for cell_vals, cell in table.groupby(keys, sort=True):
        per_arm = cell.groupby("arm")[col].apply(lambda g: _aggregate(g, criterion))
        # Arms with no defined value (e.g. non-estimating arms under 'bias') cannot
        # win — drop them before ranking rather than crown a nan.
        ranked = per_arm.dropna().sort_values()  # ascending: lower is better
        if ranked.empty:
            winner, best_val = "tie", float("nan")
        else:
            best_val = float(ranked.iloc[0])
            winner = str(ranked.index[0])
            if len(ranked) > 1:
                second_val = float(ranked.iloc[1])
                denom = abs(best_val) + 1e-9
                if (second_val - best_val) / denom <= tie_rel:
                    winner = "tie"
        vals = cell_vals if isinstance(cell_vals, tuple) else (cell_vals,)
        row = dict(zip(keys, vals, strict=True))
        row.update({"arm": winner, "winner_value": best_val, "criterion": criterion})
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=[*keys, "arm", "winner_value", "criterion"])


# --- Paper export (§C2.4) ---------------------------------------------------


def _scenario_forecaster_pairs(table: pd.DataFrame) -> list[tuple[str, str]]:
    """Distinct ``(scenario, forecaster)`` maps in a (possibly merged) table."""
    pairs = table[["scenario", "forecaster"]].drop_duplicates()
    return [(str(s), str(f)) for s, f in pairs.itertuples(index=False)]


def _winner_summary_lines(table: pd.DataFrame, scenario: str, forecaster: str) -> list[str]:
    """Prose: which arm wins how many cells under each criterion, for one map."""
    from fploop.crossover import _arm_label  # lazy: keeps the engine import-light

    lines = []
    for criterion in ("mean", "cvar", "bias"):
        win = winning_arm(table, criterion=criterion)
        counts = win["arm"].value_counts()
        total = int(counts.sum())
        parts = [f"{_arm_label(a)} ({n}/{total})" for a, n in counts.items()]
        lines.append(f"- **{criterion}**: " + ", ".join(parts))
    return lines


def _build_summary_md(result: SweepResult, calibration: pd.DataFrame | None) -> str:
    """Generate the prose ``summary.md`` stating which arm wins where (§C2.4)."""
    table = result.table
    md = ["# Crossover-map summary", ""]
    commit = result.manifest.get("git_commit")
    if commit:
        md.append(f"Generated from sweep at commit `{commit[:12]}`.")
        md.append("")
    for scenario, forecaster in _scenario_forecaster_pairs(table):
        sub = table[(table["scenario"] == scenario) & (table["forecaster"] == forecaster)]
        md.append(f"## {scenario} / {forecaster}")
        md.append("")
        md.append("Winning arm by cell count:")
        md.extend(_winner_summary_lines(sub, scenario, forecaster))
        md.append("")
    if calibration is not None and not calibration.empty:
        md.append("## Calibrated markets (you-are-here)")
        md.append("")
        for _, row in calibration.iterrows():
            md.append(
                f"- **{row['category']}**: λ≈{float(row['lambda_implied']):.2f}, "
                f"first-stage R²≈{float(row['first_stage_r2']):.2f}"
            )
        md.append("")
    else:
        md.append("_No `calibrated_markets.csv` present — partition shown without overlay._")
        md.append("")
    return "\n".join(md)


def export_sweep(
    in_dir: str | Path,
    out_dir: str | Path,
    *,
    criterion: str = "mean",
    formats: Sequence[str] = ("png", "svg"),
    calibration_path: str | Path | None = None,
) -> dict:
    """Render the paper artifacts for a cached sweep (§C2.4).

    For each ``(scenario, forecaster)`` in the cached table, writes the crossover
    map (PNG + SVG via plotly+kaleido), a per-cell risk-metric CSV, and one shared
    ``summary.md``. Static images need the optional ``viz`` extra (kaleido); when
    it is unavailable the images are skipped with a note but the CSV and markdown
    are always produced. Returns a manifest of the files written.
    """
    from fploop.crossover import (
        CALIBRATION_DEFAULT,
        cell_metrics_table,
        crossover_map_figure,
        load_calibration,
        vulnerability_atlas_figure,
    )

    result = SweepResult.load(in_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(calibration_path or CALIBRATION_DEFAULT)

    written: dict[str, list[str]] = {"figures": [], "tables": [], "summary": []}
    warned = [False]

    def _write_fig(fig, name: str) -> None:
        for fmt in formats:
            path = out / f"{name}.{fmt}"
            try:
                fig.write_image(str(path))
                written["figures"].append(str(path))
            except Exception as exc:  # noqa: BLE001 — kaleido missing is non-fatal
                if not warned[0]:
                    print(f"  (skipping static images: {exc}; pip install -e '.[viz]')")
                    warned[0] = True

    for scenario, forecaster in _scenario_forecaster_pairs(result.table):
        stem = f"{scenario}_{forecaster}"
        _write_fig(
            crossover_map_figure(
                result,
                scenario=scenario,
                forecaster=forecaster,
                criterion=criterion,
                calibration=calibration,
            ),
            f"crossover_{stem}",
        )
        # The vulnerability atlas only renders when calibrated markets are present.
        if calibration is not None and not calibration.empty:
            _write_fig(
                vulnerability_atlas_figure(
                    result,
                    calibration,
                    scenario=scenario,
                    forecaster=forecaster,
                    criterion=criterion,
                ),
                f"atlas_{stem}",
            )

        csv_path = out / f"risk_{stem}.csv"
        cell_metrics_table(result, scenario=scenario, forecaster=forecaster).to_csv(
            csv_path, index=False
        )
        written["tables"].append(str(csv_path))

    summary_path = out / "summary.md"
    summary_path.write_text(_build_summary_md(result, calibration))
    written["summary"].append(str(summary_path))
    return written


# --- CLI (§C1.2 run / §C2.4 export) -----------------------------------------


def _parse_grid(text: str | None, default: Sequence[float]) -> list[float]:
    """Parse a comma-separated float grid, falling back to ``default``."""
    if not text:
        return list(default)
    return [float(x) for x in text.split(",")]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fploop.sweep",
        description="Precompute and cache a Phase-4c regime sweep (§C1).",
    )
    p.add_argument("--scenario", default="base", choices=list(SCENARIOS))
    p.add_argument("--forecaster", default="gbt", help="gbt (default) | ff | rnn")
    p.add_argument("--seeds", type=int, default=20, help="number of seeds, range(N)")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--arms", default=None, help="comma-separated arm keys (default one per family)")
    p.add_argument("--lambda-grid", default=None, help="comma-separated endogeneity grid")
    p.add_argument("--cost-std-grid", default=None, help="comma-separated cost-shifter grid")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--out", required=True, help="output directory for the cached sweep")
    return p


def _build_export_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fploop.sweep export",
        description="Render the paper figures/tables/summary for a cached sweep (§C2.4).",
    )
    p.add_argument("--in", dest="in_dir", required=True, help="cached sweep directory")
    p.add_argument("--out", required=True, help="output directory for figures/tables")
    p.add_argument("--criterion", default="mean", choices=["mean", "cvar", "bias"])
    p.add_argument("--calibration", default=None, help="path to calibrated_markets.csv")
    return p


def _export_main(argv: Sequence[str]) -> None:
    args = _build_export_parser().parse_args(argv)
    written = export_sweep(
        args.in_dir,
        args.out,
        criterion=args.criterion,
        calibration_path=args.calibration,
    )
    n = sum(len(v) for v in written.values())
    print(f"Exported {n} files to {args.out}/ (figures + risk CSVs + summary.md)")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: ``export ...`` renders a cached sweep; otherwise run one."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        _export_main(argv[1:])
        return
    args = _build_parser().parse_args(argv)
    lambda_grid = _parse_grid(args.lambda_grid, LAMBDA_GRID)
    cost_std_grid = _parse_grid(args.cost_std_grid, COST_STD_GRID)
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(DEFAULT_ARMS)
    seeds = range(args.seeds)
    n_runs = len(lambda_grid) * len(cost_std_grid) * len(arms) * len(seeds)
    print(
        f"Sweep: scenario={args.scenario} forecaster={args.forecaster} "
        f"grid={len(lambda_grid)}x{len(cost_std_grid)} arms={len(arms)} "
        f"seeds={args.seeds} horizon={args.horizon} -> {n_runs} runs"
    )
    result = run_sweep(
        scenario=args.scenario,
        lambda_grid=lambda_grid,
        cost_std_grid=cost_std_grid,
        arms=arms,
        forecaster=args.forecaster,
        seeds=seeds,
        horizon=args.horizon,
        n_jobs=args.n_jobs,
        verbose=10,  # joblib prints a progress bar
    )
    out = result.save(args.out)
    print(f"Saved {len(result.table)} rows to {out}/ (results.parquet + manifest.json)")


if __name__ == "__main__":
    main()

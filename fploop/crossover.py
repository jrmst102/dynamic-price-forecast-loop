"""
Forecast–Price Feedback Loop
File: fploop/crossover.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Crossover-map presentation (Phase 4c §C2): plotly figures + tidy tables built
from a cached :class:`~fploop.sweep.SweepResult`.

This module is **torch-free** on purpose. The architectural rule of Phase 4c is that
the engine precomputes offline and the presentation layer only renders a cached
result. The figure logic sits here, in ``fploop``, so consumers can share it without
pulling in the sweep engine's heavy deps: the export CLI
(:func:`fploop.sweep.export_sweep`) renders the paper figures from a cached sweep.

It calls the functions below; it never re-runs a sweep. The two map axes are
endogeneity strength ``lambda`` (x) and the cost-shifter level ``cost_std`` (y),
whose economic read is the realized first-stage R^2 drawn as contour lines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fploop.metrics import cvar_regret
from fploop.sweep import ARM_REGISTRY, SweepResult, winning_arm

# Per-arm colours, themed by family so the partition reads as families while still
# distinguishing arms within one (2SLS vs DML are both "causal" purples, but the
# map must show DML taking the regions 2SLS cannot — §C2.1). Mirrors app.ARMS.
ARM_COLORS: dict[str, str] = {
    "greedy": "#808080",  # baseline — grey
    "controlled_variance": "#17a2a2",  # exploration — teal
    "spsa": "#0a7a7a",  # exploration — dark teal
    "twosls": "#7030a0",  # causal — purple
    "dml": "#c060ff",  # causal — light purple (distinct from 2SLS)
    "dro": "#e0552b",  # decision-focused — orange
    "censoring_aware": "#d6336c",  # censoring-aware EM — magenta (distinct from the purples)
}
TIE_COLOR = "#dddddd"  # neutral grey for statistically indistinguishable cells

# Short display labels for the colourbar / hover / summary prose.
ARM_LABELS: dict[str, str] = {
    "greedy": "greedy",
    "controlled_variance": "controlled-variance",
    "spsa": "SPSA",
    "twosls": "2SLS control-fn",
    "dml": "DML cross-fit",
    "dro": "DRO robust",
    "censoring_aware": "censoring-aware EM",
    "tie": "tie",
}

# Default location the export looks for the calibration overlay file.
CALIBRATION_DEFAULT = Path("sweeps/calibrated_markets.csv")
_CALIBRATION_COLUMNS = ["category", "lambda_implied", "first_stage_r2", "beta_iv", "beta_ols"]


def _arm_color(arm: str) -> str:
    return TIE_COLOR if arm == "tie" else ARM_COLORS.get(arm, "#000000")


def _arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def _resolve(result: SweepResult, scenario: str | None, forecaster: str | None) -> tuple[str, str]:
    """Fill in scenario/forecaster from the table when not given (single-valued)."""
    table = result.table
    scenario = scenario if scenario is not None else str(table["scenario"].iloc[0])
    forecaster = forecaster if forecaster is not None else str(table["forecaster"].iloc[0])
    return scenario, forecaster


def _slice(table: pd.DataFrame, scenario: str, forecaster: str) -> pd.DataFrame:
    return table[(table["scenario"] == scenario) & (table["forecaster"] == forecaster)]


def _arms_in_order(arms: set[str]) -> list[str]:
    """Arms present, in the canonical registry order (stable colours/legend)."""
    return [a for a in ARM_REGISTRY if a in arms]


def _r2_by_cost(sub: pd.DataFrame, cost_vals: list[float]) -> np.ndarray:
    """Mean realized first-stage R^2 per cost_std (averaged over lambda/arm/seed)."""
    by_cost = sub.groupby("cost_std")["first_stage_r2"].mean()
    return np.array([by_cost.get(c, np.nan) for c in cost_vals], dtype=float)


# --- The headline figure (§C2.1) --------------------------------------------


def crossover_map_figure(
    result: SweepResult,
    *,
    scenario: str | None = None,
    forecaster: str | None = None,
    criterion: str = "mean",
    tie_rel: float = 0.05,
    calibration: pd.DataFrame | None = None,
    show_contours: bool = True,
) -> go.Figure:
    """Categorical crossover map over ``(lambda, cost_std)`` for one scenario.

    Each cell is coloured by its winning arm (:func:`fploop.sweep.winning_arm`),
    with a neutral colour where arms tie. Realized first-stage R^2 is overlaid as
    contour lines so the instrument-strength axis is readable, and any calibrated
    markets are dropped on as labelled "you are here" dots.

    Parameters
    ----------
    result : SweepResult
        A cached (or in-memory) sweep.
    scenario, forecaster : str, optional
        Which map to draw; inferred from the table when it holds only one.
    criterion : str
        ``'mean'`` or ``'cvar'`` of final regret, or ``'bias'`` (mean
        ``|elasticity error|``) — re-colours the partition. The bias map is what
        shows DML taking the live-instrument region where 2SLS is already
        regret-optimal; the cvar ranking can also diverge from the mean.
    tie_rel : float
        Relative-regret threshold for the ``tie`` rule.
    calibration : pd.DataFrame, optional
        Rows of :data:`_CALIBRATION_COLUMNS`; overlaid as dots. ``None`` → no overlay.
    show_contours : bool
        Draw the first-stage-R^2 contour lines (off makes a cleaner unit test).
    """
    scenario, forecaster = _resolve(result, scenario, forecaster)
    sub = _slice(result.table, scenario, forecaster)
    if sub.empty:
        return _empty_figure(f"no rows for scenario={scenario!r} forecaster={forecaster!r}")

    win = winning_arm(sub, criterion=criterion, tie_rel=tie_rel)
    lambda_vals = sorted(sub["lambda"].unique())
    cost_vals = sorted(sub["cost_std"].unique())
    categories = _arms_in_order(set(win["arm"]) - {"tie"})
    if (win["arm"] == "tie").any():
        categories = [*categories, "tie"]
    code = {c: i for i, c in enumerate(categories)}

    # z-code matrix [row = cost_std, col = lambda]; nan where a cell is absent.
    win_by_cell = {(r["lambda"], r["cost_std"]): r["arm"] for _, r in win.iterrows()}
    z = np.full((len(cost_vals), len(lambda_vals)), np.nan)
    text = [["" for _ in lambda_vals] for _ in cost_vals]
    for i, c in enumerate(cost_vals):
        for j, lam in enumerate(lambda_vals):
            arm = win_by_cell.get((lam, c))
            if arm is not None:
                z[i, j] = code[arm]
                text[i][j] = _arm_label(arm)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=lambda_vals,
            y=cost_vals,
            colorscale=_discrete_colorscale(categories),
            zmin=-0.5,
            zmax=len(categories) - 0.5,
            colorbar=dict(
                title="winning arm",
                tickmode="array",
                tickvals=list(range(len(categories))),
                ticktext=[_arm_label(c) for c in categories],
            ),
            text=text,
            hovertemplate="λ=%{x}<br>cost_std=%{y}<br>winner=%{text}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )

    if show_contours:
        r2 = _r2_matrix(sub, lambda_vals, cost_vals)
        if np.isfinite(r2).any():
            fig.add_trace(
                go.Contour(
                    z=r2,
                    x=lambda_vals,
                    y=cost_vals,
                    contours=dict(coloring="lines", showlabels=True),
                    line=dict(color="rgba(0,0,0,0.45)", width=1),
                    colorscale="Greys",
                    showscale=False,
                    hoverinfo="skip",
                    name="first-stage R²",
                )
            )

    if calibration is not None and not calibration.empty:
        fig.add_trace(_overlay_trace(calibration, sub, cost_vals))

    crit_label = {"mean": "mean regret", "cvar": "CVaR regret", "bias": "mean |bias|"}.get(
        criterion, criterion
    )
    fig.update_layout(
        title=f"Crossover map — {scenario} / {forecaster} ({crit_label})",
        xaxis_title="endogeneity strength λ",
        yaxis_title="cost-shifter std (→ first-stage R²)",
        margin=dict(l=60, r=20, t=50, b=50),
        plot_bgcolor="white",
    )
    return fig


def _discrete_colorscale(categories: list[str]) -> list[list]:
    """A stepped plotly colorscale: one flat band per category, registry colours."""
    n = max(len(categories), 1)
    scale: list[list] = []
    for i, cat in enumerate(categories):
        col = _arm_color(cat)
        scale.append([i / n, col])
        scale.append([(i + 1) / n, col])
    return scale


def _r2_matrix(sub: pd.DataFrame, lambda_vals: list[float], cost_vals: list[float]) -> np.ndarray:
    """Mean first-stage R^2 per cell [row = cost_std, col = lambda] for the contours."""
    grid = sub.groupby(["cost_std", "lambda"])["first_stage_r2"].mean()
    r2 = np.full((len(cost_vals), len(lambda_vals)), np.nan)
    for i, c in enumerate(cost_vals):
        for j, lam in enumerate(lambda_vals):
            r2[i, j] = grid.get((c, lam), np.nan)
    return r2


def _overlay_trace(
    calibration: pd.DataFrame, sub: pd.DataFrame, cost_vals: list[float]
) -> go.Scatter:
    """Place each calibrated market at (lambda_implied, cost_std matching its R^2).

    The y-coordinate is read off the realized-R^2 axis: invert the mean
    R^2-vs-cost_std curve so the dot lands on the contour matching the market's
    measured first-stage R^2 (§C2.2).
    """
    r2_curve = _r2_by_cost(sub, cost_vals)
    finite = np.isfinite(r2_curve)
    xp = np.asarray(r2_curve)[finite]
    fp = np.asarray(cost_vals)[finite]
    order = np.argsort(xp)  # np.interp needs ascending xp
    xp, fp = xp[order], fp[order]

    xs, ys, labels = [], [], []
    for _, row in calibration.iterrows():
        target = float(row["first_stage_r2"])
        y = float(np.interp(target, xp, fp)) if xp.size else float(np.nanmin(cost_vals))
        xs.append(float(row["lambda_implied"]))
        ys.append(y)
        labels.append(str(row["category"]))
    return go.Scatter(
        x=xs,
        y=ys,
        text=labels,
        mode="markers+text",
        textposition="top center",
        marker=dict(color="black", size=10, symbol="circle", line=dict(color="white", width=1)),
        name="calibrated markets",
        hovertemplate="%{text}<br>λ=%{x}<br>cost_std=%{y:.3f}<extra></extra>",
    )


# --- The distributional view (§C2.3) ----------------------------------------


def regret_distribution_figure(
    result: SweepResult,
    *,
    lam: float,
    cost_std: float,
    scenario: str | None = None,
    forecaster: str | None = None,
) -> go.Figure:
    """Per-seed final-regret box plot, one box per arm, for the selected cell.

    This is the point of having risk metrics: the spread and the upper tail are
    visible here in a way the mean-coloured map cannot show.
    """
    scenario, forecaster = _resolve(result, scenario, forecaster)
    sub = _slice(result.table, scenario, forecaster)
    cell = sub[np.isclose(sub["lambda"], lam) & np.isclose(sub["cost_std"], cost_std)]
    if cell.empty:
        return _empty_figure(f"no data at λ={lam}, cost_std={cost_std}")

    fig = go.Figure()
    for arm in _arms_in_order(set(cell["arm"])):
        vals = cell[cell["arm"] == arm]["final_regret"].to_numpy()
        fig.add_trace(
            go.Box(
                y=vals,
                name=_arm_label(arm),
                marker_color=_arm_color(arm),
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
            )
        )
    fig.update_layout(
        title=f"Per-seed final regret — λ={lam}, cost_std={cost_std}",
        yaxis_title="final cumulative regret",
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def cell_metrics_table(
    result: SweepResult,
    *,
    scenario: str | None = None,
    forecaster: str | None = None,
    lam: float | None = None,
    cost_std: float | None = None,
    cvar_alpha: float = 0.9,
) -> pd.DataFrame:
    """Per-arm risk metrics (mean, CVaR, worst-case) for each cell, or one cell.

    With ``lam``/``cost_std`` given, returns the rows for just that cell; without
    them, every cell (the export CSV).
    ``worst_case`` is the worst single seed in the cell — the tail the mean hides.
    """
    scenario, forecaster = _resolve(result, scenario, forecaster)
    sub = _slice(result.table, scenario, forecaster)
    if lam is not None and cost_std is not None:
        sub = sub[np.isclose(sub["lambda"], lam) & np.isclose(sub["cost_std"], cost_std)]

    rows = []
    for (lv, cv, arm), g in sub.groupby(["lambda", "cost_std", "arm"], sort=True):
        regrets = g["final_regret"].to_numpy()
        rows.append(
            {
                "scenario": scenario,
                "forecaster": forecaster,
                "lambda": float(lv),
                "cost_std": float(cv),
                "arm": arm,
                "mean": float(np.mean(regrets)),
                "cvar": cvar_regret(regrets, alpha=cvar_alpha),
                "worst_case": float(np.max(regrets)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "forecaster",
            "lambda",
            "cost_std",
            "arm",
            "mean",
            "cvar",
            "worst_case",
        ],
    )


# --- Vulnerability atlas (forward-looking-lambda view) ----------------------


def _cost_for_r2(target: float, r2_curve: np.ndarray, cost_vals: list[float]) -> float:
    """The swept ``cost_std`` whose realized first-stage R^2 is closest to ``target``."""
    arr = np.asarray(r2_curve, dtype=float)
    if not np.isfinite(arr).any():
        return cost_vals[0]
    return cost_vals[int(np.nanargmin(np.abs(arr - target)))]


def _crossover_lambda(winners: list, lambda_vals: list[float]) -> float | None:
    """First lambda at which the winning arm leaves its lambda=0 arm (or None)."""
    base = next((w for w in winners if w), None)
    for lam, w in zip(lambda_vals, winners, strict=True):
        if w and w != base:
            return lam
    return None


def vulnerability_atlas_figure(
    result: SweepResult,
    calibration: pd.DataFrame | None,
    *,
    scenario: str | None = None,
    forecaster: str | None = None,
    criterion: str = "mean",
    tie_rel: float = 0.05,
) -> go.Figure:
    """Per-category horizontal slice through the map: the winning arm as lambda rises.

    Operationalizes the forward-looking-lambda framing the calibration forced. Each
    calibrated category is placed on its first-stage-R^2 row of the map; the slice
    then reads "if a loop pushed this market's lambda up, which intervention would it
    need?". Strong-instrument categories (e.g. soup, oatmeal) stay causal-rescuable
    far up the axis; weak-instrument ones (beer, juices) flip to exploration/DML at
    low lambda. Each row is marked with its crossover lambda (where the winner first
    leaves its lambda=0 arm). Returns a placeholder figure if the sweep or the
    calibration CSV is absent.
    """
    scenario, forecaster = _resolve(result, scenario, forecaster)
    sub = _slice(result.table, scenario, forecaster)
    if sub.empty or calibration is None or calibration.empty:
        return _empty_figure("atlas needs a cached sweep and calibrated_markets.csv")

    win = winning_arm(sub, criterion=criterion, tie_rel=tie_rel)
    win_by_cell = {(r["lambda"], r["cost_std"]): r["arm"] for _, r in win.iterrows()}
    lambda_vals = sorted(sub["lambda"].unique())
    cost_vals = sorted(sub["cost_std"].unique())
    r2_curve = _r2_by_cost(sub, cost_vals)

    cats: list[str] = []
    winners_grid: list[list] = []
    crossovers: list[float | None] = []
    for _, row in calibration.sort_values("first_stage_r2").iterrows():
        cstd = _cost_for_r2(float(row["first_stage_r2"]), r2_curve, cost_vals)
        winners = [win_by_cell.get((lam, cstd)) for lam in lambda_vals]
        cats.append(str(row["category"]))
        winners_grid.append(winners)
        crossovers.append(_crossover_lambda(winners, lambda_vals))

    present = _arms_in_order({w for wr in winners_grid for w in wr if w and w != "tie"})
    if any(w == "tie" for wr in winners_grid for w in wr):
        present = [*present, "tie"]
    code = {c: i for i, c in enumerate(present)}
    z = [[code[w] if w in code else np.nan for w in wr] for wr in winners_grid]
    text = [[_arm_label(w) if w else "" for w in wr] for wr in winners_grid]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=lambda_vals,
            y=cats,
            colorscale=_discrete_colorscale(present),
            zmin=-0.5,
            zmax=len(present) - 0.5,
            colorbar=dict(
                title="winning arm",
                tickmode="array",
                tickvals=list(range(len(present))),
                ticktext=[_arm_label(c) for c in present],
            ),
            text=text,
            hovertemplate="%{y}<br>λ=%{x}<br>winner=%{text}<extra></extra>",
            xgap=1,
            ygap=2,
        )
    )
    xs = [c for c in crossovers if c is not None]
    ys = [cat for cat, c in zip(cats, crossovers, strict=True) if c is not None]
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(symbol="line-ns-open", color="black", size=16, line=dict(width=2)),
                name="crossover λ",
                hovertemplate="%{y} crosses at λ=%{x}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Vulnerability atlas — {scenario} / {forecaster} ({criterion})",
        xaxis_title="endogeneity strength λ  (if a loop pushed it up →)",
        yaxis_title="category  (↑ stronger instrument)",
        margin=dict(l=110, r=20, t=50, b=50),
        plot_bgcolor="white",
    )
    return fig


# --- Calibration overlay loading (§C2.2) ------------------------------------


def load_calibration(path: str | Path = CALIBRATION_DEFAULT) -> pd.DataFrame | None:
    """Load ``calibrated_markets.csv`` if present, else ``None`` (never hard-fail).

    The overlay is optional and produced by a separate calibration workstream, so
    a missing file is the normal case — the map then renders the partition alone.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing = [c for c in _CALIBRATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing calibration columns: {missing}")
    return df


# --- Shared empty/placeholder figure ----------------------------------------


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#888888"))
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
    )
    return fig

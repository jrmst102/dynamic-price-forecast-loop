"""
Forecast–Price Feedback Loop
File: fploop/figures/pre_post.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Presentation-quality *before/after* intervention figures for the ISF talk. For a
fixed ``WorldConfig`` and seed, the naive (greedy / Arm 0) policy and each chosen
intervention are run on the **same world and seed** — only the policy differs, so
the gap between their traces *is* the intervention's effect. Reuses
:func:`run_simulation` and the oracle; computes no new science.

Figure 1 (per intervention) stacks two panels on a shared cycle axis: the
elasticity readout vs the true elasticity (the bias), and cumulative regret (the
cost). Figure 2 is the headline before/after bars across interventions: final
``|elasticity bias|`` and total regret. Static export via the ``viz`` extra's
kaleido; missing kaleido is non-fatal (the figure objects still build).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fploop
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.metrics import cumulative_regret
from fploop.sweep import ARM_REGISTRY, _forecaster_factory, _git_commit, _make_world_config
from fploop.types import RunResult, WorldConfig

# Slide palette: a muted "problem" colour for greedy, a vivid "fix" colour for the
# intervention, kept consistent across both panels and the bars.
GREEDY_COLOR = "#9aa0a6"
FIX_COLOR = "#1f77b4"
TRUTH_COLOR = "#444444"

# Friendly names; CLI also accepts "2sls" as an alias of the "twosls" registry key.
_ALIASES = {"2sls": "twosls"}
ARM_LABELS = {
    "greedy": "Naive (greedy)",
    "controlled_variance": "Controlled-variance",
    "spsa": "SPSA",
    "twosls": "2SLS",
    "dml": "DML",
    "dro": "DRO",
}
REGIME_LABELS = {
    "base": "Base",
    "reference": "Reference",
    "competition": "Competition",
    "drift": "Drift",
    "censoring": "Censoring",
}

# Defaults chosen so the contrast appears: a high-endogeneity, live-instrument
# regime where the causal and exploration arms have something to fix.
DEFAULT_SEED = 2
DEFAULT_LAMBDA = 0.6
DEFAULT_COST_STD = 0.3
DEFAULT_HORIZON = 200


def _resolve_arm(name: str) -> str:
    """Map a CLI intervention name to a registry key (rejecting the baseline)."""
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if key == "greedy":
        raise ValueError("'greedy' is the baseline, not an intervention")
    if key not in ARM_REGISTRY:
        raise ValueError(f"unknown intervention {name!r}; choose from {sorted(ARM_REGISTRY)}")
    return key


def _run(scenario: str, arm: str, *, lam: float, cost_std: float, seed: int, horizon: int):
    """Run one policy on the scenario's world at a fixed seed (same world for all)."""
    cfg = _make_world_config(WorldConfig(), scenario, lam, cost_std, horizon)
    world = LinearLogitWorld(cfg)
    policy = ARM_REGISTRY[arm][0](_forecaster_factory("gbt")(), rng=np.random.default_rng(seed))
    return run_simulation(world, policy, seed=seed)


def _cum_regret(res: RunResult) -> np.ndarray:
    return cumulative_regret(np.asarray(res.realized_revenue), np.asarray(res.oracle_revenue))


def _final_abs_bias(res: RunResult) -> float:
    """The settled ``|estimated - true|`` elasticity gap (mean over the last decile).

    A tail mean rather than the single last cycle so the headline bar is stable;
    falls back to the last finite value if the tail is all-nan (non-estimating arm).
    """
    bias = np.abs(res.estimated_elasticity.ravel() - res.true_elasticity.ravel())
    tail = bias[max(0, len(bias) - max(1, len(bias) // 10)) :]
    finite = tail[np.isfinite(tail)]
    if finite.size:
        return float(finite.mean())
    allf = bias[np.isfinite(bias)]
    return float(allf[-1]) if allf.size else float("nan")


def _total_regret(res: RunResult) -> float:
    return float(_cum_regret(res)[-1])


def _slide_layout(fig: go.Figure, title: str, caption: str) -> None:
    """Apply the shared slide theme: large fonts, minimal gridlines, a caption."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=22), x=0.5, xanchor="center"),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=14)),
        margin=dict(l=80, r=40, t=90, b=90),
        width=900,
        height=720,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=-0.13,
        showarrow=False,
        align="left",
        text=caption,
        font=dict(size=13, color="#333"),
    )


def mechanism_figure(
    scenario: str, intervention: str, greedy_res: RunResult, interv_res: RunResult, *, lam: float
) -> go.Figure:
    """Two stacked panels: elasticity-vs-truth (bias) and cumulative regret (cost)."""
    arm = _resolve_arm(intervention)
    label = ARM_LABELS.get(arm, arm)
    regime = REGIME_LABELS.get(scenario, scenario.title())

    g_est = greedy_res.estimated_elasticity.ravel()
    i_est = interv_res.estimated_elasticity.ravel()
    true = float(greedy_res.true_elasticity.ravel()[0])
    cyc = np.arange(len(g_est))
    g_reg, i_reg = _cum_regret(greedy_res), _cum_regret(interv_res)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Elasticity readout vs truth", "Cumulative regret"),
    )
    fig.add_hline(
        y=true,
        line=dict(color=TRUTH_COLOR, dash="dash", width=1.5),
        annotation_text="true elasticity",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=cyc, y=g_est, name="Naive (greedy)", line=dict(color=GREEDY_COLOR, width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=cyc, y=i_est, name=label, line=dict(color=FIX_COLOR, width=2.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=cyc, y=g_reg, line=dict(color=GREEDY_COLOR, width=2), showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=cyc, y=i_reg, line=dict(color=FIX_COLOR, width=2.5), showlegend=False),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="estimated elasticity", title_font=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="cumulative regret", title_font=dict(size=14), row=2, col=1)
    fig.update_xaxes(title_text="cycle", title_font=dict(size=14), row=2, col=1)

    gb, ib = _final_abs_bias(greedy_res), _final_abs_bias(interv_res)
    gr, ir = _total_regret(greedy_res), _total_regret(interv_res)
    caption = (
        f"Greedy settles at |bias|≈{gb:.2f} with regret≈{gr:,.0f}; "
        f"{label} holds |bias|≈{ib:.2f} and regret≈{ir:,.0f} — same world, same seed."
    )
    _slide_layout(fig, f"{regime} (λ={lam:g}) — {label}", caption)
    return fig


def before_after_figure(
    scenario: str,
    interventions: list[str],
    greedy_res: RunResult,
    interv_results: list[RunResult],
    *,
    lam: float,
) -> go.Figure:
    """Headline bars: final ``|elasticity bias|`` and total regret, naive vs each fix."""
    regime = REGIME_LABELS.get(scenario, scenario.title())
    labels = ["No intervention"] + [ARM_LABELS.get(_resolve_arm(a), a) for a in interventions]
    bias_vals = [_final_abs_bias(greedy_res)] + [_final_abs_bias(r) for r in interv_results]
    reg_vals = [_total_regret(greedy_res)] + [_total_regret(r) for r in interv_results]
    colors = [GREEDY_COLOR] + [FIX_COLOR] * len(interventions)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Final |elasticity bias|", "Total regret"))
    fig.add_trace(
        go.Bar(x=labels, y=bias_vals, marker_color=colors, showlegend=False), row=1, col=1
    )
    fig.add_trace(go.Bar(x=labels, y=reg_vals, marker_color=colors, showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="|elasticity bias|", title_font=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="total regret", title_font=dict(size=14), row=1, col=2)
    caption = "The tall bar is the naive loop; each intervention sits shorter beside it."
    _slide_layout(fig, f"{regime} (λ={lam:g}) — before / after", caption)
    fig.update_layout(height=520, width=1000)
    return fig


def _series_frame(
    scenario: str,
    arm_key: str,
    res: RunResult,
    *,
    lam: float,
    cost_std: float,
    seed: int,
    horizon: int,
) -> pd.DataFrame:
    """Tidy per-cycle trace for one policy run (one row per pricing cycle).

    Single-product: the (T, 1) RunResult arrays are flattened to length-T columns.
    """
    est = res.estimated_elasticity.ravel()
    true = res.true_elasticity.ravel()
    return pd.DataFrame(
        {
            "scenario": scenario,
            "arm": arm_key,
            "arm_label": ARM_LABELS.get(arm_key, arm_key),
            "lambda": lam,
            "cost_std": cost_std,
            "seed": seed,
            "horizon": horizon,
            "cycle": np.arange(len(est)),
            "price": res.prices.ravel(),
            "observed_demand": res.observed_demand.ravel(),
            "estimated_elasticity": est,
            "true_elasticity": true,
            "residual_bias": est - true,
            "realized_profit": np.asarray(res.realized_revenue).ravel(),
            "oracle_price": res.oracle_prices.ravel(),
            "oracle_profit": np.asarray(res.oracle_revenue).ravel(),
            "cumulative_regret": _cum_regret(res),
        }
    )


def _summary_frame(
    scenario: str,
    arms: list[str],
    greedy_res: RunResult,
    interv_results: list[RunResult],
    *,
    lam: float,
    cost_std: float,
    seed: int,
    horizon: int,
) -> pd.DataFrame:
    """Headline before/after table: settled ``|bias|`` and total regret per arm."""
    keys = ["greedy", *arms]
    runs = [greedy_res, *interv_results]
    return pd.DataFrame(
        {
            "scenario": scenario,
            "arm": keys,
            "arm_label": [ARM_LABELS.get(k, k) for k in keys],
            "lambda": lam,
            "cost_std": cost_std,
            "seed": seed,
            "horizon": horizon,
            "final_abs_bias": [_final_abs_bias(r) for r in runs],
            "total_regret": [_total_regret(r) for r in runs],
        }
    )


def _write_data_artifacts(
    out: Path,
    scenario: str,
    arms: list[str],
    greedy_res: RunResult,
    interv_results: list[RunResult],
    *,
    lam: float,
    cost_std: float,
    seed: int,
    horizon: int,
) -> list[str]:
    """Persist the tidy CSVs and a provenance manifest behind the figures.

    Always runs (independent of kaleido), so the data is reproducible even when
    static image export is unavailable. Returns the file paths written.
    """
    frames = [
        _series_frame(scenario, k, r, lam=lam, cost_std=cost_std, seed=seed, horizon=horizon)
        for k, r in [("greedy", greedy_res), *zip(arms, interv_results, strict=True)]
    ]
    series = pd.concat(frames, ignore_index=True)
    summary = _summary_frame(
        scenario,
        arms,
        greedy_res,
        interv_results,
        lam=lam,
        cost_std=cost_std,
        seed=seed,
        horizon=horizon,
    )

    series_path = out / f"series_{scenario}.csv"
    summary_path = out / f"summary_{scenario}.csv"
    manifest_path = out / f"manifest_{scenario}.json"
    series.to_csv(series_path, index=False)
    summary.to_csv(summary_path, index=False)

    cfg = _make_world_config(WorldConfig(), scenario, lam, cost_std, horizon)
    manifest = {
        "scenario": scenario,
        "interventions": list(arms),
        "forecaster": "gbt",
        "seed": seed,
        "lambda": lam,
        "cost_std": cost_std,
        "horizon": horizon,
        "world_config": asdict(cfg),
        "fploop_version": fploop.__version__,
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {"series": series_path.name, "summary": summary_path.name},
        "column_dictionary": {
            "cycle": "pricing period index, 0..horizon-1",
            "estimated_elasticity": "forecaster's believed elasticity (NaN in warm-up)",
            "true_elasticity": "ground-truth elasticity built into the world",
            "residual_bias": "estimated_elasticity - true_elasticity",
            "cumulative_regret": "cumsum(oracle_profit - realized_profit)",
            "price": "price set by the policy",
            "oracle_price": "full-information oracle price",
            "realized_profit": "per-cycle profit under the policy",
            "oracle_profit": "per-cycle profit under the oracle",
            "observed_demand": "units sold (possibly censored)",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return [str(series_path), str(summary_path), str(manifest_path)]


def export_pre_post(
    scenario: str,
    interventions: list[str],
    *,
    seed: int = DEFAULT_SEED,
    lam: float = DEFAULT_LAMBDA,
    cost_std: float = DEFAULT_COST_STD,
    horizon: int = DEFAULT_HORIZON,
    out_dir: str | Path = "figures/pre_post",
    formats: tuple[str, ...] = ("png", "svg"),
) -> dict:
    """Run baseline + interventions on one world/seed and export every figure.

    Returns a manifest ``{"written": [...paths], "greedy": RunResult,
    "interventions": {arm: RunResult}}`` so callers (and tests) can inspect the
    runs without re-executing them.
    """
    arms = [_resolve_arm(a) for a in interventions]
    greedy_res = _run(scenario, "greedy", lam=lam, cost_std=cost_std, seed=seed, horizon=horizon)
    interv_results = [
        _run(scenario, a, lam=lam, cost_std=cost_std, seed=seed, horizon=horizon) for a in arms
    ]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    written += _write_data_artifacts(
        out,
        scenario,
        arms,
        greedy_res,
        interv_results,
        lam=lam,
        cost_std=cost_std,
        seed=seed,
        horizon=horizon,
    )
    warned = [False]

    def _write(fig: go.Figure, name: str) -> None:
        for fmt in formats:
            path = out / f"{name}.{fmt}"
            try:
                fig.write_image(str(path))
                written.append(str(path))
            except Exception as exc:  # noqa: BLE001 — kaleido missing is non-fatal
                if not warned[0]:
                    print(f"  (skipping static images: {exc}; pip install -e '.[viz]')")
                    warned[0] = True

    for arm, res in zip(arms, interv_results, strict=True):
        _write(
            mechanism_figure(scenario, arm, greedy_res, res, lam=lam), f"mechanism_{scenario}_{arm}"
        )
    _write(
        before_after_figure(scenario, arms, greedy_res, interv_results, lam=lam),
        f"before_after_{scenario}",
    )

    return {
        "written": written,
        "greedy": greedy_res,
        "interventions": dict(zip(arms, interv_results, strict=True)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fploop.figures.pre_post",
        description="Slide-ready pre/post intervention figures (same world, same seed).",
    )
    p.add_argument("--scenario", default="reference", help="world scenario (default: reference)")
    p.add_argument(
        "--interventions",
        default="controlled_variance,2sls,dml",
        help="comma-separated intervention arms (default: controlled_variance,2sls,dml)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--cost-std", type=float, default=DEFAULT_COST_STD)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--out", default="figures/pre_post", help="output directory")
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    interventions = [s for s in args.interventions.split(",") if s.strip()]
    manifest = export_pre_post(
        args.scenario,
        interventions,
        seed=args.seed,
        lam=args.lam,
        cost_std=args.cost_std,
        horizon=args.horizon,
        out_dir=args.out,
    )
    print(f"Wrote {len(manifest['written'])} files to {args.out}/")


if __name__ == "__main__":
    main()

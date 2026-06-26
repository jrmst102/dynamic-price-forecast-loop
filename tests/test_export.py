"""
Forecast–Price Feedback Loop
File: tests/test_export.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4c §C2.4 — the paper export.

Export must write, for each cached scenario, the crossover-map figure (PNG/SVG
via kaleido), a per-cell risk-metric CSV, and a generated ``summary.md``. Image
bytes are skipped when kaleido is unavailable, but the CSV and markdown are
always produced.
"""

from __future__ import annotations

import importlib.util

import pandas as pd

from fploop.sweep import export_sweep
from tests._sweep_synth import synthetic_sweep

_HAVE_KALEIDO = importlib.util.find_spec("kaleido") is not None


def _cached_sweep(in_dir):
    cells = {}
    for lam in (0.0, 0.9):
        for cost in (0.05, 0.20):
            cells[(lam, cost, "greedy")] = [3.0, 3.1, 2.9]
            cells[(lam, cost, "twosls")] = [1.0, 1.1, 0.9]
    synthetic_sweep(cells).save(in_dir)


def test_export_writes_csv_and_summary(tmp_path):
    in_dir, out_dir = tmp_path / "sweep", tmp_path / "figures"
    _cached_sweep(in_dir)
    written = export_sweep(in_dir, out_dir)

    # CSV + markdown are mandatory regardless of kaleido.
    csv = out_dir / "risk_base_gbt.csv"
    summary = out_dir / "summary.md"
    assert csv.exists()
    assert summary.exists()
    assert str(csv) in written["tables"]
    assert str(summary) in written["summary"]


def test_risk_csv_has_per_arm_metrics(tmp_path):
    in_dir, out_dir = tmp_path / "sweep", tmp_path / "figures"
    _cached_sweep(in_dir)
    export_sweep(in_dir, out_dir)

    df = pd.read_csv(out_dir / "risk_base_gbt.csv")
    assert {"arm", "mean", "cvar", "worst_case"} <= set(df.columns)
    assert set(df["arm"]) == {"greedy", "twosls"}
    # CVaR is the upper-tail mean, never below the mean.
    assert (df["cvar"] >= df["mean"] - 1e-9).all()


def test_summary_names_the_winning_arm(tmp_path):
    in_dir, out_dir = tmp_path / "sweep", tmp_path / "figures"
    _cached_sweep(in_dir)
    export_sweep(in_dir, out_dir)

    text = (out_dir / "summary.md").read_text()
    assert "base / gbt" in text
    # 2SLS dominates this fixture (lower regret in every cell).
    assert "2SLS control-fn" in text


def test_images_track_kaleido_availability(tmp_path):
    in_dir, out_dir = tmp_path / "sweep", tmp_path / "figures"
    _cached_sweep(in_dir)
    export_sweep(in_dir, out_dir)

    png = out_dir / "crossover_base_gbt.png"
    svg = out_dir / "crossover_base_gbt.svg"
    if _HAVE_KALEIDO:
        assert png.exists() and svg.exists()
    else:
        assert not png.exists() and not svg.exists()

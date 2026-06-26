"""
Forecast–Price Feedback Loop
File: tests/test_sweep.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4c §C1.1-C1.2 — the offline sweep engine.

A tiny sweep must produce a right-shaped, finite, reproducible table that
round-trips through disk. The full grid is an offline CLI job, not a test.
"""

from __future__ import annotations

import pandas as pd

from fploop.sweep import TABLE_COLUMNS, SweepResult, run_sweep

# A grid small enough for CI; both cost-std values are > 0 so the realized
# first-stage R^2 is defined on every row (it is nan only on the cost_std=0 edge).
TINY = dict(
    scenario="base",
    lambda_grid=[0.0, 0.6],
    cost_std_grid=[0.05, 0.15],
    arms=["greedy", "twosls"],
    seeds=range(3),
    horizon=30,
    n_jobs=1,
)


def test_table_shape_and_finite():
    result = run_sweep(**TINY)
    # 2 lambda x 2 cost_std x 2 arms x 3 seeds = 24 rows.
    assert result.table.shape == (24, len(TABLE_COLUMNS))
    assert list(result.table.columns) == TABLE_COLUMNS
    metric_cols = ["final_regret", "mean_abs_bias_2h", "final_perf_gap", "first_stage_r2"]
    assert result.table[metric_cols].notna().all().all()
    assert pd.api.types.is_numeric_dtype(result.table["final_regret"])


def test_manifest_records_reproducibility_fields():
    manifest = run_sweep(**TINY).manifest
    assert manifest["scenario"] == "base"
    assert manifest["lambda_grid"] == [0.0, 0.6]
    assert manifest["seeds"] == [0, 1, 2]
    assert manifest["n_runs"] == 24
    assert "git_commit" in manifest and "created_utc" in manifest
    assert "base_config" in manifest


def test_reproducible_bit_for_bit():
    a = run_sweep(**TINY).table
    b = run_sweep(**TINY).table
    pd.testing.assert_frame_equal(a, b)


def test_save_load_round_trips(tmp_path):
    result = run_sweep(**TINY)
    result.save(tmp_path)
    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    loaded = SweepResult.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.table, result.table)
    assert loaded.manifest["n_runs"] == result.manifest["n_runs"]

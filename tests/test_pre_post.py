"""
Forecast–Price Feedback Loop
File: tests/test_pre_post.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Pre/post intervention figure tests. The figure objects must build for a small
config (image bytes are skipped when kaleido is absent), and — in a regime where
the intervention helps — the naive baseline's final ``|elasticity bias|`` *and*
total regret must exceed the intervention's. Uses the same-world-same-seed runner.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from fploop.figures.pre_post import (
    _final_abs_bias,
    _total_regret,
    before_after_figure,
    export_pre_post,
    mechanism_figure,
)

SCENARIO = "reference"
RUN_KW = dict(seed=2, lam=0.6, cost_std=0.3, horizon=120)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("pre_post")
    return export_pre_post(SCENARIO, ["controlled_variance"], out_dir=out, **RUN_KW)


def test_figures_build(manifest):
    g = manifest["greedy"]
    cv = manifest["interventions"]["controlled_variance"]
    mech = mechanism_figure(SCENARIO, "controlled_variance", g, cv, lam=RUN_KW["lam"])
    assert isinstance(mech, go.Figure)
    assert len(mech.data) == 4  # greedy + intervention, two panels
    bars = before_after_figure(SCENARIO, ["controlled_variance"], g, [cv], lam=RUN_KW["lam"])
    assert isinstance(bars, go.Figure)
    assert len(bars.data) == 2  # bias bars + regret bars


def test_image_files_written_when_kaleido_present(manifest):
    written = manifest["written"]
    images = [p for p in written if p.endswith((".png", ".svg"))]
    if not images:  # kaleido absent -> static image export skipped (non-fatal)
        pytest.skip("kaleido unavailable; figure objects still build (see test_figures_build)")
    for p in images:
        assert Path(p).exists()
    assert any("mechanism_reference_controlled_variance" in p for p in images)
    assert any("before_after_reference" in p for p in images)


def test_data_artifacts_always_written(manifest):
    """Tidy CSVs + manifest must persist regardless of kaleido, so every figure
    is reproducible from data alone."""
    written = manifest["written"]
    assert any(p.endswith("series_reference.csv") for p in written)
    assert any(p.endswith("summary_reference.csv") for p in written)
    assert any(p.endswith("manifest_reference.json") for p in written)
    for p in (p for p in written if p.endswith((".csv", ".json"))):
        assert Path(p).exists()


def test_intervention_beats_greedy_on_bias_and_regret(manifest):
    g = manifest["greedy"]
    cv = manifest["interventions"]["controlled_variance"]
    assert _final_abs_bias(g) > _final_abs_bias(cv)
    assert _total_regret(g) > _total_regret(cv)

"""
Forecast–Price Feedback Loop
File: tests/test_calibration.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Calibration pipeline tests (torch-free; `calib`-guarded).

The key test is **synthetic recovery**: generate a panel from the project's own
``LinearLogitWorld`` with a known elasticity and endogeneity strength, run the same
OLS/IV/implied-λ pipeline, and assert it recovers ground truth. This validates the
real-data pipeline before it ever touches Dominick's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fploop.calibration.ingest import PANEL_COLUMNS, clean_panel, load_movement

# linearmodels backs the clustered IV standard error inside estimate_panel.
pytest.importorskip("linearmodels")

from fploop.calibration.estimate import simulate_world_panel  # noqa: E402
from fploop.calibration.run import OUTPUT_COLUMNS, calibrate, calibrate_category  # noqa: E402

TRUE_BETA = -2.5
TRUE_LAMBDA = 0.4
SIGMA_XI = 0.2
SIGMA_C = 0.15


def _panel():
    return simulate_world_panel(
        beta=TRUE_BETA,
        lambda_=TRUE_LAMBDA,
        sigma_xi=SIGMA_XI,
        sigma_c=SIGMA_C,
        sigma_price_noise=0.05,
        n_units=12,
        n_weeks=200,
        seed=0,
    )


@pytest.mark.parametrize("true_lambda", [0.0, 0.3, 0.6])
def test_synthetic_recovery_via_ingest(tmp_path, true_lambda):
    """Plant a known lambda, recover it through the *real* ingest->estimate path.

    The five-category table is only trustworthy if the estimator detects a known
    lambda when one is present. This routes a LinearLogitWorld panel through the same
    load_movement -> clean_panel -> estimate_panel code path used on Dominick's (via
    calibrate_category) — not a parallel reimplementation — and checks recovery at
    each planted lambda. The printed line is the report (run with -s to see it).
    """
    panel = simulate_world_panel(
        beta=TRUE_BETA,
        lambda_=true_lambda,
        sigma_xi=SIGMA_XI,
        sigma_c=SIGMA_C,
        sigma_price_noise=0.05,
        n_units=12,
        n_weeks=200,
        seed=0,
    )
    _write_movement_csv(tmp_path / "wsyn.csv", panel)
    res = calibrate_category("SYN", tmp_path)  # full real pipeline
    gap = res["beta_ols"] - res["beta_iv"]
    print(
        f"planted lambda={true_lambda:.1f} -> beta_iv={res['beta_iv']:.3f} "
        f"(true {TRUE_BETA}), gap(ols-iv)={gap:+.3f}, lambda_implied={res['lambda_implied']:.3f}"
    )
    assert abs(res["beta_iv"] - TRUE_BETA) < 0.15  # IV recovers the true elasticity
    if true_lambda > 0:
        assert gap > 0.02  # OLS biased TOWARD zero — the loop direction
        assert abs(res["lambda_implied"] - true_lambda) < 0.2  # implied lambda recovers truth
    else:
        assert abs(gap) < 0.05  # no endogeneity -> OLS == IV
        assert res["lambda_implied"] < 0.15


def test_cleaning_filters_and_transforms():
    # Two valid rows + one of each invalid kind (OK=0, MOVE=0, PRICE=0, bad margin).
    raw = pd.DataFrame(
        {
            "STORE": [1, 1, 2, 2, 3],
            "UPC": [10, 10, 10, 10, 10],
            "WEEK": [1, 2, 3, 4, 5],
            "MOVE": [20, 30, 0, 40, 50],  # row3 MOVE=0 -> dropped
            "QTY": [2, 1, 1, 1, 1],  # row1 is a 2-unit bundle
            "PRICE": [4.0, 3.0, 5.0, 0.0, 6.0],  # row4 PRICE=0 -> dropped
            "SALE": ["", "B", "", "", ""],
            "PROFIT": [25.0, 10.0, 10.0, 10.0, 999.0],  # row5 margin absurd -> dropped
            "OK": [1, 1, 1, 1, 1],
        }
    )
    panel = clean_panel(raw)
    assert list(panel.columns) == PANEL_COLUMNS
    assert len(panel) == 2  # only rows 1 and 2 survive
    # Row 1: unit price = PRICE/QTY = 4/2 = 2.0; cost = 2*(1-0.25) = 1.5.
    first = panel.iloc[0]
    assert np.isclose(first["lp"], np.log(2.0))
    assert np.isclose(first["lc"], np.log(1.5))
    assert np.isclose(first["lq"], np.log(20))


def test_price_hex_decode(tmp_path):
    # Kilts CSV carries full precision in PRICE_HEX (big-endian IEEE-754 double).
    import struct

    price = 3.14159265358979
    hex_token = struct.pack(">d", price).hex()
    csv = tmp_path / "wxyz.csv"
    pd.DataFrame(
        {
            "STORE": [1],
            "UPC": [1],
            "WEEK": [1],
            "MOVE": [10],
            "QTY": [1],
            "PRICE": [3.14],
            "PRICE_HEX": [hex_token],
            "PROFIT": [10.0],
            "OK": [1],
            "SALE": [""],
        }
    ).to_csv(csv, index=False)
    raw = load_movement(csv)
    assert np.isclose(raw["PRICE"].iloc[0], price)  # hex overrode the truncated value


def _write_movement_csv(path, panel):
    """Round-trip a synthetic panel into a raw movement CSV ingest can read back."""
    p = np.exp(panel["lp"].to_numpy())
    c = np.exp(panel["lc"].to_numpy())
    pd.DataFrame(
        {
            "STORE": panel["store"],
            "UPC": panel["upc"],
            "WEEK": panel["week"],
            "MOVE": np.exp(panel["lq"].to_numpy()),
            "QTY": 1,
            "PRICE": p,
            "SALE": 0,
            "PROFIT": 100.0 * (1.0 - c / p),
            "OK": 1,
        }
    ).to_csv(path, index=False)


def test_run_emits_schema(tmp_path):
    data_dir = tmp_path / "dominicks"
    data_dir.mkdir()
    _write_movement_csv(data_dir / "wcso.csv", _panel())
    df = calibrate(["CSO"], data_dir)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["category"] == "CSO"
    assert np.isfinite(df.iloc[0]["lambda_implied"])
    assert df.iloc[0]["beta_iv"] < -1  # plausibly elastic

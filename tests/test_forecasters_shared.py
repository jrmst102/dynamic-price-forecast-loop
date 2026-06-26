"""
Forecast–Price Feedback Loop
File: tests/test_forecasters_shared.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Shared-base tests that need no torch (run in the core install too).

These guard the §5 refactor: ``GBTForecaster`` sits on the shared
``DesignMatrixForecaster`` base, exposes ``is_fitted``, accepts a multi-column
design matrix (Family B's ``[log_p, residual]``), and that importing the
top-level package does not pull in torch.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from fploop.forecasters import DesignMatrixForecaster, GBTForecaster


def _clean_design(beta: float = -2.0, T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    logp = rng.normal(0.0, 0.4, size=T)
    y = np.log(100.0) + beta * logp + rng.normal(0.0, 0.05, size=T)
    return logp.reshape(-1, 1), y


def test_gbt_is_design_matrix_forecaster_with_is_fitted():
    f = GBTForecaster()
    assert isinstance(f, DesignMatrixForecaster)
    assert f.is_fitted is False
    X, y = _clean_design()
    f.fit_design(X, y)
    assert f.is_fitted is True


def test_gbt_fit_design_two_column_is_finite():
    # Family B appends a control-function residual as column 1.
    X, y = _clean_design()
    rng = np.random.default_rng(1)
    resid = rng.normal(0.0, 1.0, size=X.shape[0]).reshape(-1, 1)
    design = np.column_stack([X, resid])
    f = GBTForecaster()
    f.fit_design(design, y)
    est = f.estimated_elasticity()
    assert est.shape == (1,)
    assert np.isfinite(est[0])


def test_importing_fploop_does_not_import_torch():
    # The top-level package must stay torch-free even when the nn extra is
    # installed (lazy NN imports, see fploop.forecasters.__init__). Run in a
    # clean subprocess so the assertion holds regardless of what the rest of the
    # suite has already imported into this process.
    code = "import fploop, sys; assert 'torch' not in sys.modules; print('ok')"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"

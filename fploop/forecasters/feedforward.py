"""
Forecast–Price Feedback Loop
File: fploop/forecasters/feedforward.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

PyTorch feedforward (MLP) demand forecaster. Implements
:class:`FeedforwardForecaster`, a CPU, row-wise (``window == 1``) standardised MLP
that predicts log-demand and warm-starts across retrains. Requires the optional
``nn`` (torch) extra.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from fploop.forecasters.base import DesignMatrixForecaster


class FeedforwardForecaster(DesignMatrixForecaster):
    """Standardised feedforward MLP demand forecaster (CPU, row-wise).

    Predicts ``log q`` from the design matrix one row at a time (``window == 1``).
    Inputs are standardised with per-feature mean/std fixed at the first fit;
    training warm-starts across retrains to keep repeated retraining cheap.
    """

    window = 1

    def __init__(
        self,
        *,
        hidden: tuple[int, ...] = (32, 32),
        epochs_init: int = 400,
        epochs_update: int = 60,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        fd_eps: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.epochs_init = epochs_init
        self.epochs_update = epochs_update
        self.lr = lr
        self.weight_decay = weight_decay
        self.fd_eps = fd_eps
        self.seed = seed
        torch.manual_seed(seed)
        self._net: nn.Module | None = None
        self._mu: NDArray | None = None
        self._sd: NDArray | None = None

    def _build(self, d: int) -> nn.Module:
        """Build the MLP for ``d`` input features (hidden ReLU stack, scalar head)."""
        layers: list[nn.Module] = []
        prev = d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        return nn.Sequential(*layers)

    def _fit(self, X: NDArray, y: NDArray) -> None:
        """Train (or warm-start) the net on standardised ``X`` against target ``y``."""
        fresh = self._net is None or self._net[0].in_features != X.shape[1]
        if fresh:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0) + 1e-8
            torch.manual_seed(self.seed)
            self._net = self._build(X.shape[1])
        epochs = self.epochs_init if fresh else self.epochs_update
        xs = torch.tensor((X - self._mu) / self._sd, dtype=torch.float32)
        yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(self._net(xs), yt).backward()
            opt.step()

    def predict_logq(self, X: NDArray) -> NDArray:
        """Predict log-demand for each row of ``X`` (standardised, no grad)."""
        self._net.eval()
        xs = torch.tensor((np.asarray(X, float) - self._mu) / self._sd, dtype=torch.float32)
        with torch.no_grad():
            return self._net(xs).numpy().ravel()

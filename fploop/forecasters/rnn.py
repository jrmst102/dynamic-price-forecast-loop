"""
Forecast–Price Feedback Loop
File: fploop/forecasters/rnn.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

PyTorch GRU demand forecaster. Implements :class:`RNNForecaster`, a CPU
recurrent model that predicts log-demand from a length-``window`` sliding window
of design rows, so it conditions on recent price/demand history. Requires the
optional ``nn`` (torch) extra.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from fploop.forecasters.base import DesignMatrixForecaster


class _GRURegressor(nn.Module):
    """Small GRU + linear head reading the last time step (private to this module)."""

    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Run the GRU over ``seq`` and map the final step to a scalar prediction."""
        out, _ = self.gru(seq)  # (batch, time, hidden)
        return self.head(out[:, -1, :])  # use the last step


class RNNForecaster(DesignMatrixForecaster):
    """GRU forecaster over a length-``window`` sliding window (CPU).

    Each prediction uses up to ``window`` preceding design rows (inclusive) as its
    sequence, so the model conditions on recent price/demand history — the channel
    through which the greedy feedback loop can amplify. Inputs are standardised
    with stats fixed at the first fit; training warm-starts across retrains.
    """

    def __init__(
        self,
        *,
        window: int = 8,
        hidden: int = 32,
        epochs_init: int = 300,
        epochs_update: int = 50,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        fd_eps: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.window = window
        self.hidden = hidden
        self.epochs_init = epochs_init
        self.epochs_update = epochs_update
        self.lr = lr
        self.weight_decay = weight_decay
        self.fd_eps = fd_eps
        self.seed = seed
        torch.manual_seed(seed)
        self._net: _GRURegressor | None = None
        self._mu: NDArray | None = None
        self._sd: NDArray | None = None

    def _windows(self, Xs: NDArray) -> torch.Tensor:
        """Build one sequence per row: row i -> standardised rows [i-window+1 .. i],
        left-padded with the first row when fewer than ``window`` precede."""
        n, d = Xs.shape
        seqs = np.zeros((n, self.window, d), dtype=np.float32)
        for i in range(n):
            lo = i - self.window + 1
            if lo < 0:
                pad = np.repeat(Xs[:1], -lo, axis=0)
                seqs[i] = np.vstack([pad, Xs[: i + 1]])
            else:
                seqs[i] = Xs[lo : i + 1]
        return torch.tensor(seqs)

    def _fit(self, X: NDArray, y: NDArray) -> None:
        """Train (or warm-start) the GRU on windowed, standardised ``X`` against ``y``."""
        fresh = self._net is None or self._net.gru.input_size != X.shape[1]
        if fresh:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0) + 1e-8
            torch.manual_seed(self.seed)
            self._net = _GRURegressor(X.shape[1], self.hidden)
        Xs = (X - self._mu) / self._sd
        seqs = self._windows(Xs)
        yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        epochs = self.epochs_init if fresh else self.epochs_update
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(self._net(seqs), yt).backward()
            opt.step()

    def predict_logq(self, X: NDArray) -> NDArray:
        """Predict log-demand for each row of ``X`` via its trailing window (no grad)."""
        self._net.eval()
        Xs = (np.asarray(X, float) - self._mu) / self._sd
        seqs = self._windows(Xs)
        with torch.no_grad():
            return self._net(seqs).numpy().ravel()

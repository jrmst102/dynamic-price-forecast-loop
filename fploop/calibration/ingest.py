"""
Forecast–Price Feedback Loop
File: fploop/calibration/ingest.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Load and clean Dominick's movement files into a tidy panel.

Pre-processing is **read -> clean**, all Python-native (no SAS software). The
reader accepts whatever readable form a category's movement file is in:

- ``.sas7bdat`` (SAS v7+) via :mod:`pyreadstat`;
- ``.csv`` / ``.txt`` (delimited) via :mod:`pandas`, including the Kilts
  full-precision **hex** columns (``PRICE_HEX``/``PROFIT_HEX``, 16-digit big-endian
  IEEE-754 doubles) which recover the precision the truncated ``PRICE`` column loses.

It does **not** read SAS v6 ``.sd2`` — no open-source reader does; convert those to
``.csv``/``.sas7bdat`` first (see ``data/README.md``).

The documented movement fields are ``STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE,
PROFIT, OK``. Cleaning derives unit price ``p = PRICE / QTY``, unit cost
``c = p * (1 - PROFIT/100)`` (the gross-margin field gives wholesale cost — the
instrument), quantity ``q = MOVE``, and emits ``lq, lp, lc`` plus a week-of-year
seasonality key and the deal flag.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

# Documented movement-file fields (DFF Data Manual). Confirmed per category.
MOVEMENT_FIELDS = ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"]

# Tidy panel columns the estimator consumes.
PANEL_COLUMNS = ["store", "upc", "week", "woy", "lq", "lp", "lc", "deal"]

_WEEKS_PER_YEAR = 52


def _decode_hex16(series: pd.Series) -> np.ndarray:
    """Vectorized decode of a SAS ``hex16.`` column (big-endian IEEE-754 doubles).

    Kilts CSV exports carry full precision in ``PRICE_HEX``/``PROFIT_HEX`` as 16 hex
    digits — the raw 8 bytes of the original double. Valid 16-hex tokens are decoded
    in one ``np.frombuffer`` pass (fast on millions of rows); blank/malformed tokens
    become ``nan`` so one bad row cannot abort a load. ``struct`` covers the
    single-value path used by tests.
    """
    h = series.astype(str).str.strip().str.zfill(16)
    valid = h.str.fullmatch(r"[0-9A-Fa-f]{16}").to_numpy()
    out = np.full(len(h), np.nan)
    good = h.to_numpy()[valid]
    if good.size:
        try:
            out[valid] = np.frombuffer(bytes.fromhex("".join(good)), dtype=">f8")
        except ValueError:  # fall back element-wise if the joined buffer is malformed
            out[valid] = [struct.unpack(">d", bytes.fromhex(t))[0] for t in good]
    return out


def load_movement(path: str | Path, *, row_limit: int | None = None) -> pd.DataFrame:
    """Read a category's movement file into a raw DataFrame with upper-case fields.

    Parameters
    ----------
    path : str or Path
        Movement file. ``.sas7bdat`` is read with pyreadstat; ``.csv``/``.txt`` with
        pandas. ``PRICE_HEX``/``PROFIT_HEX`` columns, if present, overwrite the
        truncated ``PRICE``/``PROFIT`` with their full-precision decode.
    row_limit : int, optional
        Read at most this many rows (a busy movement file is large; handy for a
        quick look). Applies to both readers.

    Returns
    -------
    pd.DataFrame
        Columns upper-cased; at least the subset of :data:`MOVEMENT_FIELDS` present.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".sas7bdat":
        import pyreadstat  # lazy: only when actually reading SAS

        df, _meta = pyreadstat.read_sas7bdat(str(p), row_limit=row_limit)
    elif suffix == ".csv":
        df = pd.read_csv(p, nrows=row_limit)  # C engine, comma-delimited
    elif suffix == ".txt":
        df = pd.read_csv(p, sep=None, engine="python", nrows=row_limit)  # infer whitespace
    else:
        raise ValueError(
            f"unreadable movement format {suffix!r} for {p.name}; SAS v6 .sd2 is not "
            "supported — convert to .csv/.sas7bdat first (see data/README.md)"
        )

    df.columns = [str(c).strip().upper() for c in df.columns]
    if "PRICE_HEX" in df.columns:
        df["PRICE"] = _decode_hex16(df["PRICE_HEX"])
    if "PROFIT_HEX" in df.columns:
        df["PROFIT"] = _decode_hex16(df["PROFIT_HEX"])
    return df


def clean_panel(raw: pd.DataFrame, *, max_abs_margin: float = 60.0) -> pd.DataFrame:
    """Clean a raw movement frame into the tidy ``(store, upc, week)`` panel.

    Applies the validity filter, unit-pricing, and the gross-margin -> cost
    transform, then emits log quantities/prices/costs for the elasticity
    regression.

    Steps
    -----
    1. Keep valid rows: ``OK == 1``, ``MOVE > 0``, ``PRICE > 0``, ``QTY >= 1``, and
       ``|PROFIT| <= max_abs_margin`` (drop implausible margins).
    2. Unit price ``p = PRICE / QTY`` (Dominick's prices are for ``QTY``-unit bundles).
    3. Unit cost ``c = p * (1 - PROFIT/100)`` — the wholesale cost / instrument.
       Drop ``c <= 0``.
    4. Quantity ``q = MOVE``; carry the ``SALE`` deal flag.
    5. ``lq, lp, lc`` plus week-of-year ``woy = ((WEEK - 1) % 52) + 1`` seasonality.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of :func:`load_movement` (upper-cased fields).
    max_abs_margin : float
        Drop rows whose ``|PROFIT|`` (percent gross margin) exceeds this.

    Returns
    -------
    pd.DataFrame
        Tidy panel with columns :data:`PANEL_COLUMNS`.
    """
    required = ("STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "PROFIT")
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f"movement frame missing required fields: {missing}")

    df = raw.copy()
    for col in ("MOVE", "QTY", "PRICE", "PROFIT", "WEEK", "STORE", "UPC"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = df["OK"] == 1 if "OK" in df else pd.Series(True, index=df.index)

    valid = (
        ok
        & (df["MOVE"] > 0)
        & (df["PRICE"] > 0)
        & (df["QTY"] >= 1)
        & df["PROFIT"].abs().le(max_abs_margin)
        & df[["MOVE", "QTY", "PRICE", "PROFIT", "WEEK"]].notna().all(axis=1)
    )
    df = df.loc[valid].copy()

    p = df["PRICE"] / df["QTY"]
    c = p * (1.0 - df["PROFIT"] / 100.0)
    q = df["MOVE"]
    keep = c > 0
    df, p, c, q = df.loc[keep], p.loc[keep], c.loc[keep], q.loc[keep]

    deal = df["SALE"].astype(str) if "SALE" in df else pd.Series("", index=df.index)
    panel = pd.DataFrame(
        {
            "store": df["STORE"].astype("int64"),
            "upc": df["UPC"].astype("int64"),
            "week": df["WEEK"].astype("int64"),
            "woy": ((df["WEEK"].astype("int64") - 1) % _WEEKS_PER_YEAR) + 1,
            "lq": np.log(q.to_numpy()),
            "lp": np.log(p.to_numpy()),
            "lc": np.log(c.to_numpy()),
            "deal": deal.to_numpy(),
        }
    )
    return panel.reset_index(drop=True)


def load_category_panel(
    movement_path: str | Path, *, row_limit: int | None = None, max_abs_margin: float = 60.0
) -> pd.DataFrame:
    """Convenience: :func:`load_movement` then :func:`clean_panel`."""
    raw = load_movement(movement_path, row_limit=row_limit)
    return clean_panel(raw, max_abs_margin=max_abs_margin)

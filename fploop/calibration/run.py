"""
Forecast–Price Feedback Loop
File: fploop/calibration/run.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Per-category orchestration -> ``calibrated_markets.csv`` (+ CLI).

Resolve each category's readable movement file, ingest + clean + estimate, and
emit one row per category in the schema the Phase-4c crossover-map overlay reads:

``category, lambda_implied, first_stage_r2, beta_iv, beta_ols, beta_iv_se,
sigma_xi, rho, n_obs`` (plus ``first_stage_f``/``weak_iv``, ignored by the overlay).

CLI::

    python -m fploop.calibration.run --categories RFJ,CSO \\
        --data data/raw/dominicks --out data/calibrated_markets.csv

SAS v6 ``.sd2`` files are not readable here; a category that only has a ``.sd2``
movement file raises a clear error pointing at ``data/README.md`` (e.g. canned soup
``CSO`` works from the Kilts full-precision CSV; refrigerated juice ``RFJ`` has no
CSV and needs a one-time SAS export).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fploop.calibration.estimate import estimate_panel
from fploop.calibration.ingest import load_category_panel

OUTPUT_COLUMNS = [
    "category",
    "lambda_implied",
    "first_stage_r2",
    "beta_iv",
    "beta_ols",
    "beta_iv_se",
    "sigma_xi",
    "rho",
    "n_obs",
    "first_stage_f",
    "weak_iv",
]

_READABLE_SUFFIXES = (".csv", ".txt", ".sas7bdat")


def find_movement_file(data_dir: str | Path, acronym: str) -> Path:
    """Locate a readable movement file for ``acronym`` under ``data_dir``.

    Searches ``data_dir`` and a ``<ACRONYM>/`` subfolder for ``w<acr>`` with a
    readable suffix (``.csv``/``.txt``/``.sas7bdat``). Raises if only an
    unreadable SAS v6 ``.sd2`` is present, or if nothing matches.
    """
    base = Path(data_dir)
    acr = acronym.lower()
    search_dirs = [base, base / acronym.upper(), base / acr]
    candidates: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            candidates += sorted(d.glob(f"w{acr}*"))
    readable = [p for p in candidates if p.suffix.lower() in _READABLE_SUFFIXES]
    if readable:
        return readable[0]
    sd2 = [p for p in candidates if p.suffix.lower() == ".sd2"]
    if sd2:
        raise FileNotFoundError(
            f"{acronym}: only an unreadable SAS v6 file {sd2[0].name} found. Convert it "
            f"to .csv/.sas7bdat first (see data/README.md). {acronym} has no Kilts CSV "
            "if it is refrigerated juice (RFJ) — export via SAS OnDemand."
        )
    raise FileNotFoundError(f"{acronym}: no movement file w{acr}* under {base}")


def calibrate_category(acronym: str, data_dir: str | Path, *, row_limit: int | None = None) -> dict:
    """Ingest, clean, and estimate one category -> a result row dict."""
    movement = find_movement_file(data_dir, acronym)
    panel = load_category_panel(movement, row_limit=row_limit)
    result = estimate_panel(panel)
    return {"category": acronym.upper(), **result}


def calibrate(
    categories: list[str], data_dir: str | Path, *, row_limit: int | None = None
) -> pd.DataFrame:
    """Calibrate each category; return a tidy frame in :data:`OUTPUT_COLUMNS` order."""
    rows = [calibrate_category(c, data_dir, row_limit=row_limit) for c in categories]
    return pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)


def write_calibrated_markets(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write the calibrated-markets CSV (creating parent dirs)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the calibration CLI."""
    p = argparse.ArgumentParser(
        prog="python -m fploop.calibration.run",
        description="Calibrate Dominick's categories -> calibrated_markets.csv.",
    )
    p.add_argument("--categories", required=True, help="comma-separated acronyms, e.g. RFJ,CSO")
    p.add_argument("--data", default="data/raw/dominicks", help="dir holding the movement files")
    p.add_argument("--out", default="data/calibrated_markets.csv", help="output CSV path")
    p.add_argument("--row-limit", type=int, default=None, help="cap rows read (quick look)")
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    df = calibrate(categories, args.data, row_limit=args.row_limit)
    out = write_calibrated_markets(df, args.out)
    print(f"Wrote {len(df)} row(s) to {out}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

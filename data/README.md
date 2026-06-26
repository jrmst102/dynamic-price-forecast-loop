# Data

**Forecast–Price Feedback Loop** — Dr. Jose Mendoza (jose.mendoza@nyu.edu)

This directory holds the inputs and derived outputs for the **calibration**
workstream (real grocery-market coordinates for the crossover map). Raw Dominick's
files are **not** part of the source tree — only the small derived
`calibrated_markets.csv` and the `data/reference/` decode tables are committed.

## Layout

```
data/
├── README.md                       # this file
├── calibrated_markets.csv          # derived output (committed) — overlay coordinates
├── reference/                      # small week/store decode tables (committed)
└── raw/
    └── dominicks/                  # raw Dominick's archives (gitignored except .gitkeep)
        ├── w<acr>.zip / upc<acr>.* # per-category movement + UPC files
        └── <ACR>/                  # extracted SAS files (gitignored)
```

## Dominick's Finer Foods (DFF) download

Source: the **Dominick's Database**, James M. Kilts Center, University of Chicago
Booth School of Business — public, no login. Movement files are at:

```
https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/movement_csv-files/w<acronym>.zip
```

Each zip holds `w<acronym>.csv`. **Download the CSV (not the SAS) form** and unzip
into `data/raw/dominicks/` (so the file lands at `data/raw/dominicks/w<acronym>.csv`).
**28 of 29 categories have a CSV — only analgesics (`ANA`) does not.**

### Full precision (the key detail)

The Kilts CSV truncates the `PRICE`/`PROFIT` columns, **but also carries the full
double in `PRICE_HEX`/`PROFIT_HEX`** (SAS `hex16.` — 16 hex digits, big-endian
IEEE-754). The ingest layer decodes those automatically, so a log-price regression
sees bit-exact prices. (The reader also accepts `.sas7bdat`; it does **not** read
the legacy SAS v6 `.sd2` movement form — use the CSV.)

The five categories calibrated in `calibrated_markets.csv`:

| Category            | Acronym | Movement CSV | Profile                 |
|---------------------|---------|--------------|-------------------------|
| Canned soup         | `CSO`   | `wcso.csv`   | staple                  |
| Refrigerated juice  | `RFJ`   | `wrfj.csv`   | promo-heavy             |
| Beer                | `BER`   | `wber.csv`   | promo-heavy             |
| Frozen juice        | `FRJ`   | `wfrj.csv`   | promo-heavy             |
| Oatmeal             | `OAT`   | `woat.csv`   | staple                  |

Then: `python -m fploop.calibration.run --categories CSO,RFJ,BER,FRJ,OAT
--data data/raw/dominicks --out data/calibrated_markets.csv`.

## Movement-file fields (per the DFF Data Manual)

`STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE, PROFIT, OK` — confirm against the
manual before relying on them.

## Reading the calibration (forward-looking λ)

`lambda_implied` is **descriptive, not predictive**. Calibrating against historical
Dominick's data measures the endogeneity that was *realized* in the retailer's past
pricing — i.e. where that market's historical pricing regime sat on the crossover
map's loop-strength axis. A category with a strong instrument but `lambda_implied ≈ 0`
(OLS–IV gap null/negative) is honestly telling you the feedback loop was **not biting**
there — it lands in the low-λ region where greedy/exploration suffices and IV is not
the fix. The same market could move along the axis if pricing behaviour changed; the
dot is a "you were here", not a forecast. (Canned soup is exactly this case.)

## If a category ships only as SAS v6 `.sd2`

All five calibrated categories (incl. RFJ) have a Kilts CSV, so this is rarely
needed. But the legacy **movement** download is SAS v6 `.sd2` (header
`SAS 6.12.00 WIN_NT`), which no open-source tool reads (ReadStat/pyreadstat,
pandas, R haven/foreign-without-SAS — none). To use such a file, convert it once
to a full-precision CSV via SAS:

1. Open **SAS OnDemand for Academics** (free, browser; no local install).
2. Upload `W<ACR>.SD2` and run the eurostat `sas2csv` recipe:
   ```sas
   libname dff v6 "/path/to/uploaded";          /* the V6 engine reads .sd2 */
   data work.w<acr>; set dff.w<acr>;
       PRICE_HEX=PRICE; PROFIT_HEX=PROFIT;
       format PRICE_HEX PROFIT_HEX hex16.;        /* full-precision doubles */
   run;
   proc export data=work.w<acr> outfile="w<acr>.csv" dbms=csv replace; run;
   ```
3. Drop `w<acr>.csv` in `data/raw/dominicks/` and run the calibration CLI. The
   ingest layer already decodes the `_HEX` columns; no code change needed.

## Acknowledgment

We thank the James M. Kilts Center, University of Chicago Booth School of Business,
for making the Dominick's Finer Foods dataset publicly available.

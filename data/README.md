# Input data contract

Market-price files are **not bundled**. The submitted notebook reads local datasets, but their original vendor, adjustment conventions and redistribution permissions were not established during packaging. This repository must not imply that a fresh download is the same data vintage.

Provide two UTF-8 CSV files, with ISO dates in the first column and one numeric asset-price column per ticker:

- `sp500_prices.csv`: the stock universe. The saved input has 503 stock columns and ends 2026-09-30. These are current constituents, not historical membership. Preserve missing prelisting prices as missing; do not backward-fill.
- `etf_prices.csv`: at least `SPY` and `GLD` for the script. For the notebook's auxiliary ETF benchmark, also provide `EFA`, `EEM`, `TLT`, `IEF`, `HYG`, `USO`, `DBA`, `VNQ`, `FXE`, `UUP`. The saved raw file also included another column; extra columns are permitted. `CASH` is constructed by the loader.

Dates must be unique and ascending, assets unique, observed prices finite and positive. Do not use a stock ticker named `CASH` or `GLD`, which would collide with defensive assets. All-empty stock dates are removed; ETFs are aligned to the resulting stock calendar. SPY and GLD must be present on that calendar.

The code computes price percentage changes. Whether these represent total returns depends on the supplied prices: **the project does not independently establish dividend or corporate-action treatment**. Never concatenate adjusted and unadjusted histories without documentation.

To reproduce the saved vintage, record source/vendor, download date, adjustment flags, calendar, constituent snapshot date and file hashes. The runner records SHA-256 hashes in `run_manifest.json`, but hashing alone does not validate financial correctness.

Use `--data-dir` for the script or `LRM_DATA_DIR` for the notebook. Raw CSV/parquet/pickle files in this directory are ignored by Git. Input structure examples below are schemas, not actual data:

```text
Date,AAPL,MSFT,...
YYYY-MM-DD,price,price,...
```

Future point-in-time research additionally requires constituent effective-date intervals, delisted securities and corporate-action metadata. A present-day ticker list is not a substitute.

# Published evidence

This directory is a frozen **evidence snapshot**, not the output of a full historical rerun during repository packaging.

| File | Source | Precision |
|---|---|---|
| `baseline_summary.csv` | Parsed from saved text outputs in the submitted notebook | As displayed: percent values to two decimals; Sharpe to three |
| `source_execution_log.txt` | Saved notebook stream outputs | Verbatim source log; no original machine paths |
| `experiment_summary.csv` | Author-provided comparison screenshots | Rounded display values; not full-precision daily exports |
| `provenance.json` | Packaging manifest | Source notebook SHA-256 and publication scope |

`assets/notebook_backtest.png` is the exact embedded PNG from the source notebook. Other published charts are generated from these rounded tables with `scripts/render_published_results.py`. No synthetic or interpolated daily paths are substituted for missing experiment exports.

The experiment sources are the author's risk-adjusted momentum table dated 2026-10-05 12:42:06 and volatility-control table dated 2026-10-04 23:48:55. These were supplied as screenshots during research. The table transcription preserves missing turnover for the volatility-control experiment rather than inventing a value. Experimental daily time series were not supplied with the final notebook.

The original notebook SHA is recorded locally before portable-path and English-documentation changes. A source hash is an identity record, not independent validation of the underlying market dataset.

Fresh runs belong under `results/local/` (Git-ignored), including daily returns, signals, weights, logs, summaries and input hashes. Do not overwrite the published snapshot automatically.

# Validation status

## Performed during publication

- Parsed and schema-validated the public notebook; preserved saved outputs and execution counts.
- Compared every extracted numerical function with the submitted source using Python abstract syntax trees. The HMM, signal generator, executor and metric function bodies are unchanged.
- Removed author-machine input paths from the public notebook and made figure output local to the repository.
- Parsed baseline statistics from saved notebook logs and cross-checked them against the baseline rows of the author-provided comparison table.
- Rendered the two aggregate comparison charts from published CSVs and visually inspected them, along with the exact saved notebook figure.
- Ran **10 data-free unit tests** covering next-close execution, self-financing fees, zero cash fees, passive drift, quote failures, missing held returns, negative weights, initial-capital drawdown, preserved HMM exposure in selection ablations, initial-state filtering, and Student-t final-parameter consistency. Some tests cover more than one assertion.
- Checked script import/CLI entry, source syntax, local Markdown links and the absence of original machine paths in the publication files.

## Not performed during publication

- No full weekly historical HMM refit or full historical notebook execution. Numerical source outputs are preserved rather than represented as newly reproduced results.
- No independent verification of market data vendor, adjustment convention, historical membership or delisting treatment.
- No regeneration of experiment daily series from the rounded screenshots. Published experimental charts display aggregate values only.
- No statistical-significance, selection-adjusted Sharpe, capacity or live-trading verification.
- No sector-momentum performance verification; that experiment remains pending.

## Reproduce the checks

```bash
python -m unittest discover -s tests -v
python scripts/render_published_results.py
```

Publication checks used Python with NumPy 2.5.3, pandas 3.0.6, Matplotlib 3.11.2 and nbformat 5.11.1. The requirements file supplies compatibility ranges, not a lockfile for the original source run. GitHub Actions runs the data-free checks on Python 3.12; its result is separate from the local validation described here.

For a full data-backed rerun, supply verified input files according to `data/README.md`, run `scripts/run_research.py`, and reconcile its daily exports, summaries and input hashes against the desired vintage. Running the notebook end-to-end additionally exercises the auxiliary 12-ETF benchmark.

# Limitations and prioritized next steps

## Claims the evidence supports

The supplied run shows a different historical risk/return profile after adding an HMM overlay to a stock-momentum portfolio. The implementation explicitly accounts for delayed execution, position drift and transaction costs. The tested simple volatility rule and risk-adjusted ranking do not consistently dominate the HMM baseline.

## Claims the evidence does not support

It does not establish implementable alpha, future drawdown bounds, an untouched out-of-sample result, a statistically significant Sharpe improvement, or a calibrated crash probability. A successful synthetic-state test is not a market protection success rate. Lower maximum drawdown in one interval is not proof that the HMM anticipated a specific crisis.

## Principal threats to validity

| Limitation | Why it matters | Next action |
|---|---|---|
| Current constituents | Excludes past failures and admits future index members before inclusion | Build a point-in-time universe including delistings |
| Repeated historical-test inspection | Design choices have been influenced by the evaluation interval | Freeze a prospective protocol and log all future decisions |
| Price-source uncertainty | Splits, dividends, adjusted/unadjusted prices and stale quotes can alter rankings and returns | Document vendor, adjustment settings and corporate actions; reconcile examples |
| Unequal risk exposure | HMM may outperform a full-risk comparator simply by holding less equity | Compare static and dynamic portfolios at risk budgets chosen on development data |
| Sparse tail events | A few crises can dominate drawdown conclusions | Event-level analysis, leave-one-event-out sensitivity, paired block bootstrap |
| Fixed cost model | A 7-bps assumption omits varying spreads, market impact and liquidity constraints | Turnover/capacity study and cost stress without retuning the model |
| Zero cash yield | Cash allocation and zero-risk-free Sharpe are incomplete economic conventions | Model a realizable cash instrument and excess-return Sharpe |
| Model numerical limits | Probability-space densities, deterministic local fits and parameter aging | Log-domain inference, multiple starts, stability and fallback diagnostics |
| Weekly/monthly cadence | Probability and target allocations can lag abrupt regime changes | Predeclare cadence tests on development data; avoid tailoring to one known crisis |
| Gold defense | GLD is not cash and can fall alongside stocks | Separate equity reduction from gold allocation in attribution |

## Research roadmap

**Priority 1 — trustworthy inputs.** Establish historical constituent membership, delisting treatment and price-adjustment provenance. Until then, improving a backtest's headline return is a lower priority than improving its validity.

**Priority 2 — evaluation discipline.** Freeze signal, cost, timing and model rules. Record hashes of code and data. Keep 2017–2026 as historical research; evaluate future observations without selecting designs after seeing their performance. Rolling parameter estimation is permitted only under the frozen update rule.

**Priority 3 — isolate incremental value.** Match static equity/gold/cash allocations and simple risk budgets using development data. Compare entry timing, missed upside, drawdown duration and net returns rather than optimizing the same headline metric repeatedly.

**Priority 4 — quantify uncertainty and economics.** Use paired time-series block resampling and regime/event sensitivity; distinguish uncertainty conditional on a chosen strategy from uncertainty after many strategies were tried. Test realistic spreads, turnover, capacity and cash income.

**Priority 5 — model refinement.** Only after those controls, evaluate alternative distributions, log-domain filtering, multiple starts and interpretable forward-risk calibration. Higher model complexity is not a research objective by itself.

## Proposed sector experiment

Within-sector ranking with equal sector weights changes both stock selection and sector allocation. Contemporary sector labels introduce additional historical classification bias. No verified sector performance is included in this release; it should remain pending until the data source and outputs are available.

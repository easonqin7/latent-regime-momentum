# Experiments: keep the controls and the negative results

These comparisons were run by the author during research. Tables below preserve rounded screenshot values; this publication pass has not re-estimated their full daily paths. The final supplied notebook contains the baseline HMM and SPY outputs.

## Experiment 1 — remove the overlay

Use the same current stock universe, exact lagged momentum score, top 100, monthly signal dates, next-close execution and 7-bps security cost. Allocate 100% to the stock sleeve. This isolates the full defensive overlay, including equity reduction, gold, cash and associated trading costs. It is not a pure test of regime timing at equal risk.

## Experiment 2 — replace state inference with realized volatility

Estimate annual volatility from the pure momentum portfolio's trailing 63 daily net returns. Set equity exposure to `min(1, 0.15 / volatility)` at monthly signal dates; residual cash and gold remain half each. Exclude the initial cash-only history until a full invested return window is available.

The 15% target applies to equity scaling, not a hard volatility cap for the whole portfolio. Estimation lag, monthly execution and gold risk can produce realized volatility above the reference target. Parameters were fixed for this trial, not selected by an automated validation search. In the historical diagnostic, this rule has nearly the HMM's CAGR but a much larger maximum drawdown. This does not establish that all volatility-targeting rules fail.

## Experiment 3 — risk-adjust stock ranking

Divide the original momentum score by annualized sample volatility over the same 252-return window, with both ending 21 trading days before the signal. Retain top-100 equal-weight selection. Compare with and without the same HMM allocation.

This score is not a conventional Sharpe ratio: its numerator is cumulative price return, not mean excess return. For negative momentum, division by a larger volatility can increase the score toward zero. Full-window volatility eligibility can also exclude more incomplete histories than endpoint-only momentum; selection differences are therefore not purely a ranking effect.

The source reports 81.5% average top-100 overlap during the historical diagnostic. Risk-adjusted ranking reduces volatility, but its return and Sharpe improvements are not consistent across periods. The HMM version's tiny maximum-drawdown change does not justify claiming stronger tail protection. No significance test was performed on the Sharpe difference.

## Complete rounded comparison

The original `train` and `val` labels denote development subperiods. `test` is the repeatedly observed historical diagnostic. Turnover is bought-plus-sold security notional, annualized; the volatility-control screenshot did not include it.

### Development A: 2010–2012

| Model | CAGR | Sharpe | MaxDD | Annual volatility | Annual turnover |
|---|---:|---:|---:|---:|---:|
| Momentum | 16.93% | 0.796 | -24.53% | 22.98% | 6.02 |
| RA Momentum | 17.11% | 0.899 | -19.34% | 19.74% | 6.05 |
| Momentum + HMM | 13.64% | 0.726 | -23.10% | 20.51% | 6.41 |
| RA Momentum + HMM | 13.29% | 0.793 | -19.13% | 17.33% | 6.51 |
| SPY | 10.82% | 0.651 | -18.61% | 18.37% | 0.00 |
| Vol control | 13.74% | 0.842 | -18.51% | 17.03% | Not supplied |

### Development B: 2013–2016

| Model | CAGR | Sharpe | MaxDD | Annual volatility | Annual turnover |
|---|---:|---:|---:|---:|---:|
| Momentum | 18.96% | 1.216 | -15.81% | 15.24% | 5.58 |
| RA Momentum | 16.63% | 1.160 | -13.66% | 14.12% | 6.17 |
| Momentum + HMM | 17.67% | 1.200 | -13.64% | 14.43% | 6.67 |
| RA Momentum + HMM | 15.33% | 1.133 | -12.25% | 13.39% | 7.29 |
| SPY | 14.22% | 1.102 | -13.02% | 12.80% | 0.00 |
| Vol control | 17.41% | 1.194 | -16.18% | 14.30% | Not supplied |

### Historical diagnostic: 2017–2026-09-30

| Model | CAGR | Sharpe | MaxDD | Annual volatility | Annual turnover |
|---|---:|---:|---:|---:|---:|
| Momentum | 22.36% | 1.035 | -37.44% | 21.82% | 5.98 |
| RA Momentum | 20.01% | 1.010 | -36.53% | 20.07% | 6.13 |
| Momentum + HMM | 16.54% | 1.147 | -14.52% | 14.23% | 7.30 |
| RA Momentum + HMM | 15.59% | 1.189 | -14.46% | 12.88% | 7.47 |
| SPY | 15.25% | 0.874 | -33.72% | 18.13% | 0.00 |
| Vol control | 16.65% | 0.965 | -34.30% | 17.57% | Not supplied |

## Pending — within-sector momentum

The proposed design ranks stocks within contemporary sectors and gives sectors equal capital weight while retaining 100 stocks. This changes both sector exposure and stock selection. Current sector classifications are not historical classifications. No verified output was supplied for this experiment, so it is neither a reported success nor a reported failure in this release.

## Decision discipline

Retain the original momentum + HMM as a research baseline, not a proven optimal portfolio. Keep pure momentum as a return benchmark. Do not select new lookbacks or targets simply to eliminate a known 2020 drawdown after inspecting it. The next decisive improvement is trustworthy point-in-time data and prospective evaluation, not an expanding sweep over the same historical test.

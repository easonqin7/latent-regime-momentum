"""Prespecified comparison rules. No historical-test optimization."""
import numpy as np
import pandas as pd
from .core import execute


def momentum_targets(stocks, signal_dates, columns, risk_adjusted=False):
    score = stocks.pct_change(252, fill_method=None).shift(21)
    if risk_adjusted:
        vol = (stocks.pct_change(fill_method=None).rolling(252, min_periods=252)
               .std(ddof=1).shift(21) * np.sqrt(252))
        score = score / vol.where(vol > 0)
    score = score.replace([np.inf, -np.inf], np.nan)
    targets = pd.DataFrame(0.0, index=signal_dates, columns=columns)
    for dt in signal_dates:
        available = score.loc[dt].dropna()
        available = available[stocks.loc[dt, available.index].notna()]
        if len(available) < 100:
            raise ValueError(f'{dt}: fewer than 100 eligible stocks')
        targets.loc[dt, available.nlargest(100).index] = .01
    return targets


def apply_hmm(stock_targets, hmm_targets, stock_columns):
    result = stock_targets.copy()
    result.loc[:, stock_columns] = result.loc[:, stock_columns].mul(
        hmm_targets.loc[result.index, stock_columns].sum(axis=1), axis=0)
    result.loc[:, ['CASH', 'GLD']] = hmm_targets.loc[result.index, ['CASH', 'GLD']]
    return result


def comparison_runs(stocks, returns, tradable, hmm_targets, cost=.0007):
    """Replay the user-tested momentum, 63-day vol, and RA selection variants."""
    dates = hmm_targets.dropna(how='all').index.sort_values()
    pure_targets = momentum_targets(stocks, dates, returns.columns)
    pure = execute(pure_targets, returns, cost=cost, tradable=tradable, return_details=True)
    ra_targets = momentum_targets(stocks, dates, returns.columns, risk_adjusted=True)
    ra_hmm_targets = apply_hmm(ra_targets, hmm_targets, stocks.columns)

    vol = pure['net'].rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)
    first_full_window = returns.index.get_loc(dates[0]) + 2 + 63 - 1
    if first_full_window >= len(returns):
        raise ValueError('Insufficient post-investment history for volatility rule')
    vol.iloc[:first_full_window] = np.nan
    equity = (.15 / vol.where(vol > 0)).clip(0, 1).reindex(dates).dropna()
    vol_targets = pure_targets.loc[equity.index].copy()
    vol_targets.loc[:, stocks.columns] = vol_targets.loc[:, stocks.columns].mul(equity, axis=0)
    vol_targets['CASH'] = (1-equity)/2
    vol_targets['GLD'] = (1-equity)/2

    targets = {'Momentum': pure_targets, 'RA Momentum': ra_targets,
               'RA Momentum + HMM': ra_hmm_targets, 'Vol control': vol_targets}
    runs = {'Momentum': pure}
    for name, signal in targets.items():
        if name != 'Momentum':
            runs[name] = execute(signal, returns, cost=cost, tradable=tradable, return_details=True)
    return runs, targets

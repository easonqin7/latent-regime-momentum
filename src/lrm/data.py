"""Portable input loading; valuation and execution quotes stay separate."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_prices(data_dir):
    data_dir = Path(data_dir)
    stocks = pd.read_csv(data_dir / 'sp500_prices.csv', index_col=0, parse_dates=True)
    etfs = pd.read_csv(data_dir / 'etf_prices.csv', index_col=0, parse_dates=True)
    stocks = stocks.loc[stocks.notna().any(axis=1)].copy()
    for name, frame in [('stocks', stocks), ('ETFs', etfs)]:
        if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f'{name}: empty or duplicate observations')
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing:
            raise ValueError(f'{name}: dates must be ascending datetimes')
        observed = frame.to_numpy(dtype=float)
        if np.isinf(observed).any() or (observed <= 0).any():
            raise ValueError(f'{name}: invalid price')
    if {'CASH', 'GLD'} & set(stocks.columns):
        raise ValueError('Stock columns collide with defensive assets')
    etfs = etfs.reindex(stocks.index).copy()
    if etfs[['SPY', 'GLD']].isna().any().any():
        raise ValueError('Missing SPY or GLD on stock calendar')
    etfs['CASH'] = 1.0
    stock_returns = stocks.ffill(limit=1).pct_change(fill_method=None)
    etf_returns = etfs.pct_change(fill_method=None)
    etf_returns['CASH'] = 0.0
    returns = pd.concat([stock_returns, etf_returns[['CASH', 'GLD']]], axis=1)
    tradable = pd.concat([stocks.notna(), etfs[['CASH', 'GLD']].notna()], axis=1)
    return stocks, etfs, returns, etf_returns, tradable

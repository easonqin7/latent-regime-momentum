"""Run the original weekly HMM and fixed comparisons on user-supplied prices."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from lrm.core import r4c2_signal_monthly, execute, metrics, splits
from lrm.data import load_prices
from lrm.experiments import comparison_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'local')
    parser.add_argument('--baseline-only', action='store_true', help='Skip ablation runs')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    published = (ROOT / 'results' / 'published').resolve()
    if output == published or published in output.parents:
        raise ValueError('Do not overwrite the published evidence snapshot')
    output.mkdir(parents=True, exist_ok=True)

    stocks, etfs, returns, etf_returns, tradable = load_prices(args.data_dir)
    signal, probability, log = r4c2_signal_monthly(stocks, etfs)
    baseline = execute(signal, returns, tradable=tradable, return_details=True)
    runs = {'Momentum + HMM': baseline}
    signals = {'Momentum + HMM': signal}
    if not args.baseline_only:
        extra_runs, extra_signals = comparison_runs(stocks, returns, tradable, signal)
        runs.update(extra_runs)
        signals.update(extra_signals)

    daily = pd.concat([r['net'].rename(name) for name, r in runs.items()], axis=1)
    spy = etf_returns['SPY'].copy()
    spy.iloc[0] = 0.0
    daily['SPY'] = spy
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError('Nonfinite comparison returns')
    rows = []
    for period, mask in splits(daily.index).items():
        if mask.sum() < 2:
            continue
        dates = daily.index[mask]
        for name in daily.columns:
            if name in signals:
                first_return_pos = returns.index.get_loc(signals[name].index[0]) + 2
                if first_return_pos >= len(returns) or dates[0] < returns.index[first_return_pos]:
                    raise ValueError(f'{name}: {period} overlaps initialization; align evaluation start')
            row = metrics(daily.loc[dates, name])
            row.update(period=period, model=name,
                       annual_volatility=float(daily.loc[dates, name].std(ddof=1)*np.sqrt(252)),
                       annual_turnover=float(runs[name]['turnover'].loc[dates].sum()*252/len(dates))
                       if name in runs else 0.0)
            rows.append(row)
    pd.DataFrame(rows).drop(columns=['name']).to_csv(output / 'summary.csv', index=False)
    daily.to_csv(output / 'daily_returns.csv')
    probability.to_csv(output / 'monthly_probabilities.csv')
    log.to_csv(output / 'regime_log.csv')
    for name, run in runs.items():
        slug = name.lower().replace(' + ', '_').replace(' ', '_')
        signals[name].to_csv(output / f'{slug}_signals.csv')
        run['weights'].to_csv(output / f'{slug}_weights.csv')
        pd.concat([run['net'], run['cost'], run['turnover']], axis=1).to_csv(output / f'{slug}_daily.csv')

    diag = daily.loc['2017-01-01':'2026-09-30']
    if len(diag) > 1:
        wealth = (1+diag).cumprod()
        dd = wealth / wealth.cummax().clip(lower=1) - 1
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for name in daily.columns:
            axes[0].plot(wealth.index, wealth[name], label=name)
            axes[1].plot(dd.index, dd[name], label=name)
        axes[0].set(title='Historical diagnostic — local rerun', ylabel='Wealth (initial = 1)')
        axes[1].set(ylabel='Drawdown')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / 'comparison.png', dpi=180)
        plt.close(fig)

    manifest = dict(
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        matplotlib=matplotlib.__version__,
        data_hashes={name: hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
                     for name in ['sp500_prices.csv','etf_prices.csv']},
        parameters=dict(cost=.0007, gamma=1, top_n=100, momentum_lookback=252,
                        momentum_skip=21, distribution='t', fit_frequency='weekly',
                        rebalance_frequency='monthly', cash_yield=0),
        history=dict(start=str(daily.index.min().date()), end=str(daily.index.max().date())),
        evidence='Local rerun; historical test has previously been observed')
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()

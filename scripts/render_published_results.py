"""Render aggregate evidence only; never invent daily paths from summary tables."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT/'results/published/experiment_summary.csv'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
COLORS = {'Momentum + HMM': '#2066A8', 'Momentum': '#D47A21',
          'Vol control': '#8054A3', 'SPY': '#707B85', 'RA Momentum': '#299172',
          'RA Momentum + HMM': '#8054A3'}


def main():
    table = pd.read_csv(TABLE)
    test = table[table.period.eq('test')].set_index('model')
    names = ['Momentum + HMM','Momentum','Vol control','SPY']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    specs = [('cagr','Annualized return',False), ('max_drawdown','Maximum drawdown magnitude',True),
             ('sharpe','Sharpe (risk-free rate = 0)',False)]
    for ax, (metric,title,absolute) in zip(axes,specs):
        values = test.loc[names, metric].abs() if absolute else test.loc[names,metric]
        bars=ax.barh(names,values,color=[COLORS[n] for n in names],height=.6)
        ax.invert_yaxis(); ax.set_title(title, fontsize=12, pad=14)
        ax.set_xlim(0,float(values.max())*1.28)
        ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
        if metric!='sharpe': ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for bar,val in zip(bars,values):
            label=f'{val:.3f}' if metric=='sharpe' else f'{val:.2%}'
            ax.text(val+values.max()*.025,bar.get_y()+bar.get_height()/2,label,va='center',fontsize=10)
        if ax!=axes[0]: ax.set_yticklabels([])
    fig.suptitle('The price of protection',fontsize=20,ha='left',x=.02,y=.99)
    fig.text(.02,.88,'2017–2026 historical diagnostic • rounded author-reported comparison values',color='#59636E',fontsize=11)
    fig.text(.02,.02,'Current-constituent universe; repeatedly observed historical test. Maximum drawdown shown as a positive loss magnitude.',fontsize=9,color='#59636E')
    fig.tight_layout(rect=[0,.06,1,.85]); fig.savefig(ROOT/'assets/risk_return_tradeoff.png',dpi=180); plt.close(fig)

    fig, axes=plt.subplots(1,3,figsize=(14,4.7))
    groups=['train','val','test']; labels=['Development A\n2010–2012','Development B\n2013–2016','Historical test\n2017–2026']
    for ax,(metric,title) in zip(axes,[('cagr','CAGR'),('sharpe','Sharpe'),('annual_volatility','Annual volatility')]):
        for model,offset in [('Momentum + HMM',-.18),('RA Momentum + HMM',.18)]:
            vals=table[table.model.eq(model)].set_index('period').loc[groups,metric]
            bars=ax.bar([i+offset for i in range(3)],vals,width=.34,label=model,color=COLORS[model])
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,val, f'{val:.2f}' if metric=='sharpe' else f'{val:.1%}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(3),labels,fontsize=9);ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.15)
        ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
        if metric!='sharpe':ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.02,.89), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Does risk-adjusted ranking improve the HMM portfolio?',fontsize=18,x=.02,ha='left')
    fig.text(.02,.02,'Same HMM exposure and defense; stock ranking changes. Rounded screenshot values; no claim of statistical significance.',fontsize=9,color='#59636E')
    fig.tight_layout(rect=[0,.06,1,.81]);fig.savefig(ROOT/'assets/selection_ablation.png',dpi=180);plt.close(fig)
    print('Rendered two published-table figures (no model fitting).')


if __name__=='__main__': main()

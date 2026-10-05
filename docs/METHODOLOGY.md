# Methodology: from observations to realized returns

## 1. Separate the three questions

**Selection:** which stocks have the strongest lagged price momentum? **Exposure:** how much should be allocated to that sleeve? **Execution:** when could those target positions actually have earned returns? The HMM answers the second question, not the first or third.

## 2. Stock signal

At monthly signal date $t$:

$$M_{i,t}=P_{i,t-21}/P_{i,t-273}-1.$$

Retain finite scores with an original quote at $t$, select the largest 100, and assign $u_{i,t}=1/100$ to each. No positive-momentum filter is applied. A company need not have belonged to the index at $t$ if it belongs to the current input universe; this is a known source of historical membership bias. Fewer than 100 eligible stocks produces no order rather than an incomplete target allocation.

## 3. Student-t observation model

For state $j$, location $\mu_j$, scale $\sigma_j$ and shared degrees of freedom $\nu$:

$$f_j(y)=\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\sigma_j}\left(1+\frac{(y-\mu_j)^2}{\nu\sigma_j^2}\right)^{-(\nu+1)/2}.$$

The observation is SPY daily log return multiplied by 100. The conditional variance is

$$\operatorname{Var}(y\mid s=j)=\sigma_j^2\frac{\nu}{\nu-2},\qquad \nu>2.$$

`sig2` stores $\sigma_j^2$. The degree-of-freedom bounds `[2.05, 300]` maintain finite variance and permit near-Gaussian fits. Shared $\nu$ ensures the larger scale identifies the higher-variance state. A heavy tail changes how extreme observations affect inference; it does not guarantee a smooth probability path.

## 4. Filtering, smoothing and EM

With row-stochastic transition matrix $P$ and initial state probabilities $\pi$:

$$\alpha_{1|0}=\pi,\qquad \alpha_{t|t-1}=P^\top\alpha_{t-1|t-1}\quad(t>1),$$

$$\alpha_{t|t,j}=\frac{\alpha_{t|t-1,j}f_j(y_t)}{\sum_k\alpha_{t|t-1,k}f_k(y_t)}.$$

The likelihood is the sum of logs of the filtering normalization constants. Backward smoothing supplies state responsibilities for EM. For Student-t emissions, the expected latent precision is

$$u_{t,j}=\frac{\nu+1}{\nu+(y_t-\mu_j)^2/\sigma_j^2}.$$

Given smoothed responsibilities $\gamma_{t,j}$, the location and scale updates use

$$\mu_j^{new}=\frac{\sum_t\gamma_{t,j}u_{t,j}y_t}{\sum_t\gamma_{t,j}u_{t,j}},\qquad
(\sigma_j^2)^{new}=\frac{\sum_t\gamma_{t,j}u_{t,j}(y_t-\mu_j^{new})^2}{\sum_t\gamma_{t,j}}.$$

Transition probabilities use smoothed pair counts. The initial distribution is updated from the smoothed initial state. Shared $\nu$ is updated by bounded root finding using expected latent precision and log precision.

The returned parameters are re-filtered and sorted consistently. A warm-start attempt is followed by a higher-budget deterministic cold restart on failure. A converged local solution is not a guarantee of the global likelihood maximum.

### What is causal here?

Each weekly fit uses data ending that week. Its endpoint probability is stored for that week only. Monthly allocation uses the latest stored probability no later than the signal date. Smoothing is allowed inside an available-data estimation window; it is not used to rewrite earlier trades. This addresses signal timing, **not** current-universe bias or research selection bias.

On fitting failure, parameters remain at the last valid estimate while all currently available observations are re-filtered. A first failure without a valid parameter set raises an error. Persistent failure would still imply stale parameters, so counts remain important diagnostics.

## 5. Targets and execution

$$q_{i,t}=u_{i,t}(1-p_t)^\gamma,\qquad q_{GLD,t}=q_{CASH,t}=\tfrac12[1-(1-p_t)^\gamma],\quad\gamma=1.$$

| Timestamp | Information / action |
|---|---|
| Signal-day close $t$ | Form target using information available through close $t$ |
| Trading-day $t+1$ | Old holdings earn close-to-close return |
| Close $t+1$ | Trade to target, pay fees |
| Trading-day $t+2$ | New holdings begin earning close-to-close return |

Let beginning-of-day weights be $w$ and asset returns $r$. Gross return and drifted weights are

$$g=w^\top r,\qquad d_i=\frac{w_i(1+r_i)}{1+g}.$$

The self-financing retained wealth fraction solves

$$x=1-c\sum_{i\ne CASH}|xq_i-d_i|.$$

Then $r^{net}=(1+g)x-1$, and bought-plus-sold security notional divided by beginning-of-day wealth is $(1+g)\sum_{i\ne CASH}|xq_i-d_i|$. Thus daily reported fees equal the cost rate times reported turnover. No cash fee or passive-drift turnover is added.

## 6. Data gaps and accounting scope

Stock prices may be carried one session for valuation only; a missing original quote blocks an actual trade. Held-asset nonfinite returns stop the engine. Zero-held assets with missing returns may be ignored. The account is long-only, fully allocated and unlevered; cash yields zero. These rules do not implement delisting proceeds, corporate-action reconciliation, limit-order fills or market impact.

## 7. Metrics and intervals

$$V_t=\prod_{u=1}^{t}(1+r_u),\quad CAGR=V_T^{252/T}-1,$$

$$SR=\sqrt{252}\,\bar r/s_r,\qquad DD_t=\frac{V_t}{\max(1,V_1,\ldots,V_t)}-1.$$

Sharpe uses zero risk-free rate. The notebook's ordinary $t=\sqrt{T}\bar r/s_r$ is not robust to serial dependence and does not establish alpha. Daily observations are not independent crash events.

The numerical core retains `train` 2010–2012, `val` 2013–2016 and `test` 2017–2026-09-30. Train and val are development subperiods rather than different fitting regimes. `val` does not automatically select any parameter. Test remains a descriptive historical diagnostic because it influenced development. Combining the two development slices does not change later signals or returns, but combined statistics must be recomputed from daily returns, not averaged from the table.

"""Research functions extracted from the supplied notebook.

The numerical function bodies retain the submitted implementation; see
results/published/provenance.json for source identity and validation scope.
"""
import math
import numpy as np
import pandas as pd

COST = 0.0007
GAMMA = 1.0

def month_ends(idx):
    idx = pd.DatetimeIndex(idx)
    per = idx.to_period("M")
    out = []
    for p in per.unique():
        out.append(idx[per == p][-1])
    return pd.DatetimeIndex(sorted(out))


def splits(idx):
    return {
        "train": (idx >= "2010-01-01") & (idx < "2013-01-01"),
        "val":   (idx >= "2013-01-01") & (idx < "2017-01-01"),
        "test":  (idx >= "2017-01-01") & (idx < "2026-10-01"),
    }


def _norm_pdf(y, mu, sig):
    return np.exp(-0.5 * ((y - mu) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)


def _digamma(x):
    r = 0.0
    while x < 8.0:
        r -= 1.0 / x; x += 1.0
    inv = 1.0 / x; inv2 = inv * inv
    return r + math.log(x) - 0.5 * inv - inv2 * (1/12 - inv2 * (1/120 - inv2 / 252))


def _t_pdf(y2, mu, sig, nu):
    z = (y2 - mu) / sig
    logc = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    return np.exp(logc - np.log(sig) - (nu + 1) / 2 * np.log1p(z ** 2 / nu))


def _filter_smooth(y, mu, sig2, P, nu, dist, var_floor, xi0):
    """给定参数跑一遍 Hamilton 过滤 + Kim 平滑。返回 (filt, smooth, pred, ll, ok)。"""
    T = len(y)
    sig = np.sqrt(np.maximum(sig2, var_floor))
    if dist == "t":
        eta = _t_pdf(y[:, None], mu[None, :], sig[None, :], nu) + 1e-300
    else:
        eta = _norm_pdf(y[:, None], mu[None, :], sig[None, :]) + 1e-300
    filt = np.zeros((T, 2)); pred = np.zeros((T, 2)); xi = xi0.copy(); ll = 0.0
    for t in range(T):
        # xi0 表示第一期观测对应状态的先验。
        # 第一笔观测直接使用 xi0，之后才进行状态转移。
        pr = xi0.copy() if t == 0 else P.T @ xi
        pred[t] = pr
        num = pr * eta[t]; den = num.sum()
        if den <= 0 or not np.isfinite(den):
            return filt, filt, pred, -np.inf, False
        xi = num / den; filt[t] = xi; ll += np.log(den)
    smooth = np.zeros((T, 2)); smooth[-1] = filt[-1]
    for t in range(T - 2, -1, -1):
        ratio = smooth[t + 1] / np.maximum(pred[t + 1], 1e-300)
        smooth[t] = filt[t] * ((P * ratio[None, :]).sum(axis=1))
        smooth[t] /= smooth[t].sum()
    return filt, smooth, pred, ll, True


def _em_run(y, n_iter, init, var_floor, seed, dist, nu_init):
    """单次 EM。filt/smooth/loglik 用最终参数重过滤得到，与返回参数严格一致。"""
    y = np.asarray(y, dtype=float); T = len(y)
    if init is None:
        order = np.argsort(np.abs(y))
        mu = np.array([y[order[:T//2]].mean(), y[order[T//2:]].mean()])
        sig2 = np.array([y[order[:T//2]].var(), y[order[T//2:]].var()]) + var_floor
        P = np.array([[0.95, 0.05], [0.10, 0.90]]); nu = float(nu_init)
    else:
        mu = init["mu"].copy(); sig2 = init["sig2"].copy(); P = init["P"].copy()
        nu = float(init.get("nu", nu_init))
    nu = min(max(nu, 2.05), 300.0)
    xi0 = np.array(init.get("pi", [0.5, 0.5]) if init is not None else [0.5, 0.5], dtype=float, copy=True,)

    if not np.isfinite(xi0).all() or (xi0 < 0).any() or xi0.sum() <= 0:
        raise ValueError("无效初始状态概率 pi")

    xi0 /= xi0.sum()
    ll_old, converged = -np.inf, False

    filt = smooth = pred = None
    for _ in range(n_iter):
        filt, smooth, pred, ll, ok = _filter_smooth(y, mu, sig2, P, nu, dist, var_floor, xi0)
        if not ok:
            return dict(mu=mu, sig2=sig2, P=P, nu=nu, filt=filt, smooth=filt, loglik=-np.inf, converged=False)
        # 判断当前参数对应的似然是否收敛。
        # 必须放在下一次 M 步更新之前。
        if np.isfinite(ll_old):
            change = ll - ll_old

            if change < -1e-8 * (abs(ll_old) + 1):
                # 出现明显似然下降，不能视作成功收敛。
                break

            if abs(change) < 1e-7 * (abs(ll_old) + 1):
                converged = True
                break
                
        N = smooth.sum(axis=0) + 1e-300
        if dist == "t":
            d = (y[:, None] - mu[None, :]) ** 2 / np.maximum(sig2[None, :], var_floor)
            w = (nu + 1) / (nu + d)
            elog = _digamma((nu + 1) / 2) - np.log((nu + d) / 2)
            sw = smooth * w
            mu = (sw * y[:, None]).sum(axis=0) / (sw.sum(axis=0) + 1e-300)
            sig2 = np.maximum((sw * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / N, var_floor)
            C = float((smooth * (elog - w)).sum()) / T
            def f(v):
                return math.log(v / 2) - _digamma(v / 2) + 1.0 + C
            lo, hi = 2.05, 300.0
            if f(lo) <= 0: nu = lo
            elif f(hi) >= 0: nu = hi
            else:
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if f(mid) > 0: lo = mid
                    else: hi = mid
                nu = 0.5 * (lo + hi)
        else:
            mu = (smooth * y[:, None]).sum(axis=0) / N
            sig2 = np.maximum((smooth * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / N, var_floor)
        num_j = (filt[:-1, :, None] * P[None, :, :] * (smooth[1:] / np.maximum(pred[1:], 1e-300))[:, None, :])
        P = num_j.sum(axis=0) / np.maximum(num_j.sum(axis=(0, 2))[:, None], 1e-300)
        P = np.clip(P, 1e-6, 1 - 1e-6); P /= P.sum(axis=1, keepdims=True)
        xi0 = smooth[0] / smooth[0].sum()

        ll_old = ll
    # 用最终参数重跑一遍过滤/平滑，保证返回概率与返回参数一致
    filt, smooth, pred, ll, ok = _filter_smooth(y, mu, sig2, P, nu, dist, var_floor, xi0)
    if not ok:
        return dict(mu=mu, sig2=sig2, P=P, nu=nu, filt=filt, smooth=filt, loglik=-np.inf, converged=False)
    idx = np.argsort(sig2)

    mu = mu[idx]
    sig2 = sig2[idx]
    P = P[np.ix_(idx, idx)]
    xi0 = xi0[idx]

    filt = filt[:, idx]
    smooth = smooth[:, idx]

    return dict(
        mu=mu,
        sig2=sig2,
        P=P,
        pi=xi0.copy(),
        nu=(nu if dist == "t" else float("inf")),
        filt=filt,
        smooth=smooth,
        loglik=float(ll),
        converged=bool(converged),)


def hamilton_em( y, n_iter=300, init=None, var_floor=1e-8, seed=0, dist="t", nu_init=8.0, ):
    if dist not in ("gauss", "t"):
        raise ValueError("dist 只能是 'gauss' 或 't'")

    y = np.asarray(y, dtype=float)

    if y.ndim != 1 or len(y) < 4 or not np.isfinite(y).all():
        raise ValueError("HMM输入必须是连续、有限的一维收益序列")

    if n_iter < 1:
        raise ValueError("n_iter 必须大于0")

    # 不使用 cand is init 判断预算。
    # 初次拟合也是完整预算；失败后允许一次更充分的冷重启。
    attempts = [ (init, n_iter), (None, max(n_iter, 1000)), ]

    best = None
    errors = []

    for candidate, budget in attempts:
        try:
            out = _em_run( y, budget, candidate, var_floor, seed, dist, nu_init )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            errors.append(str(exc))
            continue

        if np.isfinite(out["loglik"]):
            if best is None or out["loglik"] > best["loglik"]:
                best = out

            if out["converged"]:
                return out

    if best is not None:
        best["converged"] = False
        return best

    # 上层必须明确处理失败，不能把它当成有效模型。
    return dict( converged=False, loglik=-np.inf, error="; ".join(errors) or "全部拟合候选失败", )


def turb_prob(filt, sig2):
    """高波动状态的滤波概率。"""
    return filt[:, int(np.argmax(sig2))]


def week_ends(idx):
    idx = pd.DatetimeIndex(idx)
    per = idx.to_period("W")
    return pd.DatetimeIndex(sorted(idx[per == p][-1] for p in per.unique()))


def hamilton_pstress_weekly( px_etf, dist="t", n_iter=300, return_log=False, ):
    returns = np.log(px_etf["SPY"]).diff().iloc[1:] * 100

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("SPY收益含NaN或无穷值，请先检查行情")

    p_w = pd.Series(np.nan, index=px_etf.index)
    last_good = None
    log_rows = []

    n_fail = 0
    consecutive_failures = 0

    for dt in week_ends(px_etf.index):
        y = returns.loc[:dt].to_numpy()

        if len(y) < 252:
            continue

        out = hamilton_em( y, n_iter=n_iter, init=last_good, dist=dist, )

        good = bool( out["converged"] and np.isfinite(out["loglik"]) )

        if good:
            last_good = { k: np.array(out[k], copy=True) for k in ["mu", "sig2", "P", "pi"] }
            last_good["nu"] = float(out["nu"])
            consecutive_failures = 0
        else:
            n_fail += 1
            consecutive_failures += 1

            if last_good is None:
                raise RuntimeError( f"{dt}: 初始模型未收敛，没有有效历史参数，停止回测" )

        # 成功或失败，都用实际采用的参数过滤截至当前的全部观测。
        # 失败时参数不更新，但新观测必须进入过滤。
        filt, _, _, used_ll, ok = _filter_smooth(
            y,
            last_good["mu"],
            last_good["sig2"],
            last_good["P"],
            last_good["nu"],
            dist,
            1e-8,
            last_good["pi"],
        )

        if not ok:
            raise RuntimeError(f"{dt}: 使用有效参数重新过滤也失败")

        p = float(turb_prob(filt, last_good["sig2"])[-1])
        p_w.loc[dt] = p

        log_rows.append(
            dict(
                date=dt,
                p=p,
                converged=good,
                fallback=not good,
                consecutive_failures=consecutive_failures,
                nu=last_good["nu"],
                attempted_loglik=float(out["loglik"]),
                used_loglik=float(used_ll),
            )
        )

        if len(log_rows) % 25 == 0:
            print( dist, "已完成周数:", len(log_rows), "截至:", dt.date(), "fallback:", n_fail, flush=True, )

    if not log_rows:
        raise ValueError("历史不足，无法生成HMM概率")

    log = pd.DataFrame(log_rows).set_index("date")

    print(f"Hamilton({dist}) fallback {n_fail} 次")

    if return_log:
        return p_w, log

    return p_w


def r4c2_signal_monthly(px_spx, px_etf, gamma=GAMMA):
    me = month_ends(px_spx.index)

    # 保留原动量定义，但禁止隐式前向填充。
    mom = px_spx.pct_change(252, fill_method=None).shift(21)

    sleeve = pd.DataFrame( 0.0, index=me, columns=px_spx.columns, )

    for dt in me:
        score = ( mom.loc[dt] .replace([np.inf, -np.inf], np.nan) .dropna() )

        # 信号日必须存在原始报价。
        score = score[px_spx.loc[dt, score.index].notna()]

        if len(score) >= 100:
            sleeve.loc[dt, score.nlargest(100).index] = 1 / 100

    p_w, regime_log = hamilton_pstress_weekly( px_etf, dist="t", return_log=True, )
    p_w = p_w.dropna()

    p_m = ( p_w.reindex(p_w.index.union(me)) .sort_index() .ffill() .reindex(me) .dropna() )

    # 只有选股完整的月份才发出信号。
    # 暖启动期间保持现金，避免不完整权重或提前配置黄金。
    valid_dates = p_m.index[ np.isclose( sleeve.loc[p_m.index].sum(axis=1).to_numpy(), 1.0, ) ]
    p_m = p_m.loc[valid_dates]

    sig = pd.DataFrame( 0.0, index=valid_dates, columns=list(px_spx.columns) + ["CASH", "GLD"], )

    for dt in valid_dates:
        target_eq = float((1 - p_m.loc[dt]) ** gamma)

        sig.loc[dt, px_spx.columns] = sleeve.loc[dt] * target_eq
        sig.loc[dt, "CASH"] = (1 - target_eq) / 2
        sig.loc[dt, "GLD"] = (1 - target_eq) / 2

    assert np.isfinite(sig.to_numpy()).all()
    assert (sig.to_numpy() >= 0).all()
    assert np.allclose(sig.sum(axis=1), 1.0)

    return sig, p_m, regime_log


def execute( sig, rets, cost=COST, tradable=None, return_details=False, ):
    if not 0 <= cost < 0.01:
        raise ValueError("cost必须是收益比例，例如7bps=0.0007")

    if "CASH" not in rets.columns:
        raise ValueError("收益表必须包含显式CASH列")

    if not rets.index.is_unique or not rets.index.is_monotonic_increasing:
        raise ValueError("收益日期必须唯一且升序")

    if not sig.index.is_unique:
        raise ValueError("信号日期重复")

    if not sig.index.isin(rets.index).all():
        raise ValueError("信号日期不在收益日历中")

    if set(sig.columns) != set(rets.columns):
        raise ValueError("信号与收益资产列不一致")

    targets = sig.reindex(index=rets.index, columns=rets.columns)

    partial = targets.notna().any(axis=1) & ~targets.notna().all(axis=1)
    if partial.any():
        raise ValueError("信号必须整行有效，或整行NaN表示不交易")

    valid = targets.dropna(how="all")
    values = valid.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("目标权重含非有限值")

    if (values < 0).any():
        raise ValueError("禁止负权重、做空或负现金")

    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("完整目标权重必须合计为1")

    r = rets.to_numpy(dtype=float)
    target_array = targets.to_numpy(dtype=float)

    quotes = ( None if tradable is None else tradable.reindex_like(rets).fillna(False).to_numpy(bool) )

    n, k = r.shape
    cash_col = rets.columns.get_loc("CASH")
    securities = np.arange(k) != cash_col

    # 所有现金统一在CASH列，不再另设外部cash余额。
    w = np.zeros(k)
    w[cash_col] = 1.0

    held = np.zeros((n, k))
    net = np.zeros(n)
    turnover = np.zeros(n)
    fees = np.zeros(n)

    for t in range(n):
        held[t] = w

        missing = ~np.isfinite(r[t])
        bad_held = missing & (w > 1e-12)

        if bad_held.any():
            names = rets.columns[bad_held].tolist()
            raise ValueError( f"{rets.index[t]}: 持有资产无法估值 {names}" )

        # 仅零持仓资产的缺失收益可以忽略。
        daily = np.where(missing, 0.0, r[t])

        gross = float(w @ daily)
        if 1 + gross <= 0:
            raise ValueError("组合净值非正")

        drift = w * (1 + daily) / (1 + gross)
        retained_nav = 1.0

        # 昨日收盘信号，今日收盘执行。
        if t > 0 and np.isfinite(target_array[t - 1]).all():
            desired = target_array[t - 1]

            # 自融资费用方程：
            # x = 1 - cost * sum(abs(x*目标证券权重 - 当前证券权重))
            # CASH不属于收费证券。
            for _ in range(100):
                next_x = 1 - cost * np.abs( retained_nav * desired[securities] - drift[securities] ).sum()

                if abs(next_x - retained_nav) < 1e-14:
                    retained_nav = next_x
                    break

                retained_nav = next_x
            else:
                raise RuntimeError("费用方程未收敛")

            trades = np.abs(retained_nav * desired - drift)

            if quotes is not None:
                unavailable = ( securities & (trades > 1e-12) & ~quotes[t] )
                if unavailable.any():
                    names = rets.columns[unavailable].tolist()
                    raise ValueError( f"{rets.index[t]}: 成交报价缺失 {names}" )

            # 实际证券买卖金额相对于当日期初净值。
            turnover[t] = (1 + gross) * trades[securities].sum()
            w = desired.copy()
        else:
            w = drift

        fees[t] = (1 + gross) * (1 - retained_nav)
        net[t] = (1 + gross) * retained_nav - 1

    result = dict(
        net=pd.Series(net, index=rets.index, name="net"),
        weights=pd.DataFrame(held, index=rets.index, columns=rets.columns),
        turnover=pd.Series(turnover, index=rets.index, name="turnover"),
        cost=pd.Series(fees, index=rets.index, name="cost"),
    )

    if return_details:
        return result

    return result["net"], result["weights"]


def metrics(net, name=""):
    if len(net) < 2 or not np.isfinite(net.to_numpy()).all():
        raise ValueError("指标计算需要至少两个有效收益")

    eq = (1 + net).cumprod()

    # 包含期初净值1，避免遗漏区间第一天的损失。
    peak = np.maximum.accumulate( np.r_[1.0, eq.to_numpy()] )[1:]

    sd = float(net.std(ddof=1))
    sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan

    return dict(
        name=name,
        sharpe=sharpe,
        cagr=float(eq.iloc[-1] ** (252 / len(net)) - 1),
        maxdd=float((eq.to_numpy() / peak - 1).min()),
        t=float(net.mean() / sd * np.sqrt(len(net))) if sd > 0 else np.nan,
    )
